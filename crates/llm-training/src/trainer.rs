//! The training step and the loop around it.
//!
//! One step is: forward each micro-batch, average their objectives, check the
//! result is finite, backward, clip, and apply the optimizer at the learning
//! rate the schedule names for that step.
//!
//! # Accumulation
//!
//! Micro-batches are accumulated by summing their objectives and taking one
//! backward pass over the sum, rather than by summing per-parameter gradients
//! from several backward passes. For the gradient the two are identical, because
//! the gradient of a sum is the sum of the gradients; the difference is that the
//! sum keeps one graph alive, so it does not reduce peak memory. It is the right
//! trade at Stage 1 scale, where the reason to accumulate is to reach a larger
//! effective batch and not to fit one that is too large to hold.

use std::time::Instant;

use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, ModuleOptimizer};
use burn::tensor::{Bool, Int, Tensor};
use llm_model::{ModelError, NumericalQuantity, Transformer, causal_lm_loss, ensure_finite};

use crate::error::{NonFiniteLoss, TrainingError};
use crate::metrics::{MetricsLog, StepMetrics};
use crate::scheduler::WarmupCosineSchedule;

/// A fixed batch of token ids and the positions that contribute to the loss.
#[derive(Debug)]
pub struct TokenBatch {
    /// Token ids the model reads, `[batch, sequence]`.
    pub input: Tensor<2, Int>,
    /// Token ids the model must predict, `[batch, sequence]`.
    pub target: Tensor<2, Int>,
    /// Positions that contribute to the loss.
    pub mask: Tensor<2, Bool>,
}

impl TokenBatch {
    /// Number of token positions the batch carries.
    #[must_use]
    pub fn tokens(&self) -> u64 {
        let [batch, sequence] = self.input.dims();
        batch as u64 * sequence as u64
    }
}

/// Held-out batches and how often to evaluate them.
#[derive(Debug, Clone, Copy)]
pub struct ValidationSchedule<'a> {
    /// Batches the model is evaluated on.
    pub batches: &'a [TokenBatch],
    /// Evaluate every this many steps.
    pub interval_steps: u64,
}

/// Applies scheduled optimizer steps to a model and records what happened.
pub struct Trainer {
    optimizer: ModuleOptimizer,
    schedule: WarmupCosineSchedule,
    log: MetricsLog,
    steps: u64,
    gradient_clip_global_norm: Option<f64>,
}

/// Builds the Stage-1 optimizer: AdamW, optionally clipping by global norm.
///
/// Extracted because a resume needs a second, empty optimizer of exactly the
/// same configuration to load the stored state into.
fn build_optimizer(gradient_clip_global_norm: Option<f64>) -> ModuleOptimizer {
    let mut config = AdamWConfig::new();
    if let Some(norm) = gradient_clip_global_norm {
        // Burn's clipping threshold is `f32`; the configured value is an
        // ordinary, validated magnitude.
        config = config.with_grad_clipping(Some(GradientClippingConfig::Norm(norm as f32)));
    }
    config.init()
}

impl Trainer {
    /// Creates a trainer.
    ///
    /// AdamW is the Stage-1 optimizer; no variant is introduced until this
    /// baseline is stable, which is why the choice is not a parameter. Gradient
    /// clipping by global norm is applied by the optimizer itself, so a clipped
    /// step cannot be forgotten.
    #[must_use]
    pub fn new(schedule: WarmupCosineSchedule, gradient_clip_global_norm: Option<f64>) -> Self {
        Self {
            optimizer: build_optimizer(gradient_clip_global_norm),
            schedule,
            log: MetricsLog::new(),
            steps: 0,
            gradient_clip_global_norm,
        }
    }

    /// The schedule this trainer follows.
    #[must_use]
    pub const fn schedule(&self) -> WarmupCosineSchedule {
        self.schedule
    }

    /// What the run has measured so far.
    #[must_use]
    pub const fn log(&self) -> &MetricsLog {
        &self.log
    }

    /// The optimizer, so its state can be written to a checkpoint.
    ///
    /// AdamW keeps a moment estimate per parameter. A checkpoint that omits it
    /// produces a run that looks resumed but is not.
    #[must_use]
    pub const fn optimizer(&self) -> &ModuleOptimizer {
        &self.optimizer
    }

    /// Restores the optimizer state and the step counter from a checkpoint.
    ///
    /// Both are needed. Burn restores optimizer state into an existing
    /// optimizer, so an empty one is built from this trainer's own
    /// configuration; and the schedule is a function of the step number, so a
    /// trainer resumed at step 500 must report 500 or it will restart the
    /// learning-rate curve from the beginning.
    ///
    /// The metrics history is deliberately not restored: it is a record of what
    /// this process observed, and the checkpoint is not a log.
    pub fn resume_from(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<llm_checkpoint::CheckpointMetadata, llm_checkpoint::CheckpointError> {
        let metadata = llm_checkpoint::validate(&path)?;
        let blank = build_optimizer(self.gradient_clip_global_norm);
        self.optimizer = llm_checkpoint::load_optimizer(&path, blank)?;
        self.steps = metadata.global_step;
        self.log = MetricsLog::new();
        Ok(metadata)
    }

    /// Number of steps completed, including any that were abandoned.
    ///
    /// Counted rather than read off the log, because a resumed trainer is told
    /// its step by the checkpoint and has no history to count.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.steps
    }

    /// Runs one optimizer step over a group of micro-batches.
    ///
    /// Returns the updated model and what the step measured. The model is taken
    /// and returned by value because an optimizer step produces new parameter
    /// tensors rather than mutating the old ones.
    ///
    /// A step whose loss is not finite is **refused**: the model is returned
    /// unchanged, no optimizer step is applied, and the failure is recorded as a
    /// skipped step rather than hidden.
    pub fn step(
        &mut self,
        model: Transformer,
        batches: &[TokenBatch],
    ) -> Result<(Transformer, StepMetrics), TrainingError> {
        if batches.is_empty() {
            return Err(TrainingError::NoBatches);
        }

        let index = self.steps();
        let learning_rate = self.schedule.learning_rate_at(index);
        let tokens: u64 = batches.iter().map(TokenBatch::tokens).sum();
        let started = Instant::now();

        let objective = accumulated_loss(&model, batches);
        let objective = match objective {
            Ok(objective) => objective,
            Err(error) => {
                let metrics = StepMetrics {
                    step: index + 1,
                    learning_rate,
                    train_loss: f32::NAN,
                    validation_loss: None,
                    gradient_norm: None,
                    tokens,
                    step_millis: elapsed_millis(started),
                    skipped: true,
                };
                self.log.record(metrics);
                self.steps += 1;
                return Err(error);
            }
        };

        let gradients = objective.backward();
        let gradients = burn::optim::GradientsParams::from_grads(gradients, &model);
        let updated = self.optimizer.step(learning_rate, model, gradients);

        let metrics = StepMetrics {
            step: index + 1,
            learning_rate,
            train_loss: measure(&objective).unwrap_or(f32::NAN),
            validation_loss: None,
            // Burn applies the clipping but does not report the norm it saw.
            // Reporting it needs a per-parameter gradient iteration path that
            // does not exist yet; the field is present so that a run's record
            // does not change shape when it does.
            gradient_norm: None,
            tokens,
            step_millis: elapsed_millis(started),
            skipped: false,
        };
        self.log.record(metrics);
        self.steps += 1;

        Ok((updated, metrics))
    }

    /// Runs `steps` optimization steps over a fixed group of batches.
    pub fn fit(
        &mut self,
        model: Transformer,
        batches: &[TokenBatch],
        steps: u64,
    ) -> Result<(Transformer, MetricsLog), TrainingError> {
        self.fit_with_validation(model, batches, steps, None)
    }

    /// Runs `steps` optimization steps, evaluating held-out batches periodically.
    ///
    /// The validation loss is attached to the step that triggered it, so the
    /// history stays one row per optimizer step and a training curve and a
    /// validation curve can be plotted against the same axis.
    pub fn fit_with_validation(
        &mut self,
        mut model: Transformer,
        batches: &[TokenBatch],
        steps: u64,
        validation: Option<ValidationSchedule<'_>>,
    ) -> Result<(Transformer, MetricsLog), TrainingError> {
        for _ in 0..steps {
            let (updated, _) = self.step(model, batches)?;
            model = updated;

            if let Some(schedule) = &validation
                && schedule.interval_steps > 0
                && self.steps().is_multiple_of(schedule.interval_steps)
            {
                let loss = self.validate(&model, schedule.batches)?;
                if let Some(entry) = self.log.latest_mut() {
                    entry.validation_loss = Some(loss);
                }
            }
        }
        Ok((model, self.log.clone()))
    }

    /// Evaluates held-out batches without touching the optimizer.
    pub fn validate(
        &self,
        model: &Transformer,
        batches: &[TokenBatch],
    ) -> Result<f32, TrainingError> {
        evaluate(model, batches)
    }
}

/// Mean loss over a group of batches, with the finiteness check applied.
///
/// The mean, not the sum, so that the number a run reports does not depend on
/// how many micro-batches were grouped into a step.
fn accumulated_loss(
    model: &Transformer,
    batches: &[TokenBatch],
) -> Result<Tensor<1>, TrainingError> {
    let mut total: Option<Tensor<1>> = None;

    for batch in batches {
        let logits = model.forward(batch.input.clone(), 0)?;
        let loss = causal_lm_loss(logits, batch.target.clone(), batch.mask.clone())?;

        // The finiteness check happens before the backward pass: a non-finite
        // gradient would be written into every parameter by the step that
        // follows, and the corruption only becomes visible much later.
        if let Err(error) = ensure_finite(&loss, NumericalQuantity::Loss) {
            return Err(match error {
                ModelError::NonFinite { quantity } => NonFiniteLoss { quantity }.into(),
                other => other.into(),
            });
        }

        total = Some(match total {
            Some(sum) => sum + loss,
            None => loss,
        });
    }

    let count = batches.len() as f64;
    total.map(|sum| sum / count).ok_or(TrainingError::NoBatches)
}

/// Mean loss over a group of batches, without touching the optimizer.
///
/// This is the validation and evaluation path: a forward pass, a finiteness
/// check, and no gradient.
pub fn evaluate(model: &Transformer, batches: &[TokenBatch]) -> Result<f32, TrainingError> {
    if batches.is_empty() {
        return Err(TrainingError::NoBatches);
    }

    let mut total = 0.0_f64;
    for batch in batches {
        let logits = model.forward(batch.input.clone(), 0)?;
        let loss = causal_lm_loss(logits, batch.target.clone(), batch.mask.clone())?;

        if let Err(error) = ensure_finite(&loss, NumericalQuantity::Loss) {
            return Err(match error {
                ModelError::NonFinite { quantity } => NonFiniteLoss { quantity }.into(),
                other => other.into(),
            });
        }

        total += f64::from(measure(&loss).ok_or(NonFiniteLoss {
            quantity: NumericalQuantity::Loss,
        })?);
    }

    Ok((total / batches.len() as f64) as f32)
}

/// Reads a single-element loss tensor back as `f32`.
fn measure(loss: &Tensor<1>) -> Option<f32> {
    loss.clone()
        .to_data()
        .try_to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
}

/// Milliseconds elapsed since `started`.
fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

impl std::fmt::Debug for Trainer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Trainer")
            .field("schedule", &self.schedule)
            .field("steps", &self.steps())
            .finish_non_exhaustive()
    }
}
