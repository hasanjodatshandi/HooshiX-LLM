//! The training loop.
//!
//! This is the composition root: it decides *what* runs and in what order, and
//! every decision below it — how a batch is built, what a step does, what a
//! checkpoint contains — belongs to the crate that owns that concern.
//!
//! # What it does not do
//!
//! It does not implement the loop. [`Trainer`] owns the forward pass, the
//! objective, the backward pass, gradient accumulation and the optimizer step.
//! What is here is the order of operations around it: read the dataset, build the
//! model, step, evaluate, checkpoint, report.

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use burn::tensor::Device;
use llm_checkpoint::{CHECKPOINT_FORMAT_VERSION, CheckpointMetadata, DataCursor};
use llm_core::ExperimentConfig;
use llm_data::{DatasetManifest, Split};
use llm_model::Transformer;
use llm_tokenizer::LlmTokenizer;
use llm_training::{Trainer, WarmupCosineSchedule};

use crate::dataset::ShardBatches;
use crate::error::TrainError;

/// How many validation batches a run holds aside for evaluation.
///
/// Pulled once, so every evaluation reads the same held-out sequences and the
/// numbers are comparable across steps. Reading fresh ones would make the
/// validation curve move for reasons that have nothing to do with the model.
const VALIDATION_BATCHES: usize = 4;

/// What a run did.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// Optimizer steps completed.
    pub steps: u64,
    /// Tokens consumed by completed steps.
    pub tokens: u64,
    /// Loss on the first recorded step.
    pub first_loss: f32,
    /// Best training loss seen.
    pub best_loss: f32,
    /// Loss on the last recorded step.
    pub final_loss: f32,
    /// Steps whose loss was not finite and were refused.
    pub non_finite_losses: usize,
    /// Steps abandoned rather than applied.
    pub skipped_steps: usize,
    /// Passes the run made over the training shard.
    pub train_passes: u64,
    /// Sequences in the training shard.
    pub train_sequences: u64,
    /// Checkpoints written, in order.
    pub checkpoints: Vec<PathBuf>,
    /// Validation loss by step, where the interval landed.
    pub evaluations: Vec<(u64, f32)>,
}

impl RunReport {
    /// Whether the run stayed numerically stable.
    ///
    /// The criterion is deliberately strict: one refused step means a loss went
    /// non-finite, and a run that produced one is not a run that is fine.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.non_finite_losses == 0 && self.skipped_steps == 0 && self.final_loss.is_finite()
    }
}

/// Runs the configured training job.
pub fn run(config: &ExperimentConfig, device: &Device) -> Result<RunReport, TrainError> {
    let artifact = config
        .tokenizer
        .artifact
        .as_ref()
        .ok_or(TrainError::TokenizerArtifactMissing)?;
    let tokenizer = LlmTokenizer::load(artifact, &config.tokenizer)?;
    let identity = tokenizer.identity_hash();

    let data_root = PathBuf::from(&config.data.data_root);
    let manifest_path = data_root.join(DatasetManifest::file_name());
    if !manifest_path.is_file() {
        return Err(TrainError::DatasetMissing {
            path: manifest_path,
        });
    }
    let manifest = DatasetManifest::load(&manifest_path)?;

    if manifest.tokenizer_identity != identity {
        return Err(TrainError::TokenizerIdentityMismatch {
            manifest: manifest.tokenizer_identity,
            tokenizer: identity,
        });
    }

    let sequence_length = config.data.sequence_length;
    let micro_batch = config.data.micro_batch_size;
    let accumulation = config.training.gradient_accumulation_steps as usize;

    let mut train = open_split(
        &manifest,
        &data_root,
        Split::Train,
        &identity,
        sequence_length,
        device,
    )?;
    let mut validation = open_split(
        &manifest,
        &data_root,
        Split::Validation,
        &identity,
        sequence_length,
        device,
    )?;
    let mut validation_batches = Vec::with_capacity(VALIDATION_BATCHES);
    for _ in 0..VALIDATION_BATCHES {
        validation_batches.push(validation.next_batch(micro_batch)?);
    }

    // Seeded before anything is constructed, which is what makes a run
    // reproducible from its configuration. The model does not seed itself —
    // a module that reseeded would silently reset the run's state — so the
    // composition root is the only place this can happen, and forgetting it
    // produces two runs that differ in their first loss and nowhere obvious.
    llm_model::init::seed(device, config.training.seed);

    // The model is constructed before the loop and never rebuilt: a step returns
    // new parameter tensors, and losing that chain would silently restart
    // training from random weights.
    let model = Transformer::new(&config.model, device)?;
    let schedule = WarmupCosineSchedule::new(
        config.training.optimizer.learning_rate,
        config.training.scheduler.warmup_steps,
        config.training.scheduler.cosine_iters,
    )?;
    let mut trainer = Trainer::new(
        schedule,
        config.training.gradient_clip_global_norm,
        &config.training.optimizer,
    );

    let checkpoint_root = data_root.parent().unwrap_or(&data_root).join("checkpoints");
    let dataset_hash = manifest.content_hash()?;

    // Checked before the first step rather than at the first checkpoint. The
    // publication guard would catch it either way, but twenty-five steps into a
    // run is a poor place to learn that the output directory is occupied.
    if checkpoint_root
        .read_dir()
        .is_ok_and(|mut entries| entries.next().is_some())
    {
        return Err(TrainError::CheckpointRootNotEmpty {
            path: checkpoint_root,
        });
    }

    let mut model = model;
    let mut checkpoints = Vec::new();
    let mut evaluations = Vec::new();
    let mut batches = Vec::with_capacity(accumulation);
    let started = Instant::now();

    // Progress is reported about a thousand times over a run, so a short run still
    // shows a curve and a long one does not flood the log. A configured interval
    // would be a second thing to keep in sync with `max_steps` for no gain.
    //
    // A thousand rather than a hundred because of what the interval means in wall
    // clock: at the TinyStories configuration a step is about 5.5 s, so a
    // hundredth of a 20,000-step run is eighteen minutes of silence between lines,
    // which is indistinguishable from a hang to anyone watching. A thousandth is
    // under two minutes.
    let log_every = (config.training.max_steps / 1000).max(1);

    for _ in 0..config.training.max_steps {
        batches.clear();
        for _ in 0..accumulation {
            batches.push(train.next_batch(micro_batch)?);
        }

        let (updated, metrics) = trainer.step(model, &batches)?;
        model = updated;

        // A run reports as it goes. The configured run is measured in days, and a
        // process that prints nothing until it finishes cannot be told apart from
        // one that has hung — which is the state a long run spends most of its
        // life in.
        if metrics.step.is_multiple_of(log_every) {
            let elapsed = started.elapsed().as_secs_f64();
            let tokens = trainer.log().tokens_seen();
            let rate = if elapsed > 0.0 {
                tokens as f64 / elapsed
            } else {
                0.0
            };
            let per_step = elapsed / metrics.step.max(1) as f64;
            let remaining = config.training.max_steps.saturating_sub(metrics.step) as f64;
            println!(
                "llm-train: step {:>7}/{}  loss {:>8.4}  lr {:.3e}  {:.0} tok/s  elapsed {:.0}m  eta {:.0}m",
                metrics.step,
                config.training.max_steps,
                metrics.train_loss,
                metrics.learning_rate,
                rate,
                elapsed / 60.0,
                remaining * per_step / 60.0,
            );
        }
        if metrics
            .step
            .is_multiple_of(config.training.evaluation_interval_steps)
        {
            let loss = trainer.validate(&model, &validation_batches)?;
            evaluations.push((metrics.step, loss));

            // Printed as it happens, not only in the final report.
            //
            // A validation curve nobody can see until the run ends cannot tell
            // anyone to stop, and on the TinyStories run it could not: the curve
            // was monotone all the way down and the run had been configured for
            // exactly the number of steps it took, so nothing was lost. That was
            // luck. A run that starts to memorise shows it here first, and the
            // reader has to be able to see it while there is still time to act.
            //
            // The gap is printed rather than left to be worked out, because the
            // gap is the quantity that answers the question: negative and falling
            // means still learning, positive and rising means memorising. The
            // training figure is one step's loss and is noisy by construction;
            // the validation figure is an average over a fixed set and is not.
            println!(
                "llm-train: validation step {:>7}/{}  loss {:>8.4}  train {:>8.4}  gap {:+8.4}",
                metrics.step,
                config.training.max_steps,
                loss,
                metrics.train_loss,
                loss - metrics.train_loss,
            );
        }

        if metrics
            .step
            .is_multiple_of(config.training.checkpoint_interval_steps)
        {
            let name = format!("checkpoint-{:010}", metrics.step);
            let metadata = CheckpointMetadata {
                format_version: CHECKPOINT_FORMAT_VERSION,
                created_unix_seconds: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_secs()),
                global_step: metrics.step,
                tokens_seen: trainer.log().tokens_seen(),
                data_cursor: DataCursor {
                    epoch: train.passes(),
                    shard: 0,
                    sequence: 0,
                },
                config_hash: String::new(),
                tokenizer_identity: identity.clone(),
                dataset_manifest_hash: dataset_hash.clone(),
            };
            let path = llm_checkpoint::save(
                &checkpoint_root,
                &name,
                &model,
                trainer.optimizer(),
                &metadata,
            )?;
            checkpoints.push(path);
        }
    }

    let log = trainer.log();
    let best = log.best_train_loss().unwrap_or(f32::NAN);
    let first = log
        .steps()
        .first()
        .map_or(f32::NAN, |entry| entry.train_loss);
    let final_loss = log.last().map_or(f32::NAN, |entry| entry.train_loss);

    Ok(RunReport {
        steps: trainer.steps(),
        tokens: log.tokens_seen(),
        first_loss: first,
        best_loss: best,
        final_loss,
        non_finite_losses: log.non_finite_losses(),
        skipped_steps: log.skipped_steps(),
        train_passes: train.passes(),
        train_sequences: train.sequence_count(),
        checkpoints,
        evaluations,
    })
}

/// Opens the shard a split needs, reporting a missing split by name.
fn open_split(
    manifest: &DatasetManifest,
    data_root: &std::path::Path,
    split: Split,
    identity: &str,
    sequence_length: usize,
    device: &Device,
) -> Result<ShardBatches, TrainError> {
    let summary = manifest
        .shards
        .iter()
        .find(|shard| shard.split == split)
        .ok_or(TrainError::SplitMissing {
            split: split.as_str(),
        })?;

    ShardBatches::open(
        data_root.join(&summary.file_name),
        identity,
        sequence_length,
        device,
    )
}
