//! The configured optimizer actually configures the optimizer.
//!
//! `[training.optimizer]` names beta1, beta2, epsilon and weight_decay, and
//! until this was fixed none of them reached the optimizer: build_optimizer
//! called AdamWConfig::new() and set only the gradient-clipping threshold. The
//! A100 pilot trained at Burn's defaults for all four, nothing failed, nothing
//! warned, and the only trace was a model that learned more slowly than its own
//! configuration said it should.
//!
//! Creating a trainer with a configuration and observing no error proves
//! nothing, because the old code did that too. Each test below runs steps and
//! compares the loss, which is where a silently ignored setting shows up.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use burn::tensor::{Int, Tensor, TensorData};
use llm_core::{ModelConfig, OptimizerConfig};
use llm_model::{Transformer, all_positions, next_token_pairs};
use llm_training::{TokenBatch, Trainer, WarmupCosineSchedule};

const VOCABULARY: usize = 64;
const SEQUENCE: usize = 16;
const BATCH: usize = 2;
const LEARNING_RATE: f64 = 0.05;

/// One process-wide random source, and cargo runs a binary's tests as threads.
/// Two tests that seed and draw would otherwise interleave.
static RANDOM_SOURCE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    RANDOM_SOURCE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn tiny_config() -> ModelConfig {
    ModelConfig {
        vocab_size: VOCABULARY,
        hidden_size: 16,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        ffn_intermediate_size: 32,
        max_sequence_length: 32,
        ..ModelConfig::default()
    }
}

fn schedule() -> WarmupCosineSchedule {
    WarmupCosineSchedule::new(LEARNING_RATE, 0, 100).expect("a valid schedule")
}

/// A fixed batch, so two runs see identical data.
fn batch(device: &burn::tensor::Device) -> TokenBatch {
    let ids: Vec<i32> = (0..BATCH * (SEQUENCE + 1))
        .map(|index| i32::try_from(index * 7 % VOCABULARY).expect("a small index"))
        .collect();
    let window = Tensor::<2, Int>::from_data(TensorData::new(ids, [BATCH, SEQUENCE + 1]), device);
    let (input, target) = next_token_pairs(window).expect("more than one token");

    TokenBatch {
        input,
        target,
        mask: all_positions(BATCH, SEQUENCE, device),
    }
}

/// Trains a fresh, identically seeded model and returns the per-step losses.
fn losses_after(config: &OptimizerConfig, steps: usize) -> Vec<f32> {
    let _guard = exclusive();

    let device = burn::tensor::Device::flex().autodiff();
    llm_model::init::seed(&device, 2026);
    let mut model = Transformer::new(&tiny_config(), &device).expect("a valid tiny config");
    let mut trainer = Trainer::new(schedule(), Some(1.0), config);
    let batches = [batch(&device)];

    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (updated, metrics) = trainer
            .step(model, &batches)
            .expect("a finite step on a tiny model");
        model = updated;
        losses.push(metrics.train_loss);
    }
    losses
}

fn worst_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "same number of steps");
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn beta2_reaches_the_optimizer() {
    let at_095 = losses_after(
        &OptimizerConfig {
            beta2: 0.95,
            ..OptimizerConfig::default()
        },
        6,
    );
    let at_0999 = losses_after(
        &OptimizerConfig {
            beta2: 0.999,
            ..OptimizerConfig::default()
        },
        6,
    );

    let difference = worst_difference(&at_095, &at_0999);
    assert!(
        difference > 1e-6,
        "beta2 0.95 and 0.999 produced losses differing by only {difference:e}, \
         which means the configured value is not reaching the optimizer"
    );
}

#[test]
fn weight_decay_reaches_the_optimizer() {
    let none = losses_after(
        &OptimizerConfig {
            weight_decay: 0.0,
            ..OptimizerConfig::default()
        },
        6,
    );
    let heavy = losses_after(
        &OptimizerConfig {
            weight_decay: 0.5,
            ..OptimizerConfig::default()
        },
        6,
    );

    let difference = worst_difference(&none, &heavy);
    assert!(
        difference > 1e-6,
        "weight_decay 0.0 and 0.5 produced losses differing by only {difference:e}"
    );
}

#[test]
fn epsilon_reaches_the_optimizer() {
    let small = losses_after(
        &OptimizerConfig {
            epsilon: 1e-12,
            ..OptimizerConfig::default()
        },
        6,
    );
    let large = losses_after(
        &OptimizerConfig {
            epsilon: 1e-1,
            ..OptimizerConfig::default()
        },
        6,
    );

    let difference = worst_difference(&small, &large);
    assert!(
        difference > 1e-6,
        "epsilon 1e-12 and 1e-1 produced losses differing by only {difference:e}"
    );
}

#[test]
fn beta1_reaches_the_optimizer() {
    let low = losses_after(
        &OptimizerConfig {
            beta1: 0.5,
            ..OptimizerConfig::default()
        },
        6,
    );
    let high = losses_after(
        &OptimizerConfig {
            beta1: 0.99,
            ..OptimizerConfig::default()
        },
        6,
    );

    let difference = worst_difference(&low, &high);
    assert!(
        difference > 1e-6,
        "beta1 0.5 and 0.99 produced losses differing by only {difference:e}"
    );
}

#[test]
fn identical_configurations_produce_identical_losses() {
    // The control. Without it the tests above would pass if something else made
    // every run differ: an unseeded model, a random batch, a nondeterministic
    // step.
    let first = losses_after(&OptimizerConfig::default(), 6);
    let second = losses_after(&OptimizerConfig::default(), 6);

    assert_eq!(
        worst_difference(&first, &second),
        0.0,
        "two runs of the same configuration must be bit-identical, or the other \
         tests in this file prove nothing"
    );
}
