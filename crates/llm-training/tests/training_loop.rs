//! The validation loop, and the accounting that goes with it.
//!
//! These are the parts of the training loop that are not the optimizer: an
//! evaluation that does not touch gradients, and the periodic driver that
//! attaches its result to the step which triggered it.

// Test helpers may abort on failure: an unreadable fixture or an invalid model
// means the test cannot run, and the panic message names the cause. `clippy.toml`
// only relaxes the panic family for `#[test]` functions and `#[cfg(test)]`
// modules, which does not reach helpers in an integration test binary. The
// workspace-level denies stay in force for `src/`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::PathBuf;

use burn::tensor::{Int, Tensor, TensorData};
use llm_core::ModelConfig;
use llm_model::{Transformer, all_positions, next_token_pairs};
use llm_training::{
    TokenBatch, Trainer, TrainingError, ValidationSchedule, WarmupCosineSchedule, evaluate,
};

const VOCABULARY: usize = 256;
const SEQUENCE: usize = 24;
const LEARNING_RATE: f64 = 0.01;

fn corpus_ids() -> Vec<u32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("overfit-corpus.txt");
    std::fs::read(path)
        .expect("the fixture must be readable")
        .into_iter()
        .map(u32::from)
        .collect()
}

fn tiny_config() -> ModelConfig {
    ModelConfig {
        vocab_size: VOCABULARY,
        hidden_size: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        ffn_intermediate_size: 64,
        max_sequence_length: 128,
        ..ModelConfig::default()
    }
}

/// A batch starting at `offset` in the corpus.
fn batch_at(offset: usize, device: &burn::tensor::Device) -> TokenBatch {
    let ids = corpus_ids();
    let window: Vec<i32> = ids[offset..offset + SEQUENCE + 1]
        .iter()
        .map(|id| *id as i32)
        .collect();
    let tokens = Tensor::<2, Int>::from_data(TensorData::new(window, [1, SEQUENCE + 1]), device);
    let (input, target) = next_token_pairs(tokens).expect("more than one token");

    TokenBatch {
        input,
        target,
        mask: all_positions(1, SEQUENCE, device),
    }
}

fn schedule() -> WarmupCosineSchedule {
    WarmupCosineSchedule::new(LEARNING_RATE, 0, 1000).expect("a valid schedule")
}

#[test]
fn evaluation_produces_a_finite_loss_without_training() {
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");
    let held_out = [batch_at(0, &device)];

    let loss = evaluate(&model, &held_out).expect("the batch is usable");
    assert!(
        loss.is_finite(),
        "the validation loss must be finite: {loss}"
    );
    assert!(loss > 0.0, "a fresh model must not score zero: {loss}");
}

#[test]
fn evaluation_averages_over_every_held_out_batch() {
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");

    let first = batch_at(0, &device);
    let second = batch_at(40, &device);

    let single_first = evaluate(&model, std::slice::from_ref(&first)).expect("usable");
    let single_second = evaluate(&model, std::slice::from_ref(&second)).expect("usable");
    let both = evaluate(&model, &[first, second]).expect("usable");

    let expected = (single_first + single_second) / 2.0;
    assert!(
        (both - expected).abs() < 1e-5,
        "expected the mean {expected}, got {both}"
    );
}

#[test]
fn evaluation_refuses_an_empty_group() {
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");

    assert!(matches!(
        evaluate(&model, &[]),
        Err(TrainingError::NoBatches)
    ));
}

#[test]
fn evaluation_does_not_change_the_parameters() {
    // The whole point of a validation pass is that it is an observation. A
    // backward pass here would silently train on the held-out set.
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");
    let before = model
        .embedding_weight()
        .to_data()
        .try_to_vec::<f32>()
        .expect("f32");

    let held_out = [batch_at(0, &device)];
    let _ = evaluate(&model, &held_out).expect("usable");

    let after = model
        .embedding_weight()
        .to_data()
        .try_to_vec::<f32>()
        .expect("f32");
    assert_eq!(before, after, "evaluation must not modify the model");
}

#[test]
fn the_validation_loss_is_attached_to_the_step_that_triggered_it() {
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");
    let training = [batch_at(0, &device)];
    let held_out = [batch_at(40, &device)];

    let mut trainer = Trainer::new(schedule(), Some(1.0));
    let (model, log) = trainer
        .fit_with_validation(
            model,
            &training,
            6,
            Some(ValidationSchedule {
                batches: &held_out,
                interval_steps: 2,
            }),
        )
        .expect("no step failed");

    assert_eq!(log.len(), 6);
    for (index, entry) in log.steps().iter().enumerate() {
        let expected_to_have_run = (index + 1) % 2 == 0;
        assert_eq!(
            entry.validation_loss.is_some(),
            expected_to_have_run,
            "step {} recorded validation {}",
            entry.step,
            entry.validation_loss.is_some()
        );
        if let Some(loss) = entry.validation_loss {
            assert!(loss.is_finite(), "step {} recorded {loss}", entry.step);
        }
    }

    // The recorded value must be the one a fresh evaluation produces.
    let recomputed = evaluate(&model, &held_out).expect("usable");
    let recorded = log.last().and_then(|entry| entry.validation_loss);
    assert_eq!(recorded, Some(recomputed));
}

#[test]
fn no_validation_schedule_means_no_validation_record() {
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");
    let training = [batch_at(0, &device)];

    let mut trainer = Trainer::new(schedule(), Some(1.0));
    let (_, log) = trainer.fit(model, &training, 3).expect("no step failed");

    assert!(
        log.steps()
            .iter()
            .all(|entry| entry.validation_loss.is_none()),
        "a run without held-out data must not claim a validation loss"
    );
}

#[test]
fn a_validation_interval_larger_than_the_run_never_fires() {
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");
    let training = [batch_at(0, &device)];
    let held_out = [batch_at(40, &device)];

    let mut trainer = Trainer::new(schedule(), Some(1.0));
    let (_, log) = trainer
        .fit_with_validation(
            model,
            &training,
            3,
            Some(ValidationSchedule {
                batches: &held_out,
                interval_steps: 100,
            }),
        )
        .expect("no step failed");

    assert!(
        log.steps()
            .iter()
            .all(|entry| entry.validation_loss.is_none())
    );
}

#[test]
fn the_training_loss_falls_while_the_parameters_are_held_out_from_it() {
    // The point of the loop: training loss must move, and it must move on the
    // batches it was shown rather than on the held-out ones.
    let device = burn::tensor::Device::flex().autodiff();
    let model = Transformer::new(&tiny_config(), &device).expect("valid config");
    let training = [batch_at(0, &device)];
    let held_out = [batch_at(40, &device)];

    let mut trainer = Trainer::new(schedule(), Some(1.0));
    let (_, log) = trainer
        .fit_with_validation(
            model,
            &training,
            30,
            Some(ValidationSchedule {
                batches: &held_out,
                interval_steps: 30,
            }),
        )
        .expect("no step failed");

    let first = log.steps().first().expect("steps ran").train_loss;
    let best = log.best_train_loss().expect("steps ran");
    assert!(
        best < first,
        "training loss must fall: first {first}, best {best}"
    );
    assert_eq!(log.non_finite_losses(), 0);
    assert_eq!(log.skipped_steps(), 0);
}
