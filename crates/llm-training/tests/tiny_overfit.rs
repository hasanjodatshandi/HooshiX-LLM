//! The tiny-model overfit test.
//!
//! A very small model must be able to memorize a very small text. If it cannot,
//! the fault is structural rather than a matter of scale, and it lies in one of:
//! the target shift, the loss mask, the optimizer, the gradient flow, parameter
//! registration, or the model output. Every one of those has already been tested
//! in isolation; this test is the one that checks they agree with each other.
//!
//! It is required before any large training run, because a large run costs hours
//! and this one costs a second.

// Test helpers may abort on failure: an unreadable fixture or an empty corpus
// means the test cannot run, and the panic message names the cause. `clippy.toml`
// only relaxes the panic family for `#[test]` functions and `#[cfg(test)]`
// modules, which does not reach helpers in an integration test binary. The
// workspace-level denies stay in force for `src/`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::PathBuf;

use burn::tensor::{Int, Tensor, TensorData};
use llm_core::ModelConfig;
use llm_model::{Transformer, all_positions, next_token_pairs};
use llm_training::{TokenBatch, Trainer, WarmupCosineSchedule};

/// Vocabulary of the byte-level fixture: one id per byte.
const VOCABULARY: usize = 256;

/// Token sequence the model memorizes.
const SEQUENCE: usize = 48;

/// Steps needed to drive the loss to zero on a corpus this small; the test
/// asserts the outcome, not this number.
const STEPS: u64 = 400;

const LEARNING_RATE: f64 = 0.01;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("overfit-corpus.txt")
}

/// A byte-level id stream from the committed fixture.
///
/// The fixture is written by this repository, so it is deterministic and carries
/// no licensing question. It is read as bytes rather than through the tokenizer
/// because this test is about the model, the objective and the optimizer; the
/// tokenizer has its own contract tests.
fn corpus_ids() -> Vec<u32> {
    let bytes = std::fs::read(fixture_path()).expect("the overfit fixture must be readable");
    bytes.into_iter().map(u32::from).collect()
}

/// A small architecture that can still memorize a short text.
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

fn tensor2_int(
    values: Vec<i32>,
    shape: [usize; 2],
    device: &burn::tensor::Device,
) -> Tensor<2, Int> {
    Tensor::<2, Int>::from_data(TensorData::new(values, shape), device)
}

/// One fixed batch taken from the head of the corpus.
fn fixed_batch(device: &burn::tensor::Device) -> TokenBatch {
    let ids = corpus_ids();
    assert!(
        ids.len() > SEQUENCE,
        "the fixture must hold at least {} bytes",
        SEQUENCE + 1
    );

    let window: Vec<i32> = ids[..=SEQUENCE].iter().map(|id| *id as i32).collect();
    let tokens = tensor2_int(window, [1, SEQUENCE + 1], device);
    let (input, target) = next_token_pairs(tokens).expect("more than one token");

    TokenBatch {
        input,
        target,
        mask: all_positions(1, SEQUENCE, device),
    }
}

/// Reports the loss of a batch without applying a step.
fn measure(model: &Transformer, batch: &TokenBatch) -> f32 {
    let logits = model.forward(batch.input.clone(), 0).expect("shapes agree");
    llm_model::causal_lm_loss(logits, batch.target.clone(), batch.mask.clone())
        .expect("the mask counts something")
        .to_data()
        .try_to_vec::<f32>()
        .expect("f32")
        .first()
        .copied()
        .expect("one element")
}

/// Greedily continues a prompt from the model, one token at a time.
///
/// A minimal decoder, not the inference path: it exists to show that what the
/// model learned is the corpus and not merely a small number.
fn greedy_continue(
    model: &Transformer,
    prompt: &[u32],
    length: usize,
    device: &burn::tensor::Device,
) -> Vec<u32> {
    let mut tokens: Vec<u32> = prompt.to_vec();
    let mut produced = Vec::with_capacity(length);

    for _ in 0..length {
        let window: Vec<i32> = tokens.iter().map(|id| *id as i32).collect();
        let count = window.len();
        let input = tensor2_int(window, [1, count], device);
        let logits = model.forward(input, 0).expect("shapes agree");

        // The prediction for the next token is the last position's row.
        let next = logits
            .narrow(1, count - 1, 1)
            .reshape([VOCABULARY])
            .argmax(0)
            .into_scalar::<i32>() as u32;

        produced.push(next);
        tokens.push(next);
    }

    produced
}

#[test]
fn a_tiny_model_memorizes_a_tiny_corpus() {
    let device = burn::tensor::Device::flex().autodiff();
    let batch = fixed_batch(&device);
    let model = Transformer::new(&tiny_config(), &device).expect("the tiny config is valid");

    let initial = measure(&model, &batch);
    assert!(
        initial.is_finite(),
        "the initial loss must be finite, got {initial}"
    );

    // A flat schedule for this test: the point is memorization, not the curve.
    let schedule = WarmupCosineSchedule::new(LEARNING_RATE, 0, STEPS).expect("a valid schedule");
    let mut trainer = Trainer::new(schedule, Some(1.0));
    let (model, log) = trainer
        .fit(model, std::slice::from_ref(&batch), STEPS)
        .expect("no step failed");

    let final_loss = log
        .last()
        .map(|entry| entry.train_loss)
        .expect("the loop ran");
    let best = log.best_train_loss().expect("the loop ran");

    assert_eq!(trainer.steps(), STEPS);
    assert_eq!(log.non_finite_losses(), 0);
    assert_eq!(log.skipped_steps(), 0);
    assert_eq!(log.tokens_seen(), STEPS * batch.tokens());
    assert!(
        log.steps().iter().all(|entry| entry.train_loss.is_finite()),
        "every step must produce a finite loss"
    );

    // A model that has memorized a short text predicts it almost exactly, so the
    // loss is a small fraction of the initial value rather than merely lower.
    assert!(
        best < initial * 0.05,
        "expected clear overfitting: initial {initial}, best {best}"
    );
    assert!(
        final_loss.is_finite(),
        "the final loss must be finite, got {final_loss}"
    );

    // The learned pattern must be the corpus, not a small number that happens to
    // satisfy the loss bound.
    let ids = corpus_ids();
    let produced = greedy_continue(&model, &ids[..8], 16, &device);
    let expected: Vec<u32> = ids[8..24].to_vec();

    assert_eq!(
        produced, expected,
        "greedy continuation must reproduce the corpus; a mismatch means the model \
         learned something other than the data it was shown"
    );
}
