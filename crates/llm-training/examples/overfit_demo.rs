//! A visible demonstration that the model learns.
//!
//! The smoke model on the prepared fixture — 2 layers of width 64 over a 4 KB
//! corpus for 100 steps — cannot show this. It is far too small and sees far too
//! little, and its output is the same character repeated. That is a fine result
//! for a pipeline check and a useless one for a demonstration.
//!
//! This example shows the thing a demonstration should show: a model small
//! enough to train in seconds, on a text small enough to memorise, driven until
//! the loss reaches zero — and then asked to reproduce the text it was given. If
//! it reproduces the text, the target shift, the loss mask, the optimizer, the
//! gradient flow, parameter registration and the model output all agree with each
//! other. If it does not, one of them is broken and no amount of scale will fix
//! it.
//!
//! ```text
//! cargo run --release -p llm-training --example overfit_demo
//! ```

// An example aborts on failure rather than returning an error: there is no
// caller to hand a failure to. `clippy.toml` relaxes the panic family for tests
// only.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::PathBuf;

use burn::tensor::{Device, Int, Tensor, TensorData};
use llm_core::ModelConfig;
use llm_model::{Transformer, all_positions, next_token_pairs};
use llm_training::{TokenBatch, Trainer, WarmupCosineSchedule};

/// One id per byte, which is what the fixture is.
const VOCABULARY: usize = 256;
/// Bytes the model memorises in one window.
const SEQUENCE: usize = 48;
/// Steps to drive the loss to zero on a text this small.
const STEPS: u64 = 400;
/// Learning rate for a flat schedule; the point is memorisation, not the curve.
const LEARNING_RATE: f64 = 0.02;
/// Prompt the model is asked to continue.
const PROMPT: &str = "The tiny model ";
/// Bytes to generate.
const GENERATE: usize = 60;

fn main() {
    let corpus = std::fs::read(fixture_path()).expect("the overfit fixture is readable");
    let ids: Vec<u32> = corpus.iter().map(|byte| u32::from(*byte)).collect();

    println!("the text the model is given, {} bytes:", corpus.len());
    println!(
        "  {}",
        String::from_utf8_lossy(&corpus).replace('\n', "\n  ")
    );
    println!();

    let device = Device::flex().autodiff();
    llm_model::init::seed(&device, 0xC0FFEE);

    let batches = corpus_batches(&ids, &device);
    let model = Transformer::new(&tiny_config(), &device).expect("the tiny config is valid");

    let before = measure(&model, &batches);
    println!(
        "before training, over {} windows covering the whole text",
        batches.len()
    );
    println!("  loss            {before:.4}");
    println!(
        "  continues to    {:?}",
        render(&continue_from(&model, PROMPT.as_bytes(), GENERATE, &device))
    );
    println!();

    let schedule = WarmupCosineSchedule::new(LEARNING_RATE, 0, STEPS).expect("a valid schedule");
    let mut trainer = Trainer::new(schedule, Some(1.0));

    println!("training, {STEPS} steps over every window");
    let started = std::time::Instant::now();
    let (model, log) = trainer.fit(model, &batches, STEPS).expect("no step failed");
    let elapsed = started.elapsed();

    // A few points on the curve, so the drop is visible rather than asserted.
    let steps = log.steps();
    let stride = (STEPS / 8).max(1);
    for entry in steps.iter().filter(|entry| entry.step % stride == 0) {
        println!("  step {:>4}   loss {:.4}", entry.step, entry.train_loss);
    }
    println!();

    let after = log.last().map_or(f32::NAN, |entry| entry.train_loss);
    println!("after training, in {:.1} s", elapsed.as_secs_f64());
    println!("  loss            {after:.4}   (from {before:.4})");
    println!("  refused steps   {}", log.non_finite_losses());
    println!("  abandoned steps {}", log.skipped_steps());
    println!();

    let generated = continue_from(&model, PROMPT.as_bytes(), GENERATE, &device);
    println!("asked to continue {PROMPT:?}");
    println!("  {}", render(&generated));
    println!();

    // What the corpus actually says after the prompt, so the two can be compared
    // by eye rather than taken on trust.
    let truth = corpus_after(&corpus, PROMPT.as_bytes(), GENERATE);
    println!("what the corpus actually says there");
    println!("  {}", render(&truth));
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("overfit-corpus.txt")
}

/// A small architecture that can still memorise a short text.
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

/// The windows the model is given.
///
/// `WINDOWS=1` is the default and the cleaner demonstration: the model
/// reproduces the window it was shown, byte for byte, and stops meaning anything
/// past it — which is what overfitting a window means.
///
/// `WINDOWS=all` tiles the whole corpus. The loss ends up just as low and the
/// generation is visibly worse, which is the more interesting result: capacity is
/// finite, so spreading it over five times as much text costs fidelity per token,
/// and a slightly lower fidelity per token compounds into drift over a
/// generation. Both are honest; they say different things.
fn corpus_batches(ids: &[u32], device: &Device) -> Vec<TokenBatch> {
    let all = std::env::var("WINDOWS").is_ok_and(|value| value == "all");
    let mut batches = Vec::new();
    let mut start = 0;

    while start + 1 < ids.len() {
        let end = (start + SEQUENCE + 1).min(ids.len());
        let window: Vec<i32> = ids[start..end].iter().map(|id| *id as i32).collect();
        let count = window.len() - 1;

        let tokens = Tensor::<2, Int>::from_data(TensorData::new(window, [1, count + 1]), device);
        let (input, target) = next_token_pairs(tokens).expect("more than one token");

        batches.push(TokenBatch {
            input,
            target,
            mask: all_positions(1, count, device),
        });

        if !all {
            break;
        }
        start = end - 1;
    }

    batches
}

/// Reports the loss over a set of batches without applying a step.
fn measure(model: &Transformer, batches: &[TokenBatch]) -> f32 {
    let total: f32 = batches
        .iter()
        .map(|batch| {
            let logits = model.forward(batch.input.clone(), 0).expect("shapes agree");
            llm_model::causal_lm_loss(logits, batch.target.clone(), batch.mask.clone())
                .expect("the mask counts something")
                .try_to_data()
                .expect("readable")
                .try_to_vec::<f32>()
                .expect("f32")
                .first()
                .copied()
                .expect("one element")
        })
        .sum();

    total / batches.len() as f32
}

/// Greedily continues a prompt, one byte at a time.
fn continue_from(model: &Transformer, prompt: &[u8], length: usize, device: &Device) -> Vec<u8> {
    let mut tokens: Vec<i32> = prompt.iter().map(|byte| i32::from(*byte)).collect();

    for _ in 0..length {
        let count = tokens.len();
        let input =
            Tensor::<2, Int>::from_data(TensorData::new(tokens.clone(), [1, count]), device);
        let logits = model.forward(input, 0).expect("shapes agree");
        let next = logits
            .narrow(1, count - 1, 1)
            .reshape([VOCABULARY])
            .argmax(0)
            .into_scalar::<i32>();
        tokens.push(next);
    }

    tokens[prompt.len()..]
        .iter()
        .map(|id| u8::try_from(*id).unwrap_or(b'?'))
        .collect()
}

/// What the corpus holds after the prompt, for comparison.
fn corpus_after(corpus: &[u8], prompt: &[u8], length: usize) -> Vec<u8> {
    let at = corpus
        .windows(prompt.len())
        .position(|window| window == prompt)
        .expect("the prompt appears in the corpus");
    let start = at + prompt.len();
    corpus[start..(start + length).min(corpus.len())].to_vec()
}

/// Renders bytes for a terminal without letting control characters through.
fn render(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| match byte {
            b'\n' => '⏎',
            0x20..=0x7e => char::from(*byte),
            _ => '·',
        })
        .collect()
}
