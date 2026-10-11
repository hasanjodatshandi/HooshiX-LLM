//! Measures training throughput on the reference backend.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p llm-training --example throughput
//! BATCH=8 SEQUENCE=2048 STEPS=2 cargo run --release -p llm-training --example throughput
//! ```
//!
//! # Why this exists
//!
//! "Can this be trained on a CPU?" is an empirical question, and the answer
//! depends on the machine, the batch, the sequence length and the backend. A
//! benchmark that reports a number for the actual configured architecture is
//! worth more than an estimate, and it is what a Stage-1 run plan has to be
//! built on.
//!
//! # \(ATTENTION\)
//!
//! `ATTENTION` selects the attention path and defaults to `whole`, the path this
//! repository has always used:
//!
//! | value | what it scores with |
//! |---|---|
//! | `whole` | this repository's whole-matrix attention, `[batch, heads, s, s]` |
//! | `burn` | `burn::tensor::module::attention`, which each backend implements |
//!
//! Both are the same binary, so a comparison between them is a comparison of the
//! scoring paths and not of two builds:
//!
//! ```text
//! $env:ATTENTION="burn"; cargo run --release -p llm-training --example throughput
//! ```
//!
//! The weights are seeded from the configuration, so the runs also start from
//! the same parameters.

// An example aborts on failure rather than returning an error: there is no
// caller to hand a failure to, and the panic message names the cause.
// `clippy.toml` relaxes the panic family for tests only.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

/// Measures whether the allocator is on the critical path.
///
/// A model step allocates constantly, so a better allocator is a plausible
/// win. It is behind a feature so the same example can be built both ways.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};
use llm_core::ExperimentConfig;
use llm_model::{AttentionMode, Transformer, all_positions, next_token_pairs};
use llm_training::{TokenBatch, Trainer, WarmupCosineSchedule};

/// The backend this binary was built with.
///
/// The example is built for exactly one backend, chosen by feature, so a
/// measurement always names the path it actually ran on.
#[cfg(feature = "cuda")]
fn compute_device() -> Device {
    Device::cuda(0)
}
#[cfg(feature = "wgpu")]
fn compute_device() -> Device {
    // The WGPU backend picks its adapter from device settings rather than from
    // an index, so the default kind is the one that enumerates them.
    Device::wgpu(Default::default())
}
#[cfg(feature = "cpu")]
fn compute_device() -> Device {
    Device::cpu()
}
#[cfg(feature = "ndarray")]
fn compute_device() -> Device {
    Device::ndarray()
}
#[cfg(feature = "flex")]
fn compute_device() -> Device {
    Device::flex()
}

/// Name of the backend this binary runs on.
const BACKEND: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "wgpu") {
    "wgpu"
} else if cfg!(feature = "cpu") {
    "cpu"
} else if cfg!(feature = "ndarray") {
    "ndarray"
} else {
    "flex"
};

/// Reads an environment variable as a number, or uses the default.
fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Which attention path to build the model on.
///
/// `whole` is the whole-matrix path this repository has always used and the
/// default; `burn` hands the block to Burn's own backend-dispatched attention,
/// which is the one that can reach a fused kernel. Anything else is refused
/// rather than silently measured as one of them.
fn attention_mode() -> AttentionMode {
    match std::env::var("ATTENTION").as_deref() {
        Ok("burn") => AttentionMode::Burn,
        Ok("whole") | Err(_) => AttentionMode::Whole,
        Ok(other) => panic!("ATTENTION must be `whole` or `burn`, not {other:?}"),
    }
}

fn main() {
    let config_path =
        std::env::var("CONFIG").unwrap_or_else(|_| "configs/stage1-110m.toml".to_owned());
    let config = ExperimentConfig::load(&config_path)
        .unwrap_or_else(|error| panic!("{config_path}: {error}"));

    let batch = setting("BATCH", config.data.micro_batch_size as usize);
    let sequence = setting("SEQUENCE", config.data.sequence_length as usize);
    let steps = setting("STEPS", 2);
    let warmup = setting("WARMUP", 0);

    let device = compute_device().autodiff();
    llm_model::init::seed(&device, config.training.seed);

    let parameters = config
        .model
        .parameter_count()
        .expect("the configured model fits in u64");
    println!(
        "model: {} parameters ({:.1}M), backend {BACKEND}, precision {:?}",
        parameters,
        parameters as f64 / 1e6,
        config.runtime.precision
    );
    println!("batch {batch}, sequence {sequence}, {steps} timed steps");

    let mode = attention_mode();
    println!("attention: {mode:?}");

    let model = Transformer::with_attention_mode(&config.model, mode, &device)
        .unwrap_or_else(|error| panic!("the configured model is not valid: {error}"));

    // A synthetic batch. This is a throughput measurement, not a data test, so
    // the ids only need to be in range.
    //
    // `ACCUM` splits the same total number of tokens into that many
    // micro-batches, which is how gradient accumulation is exercised: the
    // effective batch is unchanged, but only one micro-batch's activations are
    // alive at a time.
    let accumulation = setting("ACCUM", 1);
    assert!(
        accumulation >= 1 && batch.is_multiple_of(accumulation),
        "ACCUM must divide the batch evenly"
    );
    let micro = batch / accumulation;
    let batch_tokens = batch * sequence;

    let mut example = Vec::with_capacity(accumulation);
    for part in 0..accumulation {
        let tokens: Vec<i32> = (0..(micro * (sequence + 1)))
            .map(|index| ((part * micro * sequence + index) % config.model.vocab_size) as i32)
            .collect();
        let window =
            Tensor::<2, Int>::from_data(TensorData::new(tokens, [micro, sequence + 1]), &device);
        let (input, target) = next_token_pairs(window).expect("more than one token");
        example.push(TokenBatch {
            input,
            target,
            mask: all_positions(micro, sequence, &device),
        });
    }

    if accumulation > 1 {
        println!("accumulating {accumulation} micro-batches of {micro}");
    }

    let schedule = WarmupCosineSchedule::new(
        config.training.optimizer.learning_rate,
        config.training.scheduler.warmup_steps,
        config.training.scheduler.cosine_iters,
    )
    .expect("the configured schedule is valid");
    let clip = config.training.gradient_clip_global_norm;
    let mut trainer = Trainer::new(schedule, clip);

    let mut model = model;
    let mut timings = Vec::new();
    for step in 0..(warmup + steps) {
        let started = Instant::now();
        let (updated, metrics) = trainer
            .step(model, &example)
            .unwrap_or_else(|error| panic!("step {step} failed: {error}"));
        model = updated;
        let elapsed = started.elapsed().as_secs_f64();

        if step < warmup {
            println!("step {step}: {elapsed:.3}s (warmup, not counted)");
            continue;
        }
        let per_token = elapsed / batch_tokens as f64;
        println!(
            "step {step}: {elapsed:.4}s  loss {:.4}  {:.0} tokens/s  {:.3} ms/token",
            metrics.train_loss,
            1.0 / per_token,
            per_token * 1e3
        );
        timings.push(elapsed);
    }

    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    let tokens_per_second = batch_tokens as f64 / mean;
    println!();
    println!(
        "at batch {batch}, sequence {sequence}: {mean:.4}s per step, \
         {tokens_per_second:.0} tokens/s"
    );
    println!(
        "one pass over 1B tokens at this shape: {:.1} hours",
        1e9 / tokens_per_second / 3600.0
    );

    // The configured run's own shapes, not this measurement's. Attention grows
    // quadratically in sequence length, so a throughput measured at one shape
    // does not transfer to another, and this prints the comparison rather than
    // extrapolating from it.
    let configured_tokens = (config.data.micro_batch_size as usize)
        * (config.data.sequence_length as usize)
        * (config.training.max_steps as usize);
    println!(
        "the configured run is {} steps at batch {} and sequence {}, which is {:.2}B tokens",
        config.training.max_steps,
        config.data.micro_batch_size,
        config.data.sequence_length,
        configured_tokens as f64 / 1e9
    );
    if batch != config.data.micro_batch_size as usize
        || sequence != config.data.sequence_length as usize
    {
        println!(
            "note: these are not the shapes measured above, so the rate reported \
             here does not apply to the configured run"
        );
    }
}
