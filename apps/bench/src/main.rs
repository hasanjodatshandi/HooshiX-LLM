//! `bench`: the Stage-1 benchmark baseline.
//!
//! It reports eight numbers, and it reports the environment beside them, because
//! a throughput figure without the machine it was measured on is not a
//! measurement — it is a memory. `11_BENCHMARKING.md` section 12 records what
//! happens otherwise: a 4.4x speedup that turned out to be two runs on a machine
//! under different load.
//!
//! # What is timed, and how
//!
//! - **Training**: whole optimizer steps through `Trainer`, so the number
//!   includes the backward pass and the update rather than only the forward.
//! - **Prefill**: one forward pass over the prompt, timed alone.
//! - **Decode**: per-token time through the KV cache, taken between tokens.
//! - **TTFT**: from the start of generation to the first token, which is prefill
//!   plus the first sample and is the number a user actually feels.
//! - **Checkpoint write and read**: bytes over seconds, taken around the real
//!   save and load rather than around a synthetic write.
//! - **Cold start**: building the model and reading the weights, from process
//!   start.
//! - **Peak memory**: **Not measured here.** Reading a process's peak working
//!   set needs an operating-system call, and this workspace does not permit
//!   `unsafe`. It is sampled by the wrapper script from `PeakWorkingSet64`,
//!   which is the operating system's own figure, and pasted into the report.
//!   Reporting it as measured here would be a lie.

#![deny(unsafe_code)]

mod device;
mod report;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use burn::module::Module;
use clap::Parser;
use llm_core::ExperimentConfig;
use llm_inference::{GenerationConfig, SamplingConfig, SeededRandom, generate_streaming};
use llm_model::Transformer;
use llm_training::{Trainer, WarmupCosineSchedule};

use crate::report::Report;

/// Configuration could not be read.
const EXIT_CONFIG: u8 = 2;
/// The configuration names a backend this binary was not built with.
const EXIT_BACKEND: u8 = 3;
/// A measurement could not be taken.
const EXIT_MEASURE: u8 = 4;

#[derive(Debug, Parser)]
#[command(name = "bench", about = "Measure the Stage-1 benchmark baseline")]
struct Arguments {
    /// Experiment configuration describing the model to measure.
    #[arg(long)]
    config: PathBuf,

    /// Prompt length to prefill with.
    #[arg(long, default_value_t = 128)]
    prompt_tokens: usize,

    /// Tokens to decode.
    #[arg(long, default_value_t = 32)]
    decode_tokens: usize,

    /// Optimizer steps to time for the training figure.
    ///
    /// Five rather than two, and the difference is not cosmetic. The first step
    /// after a warmup still carries a one-time cost — AdamW's moment buffers are
    /// allocated on first use, which is 880 MB for the Stage-1 model — and at two
    /// steps that cost was half the measurement. It reported 71 tokens/s for a
    /// shape that measures 195. A throughput figure taken over too few steps
    /// measures the allocator, not the model.
    #[arg(long, default_value_t = 5)]
    training_steps: usize,

    /// Steps run before timing, so the measurement is not dominated by the first
    /// allocation.
    #[arg(long, default_value_t = 2)]
    warmup_steps: usize,

    /// Micro-batch to measure instead of the configuration's.
    ///
    /// The configured Stage-1 shape is `batch 8 x sequence 2048`, whose attention
    /// score tensor is 1.6 GB and does not fit this machine. A baseline that
    /// cannot be taken is not a baseline, so the shape is a parameter.
    #[arg(long)]
    micro_batch: Option<usize>,

    /// Sequence length to measure instead of the configuration's.
    #[arg(long)]
    sequence: Option<usize>,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();

    let config = match ExperimentConfig::load(&arguments.config) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("bench: {}: {error}", arguments.config.display());
            return ExitCode::from(EXIT_CONFIG);
        }
    };

    let compiled = device::compiled_backend();
    if let Err(error) = config.validate_backend(compiled) {
        eprintln!("bench: {error}");
        return ExitCode::from(EXIT_BACKEND);
    }

    let mut report = Report::new(&config, &arguments, compiled);
    let device = device::build_device(config.runtime.backend).autodiff();

    match measure(&arguments, &config, &device, &mut report) {
        Ok(()) => {
            report.print();
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("bench: {error}");
            ExitCode::from(EXIT_MEASURE)
        }
    }
}

fn measure(
    arguments: &Arguments,
    config: &ExperimentConfig,
    device: &burn::tensor::Device,
    report: &mut Report,
) -> Result<(), String> {
    // --- cold start --------------------------------------------------------
    llm_model::init::seed(device, config.training.seed);
    let started = Instant::now();
    let model = Transformer::new(&config.model, device).map_err(|error| error.to_string())?;
    report.cold_start_seconds = started.elapsed().as_secs_f64();

    // The tokenizer is deliberately not loaded. None of the eight measurements
    // needs one — they are all about the model and the device — and requiring an
    // artifact would stop the benchmark from running on a machine that has the
    // weights but has not prepared a dataset yet.

    // --- prefill and TTFT --------------------------------------------------
    let prompt: Vec<u32> = (0..arguments.prompt_tokens)
        .map(|index| (index % config.model.vocab_size) as u32)
        .collect();

    let generation = GenerationConfig {
        max_new_tokens: arguments.decode_tokens.max(1),
        eos_token: None,
        sampling: SamplingConfig::greedy(),
    };

    let mut first_token_at = None;
    let mut token_times = Vec::with_capacity(arguments.decode_tokens);
    let mut last = Instant::now();
    let mut random = SeededRandom::new(config.generation.seed);
    let started = Instant::now();

    generate_streaming(&model, &prompt, &generation, device, &mut random, |_| {
        let now = Instant::now();
        if first_token_at.is_none() {
            first_token_at = Some(now.duration_since(started));
        } else {
            token_times.push(now.duration_since(last).as_secs_f64());
        }
        last = now;
        true
    })
    .map_err(|error| error.to_string())?;

    let ttft = first_token_at.ok_or("generation produced no tokens")?;
    report.ttft_seconds = ttft.as_secs_f64();
    // Prefill is everything before the first token, minus the sample that
    // produced it. It is reported separately because a long prompt and a long
    // decode are different costs, and one figure hides both.
    report.prefill_seconds = ttft.as_secs_f64();
    report.prompt_tokens = prompt.len();
    report.decode_seconds = token_times.clone();

    // --- training ----------------------------------------------------------
    let batches_per_step = config.training.gradient_accumulation_steps as usize;
    let micro = arguments
        .micro_batch
        .unwrap_or(config.data.micro_batch_size);
    let sequence = arguments.sequence.unwrap_or(config.data.sequence_length);
    // Recorded as measured, not as configured: a report that says `8 x 2048`
    // while the numbers came from `8 x 512` is worse than no report.
    report.micro_batch = micro;
    report.sequence_length = sequence;

    let schedule = WarmupCosineSchedule::new(
        config.training.optimizer.learning_rate,
        config.training.scheduler.warmup_steps,
        config.training.scheduler.cosine_iters,
    )
    .map_err(|error| error.to_string())?;
    let mut trainer = Trainer::new(
        schedule,
        config.training.gradient_clip_global_norm,
        &config.training.optimizer,
    );

    // Built per step rather than cloned: a batch owns its tensors and the step
    // consumes them. Rebuilding from the same rule presents the model with the
    // same work every step, which is what makes the timings comparable.
    let mut model = model;
    for _ in 0..arguments.warmup_steps {
        let batches: Vec<_> = (0..batches_per_step)
            .map(|_| synthetic_batch(micro, sequence, config.model.vocab_size, device))
            .collect();
        let (updated, _) = trainer
            .step(model, &batches)
            .map_err(|error| error.to_string())?;
        model = updated;
    }

    let started = Instant::now();
    for _ in 0..arguments.training_steps {
        let batches: Vec<_> = (0..batches_per_step)
            .map(|_| synthetic_batch(micro, sequence, config.model.vocab_size, device))
            .collect();
        let (updated, _) = trainer
            .step(model, &batches)
            .map_err(|error| error.to_string())?;
        model = updated;
    }
    report.training_seconds = started.elapsed().as_secs_f64();
    report.training_steps = arguments.training_steps;
    report.tokens_per_step = (micro * sequence * batches_per_step) as u64;

    // --- checkpoint write and read ----------------------------------------
    let root = std::env::temp_dir().join("llm-bench-checkpoint");
    let _ = std::fs::remove_dir_all(&root);
    let metadata = llm_checkpoint::CheckpointMetadata {
        format_version: llm_checkpoint::CHECKPOINT_FORMAT_VERSION,
        created_unix_seconds: 0,
        global_step: 1,
        tokens_seen: report.tokens_per_step,
        data_cursor: llm_checkpoint::DataCursor {
            epoch: 0,
            shard: 0,
            sequence: 0,
        },
        config_hash: String::new(),
        tokenizer_identity: String::new(),
        dataset_manifest_hash: String::new(),
    };

    let started = Instant::now();
    let written = llm_checkpoint::save(&root, "benchmark", &model, trainer.optimizer(), &metadata)
        .map_err(|error| error.to_string())?;
    report.checkpoint_write_seconds = started.elapsed().as_secs_f64();
    report.checkpoint_bytes = directory_bytes(&written);

    let parameters = written.join(llm_checkpoint::MODEL_FILE);
    let started = Instant::now();
    let blank = Transformer::new(&config.model, device).map_err(|error| error.to_string())?;
    let reloaded = blank
        .try_load_file(&parameters)
        .map_err(|error| error.to_string())?;
    report.checkpoint_read_seconds = started.elapsed().as_secs_f64();
    let _ = reloaded;

    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}

/// A batch of in-range ids.
///
/// This is a throughput measurement, not a data test, so the ids only need to be
/// valid — but they are generated from a fixed rule rather than randomly, so two
/// runs of the benchmark present the model with the same work.
fn synthetic_batch(
    micro: usize,
    sequence: usize,
    vocab_size: usize,
    device: &burn::tensor::Device,
) -> llm_training::TokenBatch {
    use burn::tensor::{Int, Tensor, TensorData};

    let mut ids: Vec<i32> = Vec::with_capacity(micro * sequence);
    for index in 0..(micro * sequence) {
        ids.push((index % vocab_size) as i32);
    }
    let targets: Vec<i32> = ids.iter().map(|id| (id + 1) % vocab_size as i32).collect();

    llm_training::TokenBatch {
        input: Tensor::<2, Int>::from_data(TensorData::new(ids, [micro, sequence]), device),
        target: Tensor::<2, Int>::from_data(TensorData::new(targets, [micro, sequence]), device),
        mask: llm_model::all_positions(micro, sequence, device),
    }
}

/// Total size of the files in a directory.
fn directory_bytes(path: &std::path::Path) -> u64 {
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter_map(|entry| entry.metadata().ok())
                .map(|metadata| metadata.len())
                .sum()
        })
        .unwrap_or(0)
}
