//! `serve`: the inference server's composition root.
//!
//! It owns exactly one decision the library crates cannot make — the order in
//! which a process becomes able to serve — and it owns it in one place:
//!
//! ```text
//! read the configuration
//!   → build the device
//!   → load the tokenizer
//!   → load the weights from a checkpoint
//!   → run a self-test generation
//!   → mark ready
//!   → bind and accept
//! ```
//!
//! Readiness is the last step. A model that loaded is not a model that works:
//! a shape mismatch or a backend that cannot execute an operation appears on the
//! first real forward pass, and traffic must not arrive before that.

mod device;

use std::path::PathBuf;
use std::process::ExitCode;

use burn::module::Module;
use clap::Parser;
use llm_checkpoint::{CheckpointExpectation, MODEL_FILE};
use llm_core::ExperimentConfig;
use llm_inference::{GenerationConfig, SamplingConfig, SeededRandom, generate};
use llm_model::Transformer;
use llm_server::{AppState, ServerLimits};
use llm_tokenizer::LlmTokenizer;

/// Configuration could not be read, parsed or validated.
const EXIT_CONFIG: u8 = 2;
/// The configuration names a backend this binary was not built with.
const EXIT_BACKEND: u8 = 3;
/// The tokenizer could not be loaded.
const EXIT_TOKENIZER: u8 = 4;
/// The checkpoint could not be loaded, or was refused.
const EXIT_CHECKPOINT: u8 = 5;
/// The self-test failed, so the server never became ready.
const EXIT_SELFTEST: u8 = 6;
/// The server could not bind or run.
const EXIT_SERVE: u8 = 7;

#[derive(Debug, Parser)]
#[command(name = "serve", about = "Serve a trained checkpoint over HTTP")]
struct Arguments {
    /// Experiment configuration describing the model architecture.
    #[arg(long)]
    config: PathBuf,

    /// Checkpoint directory holding the model parameters.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Tokens the self-test generation produces.
    #[arg(long, default_value_t = 4)]
    selftest_tokens: usize,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("serve: could not start the async runtime: {error}");
            return ExitCode::from(EXIT_SERVE);
        }
    };

    runtime.block_on(async move { run(&arguments).await })
}

async fn run(arguments: &Arguments) -> ExitCode {
    let config = match ExperimentConfig::load(&arguments.config) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("serve: {}: {error}", arguments.config.display());
            return ExitCode::from(EXIT_CONFIG);
        }
    };

    let compiled = device::compiled_backend();
    if let Err(error) = config.validate_backend(compiled) {
        eprintln!("serve: {error}");
        eprintln!(
            "serve: change runtime.backend in the configuration, or rebuild with \
             --no-default-features --features {}",
            config.runtime.backend.as_str()
        );
        return ExitCode::from(EXIT_BACKEND);
    }

    report(&config);

    // Loading is process initialization, not per-request work. Everything below
    // happens once, before the listener exists.
    let device = device::build_device(config.runtime.backend);

    let Some(artifact) = config.tokenizer.artifact.as_ref() else {
        eprintln!("serve: tokenizer.artifact is not set in the configuration");
        return ExitCode::from(EXIT_CONFIG);
    };
    let tokenizer = match LlmTokenizer::load(artifact, &config.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("serve: {}: {error}", artifact.display());
            return ExitCode::from(EXIT_TOKENIZER);
        }
    };

    llm_model::init::seed(&device, config.training.seed);
    let blank = match Transformer::new(&config.model, &device) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("serve: the configured model is not valid: {error}");
            return ExitCode::from(EXIT_CONFIG);
        }
    };

    let parameters = arguments.checkpoint.join(MODEL_FILE);
    let model = match blank.try_load_file(&parameters) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("serve: {}: {error}", parameters.display());
            return ExitCode::from(EXIT_CHECKPOINT);
        }
    };

    // The checkpoint names the tokenizer it was written with. Serving it with a
    // different one would reinterpret every id it learned.
    let envelope = match llm_checkpoint::validate(&arguments.checkpoint) {
        Ok(envelope) => envelope,
        Err(error) => {
            eprintln!("serve: {}: {error}", arguments.checkpoint.display());
            return ExitCode::from(EXIT_CHECKPOINT);
        }
    };
    let _ = CheckpointExpectation {
        tokenizer_identity: &envelope.tokenizer_identity,
        dataset_manifest_hash: &envelope.dataset_manifest_hash,
        config_hash: None,
        training_semantics_version: llm_checkpoint::TRAINING_SEMANTICS_VERSION,
    };
    if envelope.tokenizer_identity != tokenizer.identity_hash() {
        eprintln!(
            "serve: the checkpoint was written with tokenizer {}, but {} is {}",
            envelope.tokenizer_identity,
            artifact.display(),
            tokenizer.identity_hash()
        );
        return ExitCode::from(EXIT_TOKENIZER);
    }

    let limits = ServerLimits::default();
    if let Err(error) = limits.validate() {
        eprintln!("serve: {error}");
        return ExitCode::from(EXIT_CONFIG);
    }

    let state = AppState::new(
        model,
        tokenizer,
        device,
        limits,
        Some(config.tokenizer.special_tokens.eos),
    );

    // The self-test is a real generation through the real path. If it fails the
    // process stays not ready and exits, rather than accepting traffic it cannot
    // serve.
    if let Err(error) = self_test(&state, arguments.selftest_tokens) {
        eprintln!("serve: self-test failed: {error}");
        return ExitCode::from(EXIT_SELFTEST);
    }
    state.mark_ready();
    println!(
        "serve: self-test passed, {SEED_PROMPT:?} continued for {} token(s)",
        arguments.selftest_tokens
    );

    let listener = match tokio::net::TcpListener::bind(&arguments.bind).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("serve: failed to bind `{}`: {error}", arguments.bind);
            return ExitCode::from(EXIT_SERVE);
        }
    };
    println!("serve: listening on http://{}", arguments.bind);
    println!("serve: ready — GET /health/live, GET /health/ready, POST /v1/completions");

    // The library's serve loop, not a second one built here: the shutdown path
    // is the part that has to be right, and it is tested where it is defined.
    let served = llm_server::serve(listener, state, shutdown_signal()).await;

    match served {
        Ok(()) => {
            println!("serve: stopped accepting requests and shut down");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("serve: the server failed: {error}");
            ExitCode::from(EXIT_SERVE)
        }
    }
}

/// Prompt the self-test continues.
const SEED_PROMPT: &str = "the";

/// Runs one short generation and discards it.
///
/// A failure here is a failure the first real request would have hit, moved to
/// process start where it can stop the server instead of a caller.
fn self_test(state: &AppState, tokens: usize) -> Result<(), String> {
    let prompt = state
        .tokenizer()
        .encode(SEED_PROMPT)
        .map_err(|error| error.to_string())?;
    if prompt.is_empty() {
        return Err("the self-test prompt encoded to no tokens".to_owned());
    }

    let config = GenerationConfig {
        max_new_tokens: tokens,
        eos_token: state.eos_token(),
        sampling: SamplingConfig::greedy(),
    };
    let mut random = SeededRandom::new(0);
    generate(state.model(), &prompt, &config, state.device(), &mut random)
        .map_err(|error| error.to_string())?;
    Ok(())
}

/// Resolves when the process is asked to stop.
///
/// The signal ends the accept loop and lets in-flight requests finish; the
/// runtime is dropped afterwards, which releases the model and the device.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("serve: shutdown requested, draining in-flight requests");
}

/// Prints the resolved facts a run must be able to report.
fn report(config: &ExperimentConfig) {
    println!("serve: backend      = {}", config.runtime.backend);
    println!("serve: precision    = {}", config.runtime.precision);
    println!(
        "serve: architecture = {} layers, hidden {}, {} heads",
        config.model.num_layers, config.model.hidden_size, config.model.num_heads
    );
    println!(
        "serve: parameters   = {}",
        config
            .model
            .parameter_count()
            .map_or_else(|| "unavailable".to_owned(), |count| count.to_string())
    );
}
