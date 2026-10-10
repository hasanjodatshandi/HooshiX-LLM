//! Composition root for the training application.
//!
//! This binary owns process-level concerns only: locating and validating
//! configuration, checking the requested backend against the one it was built
//! with, constructing the device, and choosing the exit code. Training
//! mathematics belongs to the `llm-training` crate, which does not exist yet.
//!
//! Exit codes:
//!
//! | code | meaning |
//! |---|---|
//! | 0 | success |
//! | 2 | configuration could not be read, parsed or validated |
//! | 3 | configuration names a backend family this binary was not built with |
//! | 4 | the device failed its preflight |
//!
//! The exit code is distinct per failure class so that a supervisor or CI job
//! can distinguish a bad configuration from a broken device.

mod device;

use std::path::PathBuf;
use std::process::ExitCode;

use llm_core::ExperimentConfig;

/// Configuration could not be read, parsed or validated.
const EXIT_CONFIG: u8 = 2;
/// Configuration names a backend family this binary was not built with.
const EXIT_BACKEND: u8 = 3;
/// The device failed its preflight.
const EXIT_DEVICE: u8 = 4;

fn main() -> ExitCode {
    let config = match load_config() {
        Ok(config) => config,
        Err(code) => return ExitCode::from(code),
    };

    let compiled = device::compiled_backend();
    if let Err(error) = config.validate_backend(compiled) {
        eprintln!("llm-train: {error}");
        eprintln!(
            "llm-train: change runtime.backend in the configuration, \
             or rebuild with --no-default-features --features {}",
            config.runtime.backend.as_str()
        );
        return ExitCode::from(EXIT_BACKEND);
    }

    report(&config);

    let compute_device = device::build_device(config.runtime.backend);
    match device::preflight(&compute_device) {
        Ok(()) => {
            println!("llm-train: device preflight passed on {compiled}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("llm-train: device preflight failed: {error}");
            ExitCode::from(EXIT_DEVICE)
        }
    }
}

/// Resolves the configuration to run.
///
/// The path may be given as the first positional argument; otherwise the
/// built-in Stage-1 defaults are used. A full CLI parser arrives with the
/// training-loop section, when there is more than one argument to parse.
fn load_config() -> Result<ExperimentConfig, u8> {
    match std::env::args_os().nth(1) {
        None => {
            let config = ExperimentConfig::default();
            if let Err(error) = config.validate() {
                eprintln!("llm-train: built-in defaults are invalid: {error}");
                return Err(EXIT_CONFIG);
            }
            println!("llm-train: using built-in Stage-1 defaults");
            Ok(config)
        }
        Some(path) => {
            let path = PathBuf::from(path);
            match ExperimentConfig::load(&path) {
                Ok(config) => {
                    println!("llm-train: loaded configuration from {}", path.display());
                    Ok(config)
                }
                Err(error) => {
                    eprintln!("llm-train: {error}");
                    Err(EXIT_CONFIG)
                }
            }
        }
    }
}

/// Prints the resolved facts a run must be able to report.
fn report(config: &ExperimentConfig) {
    let parameters = config.model.parameter_count().map_or_else(
        || "unavailable (overflow)".to_owned(),
        |count| count.to_string(),
    );

    println!("llm-train: backend        = {}", config.runtime.backend);
    println!("llm-train: precision      = {}", config.runtime.precision);
    println!(
        "llm-train: architecture   = {} layers, hidden {}, {} query heads, {} kv heads, head_dim {}",
        config.model.num_layers,
        config.model.hidden_size,
        config.model.num_heads,
        config.model.num_kv_heads,
        config
            .model
            .head_dim()
            .map_or_else(|| "unknown".to_owned(), |dim| dim.to_string())
    );
    println!("llm-train: parameters     = {parameters}");
    println!(
        "llm-train: context        = {} tokens, packed sequence {}",
        config.model.max_sequence_length, config.data.sequence_length
    );
    println!(
        "llm-train: effective batch = {} sequences, {} tokens per optimizer step",
        config
            .effective_batch_sequences()
            .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
        config
            .effective_tokens_per_step()
            .map_or_else(|| "unavailable".to_owned(), |value| value.to_string())
    );
    println!("llm-train: max steps      = {}", config.training.max_steps);
}
