//! `generate-cli`: continue a prompt with a trained checkpoint.
//!
//! The command line owns argument parsing, file loading and output; it owns no
//! model mathematics and no sampling policy. Deciding what the next token is
//! belongs to `llm-inference`, and the model belongs to `llm-model`.
//!
//! # Why the prompt is encoded here
//!
//! The tokenizer is a file, not a property of the model. Loading it here and
//! checking its identity against the checkpoint is what stops a run from being
//! continued with a tokenizer it was not trained with, which would silently
//! reinterpret every id.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn::module::Module;
use burn::tensor::Device;
use clap::Parser;
use llm_checkpoint::{CheckpointError, CheckpointMetadata, MODEL_FILE};
use llm_core::{BackendChoice, ExperimentConfig};
use llm_inference::{GenerationConfig, SamplingConfig, SeededRandom, StopReason, generate};
use llm_model::Transformer;
use llm_tokenizer::{LlmTokenizer, SpecialTokensInOutput};

/// Configuration could not be read, parsed or validated.
const EXIT_CONFIG: u8 = 2;
/// The configuration names a backend this binary does not provide.
const EXIT_BACKEND: u8 = 3;
/// A checkpoint could not be read or was refused.
const EXIT_CHECKPOINT: u8 = 5;
/// The tokenizer could not be read or does not match the configuration.
const EXIT_TOKENIZER: u8 = 6;
/// Generation failed.
const EXIT_GENERATION: u8 = 7;

#[derive(Debug, Parser)]
#[command(
    name = "generate-cli",
    about = "Continue a prompt with a trained checkpoint"
)]
struct Arguments {
    /// Experiment configuration describing the model architecture.
    #[arg(long)]
    config: PathBuf,

    /// Checkpoint directory holding the model parameters.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Tokenizer artifact the model was trained with.
    #[arg(long)]
    tokenizer: PathBuf,

    /// Text to continue.
    #[arg(long)]
    prompt: String,

    /// Maximum number of tokens to produce.
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    /// Softmax temperature. Zero means greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Keep only the highest-scoring this many candidates.
    #[arg(long)]
    top_k: Option<usize>,

    /// Keep the smallest set of candidates whose probability sums to this.
    #[arg(long)]
    top_p: Option<f32>,

    /// Seed for sampling. Ignored by a greedy decoder.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Token id that ends generation.
    #[arg(long)]
    eos_token: Option<u32>,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    run(&arguments)
}

fn run(arguments: &Arguments) -> ExitCode {
    let config = match ExperimentConfig::load(&arguments.config) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("generate-cli: {}: {error}", arguments.config.display());
            return ExitCode::from(EXIT_CONFIG);
        }
    };

    let device = match build_device(config.runtime.backend) {
        Ok(device) => device,
        Err(message) => {
            eprintln!("generate-cli: {message}");
            return ExitCode::from(EXIT_BACKEND);
        }
    };

    let tokenizer = match LlmTokenizer::load(&arguments.tokenizer, &config.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("generate-cli: {}: {error}", arguments.tokenizer.display());
            return ExitCode::from(EXIT_TOKENIZER);
        }
    };

    let prompt = match tokenizer.encode(&arguments.prompt) {
        Ok(ids) => ids,
        Err(error) => {
            eprintln!("generate-cli: the prompt could not be encoded: {error}");
            return ExitCode::from(EXIT_TOKENIZER);
        }
    };

    let metadata = match checkpoint_metadata(&arguments.checkpoint) {
        Ok(metadata) => metadata,
        Err(error) => {
            eprintln!("generate-cli: {}: {error}", arguments.checkpoint.display());
            return ExitCode::from(EXIT_CHECKPOINT);
        }
    };

    // A checkpoint names the tokenizer it was written with. Continuing it with a
    // different one would reinterpret every id it learned.
    if metadata.tokenizer_identity != tokenizer.identity_hash() {
        eprintln!(
            "generate-cli: the checkpoint was written with tokenizer {}, but {} is {}",
            metadata.tokenizer_identity,
            arguments.tokenizer.display(),
            tokenizer.identity_hash()
        );
        return ExitCode::from(EXIT_TOKENIZER);
    }

    let blank = match Transformer::new(&config.model, &device) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("generate-cli: the configured model is not valid: {error}");
            return ExitCode::from(EXIT_CONFIG);
        }
    };

    let model = match blank.try_load_file(arguments.checkpoint.join(MODEL_FILE)) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("generate-cli: the parameters could not be loaded: {error}");
            return ExitCode::from(EXIT_CHECKPOINT);
        }
    };

    let generation_config = GenerationConfig {
        max_new_tokens: arguments.max_new_tokens,
        eos_token: arguments.eos_token,
        sampling: SamplingConfig {
            temperature: arguments.temperature,
            top_k: arguments.top_k,
            top_p: arguments.top_p,
        },
    };

    let mut random = SeededRandom::new(arguments.seed);
    let generation = match generate(&model, &prompt, &generation_config, &device, &mut random) {
        Ok(generation) => generation,
        Err(error) => {
            eprintln!("generate-cli: {error}");
            return ExitCode::from(EXIT_GENERATION);
        }
    };

    // Special tokens are skipped: a completion ending with a control marker is
    // not text, and echoing one into stdout would be noise.
    let text = match tokenizer.decode(&generation.tokens, SpecialTokensInOutput::Skip) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("generate-cli: the produced tokens could not be decoded: {error}");
            return ExitCode::from(EXIT_TOKENIZER);
        }
    };

    println!("{text}");

    // The reason is reported on stderr so stdout stays exactly the completion,
    // which is what a caller piping the output wants.
    let reason = match generation.stop {
        StopReason::EndOfSequence => "end of sequence",
        StopReason::MaxNewTokens => "token budget reached",
    };
    eprintln!(
        "generate-cli: {} tokens from step {}, stopped by {reason}",
        generation.len(),
        metadata.global_step
    );

    ExitCode::SUCCESS
}

/// Reads a checkpoint's envelope.
fn checkpoint_metadata(path: &Path) -> Result<CheckpointMetadata, CheckpointError> {
    llm_checkpoint::validate(path)
}

/// Builds the compute device named by the configuration.
///
/// Only the reference backend is provided. The others are refused rather than
/// silently substituted: a run that asked for an accelerator and quietly got the
/// reference path would produce correct text far too slowly to be useful, and
/// the caller would have no way to tell.
fn build_device(choice: BackendChoice) -> Result<Device, String> {
    match choice {
        BackendChoice::Flex => Ok(Device::flex()),
        other => Err(format!(
            "this binary provides the reference backend only; the configuration asks \
             for {}. Change runtime.backend to flex, or rebuild with the {} feature.",
            other.as_str(),
            other.as_str()
        )),
    }
}
