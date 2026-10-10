//! Command-line entry point for training and inspecting tokenizer artifacts.
//!
//! Exit codes are distinct per failure class so a script or CI job can tell a
//! bad configuration from a bad corpus from a bad artifact:
//!
//! | code | meaning |
//! |---|---|
//! | 0 | success |
//! | 2 | configuration could not be read, parsed or validated |
//! | 5 | corpus file could not be opened or read |
//! | 6 | tokenizer training, loading or saving failed |
//! | 7 | a stored identity hash does not match the artifact |

mod corpus;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use llm_core::{ExperimentConfig, TokenizerConfig};
use llm_tokenizer::{LlmTokenizer, TokenizerError};

use crate::corpus::CorpusLines;

/// Configuration could not be read, parsed or validated.
const EXIT_CONFIG: u8 = 2;
/// A corpus file could not be opened or read.
const EXIT_CORPUS: u8 = 5;
/// Tokenizer training, loading or saving failed.
const EXIT_TOKENIZER: u8 = 6;
/// A stored identity hash does not match the artifact.
const EXIT_HASH_MISMATCH: u8 = 7;

#[derive(Parser)]
#[command(
    name = "tokenizer-cli",
    about = "Train and inspect Rust-native byte-level BPE tokenizer artifacts."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a byte-level BPE tokenizer from UTF-8 corpus files.
    ///
    /// Sequences are consumed in file order and line order, and the BPE trainer
    /// has no random seed, so the same corpus and parameters always produce the
    /// same artifact identity.
    Train {
        /// Experiment configuration describing the vocabulary contract.
        #[arg(long, value_name = "PATH")]
        config: PathBuf,
        /// UTF-8 corpus file. Repeat to concatenate several files in order.
        #[arg(long = "corpus", value_name = "PATH", required = true)]
        corpora: Vec<PathBuf>,
        /// Where to write the tokenizer artifact.
        #[arg(long, value_name = "PATH")]
        output: PathBuf,
        /// Minimum frequency a pair must reach before it is merged.
        #[arg(long, value_name = "N", default_value_t = 2)]
        min_frequency: u64,
    },

    /// Print the identity hash of an existing artifact and verify its sidecar.
    Hash {
        /// Experiment configuration describing the vocabulary contract.
        #[arg(long, value_name = "PATH")]
        config: PathBuf,
        /// Tokenizer artifact to inspect.
        #[arg(long, value_name = "PATH")]
        artifact: PathBuf,
    },
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Train {
            config,
            corpora,
            output,
            min_frequency,
        } => train(&config, &corpora, &output, min_frequency),
        Command::Hash { config, artifact } => hash(&config, &artifact),
    }
}

/// Loads an experiment configuration and returns its tokenizer section.
fn tokenizer_config(path: &Path) -> Result<TokenizerConfig, u8> {
    match ExperimentConfig::load(path) {
        Ok(config) => Ok(config.tokenizer),
        Err(error) => {
            eprintln!("tokenizer-cli: {error}");
            Err(EXIT_CONFIG)
        }
    }
}

fn train(config_path: &Path, corpora: &[PathBuf], output: &Path, min_frequency: u64) -> ExitCode {
    let config = match tokenizer_config(config_path) {
        Ok(config) => config,
        Err(code) => return ExitCode::from(code),
    };

    let mut corpus = match CorpusLines::open(corpora) {
        Ok(corpus) => corpus,
        Err(error) => {
            eprintln!("tokenizer-cli: {error}");
            return ExitCode::from(EXIT_CORPUS);
        }
    };
    let files = corpora.len();

    let tokenizer = match LlmTokenizer::train(corpus.by_ref(), &config, min_frequency) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            if let Some(corpus_error) = corpus.take_failure() {
                eprintln!("tokenizer-cli: {corpus_error}");
                return ExitCode::from(EXIT_CORPUS);
            }
            eprintln!("tokenizer-cli: {error}");
            return ExitCode::from(EXIT_TOKENIZER);
        }
    };

    if let Some(corpus_error) = corpus.take_failure() {
        eprintln!("tokenizer-cli: {corpus_error}");
        return ExitCode::from(EXIT_CORPUS);
    }

    if let Err(error) = tokenizer.save(output) {
        eprintln!("tokenizer-cli: {error}");
        return ExitCode::from(EXIT_TOKENIZER);
    }

    let identity = tokenizer.identity_hash();
    if let Err(error) = write_sidecar(output, &identity) {
        eprintln!("tokenizer-cli: {error}");
        return ExitCode::from(EXIT_TOKENIZER);
    }

    println!("tokenizer-cli: corpus files   = {files}");
    println!("tokenizer-cli: vocab size     = {}", tokenizer.vocab_size());
    println!("tokenizer-cli: artifact       = {}", output.display());
    println!("tokenizer-cli: identity hash  = {identity}");

    ExitCode::SUCCESS
}

fn hash(config_path: &Path, artifact: &Path) -> ExitCode {
    let config = match tokenizer_config(config_path) {
        Ok(config) => config,
        Err(code) => return ExitCode::from(code),
    };

    let tokenizer = match LlmTokenizer::load(artifact, &config) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("tokenizer-cli: {error}");
            return ExitCode::from(EXIT_TOKENIZER);
        }
    };

    let identity = tokenizer.identity_hash();
    println!("tokenizer-cli: vocab size     = {}", tokenizer.vocab_size());
    println!("tokenizer-cli: identity hash  = {identity}");

    match read_sidecar(artifact) {
        Ok(Some(stored)) if stored != identity => {
            eprintln!("tokenizer-cli: stored identity {stored} does not match the artifact");
            ExitCode::from(EXIT_HASH_MISMATCH)
        }
        Ok(Some(stored)) => {
            println!("tokenizer-cli: stored identity matches ({stored})");
            ExitCode::SUCCESS
        }
        Ok(None) => {
            println!("tokenizer-cli: no stored identity beside the artifact");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("tokenizer-cli: {error}");
            ExitCode::from(EXIT_TOKENIZER)
        }
    }
}

/// Path of the identity sidecar written next to an artifact.
fn sidecar_path(artifact: &Path) -> PathBuf {
    let mut path = artifact.as_os_str().to_owned();
    path.push(".sha256");
    PathBuf::from(path)
}

/// Writes the identity hash in the conventional `sha256sum` text format.
fn write_sidecar(artifact: &Path, identity: &str) -> Result<(), TokenizerError> {
    let path = sidecar_path(artifact);
    std::fs::write(&path, format!("{identity}\n"))
        .map_err(|source| TokenizerError::Write { path, source })
}

/// Reads a sidecar identity, if one exists.
fn read_sidecar(artifact: &Path) -> Result<Option<String>, TokenizerError> {
    let path = sidecar_path(artifact);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(text.trim().to_owned())),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(TokenizerError::Read { path, source }),
    }
}
