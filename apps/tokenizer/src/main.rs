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

use std::fs::File;
use std::io::{BufRead, BufReader};
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

    /// Report how efficiently an artifact encodes each corpus file.
    ///
    /// Exists because a shared tokenizer over three writing systems can look
    /// perfectly healthy — a vocab size, an identity hash, no errors — while
    /// having quietly allocated almost none of its vocabulary to one of them.
    /// The number that shows it is characters per token, and it has to be read
    /// per language: a Persian text needing three times the tokens of an English
    /// text of the same length has a tokenizer that was trained mostly on
    /// English, whatever its configuration says.
    Stats {
        /// Experiment configuration describing the vocabulary contract.
        #[arg(long, value_name = "PATH")]
        config: PathBuf,
        /// Tokenizer artifact to measure.
        #[arg(long, value_name = "PATH")]
        artifact: PathBuf,
        /// A UTF-8 file to measure. Repeat for several; each is reported apart.
        #[arg(long = "corpus", value_name = "PATH", required = true)]
        corpora: Vec<PathBuf>,
        /// Stop after this many lines of each file.
        #[arg(long, value_name = "N", default_value_t = 2000)]
        lines: usize,
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
        Command::Stats {
            config,
            artifact,
            corpora,
            lines,
        } => stats(&config, &artifact, &corpora, lines),
    }
}

/// Reports characters per token and tokens per line, per corpus file.
///
/// Characters rather than bytes, because a byte ratio would flatter Persian: the
/// Arabic script costs two bytes per character in UTF-8, so a Persian text that
/// tokenizes badly can still look acceptable when measured in bytes. Characters
/// are the honest denominator across scripts.
fn stats(config_path: &Path, artifact: &Path, corpora: &[PathBuf], limit: usize) -> ExitCode {
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

    println!("tokenizer-cli: vocab size = {}", tokenizer.vocab_size());
    println!(
        "tokenizer-cli: {:<28} {:>10} {:>10} {:>12}",
        "file", "chars", "tokens", "chars/token"
    );

    for path in corpora {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("tokenizer-cli: {}: {error}", path.display());
                return ExitCode::from(EXIT_CORPUS);
            }
        };

        let mut characters: u64 = 0;
        let mut tokens: u64 = 0;

        for line in BufReader::new(file).lines().take(limit) {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    eprintln!("tokenizer-cli: {}: {error}", path.display());
                    return ExitCode::from(EXIT_CORPUS);
                }
            };
            characters += line.chars().count() as u64;
            match tokenizer.encode(&line) {
                Ok(encoded) => tokens += encoded.len() as u64,
                Err(error) => {
                    eprintln!("tokenizer-cli: {}: {error}", path.display());
                    return ExitCode::from(EXIT_TOKENIZER);
                }
            }
        }

        let per_token = if tokens == 0 {
            0.0
        } else {
            characters as f64 / tokens as f64
        };
        let label = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        println!("tokenizer-cli: {label:<28} {characters:>10} {tokens:>10} {per_token:>12.2}");
    }

    ExitCode::SUCCESS
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
