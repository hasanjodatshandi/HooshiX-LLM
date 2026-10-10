//! Command-line entry point for preparing a dataset from a corpus.
//!
//! This is a composition root: it resolves configuration and declaration files,
//! loads the tokenizer artifact, invokes the pipeline, and reports what was
//! produced. No pipeline logic lives here.
//!
//! Exit codes are distinct per failure class:
//!
//! | code | meaning |
//! |---|---|
//! | 0 | success |
//! | 2 | configuration could not be read, parsed or validated |
//! | 5 | the source declaration file could not be read or parsed |
//! | 6 | the tokenizer artifact could not be loaded |
//! | 7 | corpus preparation failed |
//! | 8 | a produced shard failed validation |

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use llm_core::ExperimentConfig;
use llm_data::{
    DatasetManifest, InvalidUtf8Policy, PrepareRequest, ShardExpectation, SourceSpec, TailPolicy,
    prepare, validate_shard,
};
use llm_tokenizer::LlmTokenizer;

/// Configuration could not be read, parsed or validated.
const EXIT_CONFIG: u8 = 2;
/// The source declaration file could not be read or parsed.
const EXIT_SOURCES: u8 = 5;
/// The tokenizer artifact could not be loaded.
const EXIT_TOKENIZER: u8 = 6;
/// Corpus preparation failed.
const EXIT_PIPELINE: u8 = 7;
/// A produced shard failed validation.
const EXIT_VALIDATION: u8 = 8;

#[derive(Parser)]
#[command(
    name = "prepare-cli",
    about = "Prepare checksummed token shards from declared corpus sources."
)]
struct Cli {
    /// Experiment configuration describing the vocabulary and data contract.
    #[arg(long, value_name = "PATH")]
    config: PathBuf,
    /// Tokenizer artifact the shards are bound to.
    #[arg(long, value_name = "PATH")]
    tokenizer: PathBuf,
    /// JSON array of source declarations, each with provenance and a path.
    #[arg(long, value_name = "PATH")]
    sources: PathBuf,
    /// Directory the shards and the dataset manifest are written into.
    #[arg(long, value_name = "PATH")]
    output: PathBuf,
    /// Minimum document length in characters.
    #[arg(long, value_name = "N", default_value_t = 32)]
    min_chars: usize,
    /// Keep an incomplete final window as a shorter sequence.
    ///
    /// Dropping it is the default, so every emitted sequence has the configured
    /// length and batches need no padding.
    #[arg(long)]
    keep_tail: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let config = match ExperimentConfig::load(&cli.config) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("prepare-cli: {error}");
            return ExitCode::from(EXIT_CONFIG);
        }
    };

    let tokenizer = match LlmTokenizer::load(&cli.tokenizer, &config.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("prepare-cli: {error}");
            return ExitCode::from(EXIT_TOKENIZER);
        }
    };

    let sources = match load_sources(&cli.sources) {
        Ok(sources) => sources,
        Err(message) => {
            eprintln!("prepare-cli: {message}");
            return ExitCode::from(EXIT_SOURCES);
        }
    };

    let request = PrepareRequest {
        output_root: cli.output,
        sources,
        data: config.data.clone(),
        filters: llm_data::FilterConfig {
            min_chars: cli.min_chars,
            ..llm_data::FilterConfig::default()
        },
        invalid_utf8: InvalidUtf8Policy::Reject,
        tail: if cli.keep_tail {
            TailPolicy::Keep
        } else {
            TailPolicy::Drop
        },
    };

    let manifest = match prepare(&request, &tokenizer) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("prepare-cli: {error}");
            return ExitCode::from(EXIT_PIPELINE);
        }
    };

    report(&manifest, &request.output_root);
    verify_shards(&manifest, &request.output_root)
}

/// Reads the declared sources, naming the file when it cannot be used.
fn load_sources(path: &PathBuf) -> Result<Vec<SourceSpec>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read sources from `{}`: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse sources in `{}`: {error}", path.display()))
}

/// Prints the resolved facts a prepared dataset must report.
fn report(manifest: &DatasetManifest, root: &Path) {
    println!("prepare-cli: dataset root  = {}", root.display());
    println!(
        "prepare-cli: tokenizer     = {}",
        manifest.tokenizer_identity
    );
    println!("prepare-cli: sequence len  = {}", manifest.sequence_length);
    println!(
        "prepare-cli: documents     = {} read, {} kept, {} filtered, {} duplicate",
        manifest.counts.documents_read,
        manifest.counts.documents_kept,
        manifest.counts.documents_dropped,
        manifest.counts.documents_duplicate
    );

    for split in &manifest.splits {
        println!(
            "prepare-cli: {:<10} = {} documents, {} sequences, {} tokens, \
             usable ratio {:.4}, {:.2} documents per sequence",
            split.split.as_str(),
            split.documents,
            split.sequences,
            split.tokens,
            split.usable_token_ratio(),
            split.documents_per_sequence()
        );
    }

    for source in &manifest.sources {
        println!(
            "prepare-cli: source `{}` = {} bytes, {} documents, digest {}",
            source.declared.source_id,
            source.raw_bytes,
            source.documents,
            &source.raw_digest[..16]
        );
        for reason in &source.documents_dropped {
            println!(
                "prepare-cli:   dropped {} for {}",
                reason.count, reason.reason
            );
        }
    }

    match manifest.content_hash() {
        Ok(hash) => println!("prepare-cli: content hash  = {hash}"),
        Err(error) => eprintln!("prepare-cli: could not hash the manifest: {error}"),
    }
}

/// Re-reads every produced shard, so a silent write defect cannot pass.
fn verify_shards(manifest: &DatasetManifest, root: &Path) -> ExitCode {
    let expectation = ShardExpectation {
        tokenizer_identity: &manifest.tokenizer_identity,
        sequence_length: manifest.sequence_length,
    };

    for summary in &manifest.shards {
        let path = root.join(&summary.file_name);
        match validate_shard(&path, &expectation) {
            Ok(validated) if validated == *summary => {
                println!(
                    "prepare-cli: verified {} = {} sequences, {} tokens",
                    summary.file_name, summary.sequence_count, summary.token_count
                );
            }
            Ok(validated) => {
                eprintln!(
                    "prepare-cli: `{}` validated but reports {} sequences instead of {}",
                    summary.file_name, validated.sequence_count, summary.sequence_count
                );
                return ExitCode::from(EXIT_VALIDATION);
            }
            Err(error) => {
                eprintln!("prepare-cli: {error}");
                return ExitCode::from(EXIT_VALIDATION);
            }
        }
    }

    ExitCode::SUCCESS
}
