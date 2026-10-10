//! Typed data-pipeline failures.
//!
//! Raw corpora are untrusted input, so every failure here names the artifact it
//! came from — a path, a source, a line — and never embeds the record itself.

use std::path::PathBuf;

use thiserror::Error;

/// Failure while reading, preparing or validating corpus data.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DataError {
    /// A corpus file could not be opened.
    #[error("failed to open corpus source `{path}`")]
    SourceOpen {
        /// Path that was opened.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A corpus file could not be read to the end.
    #[error("failed while reading corpus source `{path}`")]
    SourceRead {
        /// Path that was being read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A corpus file is not valid UTF-8 and the policy rejects such input.
    #[error("source `{path}` is not valid UTF-8 at byte offset {offset}")]
    InvalidUtf8 {
        /// Path that was being read.
        path: PathBuf,
        /// Byte offset of the first invalid sequence.
        offset: u64,
    },

    /// A source declares provenance this project cannot accept.
    #[error("source `{source_id}` has unusable provenance: {reason}")]
    UnusableProvenance {
        /// Identifier of the offending source.
        source_id: String,
        /// Why the declaration was rejected.
        reason: &'static str,
    },

    /// A JSON Lines document could not be parsed.
    #[error("source `{path}` line {line} is not a valid JSON object")]
    MalformedJsonLine {
        /// Path that was being read.
        path: PathBuf,
        /// One-based line number.
        line: u64,
        /// The underlying parser failure.
        #[source]
        source: serde_json::Error,
    },

    /// A JSON Lines document does not carry the configured text field.
    #[error("source `{path}` line {line} has no non-empty `{field}` string")]
    MissingTextField {
        /// Path that was being read.
        path: PathBuf,
        /// One-based line number.
        line: u64,
        /// Field name the source declared.
        field: String,
    },

    /// A directory needed by the pipeline could not be created.
    #[error("failed to create dataset directory `{path}`")]
    DirectoryCreate {
        /// Directory that was requested.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A dataset artifact could not be written.
    #[error("failed to write dataset artifact `{path}`")]
    Write {
        /// Path that was written.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A dataset artifact could not be read.
    #[error("failed to read dataset artifact `{path}`")]
    Read {
        /// Path that was read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The file does not begin with the shard magic.
    #[error("`{path}` is not a dataset shard")]
    NotAShard {
        /// Path that was inspected.
        path: PathBuf,
    },

    /// The shard was written by a different format version.
    #[error("shard `{path}` has format version {found}, but this build reads {supported}")]
    ShardVersion {
        /// Path that was inspected.
        path: PathBuf,
        /// Version found in the shard.
        found: u32,
        /// Version this build reads and writes.
        supported: u32,
    },

    /// The shard was prepared with a different tokenizer.
    ///
    /// Token ids only mean something relative to the tokenizer that produced
    /// them, so this is a hard incompatibility rather than a warning.
    #[error("shard `{path}` was prepared with tokenizer {found}, but {expected} was required")]
    TokenizerMismatch {
        /// Path that was inspected.
        path: PathBuf,
        /// Identity the reader required.
        expected: String,
        /// Identity recorded in the shard.
        found: String,
    },

    /// The shard is shorter than its own header and trailer require.
    #[error("shard `{path}` is truncated: expected at least {expected} bytes, found {found}")]
    ShardTruncated {
        /// Path that was inspected.
        path: PathBuf,
        /// Smallest valid length.
        expected: u64,
        /// Actual length.
        found: u64,
    },

    /// The payload length is not a whole number of sequences.
    #[error("shard `{path}` holds {tokens} tokens, which is not a multiple of {sequence_length}")]
    ShardNotAligned {
        /// Path that was inspected.
        path: PathBuf,
        /// Token count found.
        tokens: u64,
        /// Sequence length the shard declares.
        sequence_length: usize,
    },

    /// The stored checksum does not match the contents.
    #[error("shard `{path}` checksum does not match its contents")]
    ShardChecksum {
        /// Path that was inspected.
        path: PathBuf,
    },

    /// A sequence was pushed to a shard with the wrong length.
    #[error("shard `{path}` received a sequence of {found} tokens, expected {expected}")]
    ShardSequenceLength {
        /// Path that was written.
        path: PathBuf,
        /// Sequence length the shard declares.
        expected: usize,
        /// Length that was pushed.
        found: usize,
    },

    /// The dataset manifest could not be parsed.
    #[error("failed to parse dataset manifest `{path}`")]
    ManifestParse {
        /// Path that was read.
        path: PathBuf,
        /// The underlying parser failure.
        #[source]
        source: serde_json::Error,
    },

    /// The dataset manifest could not be serialized.
    #[error("failed to serialize the dataset manifest")]
    ManifestSerialize(#[source] serde_json::Error),

    /// The tokenizer configuration is invalid.
    #[error(transparent)]
    Config(#[from] llm_core::ConfigError),

    /// The tokenizer failed.
    #[error(transparent)]
    Tokenizer(#[from] llm_tokenizer::TokenizerError),
}
