//! Typed checkpoint failures.
//!
//! Every mismatch is a rejection rather than a warning: a checkpoint accepted
//! against the wrong tokenizer, dataset or configuration would silently train
//! something other than the run it claims to continue.

use std::path::PathBuf;

use thiserror::Error;
/// Failure while writing, reading or accepting a checkpoint.
///
/// Every mismatch is a rejection rather than a warning: a checkpoint accepted
/// against the wrong tokenizer, dataset or configuration would silently train
/// something other than the run it claims to continue.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// The requested checkpoint name is not usable as a directory name.
    #[error("`{name}` is not a usable checkpoint name")]
    InvalidName {
        /// The rejected name.
        name: String,
    },

    /// A checkpoint directory could not be created or removed.
    #[error("failed to prepare checkpoint directory `{path}`")]
    Directory {
        /// Directory involved.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A checkpoint with that name already exists.
    #[error("checkpoint `{path}` already exists")]
    AlreadyExists {
        /// Path that already exists.
        path: PathBuf,
    },

    /// A checkpoint file could not be written.
    #[error("failed to write `{path}`")]
    Write {
        /// Path that was written.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A checkpoint file could not be read.
    #[error("failed to read `{path}`")]
    Read {
        /// Path that was read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The envelope could not be serialized.
    #[error("failed to serialize the checkpoint envelope")]
    Encode(#[source] serde_json::Error),

    /// The envelope could not be parsed.
    #[error("failed to parse the checkpoint envelope `{path}`")]
    Decode {
        /// Path that was parsed.
        path: PathBuf,
        /// The underlying parser failure.
        #[source]
        source: serde_json::Error,
    },

    /// The `latest` pointer could not be parsed.
    #[error("failed to parse the checkpoint pointer `{path}`")]
    Latest {
        /// Path that was parsed.
        path: PathBuf,
        /// The underlying parser failure.
        #[source]
        source: serde_json::Error,
    },

    /// The `latest` pointer could not be serialized.
    #[error("failed to serialize the checkpoint pointer")]
    LatestEncode(#[source] serde_json::Error),

    /// The model parameters could not be saved.
    #[error("failed to save the model parameters")]
    ModelSave {
        /// The underlying store failure.
        #[source]
        source: burn::store::RecordError,
    },

    /// The model parameters could not be loaded.
    #[error("failed to load the model parameters")]
    ModelLoad {
        /// The underlying store failure.
        #[source]
        source: burn::store::RecordError,
    },

    /// The optimizer state could not be saved.
    #[error("failed to save the optimizer state")]
    OptimizerSave {
        /// The underlying store failure.
        #[source]
        source: burn::store::RecordError,
    },

    /// The optimizer state could not be loaded.
    #[error("failed to load the optimizer state")]
    OptimizerLoad {
        /// The underlying store failure.
        #[source]
        source: burn::store::RecordError,
    },

    /// The envelope declares a format version this build does not read.
    #[error("checkpoint `{path}` has format version {found}, but this build reads {supported}")]
    UnsupportedVersion {
        /// Path that was inspected.
        path: PathBuf,
        /// Version found.
        found: u32,
        /// Version this build reads and writes.
        supported: u32,
    },

    /// A required file is missing from the checkpoint.
    #[error("checkpoint `{path}` is incomplete: `{missing}` is missing")]
    Incomplete {
        /// Directory that was inspected.
        path: PathBuf,
        /// File that should have been present.
        missing: &'static str,
    },

    /// The checkpoint belongs to a different tokenizer.
    #[error("checkpoint was written with tokenizer {found}, but {expected} was required")]
    TokenizerMismatch {
        /// Identity the caller required.
        expected: String,
        /// Identity the checkpoint recorded.
        found: String,
    },

    /// The checkpoint belongs to a different dataset.
    #[error("checkpoint was written over dataset {found}, but {expected} was required")]
    DatasetMismatch {
        /// Hash the caller required.
        expected: String,
        /// Hash the checkpoint recorded.
        found: String,
    },

    /// The checkpoint belongs to a different configuration.
    #[error("checkpoint was written with configuration {found}, but {expected} was required")]
    ConfigMismatch {
        /// Hash the caller required.
        expected: String,
        /// Hash the checkpoint recorded.
        found: String,
    },

    /// The checkpoint could not be renamed into place.
    #[error("failed to publish `{path}`")]
    Publish {
        /// Destination that was attempted.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
}
