//! Typed failures for the training composition root.
//!
//! Every layer below this one already has its own error type, so this one exists
//! to join them in one place and to name the failures that only the composition
//! root can detect — a dataset missing the split a run asked for, or a tokenizer
//! that is not the one the dataset was prepared with.

use std::path::PathBuf;

use thiserror::Error;

/// Failure while preparing or running a training run.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TrainError {
    /// The configuration could not be read or is not valid.
    #[error(transparent)]
    Config(#[from] llm_core::ConfigError),

    /// The dataset could not be read.
    #[error(transparent)]
    Data(#[from] llm_data::DataError),

    /// The tokenizer could not be read.
    #[error(transparent)]
    Tokenizer(#[from] llm_tokenizer::TokenizerError),

    /// The configuration names no tokenizer artifact.
    ///
    /// Training cannot invent a vocabulary, and a run without one would have to
    /// guess which ids the dataset was written with.
    #[error("tokenizer.artifact is not set in the configuration")]
    TokenizerArtifactMissing,

    /// The model rejected an input or could not be built.
    #[error(transparent)]
    Model(#[from] llm_model::ModelError),

    /// A training step failed.
    #[error(transparent)]
    Training(#[from] llm_training::TrainingError),

    /// A checkpoint could not be written.
    #[error(transparent)]
    Checkpoint(#[from] llm_checkpoint::CheckpointError),

    /// The dataset manifest is not where the configuration says it is.
    #[error("no dataset manifest at `{path}`; run the prepare application first")]
    DatasetMissing {
        /// Path that was looked for.
        path: PathBuf,
    },

    /// The dataset holds no shard for a split the run needs.
    #[error("the dataset has no `{split}` shard")]
    SplitMissing {
        /// Split that was needed.
        split: &'static str,
    },

    /// The dataset was prepared with a different tokenizer.
    ///
    /// Token ids only mean something relative to the tokenizer that produced
    /// them, so this is a refusal rather than a warning: training on ids read
    /// through the wrong vocabulary produces a model that learns noise.
    #[error(
        "the dataset was prepared with tokenizer {manifest}, but {tokenizer} was loaded; \
         re-prepare the dataset with the tokenizer this run uses"
    )]
    TokenizerIdentityMismatch {
        /// Identity the manifest records.
        manifest: String,
        /// Identity of the tokenizer actually loaded.
        tokenizer: String,
    },

    /// A shard holds no sequences, so it cannot produce a batch.
    #[error("shard `{path}` holds no sequences")]
    EmptyShard {
        /// Path of the empty shard.
        path: PathBuf,
    },

    /// A checkpoint's cursor names a shard this run is not reading.
    ///
    /// Refused rather than ignored. The training reader opens one shard, so a
    /// cursor naming another one describes a different run -- and resuming from it
    /// at shard zero would silently continue from the wrong place in the data.
    // Constructed by `ShardBatches::seek_cursor`, which only the tests call until
    // the `--resume` path exists. See that method.
    #[allow(dead_code)]
    #[error("checkpoint cursor names shard {shard}, but `{path}` is the only shard this run reads")]
    CursorShardMismatch {
        /// Shard this run is reading.
        path: PathBuf,
        /// Shard the cursor names.
        shard: u64,
    },

    /// Checkpoints from an earlier run are already in the way.
    ///
    /// Checkpoint publication refuses to replace an existing directory, because
    /// two runs writing the same names would produce a directory that is neither
    /// run's state. That refusal is right; discovering it twenty-five steps into
    /// a run is not, so it is checked before the first step instead.
    #[error(
        "`{path}` already holds checkpoints from an earlier run; move or remove it first, \
         because a new run must not overwrite the state another run produced"
    )]
    CheckpointRootNotEmpty {
        /// Directory that is already in use.
        path: PathBuf,
    },
}
