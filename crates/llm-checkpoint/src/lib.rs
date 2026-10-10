//! Checkpoints: what is persisted, and how it is published.
//!
//! A checkpoint must let a run continue exactly where it stopped. Model weights
//! alone do not do that: an optimizer restarted from zero changes every update
//! that follows, and a step counter that resets changes the learning rate. The
//! envelope here therefore carries the optimizer state, the schedule position
//! (which is the step number), the token count, the data cursor and the
//! identities of everything the run depended on.
//!
//! # Atomicity
//!
//! A checkpoint is never written in place. It is built inside a temporary
//! directory, verified by reading it back, and only then renamed to its final
//! name, which is atomic on one filesystem. The `latest` pointer is published
//! last, so a crash at any point leaves either the previous good checkpoint or
//! a temporary directory that nothing refers to — never a half-written
//! checkpoint that looks valid.
//!
//! # Identity, not just weights
//!
//! A checkpoint records the hashes of the tokenizer, the dataset manifest and
//! the resolved configuration. Resuming against a different dataset or a
//! different tokenizer would silently train on something other than the run it
//! claims to continue, so those mismatches are rejections.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::optim::ModuleOptimizer;
use llm_model::Transformer;
use serde::{Deserialize, Serialize};

pub mod error;

pub use error::CheckpointError;

/// Format version of a checkpoint envelope.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Name of the file holding the envelope.
pub const METADATA_FILE: &str = "metadata.json";
/// Name of the file holding the model parameters.
pub const MODEL_FILE: &str = "model.bpk";
/// Name of the file holding the optimizer state.
pub const OPTIMIZER_FILE: &str = "optimizer.bpk";
/// Name of the pointer that names the newest complete checkpoint.
pub const LATEST_FILE: &str = "latest.json";

/// Where the data stream had reached when the checkpoint was written.
///
/// Part of the envelope from the first version: a format that cannot record
/// where reading stopped cannot resume a sharded dataset without resuming on
/// the wrong tokens, and adding the field later would invalidate every existing
/// checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataCursor {
    /// Pass over the dataset, counted from zero.
    pub epoch: u64,
    /// Shard index within the epoch.
    pub shard: u64,
    /// Sequence index within the shard.
    pub sequence: u64,
}

/// Everything a resumed run needs that is not a parameter tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointMetadata {
    /// Envelope format version.
    pub format_version: u32,
    /// When the checkpoint was written, in seconds since the Unix epoch.
    pub created_unix_seconds: u64,
    /// Optimizer steps completed. Also the schedule position.
    pub global_step: u64,
    /// Tokens consumed by the completed steps.
    pub tokens_seen: u64,
    /// Where the data stream had reached.
    pub data_cursor: DataCursor,
    /// Identity of the resolved configuration.
    pub config_hash: String,
    /// Identity of the tokenizer the run uses.
    pub tokenizer_identity: String,
    /// Identity of the prepared dataset.
    pub dataset_manifest_hash: String,
}

/// What a caller must agree with before a checkpoint may be used.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointExpectation<'a> {
    /// Tokenizer identity the caller is using.
    pub tokenizer_identity: &'a str,
    /// Dataset manifest hash the caller is using.
    pub dataset_manifest_hash: &'a str,
    /// Configuration hash the caller is using, when one is known.
    pub config_hash: Option<&'a str>,
}

/// Publishes a checkpoint atomically and returns the directory it occupies.
///
/// `name` is the final directory name, for example `checkpoint-0000000007`. The
/// checkpoint is built at `<root>/.tmp-<name>`, verified, then renamed.
pub fn save(
    root: impl AsRef<Path>,
    name: &str,
    model: &Transformer,
    optimizer: &ModuleOptimizer,
    metadata: &CheckpointMetadata,
) -> Result<PathBuf, CheckpointError> {
    let root = root.as_ref();
    if name.is_empty() || name.contains(['/', '\\']) {
        return Err(CheckpointError::InvalidName {
            name: name.to_owned(),
        });
    }

    std::fs::create_dir_all(root).map_err(|source| CheckpointError::Directory {
        path: root.to_path_buf(),
        source,
    })?;

    let temporary = root.join(format!(".tmp-{name}"));
    let final_path = root.join(name);

    // A stale temporary directory from an interrupted run is not a checkpoint.
    if temporary.exists() {
        std::fs::remove_dir_all(&temporary).map_err(|source| CheckpointError::Directory {
            path: temporary.clone(),
            source,
        })?;
    }
    if final_path.exists() {
        return Err(CheckpointError::AlreadyExists { path: final_path });
    }

    std::fs::create_dir_all(&temporary).map_err(|source| CheckpointError::Directory {
        path: temporary.clone(),
        source,
    })?;

    write_metadata(&temporary, metadata)?;

    // `save_file` consumes the module, so the caller keeps its own copy.
    model
        .clone()
        .save_file(temporary.join(MODEL_FILE))
        .map_err(|source| CheckpointError::ModelSave { source })?;

    // The optimizer carries AdamW's moment estimates. A resume that restarts
    // them from zero is not the run it claims to continue, so this is as
    // necessary as the weights.
    optimizer
        .save(temporary.join(OPTIMIZER_FILE))
        .map_err(|source| CheckpointError::OptimizerSave { source })?;

    // Verify by reading back, so a write that silently produced an unusable
    // directory is caught before it becomes the newest checkpoint.
    validate(&temporary)?;

    std::fs::rename(&temporary, &final_path).map_err(|source| CheckpointError::Publish {
        path: final_path.clone(),
        source,
    })?;

    publish_latest(root, name)?;
    Ok(final_path)
}

/// Reads a checkpoint's envelope without loading its parameters.
pub fn validate(path: impl AsRef<Path>) -> Result<CheckpointMetadata, CheckpointError> {
    let path = path.as_ref();
    let metadata = read_metadata(path)?;

    if metadata.format_version != CHECKPOINT_FORMAT_VERSION {
        return Err(CheckpointError::UnsupportedVersion {
            path: path.to_path_buf(),
            found: metadata.format_version,
            supported: CHECKPOINT_FORMAT_VERSION,
        });
    }

    for required in [MODEL_FILE, OPTIMIZER_FILE] {
        if !path.join(required).is_file() {
            return Err(CheckpointError::Incomplete {
                path: path.to_path_buf(),
                missing: required,
            });
        }
    }

    Ok(metadata)
}

/// Loads a model from a checkpoint, checking it against the caller's identities.
///
/// `blank` supplies the architecture; Burn loads a record into an existing
/// module rather than constructing one from the file.
pub fn load_model(
    path: impl AsRef<Path>,
    blank: Transformer,
    expectation: &CheckpointExpectation<'_>,
) -> Result<(Transformer, CheckpointMetadata), CheckpointError> {
    let path = path.as_ref();
    let metadata = validate(path)?;

    if metadata.tokenizer_identity != expectation.tokenizer_identity {
        return Err(CheckpointError::TokenizerMismatch {
            expected: expectation.tokenizer_identity.to_owned(),
            found: metadata.tokenizer_identity,
        });
    }
    if metadata.dataset_manifest_hash != expectation.dataset_manifest_hash {
        return Err(CheckpointError::DatasetMismatch {
            expected: expectation.dataset_manifest_hash.to_owned(),
            found: metadata.dataset_manifest_hash,
        });
    }
    if let Some(expected) = expectation.config_hash
        && metadata.config_hash != expected
    {
        return Err(CheckpointError::ConfigMismatch {
            expected: expected.to_owned(),
            found: metadata.config_hash,
        });
    }

    let model = blank
        .try_load_file(path.join(MODEL_FILE))
        .map_err(|source| CheckpointError::ModelLoad { source })?;

    Ok((model, metadata))
}

/// Restores an optimizer from a checkpoint.
///
/// Burn restores into an existing optimizer rather than constructing one from
/// the file, so the caller supplies the optimizer its configuration just built
/// and receives it with the stored state applied.
pub fn load_optimizer(
    path: impl AsRef<Path>,
    optimizer: ModuleOptimizer,
) -> Result<ModuleOptimizer, CheckpointError> {
    optimizer
        .load(path.as_ref().join(OPTIMIZER_FILE))
        .map_err(|source| CheckpointError::OptimizerLoad { source })
}

/// Name of the newest complete checkpoint, if one has been published.
pub fn latest(root: impl AsRef<Path>) -> Result<Option<String>, CheckpointError> {
    let file = root.as_ref().join(LATEST_FILE);
    match std::fs::read_to_string(&file) {
        Ok(text) => {
            let pointer: LatestPointer = serde_json::from_str(&text)
                .map_err(|source| CheckpointError::Latest { path: file, source })?;
            Ok(Some(pointer.checkpoint))
        }
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(CheckpointError::Read { path: file, source }),
    }
}

/// Resolves the newest complete checkpoint to a path.
pub fn latest_path(root: impl AsRef<Path>) -> Result<Option<PathBuf>, CheckpointError> {
    let root = root.as_ref();
    Ok(latest(root)?.map(|name| root.join(name)))
}

/// The `latest` pointer document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LatestPointer {
    checkpoint: String,
}

/// Writes the pointer to the newest checkpoint, atomically.
fn publish_latest(root: &Path, name: &str) -> Result<(), CheckpointError> {
    let pointer = LatestPointer {
        checkpoint: name.to_owned(),
    };
    let text = serde_json::to_string_pretty(&pointer).map_err(CheckpointError::LatestEncode)?;

    // Written to a temporary name and renamed, so a reader never sees a partial
    // pointer and concludes that no checkpoint exists.
    let temporary = root.join(format!(".tmp-{LATEST_FILE}"));
    std::fs::write(&temporary, text).map_err(|source| CheckpointError::Write {
        path: temporary.clone(),
        source,
    })?;
    std::fs::rename(&temporary, root.join(LATEST_FILE)).map_err(|source| CheckpointError::Publish {
        path: root.join(LATEST_FILE),
        source,
    })
}

/// Writes the envelope into a checkpoint directory.
fn write_metadata(directory: &Path, metadata: &CheckpointMetadata) -> Result<(), CheckpointError> {
    let path = directory.join(METADATA_FILE);
    let text = serde_json::to_string_pretty(metadata).map_err(CheckpointError::Encode)?;
    std::fs::write(&path, text).map_err(|source| CheckpointError::Write { path, source })
}

/// Reads the envelope from a checkpoint directory.
fn read_metadata(directory: &Path) -> Result<CheckpointMetadata, CheckpointError> {
    let path = directory.join(METADATA_FILE);
    let text = std::fs::read_to_string(&path).map_err(|source| CheckpointError::Read {
        path: path.clone(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| CheckpointError::Decode {
        path: path.clone(),
        source,
    })
}
