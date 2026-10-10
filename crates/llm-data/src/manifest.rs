//! The dataset manifest.
//!
//! A prepared dataset is only meaningful together with the rules that produced
//! it, so the manifest records every ruleset version, the tokenizer identity,
//! the split policy, the provenance of each source, the counters, and the
//! checksum of every shard.
//!
//! # Determinism
//!
//! The manifest carries a creation timestamp, which is not reproducible. What a
//! run depends on is [`DatasetManifest::content_hash`], computed over everything
//! *except* that timestamp, so two preparations of the same corpus with the same
//! configuration agree regardless of when they ran.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dedup::DEDUP_VERSION;
use crate::error::DataError;
use crate::filter::FILTER_VERSION;
use crate::normalize::NORMALIZATION_VERSION;
use crate::pack::PACKING_VERSION;
use crate::shard::ShardSummary;
use crate::source::SourceManifest;
use crate::split::{SPLIT_VERSION, Split};

/// Format version of the manifest document.
pub const MANIFEST_VERSION: u32 = 1;

/// Split policy as recorded in the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitPolicySummary {
    /// Seed mixed into the content hash.
    pub seed: u64,
    /// Share held out for validation, in basis points.
    pub validation_basis_points: u16,
    /// Share held out for testing, in basis points.
    pub test_basis_points: u16,
}

/// One counted value, so reasons survive serialization without a map's
/// unordered keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountedReason {
    /// Stable reason name.
    pub reason: String,
    /// How many documents carried it.
    pub count: u64,
}

/// What one source contributed.
///
/// Declared provenance and measured facts sit side by side, because a manifest
/// that states only one of them cannot be audited.
///
/// Unknown fields are not denied here: `serde` does not support
/// `deny_unknown_fields` together with `flatten`, and the flattened provenance
/// struct denies them itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSummary {
    /// Provenance as declared by the provider.
    #[serde(flatten)]
    pub declared: SourceManifest,
    /// Size of the source file in bytes.
    pub raw_bytes: u64,
    /// Hexadecimal SHA-256 of the raw file bytes.
    pub raw_digest: String,
    /// Documents the file yielded.
    pub documents: u64,
    /// Documents that survived normalization and filtering.
    pub documents_kept: u64,
    /// Documents dropped, by reason.
    pub documents_dropped: Vec<CountedReason>,
    /// Documents dropped as exact duplicates of an earlier document.
    pub documents_duplicate: u64,
}

/// Corpus-wide counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetCounts {
    /// Documents read from all sources.
    pub documents_read: u64,
    /// Documents kept after filtering and deduplication.
    pub documents_kept: u64,
    /// Documents dropped by a filter.
    pub documents_dropped: u64,
    /// Documents dropped as duplicates.
    pub documents_duplicate: u64,
    /// Tokens written into shards.
    pub tokens: u64,
}

/// What one split received.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitSummary {
    /// The split.
    pub split: Split,
    /// Documents assigned to it.
    pub documents: u64,
    /// Sequences written.
    pub sequences: u64,
    /// Tokens written.
    pub tokens: u64,
    /// Tokens discarded because they formed an incomplete final window.
    ///
    /// Segmentation is exact, so this is the only token loss packing can cause,
    /// and it is recorded rather than inferred.
    pub dropped_tail_tokens: u64,
}

impl SplitSummary {
    /// Share of tokenized tokens that reached a shard.
    ///
    /// One means nothing was discarded. A value below one is the cost of
    /// emitting only whole windows.
    #[must_use]
    pub fn usable_token_ratio(&self) -> f64 {
        let pushed = self.tokens + self.dropped_tail_tokens;
        if pushed == 0 {
            return 1.0;
        }
        self.tokens as f64 / pushed as f64
    }

    /// Average number of documents packed into one sequence.
    #[must_use]
    pub fn documents_per_sequence(&self) -> f64 {
        if self.sequences == 0 {
            return 0.0;
        }
        self.documents as f64 / self.sequences as f64
    }
}

/// Everything a prepared dataset asserts about itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifest {
    /// Format version of this document.
    pub manifest_version: u32,
    /// When preparation ran, in seconds since the Unix epoch.
    pub created_unix_seconds: u64,
    /// Normalization ruleset.
    pub normalization_version: String,
    /// Filter ruleset.
    pub filter_version: String,
    /// Deduplication ruleset.
    pub dedup_version: String,
    /// Split ruleset.
    pub split_version: String,
    /// Packing ruleset.
    pub packing_version: String,
    /// Tokenizer the shards were prepared with.
    pub tokenizer_identity: String,
    /// Tokens per sequence.
    pub sequence_length: usize,
    /// Split policy used.
    pub split_policy: SplitPolicySummary,
    /// Per-source provenance and measurements.
    pub sources: Vec<SourceSummary>,
    /// Corpus-wide counters.
    pub counts: DatasetCounts,
    /// Per-split counters.
    pub splits: Vec<SplitSummary>,
    /// One entry per written shard.
    pub shards: Vec<ShardSummary>,
}

impl DatasetManifest {
    /// Builds a manifest with the current ruleset versions.
    #[must_use]
    pub fn new(
        created_unix_seconds: u64,
        tokenizer_identity: String,
        sequence_length: usize,
        split_policy: SplitPolicySummary,
    ) -> Self {
        Self {
            manifest_version: MANIFEST_VERSION,
            created_unix_seconds,
            normalization_version: NORMALIZATION_VERSION.to_owned(),
            filter_version: FILTER_VERSION.to_owned(),
            dedup_version: DEDUP_VERSION.to_owned(),
            split_version: SPLIT_VERSION.to_owned(),
            packing_version: PACKING_VERSION.to_owned(),
            tokenizer_identity,
            sequence_length,
            split_policy,
            sources: Vec::new(),
            counts: DatasetCounts {
                documents_read: 0,
                documents_kept: 0,
                documents_dropped: 0,
                documents_duplicate: 0,
                tokens: 0,
            },
            splits: Vec::new(),
            shards: Vec::new(),
        }
    }

    /// Stable hash of everything except the creation timestamp.
    ///
    /// This is the value a training run records, because it identifies the data
    /// rather than the moment it was prepared. Shard paths are included, so
    /// preparing into a differently named directory deliberately produces a
    /// different identity.
    pub fn content_hash(&self) -> Result<String, DataError> {
        let mut canonical = self.clone();
        canonical.created_unix_seconds = 0;
        let bytes = serde_json::to_vec(&canonical).map_err(DataError::ManifestSerialize)?;

        let mut hasher = Sha256::new();
        hasher.update(b"llm-dataset-manifest-v1\0");
        hasher.update(&bytes);

        let digest = hasher.finalize();
        const DIGITS: [u8; 16] = *b"0123456789abcdef";
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            hex.push(char::from(DIGITS[usize::from(byte >> 4)]));
            hex.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
        }
        Ok(hex)
    }

    /// Writes the manifest as JSON.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), DataError> {
        let path = path.as_ref();
        let text = serde_json::to_string_pretty(self).map_err(DataError::ManifestSerialize)?;
        std::fs::write(path, text).map_err(|source| DataError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Reads a manifest from JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DataError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| DataError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| DataError::ManifestParse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Name of the manifest inside a dataset root.
    #[must_use]
    pub fn file_name() -> PathBuf {
        PathBuf::from("dataset-manifest.json")
    }
}

#[cfg(test)]
mod tests {
    use super::{DatasetManifest, SplitPolicySummary};

    fn manifest() -> DatasetManifest {
        DatasetManifest::new(
            1_700_000_000,
            "a".repeat(64),
            2048,
            SplitPolicySummary {
                seed: 0,
                validation_basis_points: 50,
                test_basis_points: 50,
            },
        )
    }

    #[test]
    fn the_content_hash_ignores_the_creation_timestamp() {
        let first = manifest();
        let mut second = manifest();
        second.created_unix_seconds += 86_400;

        assert_eq!(
            first.content_hash().expect("hashes"),
            second.content_hash().expect("hashes")
        );
    }

    #[test]
    fn the_content_hash_changes_with_the_tokenizer_identity() {
        let first = manifest();
        let mut second = manifest();
        second.tokenizer_identity = "b".repeat(64);

        assert_ne!(
            first.content_hash().expect("hashes"),
            second.content_hash().expect("hashes")
        );
    }

    #[test]
    fn the_content_hash_changes_with_the_split_policy() {
        let first = manifest();
        let mut second = manifest();
        second.split_policy.validation_basis_points = 500;

        assert_ne!(
            first.content_hash().expect("hashes"),
            second.content_hash().expect("hashes")
        );
    }

    #[test]
    fn every_ruleset_version_is_recorded() {
        let manifest = manifest();
        for version in [
            &manifest.normalization_version,
            &manifest.filter_version,
            &manifest.dedup_version,
            &manifest.split_version,
            &manifest.packing_version,
        ] {
            assert!(!version.is_empty(), "a ruleset version was not recorded");
        }
    }
}
