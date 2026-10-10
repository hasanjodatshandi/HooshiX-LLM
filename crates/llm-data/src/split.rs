//! Deterministic corpus splitting.
//!
//! A document's split is decided by hashing its content together with a split
//! seed, not by its position in the corpus. That makes the assignment stable
//! when the corpus grows, when sources are added, and when preparation is
//! resumed — none of which is true of "take the last five percent".

use llm_core::DataConfig;
use sha2::{Digest, Sha256};

/// Identifies the ruleset in a dataset manifest.
pub const SPLIT_VERSION: &str = "content-hash-bucket-v1";

/// Number of buckets a document can fall into.
///
/// Ten thousand gives a resolution of one basis point, which is finer than any
/// split ratio worth configuring.
const BUCKETS: u32 = 10_000;

/// Which side of the train/evaluation boundary a document belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    /// Used for training.
    Train,
    /// Held out for validation during training.
    Validation,
    /// Held out for final evaluation.
    Test,
}

impl Split {
    /// Every split, in a fixed order.
    pub const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::Test];

    /// Stable lowercase name used in manifests and shard headers.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }

    /// Stable tag stored in a shard header.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
            Self::Test => 2,
        }
    }

    /// Reads a tag written by [`Split::tag`].
    #[must_use]
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Train),
            1 => Some(Self::Validation),
            2 => Some(Self::Test),
            _ => None,
        }
    }
}

/// How the corpus is divided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitPolicy {
    /// Share held out for validation, in basis points of 10000.
    pub validation_basis_points: u16,
    /// Share held out for testing, in basis points of 10000.
    pub test_basis_points: u16,
    /// Seed mixed into the content hash.
    pub seed: u64,
}

impl SplitPolicy {
    /// Reads the split settings from a data configuration.
    #[must_use]
    pub const fn from_config(config: &DataConfig) -> Self {
        Self {
            validation_basis_points: config.validation_basis_points,
            test_basis_points: config.test_basis_points,
            seed: config.split_seed,
        }
    }

    /// Assigns a document to a split.
    ///
    /// Deterministic in the document digest and the seed alone, so the same
    /// document always lands on the same side regardless of corpus order.
    #[must_use]
    pub fn assign(&self, digest: &[u8; 32]) -> Split {
        let mut hasher = Sha256::new();
        hasher.update(self.seed.to_le_bytes());
        hasher.update(digest);
        let mixed = hasher.finalize();

        // The first eight bytes are enough to index ten thousand buckets, and
        // using a fresh hash of the seed keeps the split independent of the
        // digest's own distribution.
        let mut prefix = [0_u8; 8];
        prefix.copy_from_slice(&mixed[..8]);
        let bucket = (u64::from_le_bytes(prefix) % u64::from(BUCKETS)) as u32;

        let validation = u32::from(self.validation_basis_points);
        let test = u32::from(self.test_basis_points);

        if bucket < validation {
            Split::Validation
        } else if bucket < validation + test {
            Split::Test
        } else {
            Split::Train
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SPLIT_VERSION, Split, SplitPolicy};
    use crate::dedup::digest;

    fn policy() -> SplitPolicy {
        SplitPolicy {
            validation_basis_points: 1_000,
            test_basis_points: 1_000,
            seed: 0,
        }
    }

    #[test]
    fn assignment_is_stable_for_the_same_document() {
        let policy = policy();
        let document = digest("a document");
        assert_eq!(policy.assign(&document), policy.assign(&document));
    }

    #[test]
    fn a_different_seed_moves_documents() {
        let first = policy();
        let second = SplitPolicy {
            seed: 1,
            ..policy()
        };

        let moved = (0..200)
            .filter(|index| {
                let document = digest(&format!("document {index}"));
                first.assign(&document) != second.assign(&document)
            })
            .count();

        // With 10% validation and 10% test, roughly a third of documents change
        // side when the seed changes. Forty out of two hundred is far below the
        // expected count and far above what chance would produce.
        assert!(moved > 40, "only {moved} of 200 documents moved");
    }

    #[test]
    fn the_observed_split_share_matches_the_policy() {
        let policy = policy();
        let total = 20_000;
        let mut counts = [0_u32; 3];

        for index in 0..total {
            let document = digest(&format!("document number {index}"));
            let slot = match policy.assign(&document) {
                Split::Train => 0,
                Split::Validation => 1,
                Split::Test => 2,
            };
            counts[slot] += 1;
        }

        // A ten percent target with twenty thousand documents: the binomial
        // standard deviation is well under one percent, so a two percent window
        // is a meaningful check rather than a flaky one.
        for (slot, expected) in [(1_usize, 2_000_u32), (2, 2_000)] {
            let observed = counts[slot];
            let drift = observed.abs_diff(expected);
            assert!(
                drift < 400,
                "split {slot} observed {observed}, expected about {expected}"
            );
        }
        assert_eq!(counts.iter().sum::<u32>(), total);
    }

    #[test]
    fn extreme_ratios_are_honoured() {
        let policy = SplitPolicy {
            validation_basis_points: 0,
            test_basis_points: 0,
            seed: 0,
        };
        for index in 0..100 {
            let document = digest(&format!("document {index}"));
            assert_eq!(policy.assign(&document), Split::Train);
        }
    }

    #[test]
    fn split_tags_round_trip() {
        for split in Split::ALL {
            assert_eq!(Split::from_tag(split.tag()), Some(split));
        }
        assert_eq!(Split::from_tag(9), None);
    }

    #[test]
    fn split_names_are_distinct_and_stable() {
        assert_eq!(SPLIT_VERSION, "content-hash-bucket-v1");
        let names: Vec<&str> = Split::ALL.iter().map(|split| split.as_str()).collect();
        assert_eq!(names, ["train", "validation", "test"]);
    }
}
