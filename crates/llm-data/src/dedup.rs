//! Exact document deduplication.
//!
//! Documents are identified by the SHA-256 of their normalized text, so two
//! documents that differ only in line endings or removed control characters are
//! recognised as the same document. Deduplication runs before splitting, so a
//! duplicate can never appear on both sides of a train/evaluation boundary.
//!
//! The seen-set grows with the corpus. That is unavoidable for exact
//! deduplication and is bounded by the number of distinct documents rather than
//! by their size: each entry is a 32-byte digest, not the document.

/// Identifies the ruleset in a dataset manifest.
pub const DEDUP_VERSION: &str = "exact-sha256-v1";

use std::collections::HashSet;

use sha2::{Digest, Sha256};

/// SHA-256 of a normalized document.
#[must_use]
pub fn digest(text: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

/// Records which documents have already been seen.
#[derive(Debug, Default)]
pub struct ExactDedup {
    seen: HashSet<[u8; 32]>,
}

impl ExactDedup {
    /// Creates an empty seen-set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a digest, reporting whether it had already been seen.
    ///
    /// The first occurrence wins, so the kept document is the earliest one in
    /// corpus order — which is what makes the result reproducible.
    pub fn observe(&mut self, digest: [u8; 32]) -> bool {
        !self.seen.insert(digest)
    }

    /// Number of distinct documents observed so far.
    #[must_use]
    pub fn distinct(&self) -> usize {
        self.seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{ExactDedup, digest};

    #[test]
    fn the_first_occurrence_is_kept_and_later_ones_are_duplicates() {
        let mut dedup = ExactDedup::new();
        let first = digest("a document");
        let second = digest("a document");
        let other = digest("another document");

        assert!(!dedup.observe(first));
        assert!(dedup.observe(second));
        assert!(!dedup.observe(other));
        assert_eq!(dedup.distinct(), 2);
    }

    #[test]
    fn identical_text_produces_an_identical_digest() {
        assert_eq!(digest("same text"), digest("same text"));
        assert_ne!(digest("same text"), digest("same text "));
    }

    #[test]
    fn the_digest_is_thirty_two_bytes() {
        assert_eq!(digest("anything").len(), 32);
    }
}
