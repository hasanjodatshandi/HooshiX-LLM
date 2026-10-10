//! Sequence packing.
//!
//! Documents are tokenized separately and joined by an explicit end-of-sequence
//! token, then chopped into fixed-length windows. Two properties follow, and
//! both are required by the data specification:
//!
//! - a document boundary is never crossed without a boundary token, because the
//!   separator is what marks it;
//! - sequences are full, so batches need no padding and the loss needs no mask.
//!
//! The packer emits sequences through a callback rather than collecting them, so
//! memory use is one window regardless of corpus size.

use crate::error::DataError;

/// Identifies the ruleset in a dataset manifest.
pub const PACKING_VERSION: &str = "eos-window-v1";

/// What happens to a final window that is shorter than the configured length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TailPolicy {
    /// Discard the partial window, so every sequence has the same length.
    #[default]
    Drop,
    /// Emit the partial window as a shorter sequence.
    Keep,
}

/// Counters describing what packing produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PackingStats {
    /// Documents pushed into the packer.
    pub documents: u64,
    /// Tokens pushed, including the appended separators.
    pub tokens: u64,
    /// Sequences emitted.
    pub sequences: u64,
    /// Tokens discarded because they formed an incomplete final window.
    pub dropped_tail_tokens: u64,
}

impl PackingStats {
    /// Share of pushed tokens that ended up in an emitted sequence.
    ///
    /// One means nothing was discarded.
    #[must_use]
    pub fn usable_ratio(&self) -> f64 {
        if self.tokens == 0 {
            return 1.0;
        }
        let emitted = self.tokens.saturating_sub(self.dropped_tail_tokens);
        emitted as f64 / self.tokens as f64
    }

    /// Average number of documents per emitted sequence.
    #[must_use]
    pub fn documents_per_sequence(&self) -> f64 {
        if self.sequences == 0 {
            return 0.0;
        }
        self.documents as f64 / self.sequences as f64
    }
}

/// Accumulates tokens and emits fixed-length windows.
pub struct SequencePacker {
    sequence_length: usize,
    separator: u32,
    tail: TailPolicy,
    buffer: Vec<u32>,
    stats: PackingStats,
}

impl SequencePacker {
    /// Creates a packer.
    ///
    /// `separator` is the token appended after every document; it must be a real
    /// token id, because packing without a boundary token would concatenate
    /// documents into text that was never written.
    #[must_use]
    pub fn new(sequence_length: usize, separator: u32, tail: TailPolicy) -> Self {
        Self {
            sequence_length,
            separator,
            tail,
            buffer: Vec::with_capacity(sequence_length),
            stats: PackingStats::default(),
        }
    }

    /// Tokens currently held but not yet emitted.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Counters so far.
    #[must_use]
    pub const fn stats(&self) -> PackingStats {
        self.stats
    }

    /// Appends a tokenized document and a separator, emitting whole windows.
    pub fn push_document(
        &mut self,
        tokens: &[u32],
        emit: &mut impl FnMut(&[u32]) -> Result<(), DataError>,
    ) -> Result<(), DataError> {
        self.stats.documents += 1;
        self.stats.tokens += tokens.len() as u64 + 1;

        for token in tokens
            .iter()
            .copied()
            .chain(std::iter::once(self.separator))
        {
            self.buffer.push(token);
            if self.buffer.len() == self.sequence_length {
                emit(&self.buffer)?;
                self.stats.sequences += 1;
                self.buffer.clear();
            }
        }

        Ok(())
    }

    /// Emits whatever remains, honouring the tail policy.
    pub fn finish(
        &mut self,
        emit: &mut impl FnMut(&[u32]) -> Result<(), DataError>,
    ) -> Result<(), DataError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        match self.tail {
            TailPolicy::Drop => {
                self.stats.dropped_tail_tokens += self.buffer.len() as u64;
            }
            TailPolicy::Keep => {
                emit(&self.buffer)?;
                self.stats.sequences += 1;
            }
        }

        self.buffer.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PackingStats, SequencePacker, TailPolicy};
    use crate::error::DataError;

    /// Runs a packer over documents and returns the emitted sequences.
    fn pack(
        documents: &[Vec<u32>],
        sequence_length: usize,
        tail: TailPolicy,
    ) -> (Vec<Vec<u32>>, PackingStats) {
        let mut packer = SequencePacker::new(sequence_length, 99, tail);
        let mut emitted: Vec<Vec<u32>> = Vec::new();
        let mut sink = |sequence: &[u32]| -> Result<(), DataError> {
            emitted.push(sequence.to_vec());
            Ok(())
        };

        for document in documents {
            packer.push_document(document, &mut sink).expect("emits");
        }
        packer.finish(&mut sink).expect("emits");

        (emitted, packer.stats())
    }

    #[test]
    fn every_document_is_followed_by_the_separator() {
        let (sequences, stats) = pack(&[vec![1, 2, 3]], 8, TailPolicy::Keep);
        assert_eq!(sequences, [vec![1, 2, 3, 99]]);
        assert_eq!(stats.documents, 1);
        assert_eq!(stats.tokens, 4);
    }

    #[test]
    fn documents_are_never_concatenated_without_a_boundary_token() {
        // Two documents that would otherwise read as one run.
        let (sequences, _) = pack(&[vec![1, 2], vec![3, 4]], 8, TailPolicy::Keep);
        assert_eq!(sequences, [vec![1, 2, 99, 3, 4, 99]]);
    }

    #[test]
    fn a_document_longer_than_a_window_is_split_across_windows() {
        let document: Vec<u32> = (1..=10).collect();
        let (sequences, stats) = pack(&[document], 4, TailPolicy::Keep);

        assert_eq!(sequences.len(), 3);
        assert_eq!(sequences[0], [1, 2, 3, 4]);
        assert_eq!(sequences[1], [5, 6, 7, 8]);
        assert_eq!(sequences[2], [9, 10, 99]);
        assert_eq!(stats.sequences, 3);
    }

    #[test]
    fn dropping_the_tail_yields_uniform_sequences() {
        let document: Vec<u32> = (1..=10).collect();
        let (sequences, stats) = pack(&[document], 4, TailPolicy::Drop);

        assert_eq!(sequences.len(), 2);
        assert!(sequences.iter().all(|sequence| sequence.len() == 4));
        assert_eq!(stats.dropped_tail_tokens, 3);
    }

    #[test]
    fn every_emitted_token_is_accounted_for() {
        let documents = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let (sequences, stats) = pack(&documents, 3, TailPolicy::Keep);

        let emitted: usize = sequences.iter().map(Vec::len).sum();
        assert_eq!(emitted as u64, stats.tokens);
        assert_eq!(stats.usable_ratio(), 1.0);
        assert_eq!(stats.documents, 3);
    }

    #[test]
    fn an_exact_multiple_of_the_window_leaves_no_tail() {
        let (sequences, stats) = pack(&[vec![1, 2]], 3, TailPolicy::Drop);
        assert_eq!(sequences, [vec![1, 2, 99]]);
        assert_eq!(stats.dropped_tail_tokens, 0);
    }

    #[test]
    fn empty_input_emits_nothing() {
        let (sequences, stats) = pack(&[], 4, TailPolicy::Keep);
        assert!(sequences.is_empty());
        assert_eq!(stats, PackingStats::default());
        assert_eq!(stats.usable_ratio(), 1.0);
        assert_eq!(stats.documents_per_sequence(), 0.0);
    }

    #[test]
    fn an_empty_document_still_contributes_its_separator() {
        let (sequences, stats) = pack(&[Vec::new()], 4, TailPolicy::Keep);
        assert_eq!(sequences, [vec![99]]);
        assert_eq!(stats.documents, 1);
    }

    #[test]
    fn document_density_is_reported() {
        let (_, stats) = pack(&[vec![1], vec![2], vec![3], vec![4]], 4, TailPolicy::Keep);
        // Four separators plus four tokens is eight tokens, two windows.
        assert_eq!(stats.sequences, 2);
        assert_eq!(stats.documents_per_sequence(), 2.0);
    }
}
