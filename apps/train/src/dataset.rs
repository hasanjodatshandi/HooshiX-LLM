//! Feeding batches out of prepared shards.
//!
//! A shard holds packed sequences of exactly `sequence_length` token ids. A step
//! wants `micro_batch_size` of them side by side, and the objective wants the
//! input shifted one position against the target — which is what
//! [`next_token_pairs`] does, turning `n` tokens into `n - 1` predictions.
//!
//! This lives in the composition root rather than in `llm-data`: a shard is a
//! file of ids, and what a training step needs from it is a training decision.

use std::path::{Path, PathBuf};

use burn::tensor::{Device, Int, Tensor, TensorData};
use llm_data::{ShardExpectation, ShardReader};
use llm_model::{all_positions, next_token_pairs};
use llm_training::TokenBatch;

use crate::error::TrainError;

/// One shard, read as a stream of batches.
pub struct ShardBatches {
    path: PathBuf,
    reader: ShardReader,
    device: Device,
    passes: u64,
}

impl ShardBatches {
    /// Opens a shard, checking it against the identity and length a run expects.
    pub fn open(
        path: impl AsRef<Path>,
        tokenizer_identity: &str,
        sequence_length: usize,
        device: &Device,
    ) -> Result<Self, TrainError> {
        let path = path.as_ref().to_path_buf();
        let expectation = ShardExpectation {
            tokenizer_identity,
            sequence_length,
        };
        let reader = ShardReader::open(&path, &expectation)?;
        if reader.sequence_count() == 0 {
            return Err(TrainError::EmptyShard { path });
        }

        Ok(Self {
            path,
            reader,
            device: device.clone(),
            passes: 0,
        })
    }

    /// Sequences the shard holds.
    #[must_use]
    pub const fn sequence_count(&self) -> u64 {
        self.reader.sequence_count()
    }

    /// How many times the shard has been restarted.
    #[must_use]
    pub const fn passes(&self) -> u64 {
        self.passes
    }

    /// Builds one batch of `micro_batch_size` sequences.
    ///
    /// When the shard runs out it restarts, so a run of any length is possible
    /// over a shard of any size. The restart is counted rather than hidden: a run
    /// that has made many passes over the same few sequences is memorising them,
    /// and a report has to be able to say so.
    pub fn next_batch(&mut self, micro_batch_size: usize) -> Result<TokenBatch, TrainError> {
        let width = self.reader.sequence_length();
        let mut ids: Vec<i32> = Vec::with_capacity(micro_batch_size * width);

        for _ in 0..micro_batch_size {
            if self.reader.is_exhausted() {
                self.reader.rewind()?;
                self.passes += 1;
            }
            let sequence = self
                .reader
                .next_sequence()?
                .ok_or_else(|| TrainError::EmptyShard {
                    path: self.path.clone(),
                })?;
            ids.extend(sequence.iter().map(|id| *id as i32));
        }

        let tokens = Tensor::<2, Int>::from_data(
            TensorData::new(ids, [micro_batch_size, width]),
            &self.device,
        );
        // `n` tokens give `n - 1` predictions: the last token has no next one in
        // this sequence, and inventing one would train on a boundary that is not
        // in the data.
        let (input, target) = next_token_pairs(tokens)?;
        let positions = input.dims()[1];

        Ok(TokenBatch {
            input,
            target,
            mask: all_positions(micro_batch_size, positions, &self.device),
        })
    }
}
