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

use llm_checkpoint::DataCursor;

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

    /// Where in the data the next batch will come from.
    ///
    /// Recorded in every checkpoint so a resumed run continues rather than
    /// restarts. `shard` is always zero because this type reads exactly one
    /// shard; the field belongs to the cursor's shape rather than to this reader,
    /// and a resume from a cursor naming another shard is refused rather than
    /// ignored.
    #[must_use]
    pub const fn cursor(&self) -> DataCursor {
        DataCursor {
            epoch: self.passes,
            shard: 0,
            sequence: self.reader.position(),
        }
    }

    /// Puts the reader back where `cursor` says it was.
    ///
    /// The point of a resume is that steps after it are the steps that would have
    /// run, so this restores the pass count as well as the position: a run that
    /// resumes at sequence 500 of a shard it had already passed over once is on
    /// pass two, and reporting it as pass one would understate how much the model
    /// has seen the same data.
    ///
    /// # Errors
    ///
    /// `TrainError` if the cursor names a different shard, or a sequence past the
    /// end of this one. Both are refused rather than clamped: a cursor that does
    /// not describe this shard describes a different run.
    // Reachable from the tests below and not yet from the binary, because the
    // `--resume` path that would call it does not exist. Allowing the lint is the
    // honest form of that: the alternative is to leave the method out until the
    // caller arrives, which would mean P0-3's remaining work starts by writing
    // this again, or to pretend it is used.
    #[allow(dead_code)]
    pub fn seek_cursor(&mut self, cursor: &DataCursor) -> Result<(), TrainError> {
        if cursor.shard != 0 {
            return Err(TrainError::CursorShardMismatch {
                path: self.path.clone(),
                shard: cursor.shard,
            });
        }

        self.reader.seek(cursor.sequence)?;
        self.passes = cursor.epoch;
        Ok(())
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

#[cfg(test)]
mod tests {
    //! What a resume depends on: that a position survives a round trip through a
    //! cursor, and that the sequences after it are the ones that would have been
    //! read anyway.

    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

    use llm_data::{ShardWriter, Split};

    use super::*;

    const IDENTITY: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const WIDTH: usize = 4;

    /// A shard of `count` sequences whose token values identify both the sequence
    /// and the position within it, so a sequence read from the wrong place is
    /// visible rather than plausible.
    fn shard(name: &str, count: u32) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        // `CARGO_TARGET_TMPDIR` is defined for integration tests only, and this
        // is a unit test inside the binary.
        let directory = std::env::temp_dir().join(format!("hooshix-{name}-{unique}"));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("a scratch directory");
        let path = directory.join("train.shard");

        let mut writer =
            ShardWriter::create(&path, Split::Train, WIDTH, IDENTITY).expect("creates the shard");
        for index in 0..count {
            let tokens: Vec<u32> = (0..WIDTH as u32)
                .map(|offset| index * 10 + offset)
                .collect();
            writer.push_sequence(&tokens).expect("pushes a sequence");
        }
        writer.finish().expect("finishes the shard");
        path
    }

    fn reader(path: &Path) -> ShardBatches {
        let device = burn::tensor::Device::flex();
        ShardBatches::open(path, IDENTITY, WIDTH, &device).expect("opens the shard")
    }

    #[test]
    fn a_cursor_round_trip_reads_the_same_sequences() {
        let path = shard("cursor-round-trip", 6);
        let mut reference = reader(&path);

        // Where the stream would have been after three batches of one.
        for _ in 0..3 {
            reference.next_batch(1).expect("a batch");
        }
        let cursor = reference.cursor();
        assert_eq!(
            cursor.sequence, 3,
            "three sequences read, so position three"
        );
        assert_eq!(cursor.shard, 0, "this reader holds exactly one shard");
        assert_eq!(cursor.epoch, 0, "no pass has completed");

        let expected: Vec<TokenBatch> = (0..3)
            .map(|_| reference.next_batch(1).expect("a batch"))
            .collect();

        // A fresh reader put back at the cursor must produce the same stream.
        let mut resumed = reader(&path);
        resumed.seek_cursor(&cursor).expect("seeks to the cursor");
        assert_eq!(resumed.cursor(), cursor, "the cursor survives a round trip");

        for (index, want) in expected.iter().enumerate() {
            let got = resumed.next_batch(1).expect("a batch");
            assert_eq!(
                got.input.to_data(),
                want.input.to_data(),
                "batch {index} after resuming"
            );
            assert_eq!(got.target.to_data(), want.target.to_data());
        }
    }

    #[test]
    fn the_pass_count_survives_a_cursor() {
        // A run that resumes at sequence 2 of a shard it has already been through
        // once is on pass one, and reporting it as pass zero would understate how
        // often the model has seen the same sequences.
        let path = shard("cursor-passes", 4);
        let mut loader = reader(&path);

        // Four sequences, so four batches of one exhaust the shard exactly.
        for _ in 0..5 {
            loader.next_batch(1).expect("a batch");
        }
        let cursor = loader.cursor();
        assert_eq!(cursor.epoch, 1, "the shard restarted once");
        assert_eq!(cursor.sequence, 1, "and one sequence into the second pass");

        let mut resumed = reader(&path);
        resumed.seek_cursor(&cursor).expect("seeks");
        assert_eq!(resumed.passes(), 1, "the pass count came back too");
    }

    #[test]
    fn a_cursor_naming_another_shard_is_refused() {
        // Refused rather than ignored: continuing at shard zero from a cursor that
        // names shard one would read the wrong data and report a healthy loss.
        let path = shard("cursor-wrong-shard", 4);
        let mut loader = reader(&path);

        let elsewhere = DataCursor {
            epoch: 0,
            shard: 1,
            sequence: 0,
        };
        assert!(matches!(
            loader.seek_cursor(&elsewhere),
            Err(TrainError::CursorShardMismatch { shard: 1, .. })
        ));
    }

    #[test]
    fn a_cursor_past_the_end_is_refused() {
        let path = shard("cursor-past-end", 4);
        let mut loader = reader(&path);

        let too_far = DataCursor {
            epoch: 0,
            shard: 0,
            sequence: 5,
        };
        assert!(loader.seek_cursor(&too_far).is_err());
    }
}
