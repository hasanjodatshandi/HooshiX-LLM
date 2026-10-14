//! Dataset and batching configuration.
//!
//! Only the parameters that the Stage-1 data pipeline actually consumes are
//! declared here. Corpus provenance and shard integrity are contractual
//! properties of prepared data, not tunable settings, so they are not
//! configurable.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::error::ConfigError;

/// Where prepared token shards live, how they are batched, and how the corpus
/// is divided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// Root directory holding prepared, checksummed token shards.
    pub data_root: PathBuf,
    /// Number of tokens in one packed training sequence.
    pub sequence_length: usize,
    /// Sequences processed in one forward/backward pass.
    pub micro_batch_size: usize,
    /// Seed controlling deterministic shuffling and shard order.
    pub shuffle_seed: u64,
    /// Share of the corpus held out for validation, in basis points of 10000.
    pub validation_basis_points: u16,
    /// Share of the corpus held out for testing, in basis points of 10000.
    pub test_basis_points: u16,
    /// Seed controlling deterministic corpus splitting.
    ///
    /// Separate from `shuffle_seed` because the two answer different questions:
    /// this one decides which side of the split a document lands on, and it must
    /// be held fixed for a prepared dataset to stay reproducible.
    pub split_seed: u64,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_root: PathBuf::from("artifacts/data"),
            sequence_length: 2048,
            micro_batch_size: 8,
            shuffle_seed: 0,
            // Half a percent each: enough to detect overfitting on a Stage-1
            // corpus without materially reducing training tokens.
            validation_basis_points: 50,
            test_basis_points: 50,
            split_seed: 0,
        }
    }
}

impl DataConfig {
    /// Rejects a batching or splitting configuration that cannot be honoured.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // One, not just zero. A sequence of one token leaves no next-token pair
        // to predict, so the run would construct a model, allocate its memory and
        // fail on the first batch rather than here.
        if self.sequence_length < 2 {
            return Err(ConfigError::OutOfRange {
                field: "data.sequence_length",
                reason: "must be at least 2, because a sequence of one token has no \
                         next-token pair to predict",
            });
        }
        if self.micro_batch_size == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "data.micro_batch_size",
            });
        }
        if self.data_root.as_os_str().is_empty() {
            return Err(ConfigError::OutOfRange {
                field: "data.data_root",
                reason: "must not be empty",
            });
        }

        // Training must be left with something, and basis points cannot exceed
        // a whole corpus.
        let held_out = u32::from(self.validation_basis_points) + u32::from(self.test_basis_points);
        if held_out >= 10_000 {
            return Err(ConfigError::OutOfRange {
                field: "data.validation_basis_points + data.test_basis_points",
                reason: "must leave at least one basis point for training",
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::DataConfig;
    use crate::config::error::ConfigError;

    #[test]
    fn default_is_valid() {
        let config = DataConfig::default();
        config.validate().expect("default must be valid");
        assert_eq!(config.sequence_length, 2048);
        assert_eq!(config.micro_batch_size, 8);
    }

    #[test]
    fn a_sequence_of_one_is_rejected_as_well_as_zero() {
        // One, not only zero. A sequence of one token leaves no next-token pair
        // to predict, so a run configured that way would build a model, seed a
        // device and allocate its memory before failing on the first batch.
        for length in [0, 1] {
            let config = DataConfig {
                sequence_length: length,
                ..DataConfig::default()
            };
            assert!(
                matches!(
                    config.validate(),
                    Err(ConfigError::OutOfRange {
                        field: "data.sequence_length",
                        ..
                    })
                ),
                "sequence_length {length} must be rejected"
            );
        }

        // And two is the first length that is not.
        let two = DataConfig {
            sequence_length: 2,
            ..DataConfig::default()
        };
        assert!(
            two.validate().is_ok(),
            "a sequence of two can predict one token"
        );
    }

    #[test]
    fn zero_micro_batch_size_is_rejected() {
        let config = DataConfig {
            micro_batch_size: 0,
            ..DataConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ZeroDimension {
                field: "data.micro_batch_size"
            })
        ));
    }

    #[test]
    fn empty_data_root_is_rejected() {
        let config = DataConfig {
            data_root: PathBuf::new(),
            ..DataConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "data.data_root",
                ..
            })
        ));
    }

    #[test]
    fn the_split_must_leave_something_for_training() {
        let config = DataConfig {
            validation_basis_points: 5_000,
            test_basis_points: 5_000,
            ..DataConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "data.validation_basis_points + data.test_basis_points",
                ..
            })
        ));

        let config = DataConfig {
            validation_basis_points: 9_999,
            test_basis_points: 0,
            ..DataConfig::default()
        };
        config.validate().expect("one basis point is enough");
    }
}
