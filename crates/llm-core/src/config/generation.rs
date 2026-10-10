//! Generation and sampling configuration.
//!
//! Sampling is a closed set of strategies owned by this project, so it is
//! modelled as an enum rather than a trait. A trait would only be justified if
//! third-party strategies had to be plugged in.

use serde::{Deserialize, Serialize};

use crate::config::error::ConfigError;

/// How the next token is selected from the logits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SamplingStrategy {
    /// Take the most likely token. Deterministic.
    #[default]
    Greedy,
    /// Sample from the softmax distribution scaled by `temperature`.
    Temperature {
        /// Softmax temperature. Must be greater than zero.
        temperature: f32,
    },
    /// Restrict sampling to the `k` most likely tokens.
    TopK {
        /// Number of tokens kept.
        k: usize,
        /// Softmax temperature. Must be greater than zero.
        temperature: f32,
    },
    /// Restrict sampling to the smallest set whose cumulative probability
    /// reaches `p`.
    TopP {
        /// Cumulative-probability threshold, within `(0, 1]`.
        p: f32,
        /// Softmax temperature. Must be greater than zero.
        temperature: f32,
    },
}

impl SamplingStrategy {
    /// Rejects a strategy whose parameters cannot be honoured.
    ///
    /// A temperature of exactly zero is rejected rather than treated as greedy:
    /// `Greedy` states that intent, and zero would otherwise divide by zero.
    pub fn validate(&self, vocab_size: usize) -> Result<(), ConfigError> {
        let check_temperature = |temperature: f32| -> Result<(), ConfigError> {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(ConfigError::OutOfRange {
                    field: "generation.strategy.temperature",
                    reason: "must be finite and greater than zero; use `greedy` for argmax",
                });
            }
            Ok(())
        };

        match *self {
            Self::Greedy => Ok(()),
            Self::Temperature { temperature } => check_temperature(temperature),
            Self::TopK { k, temperature } => {
                check_temperature(temperature)?;
                if k == 0 || k > vocab_size {
                    return Err(ConfigError::TopKOutOfVocabulary { k, vocab_size });
                }
                Ok(())
            }
            Self::TopP { p, temperature } => {
                check_temperature(temperature)?;
                if !p.is_finite() || p <= 0.0 || p > 1.0 {
                    return Err(ConfigError::OutOfRange {
                        field: "generation.strategy.p",
                        reason: "must be finite and within (0, 1]",
                    });
                }
                Ok(())
            }
        }
    }
}

/// Decoding limits and sampling settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    /// Token-selection strategy.
    pub strategy: SamplingStrategy,
    /// Maximum number of tokens to generate per request.
    pub max_new_tokens: usize,
    /// Seed for reproducible sampling. Required for deterministic tests.
    pub seed: u64,
    /// Stop as soon as the end-of-sequence token is produced.
    pub stop_on_eos: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::default(),
            max_new_tokens: 128,
            seed: 0,
            stop_on_eos: true,
        }
    }
}

impl GenerationConfig {
    /// Rejects a generation configuration that cannot be served.
    ///
    /// `vocab_size` and `max_sequence_length` come from the model, so this is
    /// checked as part of whole-experiment validation.
    pub fn validate(
        &self,
        vocab_size: usize,
        max_sequence_length: usize,
    ) -> Result<(), ConfigError> {
        if self.max_new_tokens == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "generation.max_new_tokens",
            });
        }
        if self.max_new_tokens > max_sequence_length {
            return Err(ConfigError::GenerationExceedsContext {
                max_new_tokens: self.max_new_tokens,
                max_sequence_length,
            });
        }
        self.strategy.validate(vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, SamplingStrategy};
    use crate::config::error::ConfigError;

    const VOCAB: usize = 32_768;
    const CONTEXT: usize = 2048;

    #[test]
    fn default_is_valid() {
        let config = GenerationConfig::default();
        config
            .validate(VOCAB, CONTEXT)
            .expect("default must be valid");
        assert_eq!(config.strategy, SamplingStrategy::Greedy);
        assert!(config.stop_on_eos);
    }

    #[test]
    fn zero_temperature_is_rejected() {
        let config = GenerationConfig {
            strategy: SamplingStrategy::Temperature { temperature: 0.0 },
            ..GenerationConfig::default()
        };
        assert!(matches!(
            config.validate(VOCAB, CONTEXT),
            Err(ConfigError::OutOfRange {
                field: "generation.strategy.temperature",
                ..
            })
        ));
    }

    #[test]
    fn negative_temperature_is_rejected() {
        let config = GenerationConfig {
            strategy: SamplingStrategy::Temperature { temperature: -0.5 },
            ..GenerationConfig::default()
        };
        assert!(matches!(
            config.validate(VOCAB, CONTEXT),
            Err(ConfigError::OutOfRange { .. })
        ));
    }

    #[test]
    fn top_k_larger_than_the_vocabulary_is_rejected() {
        let config = GenerationConfig {
            strategy: SamplingStrategy::TopK {
                k: VOCAB + 1,
                temperature: 0.8,
            },
            ..GenerationConfig::default()
        };
        assert!(matches!(
            config.validate(VOCAB, CONTEXT),
            Err(ConfigError::TopKOutOfVocabulary { .. })
        ));
    }

    #[test]
    fn top_k_within_the_vocabulary_is_accepted() {
        let config = GenerationConfig {
            strategy: SamplingStrategy::TopK {
                k: VOCAB,
                temperature: 0.8,
            },
            ..GenerationConfig::default()
        };
        config
            .validate(VOCAB, CONTEXT)
            .expect("k == vocab_size is the inclusive bound");
    }

    #[test]
    fn top_p_outside_the_unit_interval_is_rejected() {
        for p in [0.0, -0.1, 1.5, f32::NAN] {
            let config = GenerationConfig {
                strategy: SamplingStrategy::TopP {
                    p,
                    temperature: 1.0,
                },
                ..GenerationConfig::default()
            };
            assert!(matches!(
                config.validate(VOCAB, CONTEXT),
                Err(ConfigError::OutOfRange {
                    field: "generation.strategy.p",
                    ..
                })
            ));
        }
    }

    #[test]
    fn zero_or_oversized_max_new_tokens_is_rejected() {
        let config = GenerationConfig {
            max_new_tokens: 0,
            ..GenerationConfig::default()
        };
        assert!(matches!(
            config.validate(VOCAB, CONTEXT),
            Err(ConfigError::ZeroDimension {
                field: "generation.max_new_tokens"
            })
        ));

        let config = GenerationConfig {
            max_new_tokens: CONTEXT + 1,
            ..GenerationConfig::default()
        };
        assert!(matches!(
            config.validate(VOCAB, CONTEXT),
            Err(ConfigError::GenerationExceedsContext { .. })
        ));
    }
}
