//! Training, optimizer and learning-rate-schedule configuration.
//!
//! The baseline is AdamW with linear warmup followed by cosine decay, which is
//! the Stage-1 choice. No optimizer variant is introduced until that baseline is
//! stable, so the optimizer is not a configurable strategy yet.

use serde::{Deserialize, Serialize};

use crate::config::error::ConfigError;

/// AdamW hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerConfig {
    /// Peak learning rate reached at the end of warmup.
    pub learning_rate: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
    /// Exponential decay rate for the first moment estimate.
    pub beta1: f64,
    /// Exponential decay rate for the second moment estimate.
    pub beta2: f64,
    /// Numerical-stability term added to the denominator.
    pub epsilon: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
        }
    }
}

impl OptimizerConfig {
    /// Rejects hyperparameters outside the ranges AdamW is defined for.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "training.optimizer.learning_rate",
                reason: "must be finite and greater than zero",
            });
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "training.optimizer.weight_decay",
                reason: "must be finite and not negative",
            });
        }
        for (field, value) in [
            ("training.optimizer.beta1", self.beta1),
            ("training.optimizer.beta2", self.beta2),
        ] {
            if !value.is_finite() || !(0.0..1.0).contains(&value) {
                return Err(ConfigError::OutOfRange {
                    field,
                    reason: "must be finite and within [0, 1)",
                });
            }
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "training.optimizer.epsilon",
                reason: "must be finite and greater than zero",
            });
        }
        Ok(())
    }
}

/// Learning-rate schedule shape: linear warmup, then cosine decay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Steps spent raising the rate linearly from zero to the peak.
    pub warmup_steps: u64,
    /// Steps over which the rate descends along the cosine curve.
    ///
    /// In Burn 0.22 this is the scheduler's `num_iters`: the descent to the
    /// minimum. The 0.21-era reading of "cosine with warm restarts" does not
    /// apply.
    pub cosine_iters: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 2000,
            cosine_iters: 100_000,
        }
    }
}

impl SchedulerConfig {
    /// Rejects a schedule that cannot be evaluated.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cosine_iters == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "training.scheduler.cosine_iters",
            });
        }
        if self.warmup_steps > self.cosine_iters {
            return Err(ConfigError::OutOfRange {
                field: "training.scheduler.warmup_steps",
                reason: "must not exceed training.scheduler.cosine_iters",
            });
        }
        Ok(())
    }
}

/// Training-loop configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// Number of optimizer steps to run.
    pub max_steps: u64,
    /// Number of data-parallel replicas. Stage 1 runs a single device.
    pub data_parallel_world_size: usize,
    /// Micro-batches accumulated before one optimizer step.
    pub gradient_accumulation_steps: u32,
    /// Seed for initialization, shuffling and dropout.
    pub seed: u64,
    /// Global gradient-norm clip. `None` disables clipping.
    pub gradient_clip_global_norm: Option<f64>,
    /// Steps between checkpoints.
    pub checkpoint_interval_steps: u64,
    /// Steps between validation passes.
    pub evaluation_interval_steps: u64,
    /// AdamW hyperparameters.
    pub optimizer: OptimizerConfig,
    /// Learning-rate schedule.
    pub scheduler: SchedulerConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_steps: 100_000,
            data_parallel_world_size: 1,
            gradient_accumulation_steps: 1,
            seed: 0,
            gradient_clip_global_norm: Some(1.0),
            checkpoint_interval_steps: 1000,
            evaluation_interval_steps: 500,
            optimizer: OptimizerConfig::default(),
            scheduler: SchedulerConfig::default(),
        }
    }
}

impl TrainingConfig {
    /// Rejects a training configuration that cannot run.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_steps == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "training.max_steps",
            });
        }
        if self.data_parallel_world_size == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "training.data_parallel_world_size",
            });
        }
        if self.gradient_accumulation_steps == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "training.gradient_accumulation_steps",
            });
        }
        if self.checkpoint_interval_steps == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "training.checkpoint_interval_steps",
            });
        }
        if self.evaluation_interval_steps == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "training.evaluation_interval_steps",
            });
        }

        if let Some(clip) = self.gradient_clip_global_norm
            && (!clip.is_finite() || clip <= 0.0)
        {
            return Err(ConfigError::OutOfRange {
                field: "training.gradient_clip_global_norm",
                reason: "must be finite and greater than zero when present",
            });
        }

        self.optimizer.validate()?;
        self.scheduler.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OptimizerConfig, SchedulerConfig, TrainingConfig};
    use crate::config::error::ConfigError;

    #[test]
    fn default_is_valid() {
        let config = TrainingConfig::default();
        config.validate().expect("default must be valid");
        assert_eq!(config.data_parallel_world_size, 1);
        assert_eq!(config.gradient_accumulation_steps, 1);
    }

    #[test]
    fn zero_step_counts_are_rejected() {
        for config in [
            TrainingConfig {
                max_steps: 0,
                ..TrainingConfig::default()
            },
            TrainingConfig {
                data_parallel_world_size: 0,
                ..TrainingConfig::default()
            },
            TrainingConfig {
                gradient_accumulation_steps: 0,
                ..TrainingConfig::default()
            },
            TrainingConfig {
                checkpoint_interval_steps: 0,
                ..TrainingConfig::default()
            },
        ] {
            assert!(matches!(
                config.validate(),
                Err(ConfigError::ZeroDimension { .. })
            ));
        }
    }

    #[test]
    fn non_positive_learning_rate_is_rejected() {
        for learning_rate in [0.0, -1e-4, f64::NAN, f64::INFINITY] {
            let config = TrainingConfig {
                optimizer: OptimizerConfig {
                    learning_rate,
                    ..OptimizerConfig::default()
                },
                ..TrainingConfig::default()
            };
            assert!(matches!(
                config.validate(),
                Err(ConfigError::OutOfRange {
                    field: "training.optimizer.learning_rate",
                    ..
                })
            ));
        }
    }

    #[test]
    fn beta_outside_the_unit_interval_is_rejected() {
        let config = TrainingConfig {
            optimizer: OptimizerConfig {
                beta1: 1.0,
                ..OptimizerConfig::default()
            },
            ..TrainingConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "training.optimizer.beta1",
                ..
            })
        ));
    }

    #[test]
    fn negative_weight_decay_is_rejected() {
        let config = TrainingConfig {
            optimizer: OptimizerConfig {
                weight_decay: -0.1,
                ..OptimizerConfig::default()
            },
            ..TrainingConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "training.optimizer.weight_decay",
                ..
            })
        ));
    }

    #[test]
    fn invalid_gradient_clip_is_rejected() {
        let config = TrainingConfig {
            gradient_clip_global_norm: Some(0.0),
            ..TrainingConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "training.gradient_clip_global_norm",
                ..
            })
        ));

        let config = TrainingConfig {
            gradient_clip_global_norm: None,
            ..TrainingConfig::default()
        };
        config.validate().expect("no clipping is allowed");
    }

    #[test]
    fn warmup_longer_than_the_cosine_descent_is_rejected() {
        let config = TrainingConfig {
            scheduler: SchedulerConfig {
                warmup_steps: 100,
                cosine_iters: 50,
            },
            ..TrainingConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "training.scheduler.warmup_steps",
                ..
            })
        ));
    }

    #[test]
    fn zero_cosine_iters_is_rejected() {
        let config = TrainingConfig {
            scheduler: SchedulerConfig {
                warmup_steps: 0,
                cosine_iters: 0,
            },
            ..TrainingConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ZeroDimension {
                field: "training.scheduler.cosine_iters"
            })
        ));
    }
}
