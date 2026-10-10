//! Model architecture configuration and its invariants.
//!
//! The defaults describe the Stage-1 baseline. `num_kv_heads` is present from
//! the start so that Stage-2 grouped-query attention is a configuration change
//! rather than a redesign; Stage 1 keeps it equal to `num_heads`.

use serde::{Deserialize, Serialize};

use crate::config::error::ConfigError;

/// Architecture of a decoder-only Transformer.
///
/// Fields are public so the type serializes directly to and from TOML.
/// `ModelConfig` has invariants, so a configuration must be checked with
/// [`ModelConfig::validate`] — or loaded through
/// [`ExperimentConfig::load`](crate::config::ExperimentConfig::load), which
/// validates — before its values are used to allocate tensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Tokenizer vocabulary size; also the output width of the language-model head.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Number of Transformer blocks.
    pub num_layers: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads. Equal to `num_heads` for Stage-1 MHA.
    pub num_kv_heads: usize,
    /// Width of the SwiGLU hidden layer.
    pub ffn_intermediate_size: usize,
    /// Maximum number of positions the model may attend over.
    pub max_sequence_length: usize,
    /// Base period of the rotary position embedding.
    pub rope_theta: f64,
    /// Epsilon used by RMSNorm.
    pub rms_norm_eps: f64,
    /// Tie the token embedding and the language-model head.
    pub tie_embeddings: bool,
    /// Add bias terms to the major linear layers.
    pub linear_bias: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_768,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            ffn_intermediate_size: 2048,
            max_sequence_length: 2048,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_embeddings: true,
            linear_bias: false,
        }
    }
}

impl ModelConfig {
    /// Per-head dimension.
    ///
    /// Returns `None` only for an unvalidated configuration with `num_heads`
    /// equal to zero, which [`ModelConfig::validate`] rejects.
    #[must_use]
    pub fn head_dim(&self) -> Option<usize> {
        self.hidden_size.checked_div(self.num_heads)
    }

    /// Total number of trainable parameters.
    ///
    /// Uses checked arithmetic because the value is derived from externally
    /// supplied dimensions. Returns `None` on overflow rather than wrapping to
    /// a plausible-looking number.
    ///
    /// The accounting follows the Stage-1 breakdown: token embedding, then per
    /// block the attention projections (query, key, value, output) and the
    /// three SwiGLU matrices, plus two RMSNorm weight vectors and the final
    /// normalization. A tied language-model head adds no parameters.
    #[must_use]
    pub fn parameter_count(&self) -> Option<u64> {
        let hidden = u64::try_from(self.hidden_size).ok()?;
        let vocab = u64::try_from(self.vocab_size).ok()?;
        let layers = u64::try_from(self.num_layers).ok()?;
        let heads = u64::try_from(self.num_heads).ok()?;
        let kv_heads = u64::try_from(self.num_kv_heads).ok()?;
        let head_dim = u64::try_from(self.head_dim()?).ok()?;
        let ffn = u64::try_from(self.ffn_intermediate_size).ok()?;

        // Embedding: vocab_size * hidden_size.
        let mut total = vocab.checked_mul(hidden)?;

        // Attention projections without bias:
        //   query  hidden -> num_heads    * head_dim
        //   key    hidden -> num_kv_heads * head_dim
        //   value  hidden -> num_kv_heads * head_dim
        //   output num_heads * head_dim -> hidden
        let query = hidden.checked_mul(heads.checked_mul(head_dim)?)?;
        let kv_width = kv_heads.checked_mul(head_dim)?;
        let key = hidden.checked_mul(kv_width)?;
        let value = hidden.checked_mul(kv_width)?;
        let output = heads.checked_mul(head_dim)?.checked_mul(hidden)?;
        let mut per_block = query
            .checked_add(key)?
            .checked_add(value)?
            .checked_add(output)?;

        if self.linear_bias {
            // Query, output and language-model-head-equivalent biases are
            // `hidden` wide; key and value biases are `kv_width` wide.
            let attention_bias = heads
                .checked_mul(head_dim)?
                .checked_add(kv_width.checked_mul(2)?)?
                .checked_add(hidden)?;
            per_block = per_block.checked_add(attention_bias)?;
        }

        // SwiGLU: gate, up and down matrices.
        let swiglu = hidden.checked_mul(ffn)?.checked_mul(3)?;
        per_block = per_block.checked_add(swiglu)?;

        if self.linear_bias {
            per_block = per_block.checked_add(ffn.checked_mul(2)?.checked_add(hidden)?)?;
        }

        // Two RMSNorm weight vectors per block.
        per_block = per_block.checked_add(hidden.checked_mul(2)?)?;

        total = total.checked_add(layers.checked_mul(per_block)?)?;

        // Final RMSNorm.
        if !self.tie_embeddings {
            // An untied head is a full `hidden -> vocab_size` matrix.
            total = total.checked_add(vocab.checked_mul(hidden)?)?;
        }
        total = total.checked_add(hidden)?;

        Some(total)
    }

    /// Rejects an architecture whose dimensions cannot form a coherent model.
    ///
    /// Everything here is checkable without allocating a tensor, which is the
    /// point: an invalid configuration must fail before allocation.
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (field, value) in [
            ("model.vocab_size", self.vocab_size),
            ("model.hidden_size", self.hidden_size),
            ("model.num_layers", self.num_layers),
            ("model.num_heads", self.num_heads),
            ("model.num_kv_heads", self.num_kv_heads),
            ("model.ffn_intermediate_size", self.ffn_intermediate_size),
            ("model.max_sequence_length", self.max_sequence_length),
        ] {
            if value == 0 {
                return Err(ConfigError::ZeroDimension { field });
            }
        }

        // Divisibility checks run after the zero-dimension checks above, so
        // neither divisor can be zero here.
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ConfigError::HiddenSizeNotDivisibleByHeads {
                hidden_size: self.hidden_size,
                num_heads: self.num_heads,
            });
        }

        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(ConfigError::HeadsNotDivisibleByKvHeads {
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }

        let head_dim = self.hidden_size / self.num_heads;
        if !head_dim.is_multiple_of(2) {
            return Err(ConfigError::OddRotaryDimension { head_dim });
        }

        if self.ffn_intermediate_size <= self.hidden_size {
            return Err(ConfigError::FfnNotWiderThanHidden {
                ffn_intermediate_size: self.ffn_intermediate_size,
                hidden_size: self.hidden_size,
            });
        }

        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "model.rms_norm_eps",
                reason: "must be finite and greater than zero",
            });
        }

        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "model.rope_theta",
                reason: "must be finite and greater than zero",
            });
        }

        self.parameter_count().ok_or(ConfigError::Overflow {
            field: "model parameter count",
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;
    use crate::config::error::ConfigError;

    #[test]
    fn default_matches_the_stage_one_baseline() {
        let config = ModelConfig::default();
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim(), Some(64));
        assert_eq!(config.max_sequence_length, 2048);
        assert_eq!(config.vocab_size, 32_768);
        assert!(config.tie_embeddings);
        assert!(!config.linear_bias);
        config.validate().expect("the baseline must be valid");
    }

    #[test]
    fn baseline_parameter_count_matches_the_documented_accounting() {
        let count = ModelConfig::default()
            .parameter_count()
            .expect("count fits in u64");
        // Documented Stage-1 target is approximately 110M in 100M..=150M.
        assert!(
            (100_000_000..=150_000_000).contains(&count),
            "unexpected parameter count {count}"
        );
        // Embedding alone is 32768 * 768.
        assert_eq!(32_768_u64 * 768, 25_165_824);
    }

    #[test]
    fn hides_size_not_divisible_by_heads_is_rejected() {
        let config = ModelConfig {
            hidden_size: 770,
            ..ModelConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::HiddenSizeNotDivisibleByHeads { .. })
        ));
    }

    #[test]
    fn odd_head_dimension_is_rejected() {
        let config = ModelConfig {
            hidden_size: 768,
            num_heads: 8,
            num_kv_heads: 8,
            ..ModelConfig::default()
        };
        // 768 / 8 = 96, which is even; make it odd via a different split.
        let config = ModelConfig {
            num_heads: 256,
            num_kv_heads: 256,
            ..config
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OddRotaryDimension { head_dim: 3 })
        ));
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        for config in [
            ModelConfig {
                num_layers: 0,
                ..ModelConfig::default()
            },
            ModelConfig {
                num_heads: 0,
                ..ModelConfig::default()
            },
            ModelConfig {
                max_sequence_length: 0,
                ..ModelConfig::default()
            },
        ] {
            assert!(matches!(
                config.validate(),
                Err(ConfigError::ZeroDimension { .. })
            ));
        }
    }

    #[test]
    fn kv_heads_must_divide_query_heads() {
        let config = ModelConfig {
            num_heads: 12,
            num_kv_heads: 5,
            ..ModelConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::HeadsNotDivisibleByKvHeads { .. })
        ));
    }

    #[test]
    fn grouped_query_attention_layout_is_expressible() {
        // Stage 2 shape: 12 query heads, 4 key/value heads. The schema must
        // accept it without a redesign.
        let config = ModelConfig {
            num_heads: 12,
            num_kv_heads: 4,
            ..ModelConfig::default()
        };
        config.validate().expect("GQA layout must validate");
    }

    #[test]
    fn feed_forward_must_be_wider_than_hidden() {
        let config = ModelConfig {
            ffn_intermediate_size: 768,
            ..ModelConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::FfnNotWiderThanHidden { .. })
        ));
    }

    #[test]
    fn non_finite_norms_and_rope_theta_are_rejected() {
        let config = ModelConfig {
            rms_norm_eps: 0.0,
            ..ModelConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "model.rms_norm_eps",
                ..
            })
        ));

        let config = ModelConfig {
            rope_theta: f64::NAN,
            ..ModelConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                field: "model.rope_theta",
                ..
            })
        ));
    }

    #[test]
    fn parameter_count_reports_overflow_instead_of_wrapping() {
        // Only the vocabulary is extreme, so the divisibility and width rules
        // all pass and validation reaches the parameter accounting.
        let config = ModelConfig {
            vocab_size: usize::MAX,
            ..ModelConfig::default()
        };
        assert!(config.parameter_count().is_none());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::Overflow { .. })
        ));
    }
}
