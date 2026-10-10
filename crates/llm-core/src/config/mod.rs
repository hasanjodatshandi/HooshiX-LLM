//! Experiment configuration: typed, versioned and validated at startup.
//!
//! # Contract
//!
//! Configuration types have public fields so they serialize directly to TOML,
//! which means a value can exist before it has been checked. Validation is
//! therefore mandatory and is performed by every sanctioned entry point:
//!
//! - [`ExperimentConfig::load`] reads a file, parses it and validates it;
//! - [`ExperimentConfig::from_toml_str`] parses and validates a string;
//! - [`ExperimentConfig::validate`] checks a value built in code.
//!
//! Everything of consequence runs after validation and before tensor
//! allocation, so an invalid configuration fails while it is still cheap.
//!
//! Unknown fields are rejected rather than ignored: in a strict, user-authored
//! configuration file a silently ignored typo is a correctness hazard.

pub mod data;
pub mod error;
pub mod generation;
pub mod model;
pub mod runtime;
pub mod tokenizer;
pub mod training;

use std::path::Path;

use serde::{Deserialize, Serialize};

pub use data::DataConfig;
pub use error::ConfigError;
pub use generation::{GenerationConfig, SamplingStrategy};
pub use model::ModelConfig;
pub use runtime::{Precision, RuntimeConfig};
pub use tokenizer::{SpecialTokens, TokenizerConfig, TokenizerModel};
pub use training::{OptimizerConfig, SchedulerConfig, TrainingConfig};

use crate::backend::BackendChoice;

/// Version of the configuration document format.
///
/// A persisted resolved configuration records this value so that a file written
/// by a different build is rejected instead of being reinterpreted.
///
/// Version 2 added the deterministic corpus-split settings to `DataConfig`.
/// Adding required fields is an incompatible change for a strict document, so
/// the version was raised rather than those fields being silently defaulted.
pub const CONFIG_VERSION: u32 = 2;

/// The complete, resolved configuration of one experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    /// Document-format version. Must equal [`CONFIG_VERSION`].
    pub version: u32,
    /// Model architecture.
    pub model: ModelConfig,
    /// Tokenizer contract.
    pub tokenizer: TokenizerConfig,
    /// Dataset and batching settings.
    pub data: DataConfig,
    /// Training-loop settings.
    pub training: TrainingConfig,
    /// Decoding settings.
    pub generation: GenerationConfig,
    /// Backend and precision.
    pub runtime: RuntimeConfig,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            model: ModelConfig::default(),
            tokenizer: TokenizerConfig::default(),
            data: DataConfig::default(),
            training: TrainingConfig::default(),
            generation: GenerationConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }
}

impl ExperimentConfig {
    /// Reads, parses and validates a configuration file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let config = Self::from_toml_str(&text).map_err(|error| match error {
            ConfigError::Parse { source, .. } => ConfigError::Parse {
                origin: format!("`{}`", path.display()),
                source,
            },
            other => other,
        })?;
        Ok(config)
    }

    /// Parses and validates a configuration document held in memory.
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text).map_err(|source| ConfigError::Parse {
            origin: "the provided document".to_owned(),
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Serializes the configuration as a TOML document.
    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self).map_err(ConfigError::Serialize)
    }

    /// Writes the resolved configuration next to a run's checkpoint.
    ///
    /// The resolved configuration, not the file the user edited, is what a run
    /// must be reproducible from, so defaults and every derived decision are
    /// materialized here.
    pub fn save_resolved(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let text = self.to_toml_string()?;
        std::fs::write(path, text).map_err(|source| ConfigError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Checks every invariant that does not depend on the running binary.
    ///
    /// Cross-subsystem rules live here because they belong to the experiment as
    /// a whole: the tokenizer vocabulary must match the model vocabulary, a
    /// packed sequence must fit the context window, and reserved special-token
    /// ids must exist in the embedding table.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != CONFIG_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                found: self.version,
                supported: CONFIG_VERSION,
            });
        }

        self.model.validate()?;
        self.tokenizer.validate()?;
        self.data.validate()?;
        self.training.validate()?;
        self.runtime.validate()?;

        if self.tokenizer.vocab_size != self.model.vocab_size {
            return Err(ConfigError::VocabularyMismatch {
                model_vocab_size: self.model.vocab_size,
                tokenizer_vocab_size: self.tokenizer.vocab_size,
            });
        }

        // The embedding table must cover every reserved special-token id.
        if usize::try_from(self.tokenizer.special_tokens.max_id())
            .map_or(true, |max_id| max_id >= self.model.vocab_size)
        {
            return Err(ConfigError::SpecialTokenOutOfVocabulary {
                name: "max",
                id: self.tokenizer.special_tokens.max_id(),
                vocab_size: self.model.vocab_size,
            });
        }

        if self.data.sequence_length > self.model.max_sequence_length {
            return Err(ConfigError::SequenceLengthExceedsContext {
                sequence_length: self.data.sequence_length,
                max_sequence_length: self.model.max_sequence_length,
            });
        }

        self.generation
            .validate(self.model.vocab_size, self.model.max_sequence_length)?;

        // Derived quantities must be computable without overflow.
        self.effective_tokens_per_step()
            .ok_or(ConfigError::Overflow {
                field: "effective tokens per step",
            })?;

        Ok(())
    }

    /// Checks the requested backend against the backend compiled into the binary.
    ///
    /// Kept separate from [`ExperimentConfig::validate`] because only the
    /// composition root knows which Cargo feature produced this build.
    pub fn validate_backend(&self, compiled: BackendChoice) -> Result<(), ConfigError> {
        self.runtime.validate_against_compiled(compiled)
    }

    /// Sequences processed per optimizer step.
    #[must_use]
    pub fn effective_batch_sequences(&self) -> Option<u64> {
        let micro = u64::try_from(self.data.micro_batch_size).ok()?;
        let accumulation = u64::from(self.training.gradient_accumulation_steps);
        let world = u64::try_from(self.training.data_parallel_world_size).ok()?;
        micro.checked_mul(accumulation)?.checked_mul(world)
    }

    /// Tokens processed per optimizer step.
    ///
    /// This is the quantity a training report must state, because an ambiguous
    /// "batch size" is not comparable between runs.
    #[must_use]
    pub fn effective_tokens_per_step(&self) -> Option<u64> {
        let sequences = self.effective_batch_sequences()?;
        let sequence_length = u64::try_from(self.data.sequence_length).ok()?;
        sequences.checked_mul(sequence_length)
    }
}

#[cfg(test)]
mod tests {
    use super::{CONFIG_VERSION, ExperimentConfig};
    use crate::config::error::ConfigError;

    #[test]
    fn default_is_valid_and_reports_a_stage_one_scale() {
        let config = ExperimentConfig::default();
        config.validate().expect("default must be valid");
        assert_eq!(config.effective_batch_sequences(), Some(8));
        assert_eq!(config.effective_tokens_per_step(), Some(8 * 2048));
    }

    #[test]
    fn a_version_from_another_build_is_rejected() {
        let config = ExperimentConfig {
            version: CONFIG_VERSION + 1,
            ..ExperimentConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn tokenizer_and_model_vocabularies_must_agree() {
        let mut config = ExperimentConfig::default();
        config.tokenizer.vocab_size = 16_000;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::VocabularyMismatch { .. })
        ));
    }

    #[test]
    fn a_sequence_longer_than_the_context_window_is_rejected() {
        let mut config = ExperimentConfig::default();
        config.data.sequence_length = config.model.max_sequence_length + 1;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::SequenceLengthExceedsContext { .. })
        ));
    }

    #[test]
    fn generation_budget_beyond_the_context_is_rejected() {
        let mut config = ExperimentConfig::default();
        config.generation.max_new_tokens = config.model.max_sequence_length + 1;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::GenerationExceedsContext { .. })
        ));
    }

    #[test]
    fn a_toml_document_round_trips_through_serialization() {
        let config = ExperimentConfig::default();
        let text = config.to_toml_string().expect("serialization succeeds");
        let parsed = ExperimentConfig::from_toml_str(&text).expect("round trip parses");
        assert_eq!(parsed, config);
    }

    #[test]
    fn unknown_fields_are_rejected_rather_than_ignored() {
        let text = r#"
version = 1
typo_field = true
"#;
        assert!(matches!(
            ExperimentConfig::from_toml_str(text),
            Err(ConfigError::Parse { .. })
        ));
    }

    #[test]
    fn a_negative_temperature_survives_parsing_but_fails_validation() {
        let text = r#"
version = 2

[model]
vocab_size = 32768
hidden_size = 768
num_layers = 12
num_heads = 12
num_kv_heads = 12
ffn_intermediate_size = 2048
max_sequence_length = 2048
rope_theta = 10000.0
rms_norm_eps = 1e-5
tie_embeddings = true
linear_bias = false

[tokenizer]
vocab_size = 32768
model = "bpe"
artifact = "artifacts/tokenizer.json"

[tokenizer.special_tokens]
bos = 0
eos = 1
pad = 2
unk = 3

[data]
data_root = "artifacts/data"
sequence_length = 2048
micro_batch_size = 8
shuffle_seed = 0
validation_basis_points = 50
test_basis_points = 50
split_seed = 0

[training]
max_steps = 10
data_parallel_world_size = 1
gradient_accumulation_steps = 1
seed = 0
checkpoint_interval_steps = 5
evaluation_interval_steps = 5

[training.optimizer]
learning_rate = 0.0003
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-8

[training.scheduler]
warmup_steps = 2
cosine_iters = 10

[generation]
max_new_tokens = 16
seed = 0
stop_on_eos = true

[generation.strategy]
kind = "temperature"
temperature = -1.0

[runtime]
backend = "flex"
precision = "f32"
"#;
        assert!(matches!(
            ExperimentConfig::from_toml_str(text),
            Err(ConfigError::OutOfRange {
                field: "generation.strategy.temperature",
                ..
            })
        ));
    }

    #[test]
    fn missing_required_sections_are_rejected() {
        assert!(matches!(
            ExperimentConfig::from_toml_str("version = 1\n"),
            Err(ConfigError::Parse { .. })
        ));
    }
}
