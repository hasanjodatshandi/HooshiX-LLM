//! Shared core types for the Rust-native LLM stack.
//!
//! This crate owns cross-crate configuration and value types, validated
//! identifiers and units, and the error foundation shared by the inner layers.
//!
//! Rules this crate follows:
//!
//! - it depends on no other workspace crate, so it cannot invert the dependency
//!   direction;
//! - it owns no tensor, dataset, request or checkpoint behaviour;
//! - it must not become a miscellaneous dumping ground.

pub mod backend;
pub mod config;
pub mod error;

pub use backend::BackendChoice;
pub use config::{
    CONFIG_VERSION, ConfigError, DataConfig, ExperimentConfig, GenerationConfig, ModelConfig,
    OptimizerConfig, Precision, RuntimeConfig, SamplingStrategy, SchedulerConfig, SpecialTokens,
    TokenizerConfig, TokenizerModel, TrainingConfig,
};
pub use error::CoreError;
