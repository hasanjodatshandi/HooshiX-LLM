//! Typed configuration failures.
//!
//! Validation failures are structured rather than stringly typed, so a caller
//! can report exactly which rule rejected a configuration file. Messages name
//! the offending field and the values involved, and never embed file contents.

use std::path::PathBuf;

use thiserror::Error;

use crate::backend::BackendChoice;
use crate::config::runtime::Precision;

/// Failure while loading, parsing or validating experiment configuration.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// The configuration file could not be read.
    #[error("failed to read configuration file `{path}`")]
    Read {
        /// Path that was being read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The resolved configuration could not be written.
    #[error("failed to write resolved configuration to `{path}`")]
    Write {
        /// Path that was being written.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The input is not valid TOML, or it contains a field this build rejects.
    ///
    /// Unknown fields are rejected on purpose: a silently ignored typo in a
    /// strict, user-authored configuration is a correctness hazard.
    #[error("invalid configuration syntax or unknown field in {origin}")]
    Parse {
        /// Where the parsed text came from.
        origin: String,
        /// The underlying parser failure.
        #[source]
        source: toml::de::Error,
    },

    /// The configuration could not be serialized.
    #[error("failed to serialize the resolved configuration")]
    Serialize(#[source] toml::ser::Error),

    /// The file declares a configuration version this build does not understand.
    #[error(
        "unsupported configuration version {found}; this build understands version {supported}"
    )]
    UnsupportedVersion {
        /// Version found in the document.
        found: u32,
        /// Version this build reads and writes.
        supported: u32,
    },

    /// `hidden_size` must be divisible by `num_heads`.
    #[error("model.hidden_size ({hidden_size}) must be divisible by model.num_heads ({num_heads})")]
    HiddenSizeNotDivisibleByHeads {
        /// Configured hidden size.
        hidden_size: usize,
        /// Configured number of query heads.
        num_heads: usize,
    },

    /// `num_heads` must be divisible by `num_kv_heads`.
    #[error(
        "model.num_heads ({num_heads}) must be divisible by model.num_kv_heads ({num_kv_heads})"
    )]
    HeadsNotDivisibleByKvHeads {
        /// Configured number of query heads.
        num_heads: usize,
        /// Configured number of key/value heads.
        num_kv_heads: usize,
    },

    /// A dimension that must be positive was zero.
    #[error("{field} must be greater than zero")]
    ZeroDimension {
        /// Dotted path of the offending field.
        field: &'static str,
    },

    /// The feed-forward width must exceed the hidden size.
    #[error(
        "model.ffn_intermediate_size ({ffn_intermediate_size}) must exceed \
         model.hidden_size ({hidden_size})"
    )]
    FfnNotWiderThanHidden {
        /// Configured feed-forward width.
        ffn_intermediate_size: usize,
        /// Configured hidden size.
        hidden_size: usize,
    },

    /// The per-head dimension used by rotary embedding must be even.
    #[error("model.head_dim ({head_dim}) must be even for rotary position embedding")]
    OddRotaryDimension {
        /// Per-head dimension derived from `hidden_size / num_heads`.
        head_dim: usize,
    },

    /// A special-token id does not fit inside the tokenizer vocabulary.
    #[error(
        "tokenizer special token `{name}` has id {id}, \
         which is outside vocabulary size {vocab_size}"
    )]
    SpecialTokenOutOfVocabulary {
        /// Name of the offending special token.
        name: &'static str,
        /// Configured id.
        id: u32,
        /// Configured vocabulary size.
        vocab_size: usize,
    },

    /// Two special tokens were assigned the same id.
    #[error("tokenizer special-token id {id} is assigned more than once")]
    DuplicateSpecialTokenId {
        /// The duplicated id.
        id: u32,
    },

    /// The tokenizer and model vocabularies disagree.
    #[error(
        "tokenizer.vocab_size ({tokenizer_vocab_size}) \
         must equal model.vocab_size ({model_vocab_size})"
    )]
    VocabularyMismatch {
        /// Vocabulary declared by the model.
        model_vocab_size: usize,
        /// Vocabulary declared by the tokenizer.
        tokenizer_vocab_size: usize,
    },

    /// A packed training sequence would exceed the model context window.
    #[error(
        "data.sequence_length ({sequence_length}) \
         must not exceed model.max_sequence_length ({max_sequence_length})"
    )]
    SequenceLengthExceedsContext {
        /// Configured packed sequence length.
        sequence_length: usize,
        /// Configured model context length.
        max_sequence_length: usize,
    },

    /// A generation request could exceed the model context window.
    #[error(
        "generation.max_new_tokens ({max_new_tokens}) \
         must not exceed model.max_sequence_length ({max_sequence_length})"
    )]
    GenerationExceedsContext {
        /// Configured generation budget.
        max_new_tokens: usize,
        /// Configured model context length.
        max_sequence_length: usize,
    },

    /// A numeric hyperparameter is outside its allowed range.
    #[error("{field} is out of range: {reason}")]
    OutOfRange {
        /// Dotted path of the offending field.
        field: &'static str,
        /// Why the value was rejected.
        reason: &'static str,
    },

    /// The configured top-k exceeds the model vocabulary.
    #[error("generation top-k ({k}) must be within 1..={vocab_size}")]
    TopKOutOfVocabulary {
        /// Configured k.
        k: usize,
        /// Vocabulary size of the model.
        vocab_size: usize,
    },

    /// The requested backend is not the backend this binary was compiled with.
    #[error("runtime.backend is `{requested}` but this binary was built with `{compiled}`")]
    BackendMismatch {
        /// Backend named by the configuration.
        requested: BackendChoice,
        /// Backend compiled into the running binary.
        compiled: BackendChoice,
    },

    /// The requested numeric precision is unavailable on the requested backend.
    #[error("runtime.precision `{precision}` is not supported on backend `{backend}`")]
    UnsupportedPrecision {
        /// Backend the precision was requested for.
        backend: BackendChoice,
        /// The rejected precision.
        precision: Precision,
    },

    /// Computing a derived quantity overflowed.
    #[error("computing {field} overflowed")]
    Overflow {
        /// Dotted path or name of the derived quantity.
        field: &'static str,
    },

    /// An execution-backend name inside the configuration could not be parsed.
    #[error("invalid runtime.backend: {0}")]
    Backend(#[from] crate::error::CoreError),
}
