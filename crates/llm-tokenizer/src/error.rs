//! Typed tokenizer failures.

use std::path::PathBuf;

use thiserror::Error;

/// Failure while training, loading or using a tokenizer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TokenizerError {
    /// The artifact could not be read from disk.
    #[error("failed to read tokenizer artifact `{path}`")]
    Read {
        /// Path that was being read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The artifact could not be written to disk.
    #[error("failed to write tokenizer artifact `{path}`")]
    Write {
        /// Path that was being written.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The artifact is not a tokenizer document this build can read.
    #[error("`{path}` is not a readable tokenizer document")]
    Load {
        /// Path that was loaded.
        path: PathBuf,
        /// The underlying deserialization failure.
        #[source]
        source: tokenizers::Error,
    },

    /// The artifact could not be serialized.
    #[error("failed to serialize the tokenizer artifact")]
    Save(#[source] tokenizers::Error),

    /// Training failed.
    #[error("tokenizer training failed")]
    Train(#[source] tokenizers::Error),

    /// Encoding failed.
    #[error("tokenizer encoding failed")]
    Encode(#[source] tokenizers::Error),

    /// Decoding failed.
    #[error("tokenizer decoding failed")]
    Decode(#[source] tokenizers::Error),

    /// The corpus could not fill the configured vocabulary.
    ///
    /// This is a data problem, not a tuning problem: a vocabulary that is not
    /// filled means token ids do not mean what the configuration says they mean.
    #[error(
        "the corpus produced {actual} tokens but {expected} were configured; \
         the corpus is too small or too repetitive for this vocabulary size"
    )]
    VocabularyNotFilled {
        /// Vocabulary size the configuration asked for.
        expected: usize,
        /// Vocabulary size the corpus actually produced.
        actual: usize,
    },

    /// The configured vocabulary cannot hold the byte alphabet and reserved ids.
    ///
    /// A byte-level tokenizer must be able to represent every byte from the
    /// start; otherwise some valid UTF-8 input has no token sequence at all.
    #[error(
        "vocabulary size {configured} cannot hold the {required} required tokens \
         (the byte alphabet plus the reserved tokens)"
    )]
    VocabularyTooSmall {
        /// Vocabulary size the configuration asked for.
        configured: usize,
        /// Smallest vocabulary that can represent every byte and reserved token.
        required: usize,
    },

    /// The artifact vocabulary disagrees with the configured vocabulary.
    #[error("tokenizer vocabulary size {actual} does not match the configured {expected}")]
    VocabularySizeMismatch {
        /// Size the configuration requires.
        expected: usize,
        /// Size the artifact actually has.
        actual: usize,
    },

    /// A reserved token is not bound to its configured id in the artifact.
    ///
    /// Ids are part of the model's contract: the embedding rows are tied to
    /// them, so a mismatch is a hard incompatibility rather than a warning.
    #[error(
        "special token `{literal}` is bound to id {actual:?} in the artifact, \
         but id {expected} is configured"
    )]
    SpecialTokenMismatch {
        /// Literal text of the reserved token.
        literal: &'static str,
        /// Id the configuration requires.
        expected: u32,
        /// Id the artifact actually uses, if the token is present at all.
        actual: Option<u32>,
    },

    /// Reserved ids are not the contiguous block the trainer can produce.
    #[error(
        "special-token ids must be exactly the contiguous range 0..{count} \
         so the trainer can place them at the head of the vocabulary, but got {ids:?}"
    )]
    SpecialTokenIdsNotContiguous {
        /// Number of reserved tokens.
        count: usize,
        /// The ids that were configured.
        ids: Vec<u32>,
    },

    /// The input produced more tokens than the caller allowed.
    #[error("input produced {actual} tokens, which exceeds the limit of {limit}")]
    InputTooLong {
        /// The limit the caller set.
        limit: usize,
        /// How many tokens the input actually produced.
        actual: usize,
    },

    /// The tokenizer configuration itself is invalid.
    #[error(transparent)]
    Config(#[from] llm_core::ConfigError),
}
