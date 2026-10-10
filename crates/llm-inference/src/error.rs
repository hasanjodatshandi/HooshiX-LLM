//! Typed inference failures.

use thiserror::Error;

/// Failure while sampling or generating.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InferenceError {
    /// The model rejected an input.
    #[error(transparent)]
    Model(#[from] llm_model::ModelError),

    /// A sampling setting cannot be used.
    #[error("invalid sampling configuration: {reason}")]
    InvalidSampling {
        /// Why the configuration was rejected.
        reason: String,
    },

    /// The prompt carried no tokens.
    #[error("the prompt is empty, so there is nothing to continue")]
    EmptyPrompt,

    /// The vocabulary the model predicts is empty.
    #[error("the model produced logits with no vocabulary axis")]
    EmptyVocabulary,
}
