//! Typed training failures.

use thiserror::Error;

use llm_checkpoint::CheckpointError;

/// Failure while running a training step or loop.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TrainingError {
    /// The model rejected the batch.
    #[error(transparent)]
    Model(#[from] llm_model::ModelError),

    /// The loss was not finite, so the optimizer step was abandoned.
    ///
    /// A non-finite loss that reaches the optimizer writes `NaN` into every
    /// parameter, and the corruption only becomes visible long afterwards. The
    /// step is therefore refused rather than applied.
    #[error("refusing to apply an optimizer step: {0}")]
    NonFinite(#[from] NonFiniteLoss),

    /// The learning-rate schedule cannot be built.
    #[error("invalid learning-rate schedule: {reason}")]
    InvalidSchedule {
        /// Why the schedule was rejected.
        reason: String,
    },

    /// A step was asked for with nothing to average over.
    #[error("a training step needs at least one batch")]
    NoBatches,

    /// A checkpoint operation failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
}

/// Marker for a step abandoned because the loss was not finite.
#[derive(Debug, Error)]
#[error("the loss is not finite ({quantity})")]
pub struct NonFiniteLoss {
    /// Which quantity failed the check.
    pub quantity: llm_model::NumericalQuantity,
}
