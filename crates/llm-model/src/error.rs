//! Typed model failures.

use std::fmt;

use thiserror::Error;

/// A quantity whose finiteness is checked during a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericalQuantity {
    /// The training objective.
    Loss,
    /// A parameter gradient.
    Gradients,
}

impl NumericalQuantity {
    /// Stable lowercase name used in messages and metrics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Loss => "loss",
            Self::Gradients => "gradients",
        }
    }
}

impl fmt::Display for NumericalQuantity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Failure while constructing or running the model.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The configuration cannot describe a model this code can build.
    #[error("invalid model configuration: {reason}")]
    InvalidConfig {
        /// Why the configuration was rejected.
        reason: String,
    },

    /// A tensor shape did not match what the layer requires.
    #[error("{operation} expected {expected}, but received {actual}")]
    Shape {
        /// Operation that rejected the input.
        operation: &'static str,
        /// Shape the operation requires.
        expected: String,
        /// Shape it received.
        actual: String,
    },

    /// A checked quantity contained `NaN` or an infinity.
    ///
    /// Reported rather than propagated: a non-finite loss that reaches the
    /// optimizer corrupts every parameter, and the corruption is only visible
    /// long afterwards.
    #[error("{quantity} is not finite: it contains a NaN or an infinity")]
    NonFinite {
        /// Which quantity was checked.
        quantity: NumericalQuantity,
    },

    /// Every position in the loss mask was ignored.
    ///
    /// A fully masked batch means the batch carries no learning signal. It is a
    /// batching defect, so it is reported instead of producing a `NaN` that
    /// would be attributed to numerical instability.
    #[error("every position in the loss mask was ignored, so the batch carries no loss")]
    EmptyLossMask,

    /// A cache was asked for a block it does not hold state for.
    ///
    /// Means the cache was built for a model of a different depth, which is a
    /// configuration defect rather than a runtime condition to tolerate.
    #[error("the cache holds {available} layers, but layer {requested} was requested")]
    CacheLayer {
        /// Layer index that was requested.
        requested: usize,
        /// Layers the cache holds.
        available: usize,
    },
}
