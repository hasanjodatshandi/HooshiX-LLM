//! Typed errors for the shared core layer.
//!
//! Each library crate owns a focused error enum; outer layers translate these
//! into CLI or HTTP representations. Messages identify the failed operation and
//! never embed raw data payloads.

use thiserror::Error;

/// Failure raised by the shared core layer.
///
/// Equality is derived so that parsing and validation can be asserted directly
/// in tests without matching every field.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoreError {
    /// An execution-backend name could not be recognised.
    #[error(
        "unknown execution backend `{value}`; expected one of: flex, cpu, wgpu, cuda, cuda:<index>"
    )]
    InvalidBackend {
        /// The rejected input value.
        value: String,
    },

    /// A backend index was present but is not a valid device index.
    #[error("invalid device index `{value}` for backend `{backend}`")]
    InvalidDeviceIndex {
        /// Backend the index was supplied for.
        backend: &'static str,
        /// The rejected input value.
        value: String,
    },
}
