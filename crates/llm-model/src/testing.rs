//! Helpers shared by this crate's unit tests.
//!
//! The reference backend is named in exactly one place so that a test never
//! accidentally measures a different device from the one another test used.

use burn::tensor::Device;

/// Device used by unit tests: the pure-Rust reference backend.
///
/// Deterministic and free of just-in-time compilation, which is what makes it
/// the reference path.
pub(crate) fn reference_device() -> Device {
    Device::flex()
}
