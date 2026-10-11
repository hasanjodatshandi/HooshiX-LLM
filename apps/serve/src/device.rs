//! Choosing and building the execution backend.
//!
//! The same rules as the training composition root: the family comes from the
//! compiled-in feature, the configuration must agree with it, and the device is
//! built explicitly. `Device::default()` is never used, because it chooses from
//! the backends compiled into the binary rather than from the one the
//! configuration asked for.

use burn::tensor::Device;
use llm_core::BackendChoice;

/// The backend family this binary was compiled with.
#[must_use]
pub const fn compiled_backend() -> BackendChoice {
    #[cfg(feature = "cuda")]
    {
        BackendChoice::Cuda { index: 0 }
    }
    #[cfg(feature = "wgpu")]
    {
        BackendChoice::Wgpu { index: 0 }
    }
    #[cfg(feature = "cpu")]
    {
        BackendChoice::Cpu
    }
    #[cfg(feature = "flex")]
    {
        BackendChoice::Flex
    }
}

/// Builds the device the configuration asks for.
///
/// Inference does not need an autodiff device: nothing here computes a gradient,
/// and tracking one would retain intermediates for the life of the process.
#[must_use]
pub fn build_device(requested: BackendChoice) -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(requested.device_index())
    }
    #[cfg(feature = "wgpu")]
    {
        let _ = requested;
        Device::wgpu(Default::default())
    }
    #[cfg(feature = "cpu")]
    {
        let _ = requested;
        Device::cpu()
    }
    #[cfg(feature = "flex")]
    {
        let _ = requested;
        Device::flex()
    }
}

#[cfg(test)]
mod tests {
    use super::compiled_backend;
    use llm_core::BackendChoice;

    /// The feature that was enabled and the backend that is reported must agree,
    /// or a run would be served on a path the configuration did not name.
    #[test]
    fn the_reported_backend_matches_the_enabled_feature() {
        let expected = if cfg!(feature = "cuda") {
            BackendChoice::Cuda { index: 0 }
        } else if cfg!(feature = "wgpu") {
            BackendChoice::Wgpu
        } else if cfg!(feature = "cpu") {
            BackendChoice::Cpu
        } else {
            BackendChoice::Flex
        };
        assert!(compiled_backend().same_family(expected));
    }
}
