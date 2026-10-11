//! Choosing the execution backend.
//!
//! The same rules as the other composition roots: the family comes from the
//! compiled-in feature, the configuration must agree with it, and the device is
//! built explicitly rather than through `Device::default()`.

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
        BackendChoice::Wgpu
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
/// Not an autodiff device here; the caller adds `.autodiff()` because only the
/// training measurement needs a graph, and tracking one through prefill and
/// decode would retain intermediates and misreport memory.
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
