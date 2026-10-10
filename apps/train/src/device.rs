//! Backend feature selection, device construction and device preflight.
//!
//! The backend *family* is chosen at compile time through this crate's Cargo
//! features; the *device index* within that family is a runtime choice that
//! comes from configuration. Constructors are always explicit: `Device::default()`
//! selects from the backends compiled into the binary rather than from the
//! hardware present, so enabling an extra backend could silently change which
//! device a run uses.

use burn::tensor::{Device, Tensor};
use llm_core::BackendChoice;

// Two accelerator backends in one binary is always a mistake: it doubles build
// cost and makes the compiled-in backend ambiguous. `flex` is deliberately
// allowed alongside an accelerator, because it is the reference path and test
// builds pull it in as a dev-dependency; `compiled_backend` then states which
// one wins, and every device is constructed explicitly anyway.
#[cfg(all(feature = "cpu", feature = "cuda"))]
compile_error!("features `cpu` and `cuda` are mutually exclusive; enable one accelerator backend");
#[cfg(all(feature = "cpu", feature = "wgpu"))]
compile_error!("features `cpu` and `wgpu` are mutually exclusive; enable one accelerator backend");
#[cfg(all(feature = "cuda", feature = "wgpu"))]
compile_error!("features `cuda` and `wgpu` are mutually exclusive; enable one accelerator backend");

#[cfg(not(any(feature = "flex", feature = "cpu", feature = "cuda", feature = "wgpu")))]
compile_error!(
    "no execution backend selected; enable exactly one of `flex` (reference), `cpu`, `cuda`, `wgpu`"
);

/// The backend family this binary was compiled with.
///
/// The device index is a runtime selection, so the returned value carries index
/// zero as a placeholder. Capability checks compare families, not indices.
///
/// When more than one backend is compiled in — which happens in test builds,
/// where the reference path arrives as a dev-dependency — the accelerator wins,
/// so the value names the backend the run was actually built for.
#[must_use]
pub const fn compiled_backend() -> BackendChoice {
    if cfg!(feature = "cuda") {
        BackendChoice::Cuda { index: 0 }
    } else if cfg!(feature = "wgpu") {
        BackendChoice::Wgpu
    } else if cfg!(feature = "cpu") {
        BackendChoice::Cpu
    } else {
        BackendChoice::Flex
    }
}

/// Builds the device named by the configuration.
///
/// The family comes from the compiled-in feature and the index from the
/// request. Callers must have checked the request against
/// [`compiled_backend`] first, so the request is known to name this family.
#[must_use]
pub fn build_device(requested: BackendChoice) -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(requested.device_index())
    }
    #[cfg(feature = "wgpu")]
    {
        // WGPU picks its adapter from device settings rather than from an index.
        let _ = requested;
        Device::wgpu(Default::default())
    }
    #[cfg(feature = "cpu")]
    {
        // The CubeCL CPU runtime is single-device.
        let _ = requested;
        Device::cpu()
    }
    #[cfg(feature = "flex")]
    {
        // The Flex backend is single-device.
        let _ = requested;
        Device::flex()
    }
}

/// Failure while proving that the selected backend can execute a tensor
/// operation.
#[derive(Debug, thiserror::Error)]
pub enum PreflightError {
    /// The result could not be read back from the device.
    #[error("reading the preflight result back from the device failed")]
    Readback(#[source] burn::tensor::DataError),

    /// The device produced an unexpected value.
    #[error("preflight produced {actual:?}; expected six elements equal to 2.0")]
    UnexpectedResult {
        /// The values the device actually returned.
        actual: Vec<f32>,
    },
}

/// Proves the device can execute an operation before any real work is launched.
///
/// This is the check that must pass on a backend before a training or
/// generation run is started on it.
pub fn preflight(device: &Device) -> Result<(), PreflightError> {
    let left = Tensor::<2>::ones([2, 3], device);
    let right = Tensor::<2>::ones([2, 3], device);

    let actual = (left + right)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(PreflightError::Readback)?;

    let expected = [2.0_f32; 6];
    if actual != expected {
        return Err(PreflightError::UnexpectedResult { actual });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PreflightError, build_device, compiled_backend, preflight};

    #[test]
    fn compiled_backend_matches_the_enabled_feature() {
        let expected = if cfg!(feature = "cuda") {
            llm_core::BackendChoice::Cuda { index: 0 }
        } else if cfg!(feature = "wgpu") {
            llm_core::BackendChoice::Wgpu
        } else if cfg!(feature = "cpu") {
            llm_core::BackendChoice::Cpu
        } else {
            llm_core::BackendChoice::Flex
        };
        assert_eq!(compiled_backend(), expected);
    }

    #[test]
    fn device_passes_its_own_preflight() {
        let device = build_device(compiled_backend());
        preflight(&device).expect("the compiled backend must pass its preflight");
    }

    #[test]
    fn preflight_error_message_names_the_operation() {
        let error = PreflightError::UnexpectedResult { actual: vec![0.0] };
        assert!(error.to_string().contains("preflight produced"));
    }
}
