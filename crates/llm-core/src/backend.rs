//! Execution-backend selection.
//!
//! Burn 0.22 selects its execution backend at runtime through `Device`, so the
//! backend is a value rather than a type parameter on the model.
//!
//! `flex` and `cpu` are distinct CPU backends and are not interchangeable:
//!
//! - [`BackendChoice::Flex`] is the pure-Rust eager backend and is this
//!   project's deterministic reference path;
//! - [`BackendChoice::Cpu`] runs through CubeCL's CPU runtime with LLVM JIT and
//!   supports fusion; it is a performance path, not a reference path.
//!
//! NdArray and LibTorch are deprecated in Burn 0.22 and Candle was removed, so
//! none of them is representable here.

use core::fmt;
use core::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::CoreError;

/// Execution backend requested for a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendChoice {
    /// Pure-Rust eager CPU backend. The deterministic reference path.
    Flex,
    /// CubeCL CPU runtime with LLVM JIT. A performance path.
    Cpu,
    /// NVIDIA GPU through CubeCL's CUDA runtime.
    Cuda {
        /// Zero-based device index.
        index: usize,
    },
    /// Portable graphics API through WGPU.
    Wgpu,
}

impl BackendChoice {
    /// Stable lowercase name used in configuration files and run manifests.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Flex => "flex",
            Self::Cpu => "cpu",
            Self::Cuda { .. } => "cuda",
            Self::Wgpu => "wgpu",
        }
    }

    /// Whether this choice can only run on accelerator hardware.
    ///
    /// A backend that requires an accelerator must not be used to satisfy a
    /// correctness gate on a machine without one; such validation is reported
    /// as not run rather than passed.
    #[must_use]
    pub const fn requires_accelerator(self) -> bool {
        matches!(self, Self::Cuda { .. })
    }

    /// Device index within the backend family.
    ///
    /// Only indexed backends distinguish devices; the single-device CPU
    /// backends are always index zero.
    #[must_use]
    pub const fn device_index(self) -> usize {
        match self {
            Self::Cuda { index } => index,
            Self::Flex | Self::Cpu | Self::Wgpu => 0,
        }
    }

    /// Whether two choices belong to the same backend family.
    ///
    /// The family is a compile-time Cargo feature; the device index is a
    /// runtime selection inside it. Capability checks must compare families,
    /// because a binary built with the `cuda` feature can legitimately be asked
    /// for `cuda:3`.
    #[must_use]
    pub const fn same_family(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Flex, Self::Flex)
                | (Self::Cpu, Self::Cpu)
                | (Self::Wgpu, Self::Wgpu)
                | (Self::Cuda { .. }, Self::Cuda { .. })
        )
    }
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda { index } => write!(formatter, "cuda:{index}"),
            other => formatter.write_str(other.as_str()),
        }
    }
}

impl FromStr for BackendChoice {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();

        if value.eq_ignore_ascii_case("flex") {
            return Ok(Self::Flex);
        }
        if value.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if value.eq_ignore_ascii_case("wgpu") {
            return Ok(Self::Wgpu);
        }

        if let Some(rest) = value
            .strip_prefix("cuda")
            .or_else(|| value.strip_prefix("CUDA"))
        {
            let index_text = rest.strip_prefix(':').unwrap_or("0");
            let index = index_text
                .parse::<usize>()
                .map_err(|_| CoreError::InvalidDeviceIndex {
                    backend: "cuda",
                    value: index_text.to_owned(),
                })?;
            return Ok(Self::Cuda { index });
        }

        Err(CoreError::InvalidBackend {
            value: value.to_owned(),
        })
    }
}

impl Serialize for BackendChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for BackendChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        Self::from_str(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::BackendChoice;
    use crate::error::CoreError;

    #[test]
    fn parses_every_supported_backend_name() {
        assert_eq!("flex".parse::<BackendChoice>(), Ok(BackendChoice::Flex));
        assert_eq!("cpu".parse::<BackendChoice>(), Ok(BackendChoice::Cpu));
        assert_eq!("wgpu".parse::<BackendChoice>(), Ok(BackendChoice::Wgpu));
        assert_eq!(
            "cuda".parse::<BackendChoice>(),
            Ok(BackendChoice::Cuda { index: 0 })
        );
        assert_eq!(
            "cuda:3".parse::<BackendChoice>(),
            Ok(BackendChoice::Cuda { index: 3 })
        );
    }

    #[test]
    fn parsing_is_case_insensitive_and_trims_surrounding_space() {
        assert_eq!("  FLex ".parse::<BackendChoice>(), Ok(BackendChoice::Flex));
        assert_eq!(
            "CUDA:1".parse::<BackendChoice>(),
            Ok(BackendChoice::Cuda { index: 1 })
        );
    }

    #[test]
    fn rejects_unknown_backend_names() {
        let error = "tpu"
            .parse::<BackendChoice>()
            .expect_err("`tpu` is not a backend");
        assert!(matches!(error, CoreError::InvalidBackend { .. }));
    }

    #[test]
    fn rejects_a_non_numeric_cuda_index() {
        let error = "cuda:abc"
            .parse::<BackendChoice>()
            .expect_err("`abc` is not a device index");
        assert!(matches!(error, CoreError::InvalidDeviceIndex { .. }));
    }

    #[test]
    fn display_round_trips_through_parsing() {
        for choice in [
            BackendChoice::Flex,
            BackendChoice::Cpu,
            BackendChoice::Wgpu,
            BackendChoice::Cuda { index: 2 },
        ] {
            assert_eq!(choice.to_string().parse::<BackendChoice>(), Ok(choice));
        }
    }

    #[test]
    fn only_cuda_requires_an_accelerator() {
        assert!(BackendChoice::Cuda { index: 0 }.requires_accelerator());
        assert!(!BackendChoice::Flex.requires_accelerator());
        assert!(!BackendChoice::Cpu.requires_accelerator());
        assert!(!BackendChoice::Wgpu.requires_accelerator());
    }

    #[test]
    fn device_index_is_meaningful_only_for_indexed_backends() {
        assert_eq!(BackendChoice::Cuda { index: 3 }.device_index(), 3);
        assert_eq!(BackendChoice::Flex.device_index(), 0);
        assert_eq!(BackendChoice::Cpu.device_index(), 0);
        assert_eq!(BackendChoice::Wgpu.device_index(), 0);
    }

    #[test]
    fn same_family_ignores_the_device_index() {
        // A binary built with the `cuda` feature reports index zero as a
        // placeholder, and must still accept a request for another device.
        assert!(BackendChoice::Cuda { index: 0 }.same_family(BackendChoice::Cuda { index: 3 }));
        assert!(BackendChoice::Flex.same_family(BackendChoice::Flex));
        assert!(!BackendChoice::Flex.same_family(BackendChoice::Cpu));
        assert!(!BackendChoice::Cuda { index: 0 }.same_family(BackendChoice::Wgpu));
        assert!(!BackendChoice::Cpu.same_family(BackendChoice::Wgpu));
    }
}
