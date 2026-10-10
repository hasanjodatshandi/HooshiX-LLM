//! Execution backend and numeric precision configuration.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::backend::BackendChoice;
use crate::config::error::ConfigError;

/// Floating-point precision requested for tensor operations.
///
/// This mirrors Burn's float dtypes without depending on Burn: `llm-core` is a
/// value-type crate and must not pull the tensor framework into the inner layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// 32-bit float. The Stage-1 smoke default.
    F32,
    /// Brain float 16.
    Bf16,
    /// 16-bit float. Requires loss scaling; not a default.
    F16,
}

impl fmt::Display for Precision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        })
    }
}

/// Where computation runs and at what precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    /// Requested execution backend.
    ///
    /// The backend is a compile-time Cargo feature, so this value must agree
    /// with the backend the binary was built with. A mismatch is rejected
    /// rather than silently ignored.
    pub backend: BackendChoice,
    /// Requested floating-point precision.
    pub precision: Precision,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: BackendChoice::Flex,
            precision: Precision::F32,
        }
    }
}

impl RuntimeConfig {
    /// Rejects a backend/precision combination that is not supported.
    ///
    /// Reduced precision is accepted only on an accelerator backend, and even
    /// there it remains gated by forward/backward parity tests: a configuration
    /// being accepted is not evidence that the precision is numerically safe.
    pub fn validate(&self) -> Result<(), ConfigError> {
        match (self.backend, self.precision) {
            (_, Precision::F32) => Ok(()),
            (BackendChoice::Cuda { .. }, Precision::Bf16 | Precision::F16) => Ok(()),
            (backend, precision) => Err(ConfigError::UnsupportedPrecision { backend, precision }),
        }
    }

    /// Checks the requested backend against the family compiled into the binary.
    ///
    /// The family is fixed by a Cargo feature, so a mismatch means this binary
    /// cannot run the configuration at all and must fail rather than silently
    /// fall back. The device index is a runtime selection inside the family and
    /// is therefore not part of this check: a binary built with the `cuda`
    /// feature may legitimately be asked for `cuda:3`.
    pub fn validate_against_compiled(&self, compiled: BackendChoice) -> Result<(), ConfigError> {
        if self.backend.same_family(compiled) {
            return Ok(());
        }
        Err(ConfigError::BackendMismatch {
            requested: self.backend,
            compiled,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Precision, RuntimeConfig};
    use crate::backend::BackendChoice;
    use crate::config::error::ConfigError;

    #[test]
    fn default_targets_the_reference_backend_in_fp32() {
        let config = RuntimeConfig::default();
        config.validate().expect("default must be valid");
        assert_eq!(config.backend, BackendChoice::Flex);
        assert_eq!(config.precision, Precision::F32);
    }

    #[test]
    fn reduced_precision_is_rejected_on_cpu_backends() {
        for backend in [BackendChoice::Flex, BackendChoice::Cpu, BackendChoice::Wgpu] {
            let config = RuntimeConfig {
                backend,
                precision: Precision::Bf16,
            };
            assert!(matches!(
                config.validate(),
                Err(ConfigError::UnsupportedPrecision { .. })
            ));
        }
    }

    #[test]
    fn reduced_precision_is_accepted_on_cuda() {
        let config = RuntimeConfig {
            backend: BackendChoice::Cuda { index: 0 },
            precision: Precision::Bf16,
        };
        config.validate().expect("bf16 on cuda is allowed");
    }

    #[test]
    fn backend_must_match_the_compiled_feature() {
        let config = RuntimeConfig {
            backend: BackendChoice::Cuda { index: 0 },
            precision: Precision::F32,
        };
        assert!(matches!(
            config.validate_against_compiled(BackendChoice::Flex),
            Err(ConfigError::BackendMismatch { .. })
        ));
        config
            .validate_against_compiled(BackendChoice::Cuda { index: 0 })
            .expect("identical backends must agree");
    }

    #[test]
    fn any_device_of_the_compiled_family_is_accepted() {
        // The family is a compile-time capability; the index is a runtime
        // choice, so `cuda:3` is valid in a binary compiled with `cuda`.
        let config = RuntimeConfig {
            backend: BackendChoice::Cuda { index: 3 },
            precision: Precision::F32,
        };
        config
            .validate_against_compiled(BackendChoice::Cuda { index: 0 })
            .expect("device index must not be treated as a capability mismatch");
    }
}
