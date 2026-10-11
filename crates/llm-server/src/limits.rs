//! Request limits.
//!
//! Every limit here is checked *before* the work it bounds. Rejecting a prompt
//! after tokenizing it, or a token budget after allocating for it, means the
//! limit did not protect anything — it only reported the damage afterwards.

use serde::{Deserialize, Serialize};

use crate::error::ServerError;

/// What the server refuses to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerLimits {
    /// Longest prompt accepted, in bytes.
    ///
    /// Checked first because it can be checked without touching the tokenizer.
    pub max_prompt_bytes: usize,
    /// Longest prompt accepted, after tokenization.
    pub max_prompt_tokens: usize,
    /// Largest token budget a request may ask for.
    pub max_new_tokens: usize,
    /// Requests allowed to generate at once.
    pub max_concurrent_requests: usize,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_prompt_bytes: 64 * 1024,
            max_prompt_tokens: 2048,
            max_new_tokens: 512,
            // One model, one device. More than one concurrent generation does not
            // produce more tokens, it only divides the same cores between them
            // and makes every request slower. Raise this only with a measurement.
            max_concurrent_requests: 1,
        }
    }
}

impl ServerLimits {
    /// Rejects a limit set that cannot bound anything.
    pub fn validate(&self) -> Result<(), ServerError> {
        if self.max_prompt_bytes == 0 || self.max_prompt_tokens == 0 {
            return Err(ServerError::InvalidLimits {
                reason: "prompt limits must be greater than zero".to_owned(),
            });
        }
        if self.max_new_tokens == 0 {
            return Err(ServerError::InvalidLimits {
                reason: "max_new_tokens must be greater than zero".to_owned(),
            });
        }
        if self.max_concurrent_requests == 0 {
            return Err(ServerError::InvalidLimits {
                reason: "max_concurrent_requests must be greater than zero, \
                         which would refuse every request"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ServerLimits;
    use crate::error::ServerError;

    #[test]
    fn the_defaults_are_usable() {
        ServerLimits::default()
            .validate()
            .expect("defaults are valid");
    }

    #[test]
    fn a_zero_limit_is_refused() {
        for limits in [
            ServerLimits {
                max_prompt_bytes: 0,
                ..ServerLimits::default()
            },
            ServerLimits {
                max_prompt_tokens: 0,
                ..ServerLimits::default()
            },
            ServerLimits {
                max_new_tokens: 0,
                ..ServerLimits::default()
            },
            ServerLimits {
                max_concurrent_requests: 0,
                ..ServerLimits::default()
            },
        ] {
            assert!(matches!(
                limits.validate(),
                Err(ServerError::InvalidLimits { .. })
            ));
        }
    }
}
