//! Typed failures, and the status code each one becomes.
//!
//! The status code is decided here rather than at the handler, so a caller
//! cannot accidentally report a rejected prompt as a server fault. The
//! distinction matters: a `400` says the request was wrong and retrying it
//! unchanged will fail again, while a `503` says the server was not ready and
//! retrying might work.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use thiserror::Error;

/// Failure while serving a request.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ServerError {
    /// The configured limits cannot bound anything.
    #[error("invalid server limits: {reason}")]
    InvalidLimits {
        /// Why the limits were rejected.
        reason: String,
    },

    /// The prompt is longer than the byte limit.
    #[error("the prompt is {bytes} bytes, above the {limit}-byte limit")]
    PromptTooLong {
        /// Size of the prompt.
        bytes: usize,
        /// Configured limit.
        limit: usize,
    },

    /// The prompt is longer than the token limit.
    #[error("the prompt is {tokens} tokens, above the {limit}-token limit")]
    PromptTokensTooMany {
        /// Tokens the prompt encoded to.
        tokens: usize,
        /// Configured limit.
        limit: usize,
    },

    /// The request asked for more tokens than the limit allows.
    #[error("the request asked for {requested} new tokens, above the {limit} limit")]
    NewTokensTooMany {
        /// Tokens requested.
        requested: usize,
        /// Configured limit.
        limit: usize,
    },

    /// The sampling settings cannot be evaluated.
    #[error("{reason}")]
    InvalidSampling {
        /// Why the settings were rejected.
        reason: String,
    },

    /// The prompt could not be encoded.
    #[error("the prompt could not be encoded: {reason}")]
    Encoding {
        /// Underlying failure.
        reason: String,
    },

    /// Generation failed.
    #[error("generation failed: {reason}")]
    Generation {
        /// Underlying failure.
        reason: String,
    },

    /// Every generation slot is taken.
    #[error("the server is already generating {limit} request(s); retry shortly")]
    Busy {
        /// How many are allowed at once.
        limit: usize,
    },

    /// The model has not finished loading or has not passed its self-test.
    ///
    /// Distinct from `Busy`: this one means the server is not ready to serve at
    /// all, and no amount of retrying before readiness will help.
    #[error("the model is not ready to serve yet")]
    NotReady,

    /// The server could not be bound to an address.
    #[error("failed to bind `{address}`")]
    Bind {
        /// Address that was attempted.
        address: String,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
}

impl ServerError {
    /// The status code this failure is reported with.
    #[must_use]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::PromptTooLong { .. }
            | Self::PromptTokensTooMany { .. }
            | Self::NewTokensTooMany { .. }
            | Self::InvalidSampling { .. }
            | Self::Encoding { .. } => StatusCode::BAD_REQUEST,
            Self::Busy { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::NotReady => StatusCode::SERVICE_UNAVAILABLE,
            Self::Generation { .. } | Self::InvalidLimits { .. } | Self::Bind { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }
}

/// The body an error is reported with.
#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status();
        // The message is the error's own, which is written for a caller and
        // contains no internal paths or stack detail. A server fault still gets
        // logged where the handler can see it; the client is told what failed
        // rather than what to do about it.
        let body = Json(ErrorBody {
            error: self.to_string(),
        });
        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::ServerError;
    use axum::http::StatusCode;

    #[test]
    fn a_bad_request_is_not_reported_as_a_server_fault() {
        let error = ServerError::PromptTooLong {
            bytes: 100,
            limit: 10,
        };
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn being_busy_is_told_apart_from_being_unready() {
        // Retrying a busy server may work; retrying an unready one will not
        // until it is ready, and the two need different client behaviour.
        assert_eq!(
            ServerError::Busy { limit: 1 }.status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ServerError::NotReady.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn a_generation_failure_is_a_server_fault() {
        let error = ServerError::Generation {
            reason: "shapes disagree".to_owned(),
        };
        assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
