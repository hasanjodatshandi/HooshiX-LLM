//! The HTTP surface for inference.
//!
//! This crate owns routing, request validation, limits, streaming and the
//! readiness lifecycle. It owns no model mathematics, no sampling policy and no
//! tokenizer implementation — those belong to `llm-model`, `llm-inference` and
//! `llm-tokenizer`, and this crate only decides what a request may ask them for.
//!
//! # What it deliberately does not do
//!
//! - **No chat endpoint.** A base pretraining model is not an instruction
//!   follower, and exposing a chat API that answers with a continuation would
//!   misrepresent what the model is. `09_INFERENCE_AND_SERVING.md` section 7
//!   defers that until a chat template and a post-trained model exist.
//! - **No model loading in a handler.** State is built once by the composition
//!   root and shared. There is no code path here that constructs a `Transformer`.
//! - **No queue.** A request that cannot be served now is refused, not held.
//!   Continuous batching is a Stage-3 concern and needs benchmark evidence first.

#![deny(unsafe_code)]

pub mod error;
pub mod limits;
pub mod routes;
pub mod state;

pub use error::ServerError;
pub use limits::ServerLimits;
pub use routes::{CompletionRequest, CompletionResponse, router, serve};
pub use state::AppState;
