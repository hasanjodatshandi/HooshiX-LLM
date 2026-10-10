//! Inference for the Rust-native LLM stack.
//!
//! This crate owns turning a trained model into tokens: how the next token is
//! chosen, and how a sequence is extended one step at a time without re-reading
//! its own history.
//!
//! It owns no model mathematics — attention, the cache and the forward pass live
//! in `llm-model` — and no training, no dataset access and no checkpoint format.
//!
//! # Determinism
//!
//! Sampling takes its randomness as a parameter rather than reading a global
//! source. A greedy decoder therefore ignores randomness entirely, and a sampled
//! one replays exactly from its seed. That also keeps generation independent of
//! the random source the model was constructed with.

#![deny(unsafe_code)]

pub mod error;
pub mod generator;
pub mod sampling;

pub use error::InferenceError;
pub use generator::{Generation, GenerationConfig, StopReason, generate};
pub use sampling::{RandomSource, SamplingConfig, SeededRandom, sample};
