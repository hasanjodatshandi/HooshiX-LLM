//! Rust-native tokenizer training, loading and encoding.
//!
//! This crate owns the tokenizer artifact and the vocabulary contract it must
//! satisfy. It depends on [`llm_core`] for the configuration types and on the
//! Rust `tokenizers` crate for the BPE implementation; it owns no dataset
//! loading, no training loop and no serving behaviour.
//!
//! # Contract
//!
//! - Reserved-token *literals* are fixed by this crate; reserved-token *ids*
//!   come from [`llm_core::TokenizerConfig`]. An artifact is only accepted when
//!   both agree, because the model's embedding rows are tied to those ids.
//! - Encoding never inserts or removes tokens implicitly, and never truncates
//!   implicitly. Length limits are stated by the caller.
//! - [`LlmTokenizer::identity_hash`] is the stable identity recorded alongside
//!   dataset shards and checkpoints.
//!
//! # Backend note
//!
//! The `tokenizers` dependency is built with `default-features = false` and the
//! pure-Rust `fancy-regex` backend, so it introduces no native C or C++
//! dependency.

mod artifact;
mod error;
mod special_tokens;
mod train;

pub use artifact::{LlmTokenizer, SpecialTokensInOutput, Truncation};
pub use error::TokenizerError;
pub use special_tokens::{SpecialTokenKind, SpecialTokenRegistry};
