//! Decoder-only Transformer primitives, the model definition and the causal
//! language-model loss.
//!
//! This crate owns the model mathematics: normalization, rotary position
//! embedding, causal attention, the gated feed-forward network, the decoder
//! stack, the tied language-model head and the training objective. It owns no
//! HTTP, no CLI, no filesystem policy, no dataset acquisition and no checkpoint
//! layout.
//!
//! # Backend independence
//!
//! Burn 0.22 selects its execution backend at runtime through `Device`, so the
//! types here carry no backend parameter. Nothing in this crate names a backend,
//! which is what lets the same model run on the reference path and on an
//! accelerator without a conditional anywhere in the mathematics.
//!
//! # Order of construction
//!
//! The primitives are independent and testable one at a time — normalization,
//! rotary embedding, the causal mask, attention, the feed-forward network, a
//! block — and [`Transformer`] assembles them. Every layer validates the shapes
//! it receives and reports a [`ModelError`] rather than silently broadcasting
//! something unintended.

#![deny(unsafe_code)]

pub mod attention;
pub mod block;
pub mod causal_mask;
pub mod embedding;
pub mod error;
pub mod grad;
pub mod init;
pub mod kv_cache;
pub mod loss;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;
pub mod transformer;

#[cfg(test)]
pub(crate) mod testing;

pub use attention::MultiHeadAttention;
pub use block::TransformerBlock;
pub use causal_mask::causal_mask;
pub use embedding::TokenEmbedding;
pub use error::{ModelError, NumericalQuantity};
pub use grad::ensure_finite;
pub use kv_cache::{KvCache, LayerCache};
pub use loss::{all_positions, causal_lm_loss, next_token_pairs};
pub use rms_norm::RmsNorm;
pub use rope::Rope;
pub use swiglu::SwiGlu;
pub use transformer::Transformer;
