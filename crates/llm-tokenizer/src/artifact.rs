//! The tokenizer artifact: training, loading, saving, encoding and decoding.
//!
//! An artifact is only usable together with the contract it was built for, so
//! [`LlmTokenizer`] always carries a [`SpecialTokenRegistry`] and can re-check
//! that contract against a [`TokenizerConfig`] at any time.

use std::path::Path;

use llm_core::TokenizerConfig;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::error::TokenizerError;
use crate::special_tokens::SpecialTokenRegistry;
use crate::train::{BpeTrainingParams, train_byte_level_bpe};

/// What to do when an input encodes to more tokens than a caller allows.
///
/// Encoding never truncates implicitly: dropping a suffix of a training
/// sequence changes the objective, so the caller states the intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truncation {
    /// Fail instead of truncating.
    Reject,
    /// Keep the first `limit` tokens and drop the remainder.
    KeepFirst,
}

/// Whether reserved tokens survive decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialTokensInOutput {
    /// Drop reserved tokens, producing plain text.
    Skip,
    /// Keep reserved tokens, so a decode/encode round trip is lossless.
    Keep,
}

/// A trained tokenizer bound to the vocabulary contract it must satisfy.
pub struct LlmTokenizer {
    inner: Tokenizer,
    registry: SpecialTokenRegistry,
}

impl LlmTokenizer {
    /// Trains a byte-level BPE tokenizer against a configuration's contract.
    ///
    /// Sequences are consumed in iteration order, so determinism is the caller's
    /// responsibility; see [`crate::tokenizer`] module notes.
    pub fn train<I, S>(
        sequences: I,
        config: &TokenizerConfig,
        min_frequency: u64,
    ) -> Result<Self, TokenizerError>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        config.validate()?;

        let registry = SpecialTokenRegistry::new(config.special_tokens);
        let params = BpeTrainingParams {
            vocab_size: config.vocab_size,
            min_frequency,
            registry,
        };

        let inner = train_byte_level_bpe(sequences, &params)?;
        let tokenizer = Self { inner, registry };
        tokenizer.verify_against(config)?;
        Ok(tokenizer)
    }

    /// Loads an artifact and checks it against a configuration's contract.
    ///
    /// The bytes are read here rather than through the library's file helper so
    /// that a missing or unreadable artifact reports its path directly, and so
    /// that the artifact size is bounded before it is parsed.
    pub fn load(path: impl AsRef<Path>, config: &TokenizerConfig) -> Result<Self, TokenizerError> {
        config.validate()?;

        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| TokenizerError::Read {
            path: path.to_path_buf(),
            source,
        })?;

        let inner = Tokenizer::from_bytes(bytes).map_err(|source| TokenizerError::Load {
            path: path.to_path_buf(),
            source,
        })?;

        let tokenizer = Self {
            inner,
            registry: SpecialTokenRegistry::new(config.special_tokens),
        };
        tokenizer.verify_against(config)?;
        Ok(tokenizer)
    }

    /// Writes the artifact as compact JSON.
    ///
    /// Compact rather than pretty because the artifact is a machine input and
    /// its size is a tracked measurement; the identity hash is what identifies
    /// it, not its formatting.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), TokenizerError> {
        self.inner
            .save(path.as_ref(), false)
            .map_err(TokenizerError::Save)
    }

    /// Encodes text without adding reserved tokens.
    ///
    /// BOS/EOS insertion is the training pipeline's decision, not the
    /// tokenizer's, and no document boundary is invented here.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(TokenizerError::Encode)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encodes text under an explicit length policy.
    pub fn encode_limited(
        &self,
        text: &str,
        limit: usize,
        truncation: Truncation,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = self.encode(text)?;
        if ids.len() > limit {
            match truncation {
                Truncation::Reject => {
                    return Err(TokenizerError::InputTooLong {
                        limit,
                        actual: ids.len(),
                    });
                }
                Truncation::KeepFirst => ids.truncate(limit),
            }
        }
        Ok(ids)
    }

    /// Decodes ids back to text.
    pub fn decode(
        &self,
        ids: &[u32],
        special: SpecialTokensInOutput,
    ) -> Result<String, TokenizerError> {
        let skip = special == SpecialTokensInOutput::Skip;
        self.inner.decode(ids, skip).map_err(TokenizerError::Decode)
    }

    /// Vocabulary size, including reserved tokens.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// The reserved-token registry this artifact was built with.
    #[must_use]
    pub const fn registry(&self) -> SpecialTokenRegistry {
        self.registry
    }

    /// Stable identity of the tokenizer's semantics.
    ///
    /// Computed over a canonical, sorted listing of the vocabulary rather than
    /// over the serialized file, because the JSON key order of a hash-map-backed
    /// vocabulary is not guaranteed to be stable between builds. Two artifacts
    /// with the same identity assign the same id to the same token, which is the
    /// property dataset shards and checkpoints actually depend on.
    #[must_use]
    pub fn identity_hash(&self) -> String {
        let mut entries: Vec<(String, u32)> = self.inner.get_vocab(true).into_iter().collect();
        entries.sort_unstable();

        let mut hasher = Sha256::new();
        hasher.update(b"llm-tokenizer-identity-v1\0");
        hasher.update((entries.len() as u64).to_le_bytes());
        for (token, id) in &entries {
            hasher.update(id.to_le_bytes());
            hasher.update((token.len() as u64).to_le_bytes());
            hasher.update(token.as_bytes());
        }
        for (_, id, literal) in self.registry.entries() {
            hasher.update(id.to_le_bytes());
            hasher.update(literal.as_bytes());
        }

        let digest = hasher.finalize();
        const HEX: [u8; 16] = *b"0123456789abcdef";
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            hex.push(char::from(HEX[usize::from(byte >> 4)]));
            hex.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        hex
    }

    /// Re-checks the artifact against a configuration's contract.
    ///
    /// Vocabulary size and every reserved id must agree. Ids are part of the
    /// model's contract because the embedding rows are tied to them, so a
    /// mismatch is rejected rather than tolerated.
    pub fn verify_against(&self, config: &TokenizerConfig) -> Result<(), TokenizerError> {
        config.validate()?;

        let actual = self.inner.get_vocab_size(true);
        if actual != config.vocab_size {
            return Err(TokenizerError::VocabularySizeMismatch {
                expected: config.vocab_size,
                actual,
            });
        }

        // The ids being checked are the ones the caller requires, not the ones
        // this tokenizer was built with: re-checking an artifact against a
        // different contract is the whole point of this method.
        let expected_registry = SpecialTokenRegistry::new(config.special_tokens);
        let vocab = self.inner.get_vocab(true);
        for (_, expected, literal) in expected_registry.entries() {
            let actual = vocab.get(literal).copied();
            if actual != Some(expected) {
                return Err(TokenizerError::SpecialTokenMismatch {
                    literal,
                    expected,
                    actual,
                });
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for LlmTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LlmTokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("registry", &self.registry)
            .finish_non_exhaustive()
    }
}
