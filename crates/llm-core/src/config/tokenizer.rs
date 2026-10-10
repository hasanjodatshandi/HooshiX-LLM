//! Tokenizer configuration and its invariants.
//!
//! Chat and instruction tokens are deliberately absent: they must not be
//! introduced before the post-training format is defined.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::error::ConfigError;

/// Tokenizer family used by the Rust `tokenizers` crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerModel {
    /// Byte-level BPE, the Stage-1 default.
    Bpe,
    /// Unigram language model.
    Unigram,
    /// WordPiece.
    WordPiece,
}

/// Reserved special-token identifiers.
///
/// These ids are part of the model's contract: once a model is trained they
/// must never change, because the embedding rows are bound to them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialTokens {
    /// Beginning of sequence.
    pub bos: u32,
    /// End of sequence.
    pub eos: u32,
    /// Padding.
    pub pad: u32,
    /// Unknown token.
    pub unk: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: 0,
            eos: 1,
            pad: 2,
            unk: 3,
        }
    }
}

impl SpecialTokens {
    /// Every reserved token paired with the name used in diagnostics.
    #[must_use]
    pub const fn named(&self) -> [(&'static str, u32); 4] {
        [
            ("bos", self.bos),
            ("eos", self.eos),
            ("pad", self.pad),
            ("unk", self.unk),
        ]
    }

    /// Highest reserved id, used to check that the vocabulary covers them.
    #[must_use]
    pub fn max_id(&self) -> u32 {
        self.named().iter().map(|(_, id)| *id).max().unwrap_or(0)
    }
}

/// Tokenizer artifact and vocabulary contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerConfig {
    /// Vocabulary size; must equal the model vocabulary.
    pub vocab_size: usize,
    /// Tokenizer family.
    pub model: TokenizerModel,
    /// Reserved special-token identifiers.
    pub special_tokens: SpecialTokens,
    /// Path to the trained tokenizer artifact.
    ///
    /// The artifact hash, not this path, is what identifies the tokenizer in a
    /// run manifest.
    pub artifact: Option<PathBuf>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_768,
            model: TokenizerModel::Bpe,
            special_tokens: SpecialTokens::default(),
            artifact: None,
        }
    }
}

impl TokenizerConfig {
    /// Rejects a vocabulary contract that cannot be honoured.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size == 0 {
            return Err(ConfigError::ZeroDimension {
                field: "tokenizer.vocab_size",
            });
        }

        for (name, id) in self.special_tokens.named() {
            if usize::try_from(id).map_or(true, |id| id >= self.vocab_size) {
                return Err(ConfigError::SpecialTokenOutOfVocabulary {
                    name,
                    id,
                    vocab_size: self.vocab_size,
                });
            }
        }

        let named = self.special_tokens.named();
        for (index, (_, id)) in named.iter().enumerate() {
            if named[index + 1..].iter().any(|(_, other)| other == id) {
                return Err(ConfigError::DuplicateSpecialTokenId { id: *id });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{SpecialTokens, TokenizerConfig, TokenizerModel};
    use crate::config::error::ConfigError;

    #[test]
    fn default_reserves_four_distinct_special_tokens() {
        let config = TokenizerConfig::default();
        config.validate().expect("default must be valid");
        assert_eq!(config.special_tokens.bos, 0);
        assert_eq!(config.special_tokens.max_id(), 3);
        assert_eq!(config.model, TokenizerModel::Bpe);
        assert!(config.artifact.is_none());
    }

    #[test]
    fn special_token_outside_the_vocabulary_is_rejected() {
        let config = TokenizerConfig {
            vocab_size: 4,
            special_tokens: SpecialTokens {
                bos: 0,
                eos: 1,
                pad: 2,
                unk: 99,
            },
            ..TokenizerConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::SpecialTokenOutOfVocabulary { name: "unk", .. })
        ));
    }

    #[test]
    fn duplicate_special_token_ids_are_rejected() {
        let config = TokenizerConfig {
            special_tokens: SpecialTokens {
                bos: 0,
                eos: 0,
                pad: 2,
                unk: 3,
            },
            ..TokenizerConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::DuplicateSpecialTokenId { id: 0 })
        ));
    }

    #[test]
    fn empty_vocabulary_is_rejected() {
        let config = TokenizerConfig {
            vocab_size: 0,
            ..TokenizerConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ZeroDimension {
                field: "tokenizer.vocab_size"
            })
        ));
    }
}
