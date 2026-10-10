//! The reserved special-token registry.
//!
//! The registry fixes the *literal text* of each reserved token, because those
//! literals are part of the tokenizer artifact's identity, and reads the *ids*
//! from [`TokenizerConfig`]. Ids are configurable; literals are not, so that two
//! artifacts built by this project always agree on what `<|eos|>` means.
//!
//! Chat and instruction tokens are deliberately absent: they must not be
//! introduced before the post-training format is defined.

use llm_core::SpecialTokens;

use crate::error::TokenizerError;

/// A reserved token with a fixed meaning in the training objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialTokenKind {
    /// Beginning of sequence.
    Bos,
    /// End of sequence.
    Eos,
    /// Padding.
    Pad,
    /// Unknown token.
    Unk,
}

impl SpecialTokenKind {
    /// Every reserved kind, in the order the registry reports them.
    pub const ALL: [Self; 4] = [Self::Bos, Self::Eos, Self::Pad, Self::Unk];

    /// Configuration-field name, used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bos => "bos",
            Self::Eos => "eos",
            Self::Pad => "pad",
            Self::Unk => "unk",
        }
    }

    /// Literal text of the reserved token, fixed by this project.
    #[must_use]
    pub const fn literal(self) -> &'static str {
        match self {
            Self::Bos => "<|bos|>",
            Self::Eos => "<|eos|>",
            Self::Pad => "<|pad|>",
            Self::Unk => "<|unk|>",
        }
    }
}

/// Reserved-token literals bound to configured ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenRegistry {
    ids: SpecialTokens,
}

impl SpecialTokenRegistry {
    /// Binds the configured ids to the project's fixed literals.
    #[must_use]
    pub const fn new(ids: SpecialTokens) -> Self {
        Self { ids }
    }

    /// The configured ids.
    #[must_use]
    pub const fn ids(&self) -> SpecialTokens {
        self.ids
    }

    /// Configured id of one reserved token.
    #[must_use]
    pub const fn id(&self, kind: SpecialTokenKind) -> u32 {
        match kind {
            SpecialTokenKind::Bos => self.ids.bos,
            SpecialTokenKind::Eos => self.ids.eos,
            SpecialTokenKind::Pad => self.ids.pad,
            SpecialTokenKind::Unk => self.ids.unk,
        }
    }

    /// Every reserved token as `(kind, id, literal)`.
    #[must_use]
    pub const fn entries(&self) -> [(SpecialTokenKind, u32, &'static str); 4] {
        [
            (
                SpecialTokenKind::Bos,
                self.ids.bos,
                SpecialTokenKind::Bos.literal(),
            ),
            (
                SpecialTokenKind::Eos,
                self.ids.eos,
                SpecialTokenKind::Eos.literal(),
            ),
            (
                SpecialTokenKind::Pad,
                self.ids.pad,
                SpecialTokenKind::Pad.literal(),
            ),
            (
                SpecialTokenKind::Unk,
                self.ids.unk,
                SpecialTokenKind::Unk.literal(),
            ),
        ]
    }

    /// Kinds ordered by ascending configured id.
    ///
    /// The BPE trainer places special tokens at the head of the vocabulary in
    /// the order it is given them, so supplying this ordering is what makes the
    /// resulting ids equal the configured ids. It therefore requires the ids to
    /// be exactly the contiguous range `0..len`, and rejects anything else
    /// rather than silently producing a different mapping.
    pub fn ordered_by_id(&self) -> Result<[SpecialTokenKind; 4], TokenizerError> {
        let mut entries = self.entries();
        entries.sort_unstable_by_key(|(_, id, _)| *id);

        let contiguous = entries
            .iter()
            .enumerate()
            .all(|(expected, (_, id, _))| u32::try_from(expected) == Ok(*id));

        if !contiguous {
            return Err(TokenizerError::SpecialTokenIdsNotContiguous {
                count: entries.len(),
                ids: entries.iter().map(|(_, id, _)| *id).collect(),
            });
        }

        Ok(entries.map(|(kind, _, _)| kind))
    }
}

#[cfg(test)]
mod tests {
    use llm_core::SpecialTokens;

    use super::{SpecialTokenKind, SpecialTokenRegistry};
    use crate::error::TokenizerError;

    #[test]
    fn the_default_configuration_is_contiguous_in_registry_order() {
        let registry = SpecialTokenRegistry::new(SpecialTokens::default());
        let ordered = registry.ordered_by_id().expect("defaults are contiguous");
        assert_eq!(ordered, SpecialTokenKind::ALL);
    }

    #[test]
    fn a_permutation_of_the_reserved_ids_is_accepted_and_reordered() {
        let registry = SpecialTokenRegistry::new(SpecialTokens {
            bos: 3,
            eos: 0,
            pad: 2,
            unk: 1,
        });
        let ordered = registry
            .ordered_by_id()
            .expect("a permutation is contiguous");
        assert_eq!(
            ordered,
            [
                SpecialTokenKind::Eos,
                SpecialTokenKind::Unk,
                SpecialTokenKind::Pad,
                SpecialTokenKind::Bos,
            ]
        );
    }

    #[test]
    fn non_contiguous_ids_are_rejected() {
        let registry = SpecialTokenRegistry::new(SpecialTokens {
            bos: 0,
            eos: 1,
            pad: 2,
            unk: 9,
        });
        assert!(matches!(
            registry.ordered_by_id(),
            Err(TokenizerError::SpecialTokenIdsNotContiguous { .. })
        ));
    }

    #[test]
    fn every_kind_has_a_distinct_literal_and_a_stable_name() {
        let literals: Vec<&str> = SpecialTokenKind::ALL
            .iter()
            .map(|kind| kind.literal())
            .collect();
        let names: Vec<&str> = SpecialTokenKind::ALL
            .iter()
            .map(|kind| kind.name())
            .collect();

        for literal in &literals {
            assert_eq!(literals.iter().filter(|other| *other == literal).count(), 1);
        }
        for name in &names {
            assert_eq!(names.iter().filter(|other| *other == name).count(), 1);
        }
    }
}
