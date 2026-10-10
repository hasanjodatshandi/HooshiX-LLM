//! Byte-level BPE training.
//!
//! `tokenizers` 0.23 exposes training as `Tokenizer::train(&mut trainer, sequences)`
//! where `sequences` is an iterator of string-like items. Training streams the
//! sequences, so the corpus never has to be materialized in memory.

use tokenizers::Tokenizer;
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;

use crate::error::TokenizerError;
use crate::special_tokens::SpecialTokenRegistry;

/// Everything one training run needs beyond the corpus itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BpeTrainingParams {
    /// Total vocabulary size, including the reserved tokens.
    pub(crate) vocab_size: usize,
    /// Minimum frequency a pair must reach before it is merged.
    pub(crate) min_frequency: u64,
    /// Reserved tokens and the ids they must occupy.
    pub(crate) registry: SpecialTokenRegistry,
}

/// Trains a byte-level BPE tokenizer.
///
/// # Determinism
///
/// The BPE trainer has no random seed, so its result is a function of the
/// sequence order and these parameters alone. Determinism is therefore the
/// caller's obligation: feed sequences in a fixed order.
///
/// # Vocabulary contract
///
/// The trainer places the reserved tokens at the head of the vocabulary in the
/// order given, which is what makes their ids match the configuration. If the
/// corpus cannot fill `vocab_size`, training fails instead of yielding a
/// smaller vocabulary whose ids would not mean what the configuration says.
pub(crate) fn train_byte_level_bpe<I, S>(
    sequences: I,
    params: &BpeTrainingParams,
) -> Result<Tokenizer, TokenizerError>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
{
    let ordered = params.registry.ordered_by_id()?;
    let special_tokens = ordered
        .iter()
        .map(|kind| tokenizers::AddedToken::from(kind.literal(), true))
        .collect::<Vec<_>>();

    // A byte-level tokenizer starts from the full 256-byte alphabet. Deriving
    // the alphabet from the corpus instead would leave bytes that never occur
    // during training — a newline in a corpus read line by line, for instance —
    // with no token at all, and any input containing them would encode to
    // nothing. Reserving the whole alphabet up front is what makes every
    // possible byte, and therefore every valid UTF-8 string, representable.
    let alphabet = ByteLevelPreTokenizer::alphabet();
    let required = alphabet.len() + ordered.len();
    if params.vocab_size < required {
        return Err(TokenizerError::VocabularyTooSmall {
            configured: params.vocab_size,
            required,
        });
    }

    // `Tokenizer` is type-erased in 0.23, so training goes through
    // `TrainerWrapper`, which is the only type implementing
    // `Trainer<Model = ModelWrapper>`.
    let mut trainer = TrainerWrapper::BpeTrainer(
        BpeTrainer::builder()
            .vocab_size(params.vocab_size)
            .min_frequency(params.min_frequency)
            .show_progress(false)
            .initial_alphabet(alphabet.into_iter().collect())
            .special_tokens(special_tokens)
            .build(),
    );

    let mut tokenizer = Tokenizer::new(BPE::default());

    // `ByteLevel::default()` uses the GPT-2 convention `add_prefix_space: true`,
    // which invents a leading space on encode and strips one on decode. That
    // makes the round trip lossy for input that already starts with whitespace,
    // so it is disabled here. `use_regex` keeps the GPT-2 whitespace-splitting
    // pattern, which is what aligns merges with word boundaries; `trim_offsets`
    // stays off so an offset describes the span the token actually came from.
    let byte_level = ByteLevelPreTokenizer::new(false, false, true);
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(byte_level));

    tokenizer
        .train(&mut trainer, sequences)
        .map_err(TokenizerError::Train)?;

    let actual = tokenizer.get_vocab_size(true);
    if actual != params.vocab_size {
        return Err(TokenizerError::VocabularyNotFilled {
            expected: params.vocab_size,
            actual,
        });
    }

    Ok(tokenizer)
}
