//! Tokenizer contract tests.
//!
//! These are the invariants a tokenizer artifact must hold before any training
//! data is prepared with it. They cover the tokenizer invariants that
//! `docs/06_TOKENIZER_AND_DATA.md` section 1 requires and that
//! `docs/10_TESTING_AND_VALIDATION.md` section 2 lists.

// Test helpers may abort on failure: an unreadable fixture or an unwritable
// scratch directory means the test cannot run, and the panic message names the
// cause. `clippy.toml` only relaxes the panic family for `#[test]` functions
// and `#[cfg(test)]` modules, which does not reach helpers in an integration
// test binary. The workspace-level denies stay in force for `src/`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::PathBuf;

use llm_core::{SpecialTokens, TokenizerConfig, TokenizerModel};
use llm_tokenizer::{LlmTokenizer, SpecialTokensInOutput, TokenizerError, Truncation};

/// Vocabulary the fixture corpus is rich enough to fill.
const TEST_VOCAB_SIZE: usize = 320;

/// Minimum merge frequency; one, so a small fixture still produces merges.
const TEST_MIN_FREQUENCY: u64 = 1;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("tokenizer-corpus.txt")
}

fn corpus_lines() -> Vec<String> {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture corpus must be readable");
    text.lines().map(str::to_owned).collect()
}

fn test_config() -> TokenizerConfig {
    TokenizerConfig {
        vocab_size: TEST_VOCAB_SIZE,
        model: TokenizerModel::Bpe,
        special_tokens: SpecialTokens::default(),
        artifact: None,
    }
}

fn trained() -> LlmTokenizer {
    let config = test_config();
    LlmTokenizer::train(corpus_lines().into_iter(), &config, TEST_MIN_FREQUENCY)
        .expect("the fixture corpus must fill the test vocabulary")
}

/// Strings that must survive the pipeline unchanged.
const AWKWARD_INPUTS: &[&str] = &[
    "",
    " ",
    "\n",
    "\t\t",
    "a",
    "The quick brown fox jumps over the lazy dog.",
    // Combining marks: the accent is a separate code point.
    "e\u{0301}a\u{0300}n\u{0303}u\u{0308}",
    // Zero-width joiner sequence.
    "\u{1F469}\u{200D}\u{1F4BB}",
    // Right-to-left scripts.
    "مریم از پنجره به آسمان نگاه کرد و لبخند زد.",
    "زبان فارسی از راست به چپ نوشته می‌شود.",
    "العربية تُكتب من اليمين إلى اليسار.",
    // CJK and Hangul.
    "混合文字列：中文、日本語、한국어。",
    // Cyrillic and Greek.
    "Здравствуйте, мир! Ἡ γλῶσσα ἡ ἑλληνική.",
    // Symbols and emoji.
    "🌍 🚀 ✅ ❌ → ← ↔ ≈ ≠ ≤ ≥ ± × ÷",
    // A NUL byte inside otherwise valid UTF-8.
    "before\u{0000}after",
    // Very long single token run.
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    // Mixed direction and mixed script in one line.
    "abcمریم123日本語",
];

#[test]
fn training_fills_the_configured_vocabulary_and_binds_reserved_ids() {
    let tokenizer = trained();

    assert_eq!(tokenizer.vocab_size(), TEST_VOCAB_SIZE);

    let registry = tokenizer.registry();
    for (kind, id, literal) in registry.entries() {
        assert_eq!(id, kind as u32, "reserved ids must be contiguous from zero");
        // The literal must be bound to exactly its configured id.
        let probe = vec![id];
        let decoded = tokenizer
            .decode(&probe, SpecialTokensInOutput::Keep)
            .expect("a reserved id must decode");
        assert_eq!(decoded, literal);
    }
}

#[test]
fn reserved_ids_survive_a_save_and_load_round_trip() {
    let original = trained();
    let directory = tempfile_dir();
    let path = directory.join("tokenizer.json");

    original.save(&path).expect("saving must succeed");

    let reloaded = LlmTokenizer::load(&path, &test_config()).expect("loading must succeed");

    assert_eq!(reloaded.vocab_size(), original.vocab_size());
    assert_eq!(
        reloaded.identity_hash(),
        original.identity_hash(),
        "a reloaded artifact must keep the same identity"
    );

    std::fs::remove_dir_all(&directory).ok();
}

#[test]
fn training_is_deterministic_for_a_fixed_corpus_order() {
    let first = trained();
    let second = trained();

    assert_eq!(
        first.identity_hash(),
        second.identity_hash(),
        "two runs over the same corpus in the same order must agree"
    );
}

#[test]
fn encode_and_decode_round_trip_arbitrary_valid_utf8() {
    let tokenizer = trained();

    for input in AWKWARD_INPUTS {
        let ids = tokenizer
            .encode(input)
            .unwrap_or_else(|error| panic!("encoding {input:?} must not fail: {error}"));
        let decoded = tokenizer
            .decode(&ids, SpecialTokensInOutput::Keep)
            .unwrap_or_else(|error| panic!("decoding {input:?} must not fail: {error}"));

        assert_eq!(&decoded, input, "round trip changed the input");
    }
}

#[test]
fn empty_input_is_defined_and_produces_no_tokens() {
    let tokenizer = trained();

    let ids = tokenizer.encode("").expect("empty input is valid");
    assert!(ids.is_empty(), "empty input must produce no tokens");

    let decoded = tokenizer
        .decode(&[], SpecialTokensInOutput::Skip)
        .expect("decoding nothing is valid");
    assert!(decoded.is_empty());
}

#[test]
fn decoding_skips_reserved_tokens_only_when_asked() {
    let tokenizer = trained();
    let eos = tokenizer
        .registry()
        .id(llm_tokenizer::SpecialTokenKind::Eos);

    let ids = vec![eos, eos];
    let kept = tokenizer
        .decode(&ids, SpecialTokensInOutput::Keep)
        .expect("decoding reserved ids must succeed");
    let skipped = tokenizer
        .decode(&ids, SpecialTokensInOutput::Skip)
        .expect("decoding reserved ids must succeed");

    assert_eq!(kept, "<|eos|><|eos|>");
    assert!(skipped.is_empty(), "reserved tokens must be dropped");
}

#[test]
fn truncation_is_explicit_and_never_implicit() {
    let tokenizer = trained();
    let text = "The quick brown fox jumps over the lazy dog.";
    let full = tokenizer.encode(text).expect("encoding must succeed");

    // Rejecting is the default: no silent loss of the tail.
    let rejected = tokenizer.encode_limited(text, 3, Truncation::Reject);
    assert!(matches!(
        rejected,
        Err(TokenizerError::InputTooLong { limit: 3, .. })
    ));

    // Keeping the head is opt-in and keeps exactly the prefix.
    let kept = tokenizer
        .encode_limited(text, 3, Truncation::KeepFirst)
        .expect("truncation was explicitly requested");
    assert_eq!(kept, full[..3].to_vec());

    // A limit that is not exceeded changes nothing.
    let unlimited = tokenizer
        .encode_limited(text, usize::MAX, Truncation::Reject)
        .expect("a large limit must not reject");
    assert_eq!(unlimited, full);
}

#[test]
fn a_vocabulary_size_mismatch_is_rejected() {
    let tokenizer = trained();
    let mut config = test_config();
    config.vocab_size += 1;

    assert!(matches!(
        tokenizer.verify_against(&config),
        Err(TokenizerError::VocabularySizeMismatch { .. })
    ));
}

#[test]
fn a_reserved_id_mismatch_is_rejected() {
    let tokenizer = trained();
    let mut config = test_config();
    config.special_tokens = SpecialTokens {
        bos: 0,
        eos: 1,
        pad: 2,
        unk: 7,
    };

    assert!(matches!(
        tokenizer.verify_against(&config),
        Err(TokenizerError::SpecialTokenMismatch { .. })
    ));
}

#[test]
fn loading_a_non_artifact_file_reports_the_path() {
    let directory = tempfile_dir();
    let path = directory.join("not-a-tokenizer.json");
    std::fs::write(&path, b"{\"this\": \"is not a tokenizer\"}").expect("write must succeed");

    let error =
        LlmTokenizer::load(&path, &test_config()).expect_err("a foreign document must not load");
    match error {
        TokenizerError::Load { path: reported, .. } => assert_eq!(reported, path),
        other => panic!("expected a load failure, got {other}"),
    }

    std::fs::remove_dir_all(&directory).ok();
}

#[test]
fn loading_a_missing_artifact_reports_the_path() {
    let path = tempfile_dir().join("absent.json");

    let error =
        LlmTokenizer::load(&path, &test_config()).expect_err("a missing artifact must not load");
    match error {
        TokenizerError::Read { path: reported, .. } => assert_eq!(reported, path),
        other => panic!("expected a read failure, got {other}"),
    }
}

#[test]
fn a_corpus_too_small_for_the_vocabulary_fails_instead_of_shrinking_it() {
    let config = TokenizerConfig {
        vocab_size: TEST_VOCAB_SIZE,
        ..test_config()
    };

    let error = LlmTokenizer::train(["short corpus".to_owned()].into_iter(), &config, 1)
        .expect_err("a tiny corpus cannot fill this vocabulary");
    assert!(matches!(error, TokenizerError::VocabularyNotFilled { .. }));
}

#[test]
fn non_contiguous_reserved_ids_are_rejected_before_training() {
    let config = TokenizerConfig {
        special_tokens: SpecialTokens {
            bos: 0,
            eos: 1,
            pad: 2,
            unk: 9,
        },
        ..test_config()
    };

    let error = LlmTokenizer::train(corpus_lines().into_iter(), &config, 1)
        .expect_err("ids the trainer cannot produce must be rejected up front");
    assert!(matches!(
        error,
        TokenizerError::SpecialTokenIdsNotContiguous { .. }
    ));
}

#[test]
fn identity_hash_is_a_hex_digest_and_changes_with_the_vocabulary() {
    let tokenizer = trained();
    let hash = tokenizer.identity_hash();

    assert_eq!(hash.len(), 64, "sha256 renders as 64 hex characters");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

    // A different vocabulary contract must not share an identity.
    let other_config = TokenizerConfig {
        vocab_size: TEST_VOCAB_SIZE - 20,
        ..test_config()
    };
    let other = LlmTokenizer::train(
        corpus_lines().into_iter(),
        &other_config,
        TEST_MIN_FREQUENCY,
    )
    .expect("a smaller vocabulary still fills from this corpus");
    assert_ne!(hash, other.identity_hash());
}

/// A unique scratch directory inside the Cargo target directory.
///
/// Cargo provides `CARGO_TARGET_TMPDIR` for exactly this, and it keeps test
/// scratch files inside the workspace instead of the system temporary
/// directory.
fn tempfile_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let directory =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("tokenizer-contract-{unique}"));
    std::fs::create_dir_all(&directory).expect("scratch directory must be creatable");
    directory
}
