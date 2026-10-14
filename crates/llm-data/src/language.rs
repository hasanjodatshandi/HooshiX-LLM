//! Per-language cleaning rules, loaded from an adapter file.
//!
//! # Why one file per language rather than one parameterised ruleset
//!
//! English and Persian are not the same ruleset with different numbers. The
//! English Gopher and C4 rules applied to Persian delete Persian: their stopword
//! list is `the, be, to, of, and, …` so every Persian document scores zero
//! density, their sentence-terminator rule expects ASCII punctuation where
//! Persian uses `؟` `؛` `»`, and their symbol rule counts characters Persian uses
//! as ordinary commas. Each adapter is therefore a separate ruleset, and
//! `configs/languages/` holds one per language, next to the corpus rather than
//! compiled into this crate so that a threshold can be changed and reviewed
//! without a rebuild.
//!
//! # Where this sits
//!
//! [`script`](crate::script) answers *which* ruleset applies. This module is the
//! ruleset. [`filter`](crate::filter) applies it and names the reason for every
//! drop.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

use crate::error::DataError;
use crate::script::Script;

/// The rules for one language, as read from its adapter file.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageRules {
    /// The language tag, as the dataset spells it. `fas_Arab` for Persian.
    pub language: String,
    /// Human-readable name, for reports.
    pub display_name: String,
    /// Which writing system these rules are for.
    pub script: Script,
    /// The prose rules.
    pub text: TextRules,
    /// Function words.
    pub stopwords: Stopwords,
    /// Which normalizations this language needs.
    pub normalization: NormalizationSwitches,
}

/// The prose rules, all of them script-specific.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRules {
    /// Characters, counted in characters and not bytes.
    pub min_chars: usize,
    /// Characters.
    pub max_chars: usize,
    /// Words, after normalization and with ZWNJ respected.
    pub min_words: usize,
    /// Punctuation and symbols as a fraction of characters.
    pub max_symbol_ratio: f64,
    /// Function words as a fraction of words.
    pub min_stopword_density: f64,
    /// Lines ending in an ellipsis, as a fraction.
    pub max_ellipsis_line_ratio: f64,
    /// Lines starting with a bullet, as a fraction.
    pub max_bullet_line_ratio: f64,
    /// Lines ending in a sentence terminator, as a fraction.
    pub min_terminator_line_ratio: f64,
    /// Mean word length in characters, lower bound.
    pub min_mean_word_chars: f64,
    /// Mean word length in characters, upper bound.
    pub max_mean_word_chars: f64,
    /// Characters that end a sentence in this script.
    pub terminators: Vec<char>,
    /// Characters counted as symbols rather than as letters or punctuation.
    pub symbols: Vec<char>,
}

/// Function words.
///
/// Whole words only. An earlier version also matched a language's prefixes and
/// suffixes, and that was measured and removed — `crate::filter` carries the
/// numbers, which come down to a list of city names passing the density rule
/// because `ایلام` ends in `ام`.
#[derive(Debug, Clone, PartialEq)]
pub struct Stopwords {
    /// Whole words.
    pub words: HashSet<String>,
}

/// Which normalizations this language needs.
///
/// Kept in the adapter rather than hard-coded because the answers differ by
/// script. In particular **ZWNJ must be preserved for Persian and is meaningless
/// for English**, and a normalizer that treated it as whitespace would split
/// every `میرود` in the corpus in two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NormalizationSwitches {
    /// Keep U+200C as part of the word rather than as a separator.
    pub preserve_zwnj: bool,
    /// Fold Arabic yeh and kaf onto Persian yeh and kaf.
    pub unify_arabic_yeh_kaf: bool,
    /// Remove Arabic diacritics.
    pub strip_diacritics: bool,
    /// Map Persian and Arabic-Indic digits to ASCII.
    pub unify_digits: bool,
}

// --- Deserialization -------------------------------------------------------
//
// The adapter file carries more than this module reads: the `[source]` section
// belongs to the converter, which applies the dataset's own columns before any
// of this runs. Unknown keys are ignored deliberately, because the same file has
// two consumers with different jobs and neither should reject the other's keys.

#[derive(Debug, Deserialize)]
struct AdapterFile {
    language: String,
    display_name: String,
    script: String,
    text: TextSection,
    stopwords: StopwordSection,
    normalization: NormalizationSection,
}

#[derive(Debug, Deserialize)]
struct TextSection {
    min_chars: usize,
    max_chars: usize,
    min_words: usize,
    max_symbol_ratio: f64,
    min_stopword_density: f64,
    max_ellipsis_line_ratio: f64,
    max_bullet_line_ratio: f64,
    min_terminator_line_ratio: f64,
    min_mean_word_chars: f64,
    max_mean_word_chars: f64,
    terminators: Vec<char>,
    symbols: Vec<char>,
}

#[derive(Debug, Deserialize)]
struct StopwordSection {
    words: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct NormalizationSection {
    preserve_zwnj: bool,
    unify_arabic_yeh_kaf: bool,
    strip_diacritics: bool,
    unify_digits: bool,
}

impl LanguageRules {
    /// Reads an adapter file.
    ///
    /// A script name that is not recognised is an error rather than a default.
    /// Silently falling back would mean a typo in an adapter turned the Persian
    /// rules into the English ones, which is the failure this whole module
    /// exists to prevent and would be invisible in every counter.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DataError> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).map_err(|source| DataError::LanguageAdapterRead {
                path: path.to_path_buf(),
                source,
            })?;
        Self::parse(&text, path)
    }

    /// Parses an adapter from a string, labelled with where it came from.
    pub fn parse(text: &str, origin: &Path) -> Result<Self, DataError> {
        let file: AdapterFile =
            toml::from_str(text).map_err(|error| DataError::LanguageAdapterInvalid {
                path: origin.to_path_buf(),
                reason: error.to_string(),
            })?;

        let script = match file.script.as_str() {
            "Latin" => Script::Latin,
            "Arabic" => Script::Arabic,
            other => {
                return Err(DataError::LanguageAdapterInvalid {
                    path: origin.to_path_buf(),
                    reason: format!(
                        "unknown script `{other}`; expected `Latin` or `Arabic`. \
                         A default here would turn one language's rules into another's \
                         without any counter noticing."
                    ),
                });
            }
        };

        Ok(Self {
            language: file.language,
            display_name: file.display_name,
            script,
            text: TextRules {
                min_chars: file.text.min_chars,
                max_chars: file.text.max_chars,
                min_words: file.text.min_words,
                max_symbol_ratio: file.text.max_symbol_ratio,
                min_stopword_density: file.text.min_stopword_density,
                max_ellipsis_line_ratio: file.text.max_ellipsis_line_ratio,
                max_bullet_line_ratio: file.text.max_bullet_line_ratio,
                min_terminator_line_ratio: file.text.min_terminator_line_ratio,
                min_mean_word_chars: file.text.min_mean_word_chars,
                max_mean_word_chars: file.text.max_mean_word_chars,
                terminators: file.text.terminators,
                symbols: file.text.symbols,
            },
            stopwords: Stopwords {
                words: file.stopwords.words.into_iter().collect(),
            },
            normalization: NormalizationSwitches {
                preserve_zwnj: file.normalization.preserve_zwnj,
                unify_arabic_yeh_kaf: file.normalization.unify_arabic_yeh_kaf,
                strip_diacritics: file.normalization.strip_diacritics,
                unify_digits: file.normalization.unify_digits,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

    use super::LanguageRules;
    use crate::script::Script;

    // `r##` rather than `r#`: the fixture contains `["#", "%"]`, and a lone `"#`
    // inside a `r#"…"#` string ends it early. Two hashes makes the terminator
    // `"##`, which the fixture does not contain.
    const ENGLISH: &str = r##"
language = "eng_Latn"
display_name = "English"
script = "Latin"

[source]
min_language_score = 0.80
max_minhash_cluster_size = 100

[text]
min_chars = 100
max_chars = 200000
min_words = 20
max_symbol_ratio = 0.10
min_stopword_density = 0.02
max_ellipsis_line_ratio = 0.30
max_bullet_line_ratio = 0.90
min_terminator_line_ratio = 0.80
min_mean_word_chars = 3.0
max_mean_word_chars = 10.0
terminators = [".", "!", "?"]
symbols = ["#", "%"]

[stopwords]
words = ["the", "and", "of"]


[normalization]
preserve_zwnj = false
unify_arabic_yeh_kaf = false
strip_diacritics = false
unify_digits = false
"##;

    /// Parses the inline fixture, which has no path of its own.
    fn rules_from(text: &str) -> Result<LanguageRules, crate::error::DataError> {
        LanguageRules::parse(text, std::path::Path::new("<inline>"))
    }

    #[test]
    fn a_real_adapter_parses() {
        let rules = rules_from(ENGLISH).expect("the adapter parses");
        assert_eq!(rules.language, "eng_Latn");
        assert_eq!(rules.script, Script::Latin);
        assert_eq!(rules.text.min_chars, 100);
        assert_eq!(rules.text.terminators, vec!['.', '!', '?']);
        assert!(rules.stopwords.words.contains("the"));
        assert!(!rules.normalization.preserve_zwnj);
    }

    #[test]
    fn the_source_section_is_ignored_here() {
        // It belongs to the converter. The same file has two consumers, and one
        // rejecting the other's keys would make the file impossible to share.
        let rules = rules_from(ENGLISH).expect("the adapter parses");
        assert_eq!(rules.language, "eng_Latn");
    }

    #[test]
    fn an_unknown_script_is_an_error_and_not_a_default() {
        // The failure this guards: a typo turning the Persian rules into the
        // English ones, which every counter would report as a healthy run.
        let broken = ENGLISH.replace(r#"script = "Latin""#, r#"script = "Arabick""#);
        let error = rules_from(&broken).expect_err("it must refuse");
        let message = error.to_string();
        assert!(
            message.contains("Arabick"),
            "the message names the bad value: {message}"
        );
    }

    #[test]
    fn a_missing_required_section_is_an_error() {
        let broken = ENGLISH.replace("[stopwords]", "[not_stopwords]");
        assert!(rules_from(&broken).is_err());
    }

    #[test]
    fn the_adapters_in_the_repository_parse() {
        // The real files, not a fixture. A schema change that breaks them should
        // fail here rather than in a corpus run hours long.
        for name in ["fas_Arab", "eng_Latn"] {
            let path = format!(
                "{}/../../configs/languages/{name}.toml",
                env!("CARGO_MANIFEST_DIR")
            );
            let rules = LanguageRules::load(&path)
                .unwrap_or_else(|error| panic!("{name} must parse: {error}"));
            assert_eq!(rules.language, name);
            assert!(
                !rules.stopwords.words.is_empty(),
                "{name} must carry stopwords; an empty list scores every document at zero density"
            );
            assert!(
                !rules.text.terminators.is_empty(),
                "{name} must carry terminators; an empty list reads every sentence as unterminated"
            );
        }
    }

    #[test]
    fn the_persian_adapter_keeps_zwnj_and_the_english_one_has_no_opinion() {
        let base = format!("{}/../../configs/languages", env!("CARGO_MANIFEST_DIR"));
        let persian = LanguageRules::load(format!("{base}/fas_Arab.toml")).expect("parses");
        let english = LanguageRules::load(format!("{base}/eng_Latn.toml")).expect("parses");

        assert_eq!(persian.script, Script::Arabic);
        assert!(
            persian.normalization.preserve_zwnj,
            "Persian must keep ZWNJ"
        );
        assert!(
            persian.normalization.unify_digits,
            "Persian writes ۱۳۹۱ as well as 1391"
        );

        assert_eq!(english.script, Script::Latin);
        assert!(!english.normalization.preserve_zwnj);
    }

    #[test]
    fn the_persian_adapter_carries_the_attached_forms_as_words() {
        let path = format!(
            "{}/../../configs/languages/fas_Arab.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let persian = LanguageRules::load(path).expect("parses");
        // Persian attaches these, and the word list carries them as entries of
        // their own. That is where the attached morphology is handled, and it is
        // why the affix machinery was unnecessary as well as harmful.
        for form in ["می", "نمی", "ها", "های"] {
            assert!(
                persian.stopwords.words.contains(form),
                "`{form}` must be in the Persian word list"
            );
        }
    }
}
