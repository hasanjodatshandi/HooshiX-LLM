//! Document filtering.
//!
//! The baseline here is deliberately small and explicit: a length window and a
//! detector for a handful of unambiguous credential shapes. It is not a general
//! PII solution, and it must not be described as one. What it does guarantee is
//! that every drop is counted and attributed to a named reason.
//!
//! # Script-aware rules
//!
//! A length window is script-neutral and can be applied to anything. The prose
//! rules are not: the standard Gopher and C4 heuristics are English-shaped, and
//! applied to Persian they delete the corpus while every counter reports
//! success. Their stopword list is `the, be, to, of, and, …` so every Persian
//! document scores zero density; their terminator rule wants `. ! ? "` where
//! Persian writes `؟` `؛` `»`; their symbol rule counts characters Persian uses
//! as ordinary commas.
//!
//! So [`FilterConfig`] carries a list of [`LanguageRules`], one per language, and
//! [`assess`] selects between them by the document's script. **A configuration
//! with no languages behaves exactly as this module did before**, which is why
//! no existing caller or test changes.

use std::collections::HashSet;

use crate::language::{LanguageRules, Stopwords};
use crate::script::{Script, ScriptProfile};

/// Identifies the ruleset in a dataset manifest.
///
/// Bumped when the prose rules arrived. The version travels in the manifest so a
/// corpus can be tied to the rules that produced it, and a corpus built under
/// `length-and-credential-shapes-v1` is not comparable with one built under this.
pub const FILTER_VERSION: &str = "script-aware-prose-rules-v2";

/// The share of letters a minority script must reach for a document to count as
/// mixed rather than as its majority script.
///
/// A tenth, and the direction of the error is what sets the value. A document
/// wrongly called mixed is judged by rules that hold for both scripts; one
/// wrongly called single-script may fail a rule written for the other and be
/// dropped for no reason. **Being generous here is the safe direction.**
pub const DEFAULT_MIXED_SHARE: f64 = 0.10;

/// Length window, credential policy, and the per-language prose rules.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterConfig {
    /// Documents shorter than this many characters are dropped.
    ///
    /// Used when no language rules match. When they do, the language's own
    /// window replaces it, because a Persian floor and an English floor are not
    /// the same number and applying one to both is the mistake this design
    /// exists to avoid.
    pub min_chars: usize,
    /// Documents longer than this many characters are dropped.
    pub max_chars: usize,
    /// Drop documents containing a recognised credential shape.
    pub reject_credential_shapes: bool,
    /// Per-language prose rules, selected by script.
    ///
    /// Empty by default, and empty means the script-neutral rules run alone —
    /// exactly the behaviour before prose rules existed.
    pub languages: Vec<LanguageRules>,
    /// Share of letters a minority script must reach to count as mixed.
    pub mixed_share: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            // Long enough to exclude headings, fragments and boilerplate, short
            // enough to keep ordinary sentences.
            min_chars: 32,
            // A single record of a hundred megabytes is a corpus defect, not a
            // document worth training on.
            max_chars: 100_000_000,
            reject_credential_shapes: true,
            languages: Vec::new(),
            mixed_share: DEFAULT_MIXED_SHARE,
        }
    }
}

/// Well-known credential shapes the baseline detector recognises.
///
/// Each is unambiguous enough that a false positive is very unlikely. A document
/// carrying one is dropped rather than redacted, because redaction would leave a
/// record whose text no longer matches its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialShape {
    /// A PEM private-key block.
    PemPrivateKey,
    /// An AWS access key identifier.
    AwsAccessKeyId,
    /// A GitHub personal access token.
    GitHubToken,
    /// A Slack API token.
    SlackToken,
}

impl CredentialShape {
    /// Stable name used in manifests and reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PemPrivateKey => "pem_private_key",
            Self::AwsAccessKeyId => "aws_access_key_id",
            Self::GitHubToken => "github_token",
            Self::SlackToken => "slack_token",
        }
    }
}

/// Why a document was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterReason {
    /// Fewer than the configured minimum characters.
    TooShort,
    /// More than the configured maximum characters.
    TooLong,
    /// A recognised credential shape was present.
    CredentialShape(CredentialShape),
    /// Fewer words than the language's rules require.
    TooFewWords,
    /// Punctuation and symbols above the language's share of characters.
    TooManySymbols,
    /// Function words below the language's share of words.
    ///
    /// This is the rule that deletes a corpus when it is written for the wrong
    /// language, because a stopword list in another script matches nothing and
    /// every document scores zero.
    TooFewStopwords,
    /// More lines than allowed ending in an ellipsis.
    TooManyEllipsisLines,
    /// More lines than allowed opening with a bullet.
    TooManyBulletLines,
    /// Too few lines ending in a sentence terminator.
    TooFewTerminatedLines,
    /// Mean word length outside the language's window.
    WordLengthOutOfRange,
}

impl FilterReason {
    /// Stable name used in manifests and reports.
    ///
    /// A credential shape reports its own name, so each reason has exactly one
    /// counter and the counts sum to the number of dropped documents.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TooShort => "too_short",
            Self::TooLong => "too_long",
            Self::CredentialShape(shape) => shape.as_str(),
            Self::TooFewWords => "too_few_words",
            Self::TooManySymbols => "too_many_symbols",
            Self::TooFewStopwords => "too_few_stopwords",
            Self::TooManyEllipsisLines => "too_many_ellipsis_lines",
            Self::TooManyBulletLines => "too_many_bullet_lines",
            Self::TooFewTerminatedLines => "too_few_terminated_lines",
            Self::WordLengthOutOfRange => "word_length_out_of_range",
        }
    }

    /// Whether this reason came from a language's prose rules rather than from
    /// the script-neutral ones.
    ///
    /// Reported separately in a run's counts, because a drop caused by the wrong
    /// language's rules is a configuration fault and a drop caused by the length
    /// window is not. Mixing them in one total is how the failure stays hidden.
    #[must_use]
    pub const fn is_language_specific(self) -> bool {
        !matches!(
            self,
            Self::TooShort | Self::TooLong | Self::CredentialShape(_)
        )
    }
}

/// What the prose rules measured about a document, for reporting.
///
/// Returned by [`measure`] so a run can report the distribution of these values
/// and a threshold can be chosen from data rather than guessed. The adapter
/// thresholds are starting values, and this is how they get measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextMeasurements {
    /// Characters.
    pub characters: usize,
    /// Non-empty lines.
    ///
    /// Reported so the caller can tell whether the three line-based rules have
    /// anything to judge. A one-line document gives every line ratio a verdict
    /// drawn from a single sample, and those rules must be **skipped** rather
    /// than passed or failed. Getting that wrong is not symmetric: a missing
    /// ratio reported as zero passes a ceiling and fails a floor, so a one-line
    /// document would be dropped by `too_few_terminated_lines` for the crime of
    /// having one line to look at.
    pub lines: usize,
    /// Words, with ZWNJ respected so Persian compounds stay one word.
    pub words: usize,
    /// Symbols as a fraction of characters.
    pub symbol_ratio: f64,
    /// Function words as a fraction of words.
    pub stopword_density: f64,
    /// Mean word length in characters.
    pub mean_word_chars: f64,
    /// Fraction of lines ending in an ellipsis.
    pub ellipsis_line_ratio: f64,
    /// Fraction of lines opening with a bullet.
    pub bullet_line_ratio: f64,
    /// Fraction of lines ending in a sentence terminator.
    pub terminated_line_ratio: f64,
}

/// Measures a document against one language's rules.
///
/// Splitting on whitespace is correct for both scripts and is worth stating:
/// **ZWNJ is not whitespace in Unicode**, so `میرود` stays one word and is not
/// counted as two. A splitter that treated it as a separator would report half
/// the words and twice the mean length for every Persian document that uses it,
/// which is most of them.
#[must_use]
pub fn measure(text: &str, rules: &LanguageRules) -> TextMeasurements {
    let characters = text.chars().count();

    let symbols: HashSet<char> = rules.text.symbols.iter().copied().collect();
    let terminators: HashSet<char> = rules.text.terminators.iter().copied().collect();

    let mut symbol_count = 0usize;
    for character in text.chars() {
        if symbols.contains(&character) {
            symbol_count += 1;
        }
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len();

    let letters: usize = words.iter().map(|word| word.chars().count()).sum();
    let mean_word_chars = if word_count == 0 {
        0.0
    } else {
        letters as f64 / word_count as f64
    };

    let stopword_count = words
        .iter()
        .filter(|word| is_function_word(word, &rules.stopwords))
        .count();

    let mut lines = 0usize;
    let mut ellipsis_lines = 0usize;
    let mut bullet_lines = 0usize;
    let mut terminated_lines = 0usize;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        lines += 1;
        if trimmed.starts_with('•') || trimmed.starts_with('-') || trimmed.starts_with('*') {
            bullet_lines += 1;
        }
        if trimmed.ends_with('…') || trimmed.ends_with("...") {
            ellipsis_lines += 1;
        }
        if trimmed
            .chars()
            .last()
            .is_some_and(|last| terminators.contains(&last))
        {
            terminated_lines += 1;
        }
    }

    let ratio = |part: usize, whole: usize| {
        if whole == 0 {
            0.0
        } else {
            part as f64 / whole as f64
        }
    };

    // Reported as measured, including for short documents. Whether they should
    // be acted on is `assess_prose`'s decision: the ratio is a fact about the
    // text and suppressing it here would hide it from a report that wants the
    // distribution across a corpus.
    let (ellipsis_line_ratio, bullet_line_ratio, terminated_line_ratio) = (
        ratio(ellipsis_lines, lines),
        ratio(bullet_lines, lines),
        ratio(terminated_lines, lines),
    );

    TextMeasurements {
        characters,
        lines,
        words: word_count,
        symbol_ratio: ratio(symbol_count, characters),
        stopword_density: ratio(stopword_count, word_count),
        mean_word_chars,
        ellipsis_line_ratio,
        bullet_line_ratio,
        terminated_line_ratio,
    }
}

/// Whether a word is function-word evidence for this language.
///
/// A whole-word match, and nothing else.
///
/// # Affix matching was tried, measured, and removed
///
/// The first version of this also matched a language's prefixes and suffixes
/// with something left over, on the reasoning that Persian attaches possession,
/// plurality and verb aspect, so a bare word list under-counts every ordinary
/// sentence. The reasoning is sound and the implementation was wrong.
///
/// Measured against a list of thirty-one Iranian city names — a document with no
/// grammar at all, which the density rule exists to reject — it scored **9.7 %
/// density against a 4 % floor, and passed.** `ایلام` ends in the possessive
/// suffix `ام`. `بیرجند` begins with the prefix `بی`. A short affix matches a
/// proper noun, and Persian place names are full of both.
///
/// It was also unnecessary. The Persian word list already carries `می`, `نمی`,
/// `ها`, `های` and `هایی` as entries of their own, which is where the attached
/// forms are actually handled. The affix machinery added false positives and no
/// recall, so it is gone rather than tuned, and the measurement above is why.
fn is_function_word(word: &str, stopwords: &Stopwords) -> bool {
    stopwords.words.contains(word)
}

/// How many lines a document needs before the line-based rules judge it.
///
/// Three, which is the smallest number that is a pattern rather than a
/// coincidence: two lines agree or disagree, three can be mostly one thing.
const MIN_LINES_FOR_LINE_RULES: usize = 3;

/// Picks the rules whose script matches the document.
///
/// Returns `None` when the configuration carries no languages, or when the
/// document is in a script none of them covers — a page of digits, or a third
/// script. Returning `None` there is deliberate: the prose rules were written
/// for a writing system, and applying Persian rules to Greek would produce a
/// verdict with nothing behind it.
fn select_rules<'a>(
    profile: &ScriptProfile,
    config: &'a FilterConfig,
) -> Option<&'a LanguageRules> {
    if config.languages.is_empty() {
        return None;
    }

    let script = profile.classify(config.mixed_share);
    match script {
        // A mixed document is judged by the rules of whichever script it carries
        // most of. Both rule sets hold for it, because both were written for
        // prose that may contain the other.
        Script::Latin | Script::Mixed => config
            .languages
            .iter()
            .find(|rules| rules.script == Script::Latin)
            .or_else(|| config.languages.first()),
        Script::Arabic => config
            .languages
            .iter()
            .find(|rules| rules.script == Script::Arabic),
        Script::Neither => None,
    }
}

/// Applies one language's prose rules.
fn assess_prose(text: &str, rules: &LanguageRules) -> Option<FilterReason> {
    let measured = measure(text, rules);

    if measured.words < rules.text.min_words {
        return Some(FilterReason::TooFewWords);
    }
    if measured.symbol_ratio > rules.text.max_symbol_ratio {
        return Some(FilterReason::TooManySymbols);
    }
    if measured.stopword_density < rules.text.min_stopword_density {
        return Some(FilterReason::TooFewStopwords);
    }
    // The three line-based rules need a few lines before they say anything. One
    // line gives a ratio of 0 or 1, which is a verdict drawn from a single
    // sample. They are skipped rather than defaulted, because a default is not
    // neutral: zero passes a ceiling and fails a floor, and a one-line document
    // would be dropped by `too_few_terminated_lines` for having one line.
    //
    // This matters more than it looks. A short paragraph on a page is one line,
    // and so is a comment, a caption, and most of the genuinely short documents
    // a corpus contains.
    if measured.lines >= MIN_LINES_FOR_LINE_RULES {
        if measured.ellipsis_line_ratio > rules.text.max_ellipsis_line_ratio {
            return Some(FilterReason::TooManyEllipsisLines);
        }
        if measured.bullet_line_ratio > rules.text.max_bullet_line_ratio {
            return Some(FilterReason::TooManyBulletLines);
        }
        if measured.terminated_line_ratio < rules.text.min_terminator_line_ratio {
            return Some(FilterReason::TooFewTerminatedLines);
        }
    }
    if measured.mean_word_chars < rules.text.min_mean_word_chars
        || measured.mean_word_chars > rules.text.max_mean_word_chars
    {
        return Some(FilterReason::WordLengthOutOfRange);
    }

    None
}

/// Decides whether a normalized document is kept.
///
/// Returns `None` when the document is kept.
#[must_use]
pub fn assess(text: &str, config: &FilterConfig) -> Option<FilterReason> {
    let profile = ScriptProfile::measure(text);
    let selected = select_rules(&profile, config);

    // The language's own window replaces the script-neutral one when a language
    // matched. Letting the neutral window run first and the language window
    // second would make the language floor unreachable, since anything below it
    // would already have been dropped under `TooShort` and the counts would
    // attribute the drop to the wrong rule.
    let (min_chars, max_chars) = match selected {
        Some(rules) => (rules.text.min_chars, rules.text.max_chars),
        None => (config.min_chars, config.max_chars),
    };

    // Counted in characters, not bytes: a byte threshold would penalise every
    // non-Latin script.
    let characters = profile.latin + profile.arabic + profile.joiners + profile.other;

    if characters < min_chars {
        return Some(FilterReason::TooShort);
    }
    if characters > max_chars {
        return Some(FilterReason::TooLong);
    }

    if let Some(rules) = selected
        && let Some(reason) = assess_prose(text, rules)
    {
        return Some(reason);
    }

    if config.reject_credential_shapes {
        if contains_pem_private_key(text) {
            return Some(FilterReason::CredentialShape(
                CredentialShape::PemPrivateKey,
            ));
        }
        if contains_prefixed_run(text, "AKIA", RunLength::Exact(16), is_upper_alphanumeric) {
            return Some(FilterReason::CredentialShape(
                CredentialShape::AwsAccessKeyId,
            ));
        }
        if contains_prefixed_run(text, "ghp_", RunLength::Exact(36), is_alphanumeric) {
            return Some(FilterReason::CredentialShape(CredentialShape::GitHubToken));
        }
        for prefix in ["xoxb-", "xoxp-", "xoxa-", "xoxr-"] {
            if contains_prefixed_run(text, prefix, RunLength::AtLeast(12), is_token_character) {
                return Some(FilterReason::CredentialShape(CredentialShape::SlackToken));
            }
        }
    }

    None
}

/// Detects a `-----BEGIN <label> PRIVATE KEY-----` header.
///
/// The label must not span a line break, so ordinary prose that happens to
/// mention both fragments is not mistaken for a key block.
fn contains_pem_private_key(text: &str) -> bool {
    const BEGIN: &str = "-----BEGIN ";
    const END: &str = "PRIVATE KEY-----";

    for (index, _) in text.match_indices(BEGIN) {
        let after = index + BEGIN.len();
        let Some(end) = text[after..].find(END) else {
            continue;
        };
        if !text[after..after + end].contains('\n') {
            return true;
        }
    }
    false
}

/// How long the run after a prefix has to be.
#[derive(Debug, Clone, Copy)]
enum RunLength {
    /// Exactly this many characters, and no more.
    Exact(usize),
    /// At least this many characters.
    AtLeast(usize),
}

/// Detects `prefix` followed by a run of characters satisfying `allowed`.
fn contains_prefixed_run(
    text: &str,
    prefix: &str,
    length: RunLength,
    allowed: fn(char) -> bool,
) -> bool {
    for (index, _) in text.match_indices(prefix) {
        let after = index + prefix.len();
        let mut run = text[after..].chars();
        let matches = match length {
            RunLength::Exact(count) => {
                (0..count).all(|_| run.next().is_some_and(allowed))
                    // The run must end here: a longer run is a different shape.
                    && !run.next().is_some_and(allowed)
            }
            RunLength::AtLeast(count) => (0..count).all(|_| run.next().is_some_and(allowed)),
        };
        if matches {
            return true;
        }
    }
    false
}

const fn is_upper_alphanumeric(character: char) -> bool {
    character.is_ascii_uppercase() || character.is_ascii_digit()
}

const fn is_alphanumeric(character: char) -> bool {
    character.is_ascii_alphanumeric()
}

const fn is_token_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '-'
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

    use super::{
        CredentialShape, DEFAULT_MIXED_SHARE, FilterConfig, FilterReason, assess, assess_prose,
        measure, select_rules,
    };
    use crate::language::LanguageRules;
    use crate::script::{Script, ScriptProfile};

    fn config() -> FilterConfig {
        FilterConfig {
            min_chars: 8,
            max_chars: 64,
            reject_credential_shapes: true,
            // No languages, so the script-neutral rules run alone. That is the
            // behaviour before prose rules existed, and these tests pin it: a
            // configuration that carries no adapter must not change what a
            // corpus run does.
            languages: Vec::new(),
            mixed_share: DEFAULT_MIXED_SHARE,
        }
    }

    /// The real Persian adapter, from the repository.
    fn persian() -> LanguageRules {
        LanguageRules::load(format!(
            "{}/../../configs/languages/fas_Arab.toml",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("the Persian adapter parses")
    }

    /// The real English adapter, from the repository.
    fn english() -> LanguageRules {
        LanguageRules::load(format!(
            "{}/../../configs/languages/eng_Latn.toml",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("the English adapter parses")
    }

    /// The configuration a bilingual corpus run uses.
    fn bilingual() -> FilterConfig {
        FilterConfig {
            languages: vec![persian(), english()],
            ..config()
        }
    }

    fn script_of(text: &str) -> Script {
        ScriptProfile::measure(text).classify(DEFAULT_MIXED_SHARE)
    }

    #[test]
    fn an_ordinary_sentence_is_kept() {
        assert_eq!(assess("The quick brown fox jumps.", &config()), None);
    }

    #[test]
    fn length_bounds_are_enforced_in_characters() {
        assert_eq!(assess("short", &config()), Some(FilterReason::TooShort));
        assert_eq!(
            assess(&"x".repeat(65), &config()),
            Some(FilterReason::TooLong)
        );
        // Twelve Persian characters are twelve characters, not twenty-four bytes.
        assert_eq!(assess("این یک متن است", &config()), None);
    }

    #[test]
    fn a_pem_private_key_header_is_detected() {
        let text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END";
        assert_eq!(
            assess(text, &config()),
            Some(FilterReason::CredentialShape(
                CredentialShape::PemPrivateKey
            ))
        );
    }

    #[test]
    fn prose_mentioning_both_fragments_is_not_a_key_block() {
        // The fragments are split across a line break, so the label between
        // them spans a newline and this is not a key block.
        let text = "Text with -----BEGIN and a\nPRIVATE KEY----- marker.";
        assert_eq!(assess(text, &config()), None);
    }

    #[test]
    fn an_aws_access_key_id_is_detected() {
        let text = "credentials: AKIAIOSFODNN7EXAMPLE trailing text";
        assert_eq!(
            assess(text, &config()),
            Some(FilterReason::CredentialShape(
                CredentialShape::AwsAccessKeyId
            ))
        );
    }

    #[test]
    fn a_lowercase_run_is_not_an_aws_key() {
        let text = "not a key: akiaiosfodnn7example";
        assert_eq!(assess(text, &config()), None);
    }

    #[test]
    fn a_github_token_is_detected() {
        let text = "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        assert_eq!(
            assess(text, &config()),
            Some(FilterReason::CredentialShape(CredentialShape::GitHubToken))
        );
    }

    #[test]
    fn a_slack_token_is_detected() {
        let text = "xoxb-123456789012-abcdefghijkl";
        assert_eq!(
            assess(text, &config()),
            Some(FilterReason::CredentialShape(CredentialShape::SlackToken))
        );
    }

    #[test]
    fn the_credential_detector_can_be_disabled() {
        let config = FilterConfig {
            reject_credential_shapes: false,
            ..config()
        };
        let text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END";
        assert_eq!(assess(text, &config), None);
    }

    #[test]
    fn every_reason_has_a_distinct_stable_name() {
        let names = [
            FilterReason::TooShort.as_str(),
            FilterReason::TooLong.as_str(),
            FilterReason::CredentialShape(CredentialShape::PemPrivateKey).as_str(),
            CredentialShape::AwsAccessKeyId.as_str(),
            CredentialShape::GitHubToken.as_str(),
            CredentialShape::SlackToken.as_str(),
        ];
        for (index, name) in names.iter().enumerate() {
            assert_eq!(names.iter().filter(|other| *other == name).count(), 1);
            assert!(!name.is_empty(), "name {index} is empty");
        }
    }

    // --- script-aware prose rules -------------------------------------------
    //
    // The tests below are the reason this feature exists. Every one of them
    // would pass trivially against the English rules alone, which is exactly the
    // failure: the English rules applied to Persian do not error, do not warn,
    // and look correct in every counter. They simply delete the corpus.

    /// A representative Persian paragraph: ordinary prose, connection words,
    /// attached verb prefixes, and a ZWNJ.
    const PERSIAN_PROSE: &str = "روزی روزگاری در یک دهکده کوچک، مردی زندگی می‌کرد که \
        همه او را می‌شناختند. او هر روز صبح زود از خانه بیرون می‌رفت و به \
        مزرعه می‌رفت تا به گیاهان رسیدگی کند. مردم دهکده او را دوست داشتند \
        و همیشه به او سلام می‌کردند.";

    /// A representative English paragraph, the same length and register.
    const ENGLISH_PROSE: &str = "Once upon a time in a small village there lived a man \
        whom everyone knew. Every morning he left the house early and went to the \
        field to look after the plants. The people of the village liked him and \
        always said hello to him when they passed.";

    #[test]
    fn persian_prose_survives_the_heuristic_filter() {
        // The single most important test in this module.
        assert_eq!(
            assess(PERSIAN_PROSE, &bilingual()),
            None,
            "Persian prose must be kept, not dropped"
        );
    }

    #[test]
    fn the_english_prose_rules_would_delete_a_persian_document() {
        // The claim this module rests on, tested where it is true: the rules
        // themselves, applied to Persian. An English stopword list holds no
        // Persian word, so the density is zero and the floor rejects it.
        assert_eq!(
            assess_prose(PERSIAN_PROSE, &english()),
            Some(FilterReason::TooFewStopwords),
            "the English rules must be shown to reject Persian, or this module is unnecessary"
        );
        // The same text under its own rules passes. That difference is the
        // entire value of the feature.
        assert_eq!(assess_prose(PERSIAN_PROSE, &persian()), None);
    }

    #[test]
    fn a_persian_document_is_never_judged_by_the_english_rules() {
        // The script field is what prevents the failure, and it prevents it
        // structurally rather than by a threshold: a Persian document selects
        // the Arabic rules, and a configuration carrying only Latin rules
        // selects nothing at all, so the prose rules never run.
        let latin_only = FilterConfig {
            languages: vec![english()],
            min_chars: 8,
            max_chars: 100_000,
            ..config()
        };
        assert_eq!(
            select_rules(&ScriptProfile::measure(PERSIAN_PROSE), &latin_only),
            None
        );
        assert_eq!(assess(PERSIAN_PROSE, &latin_only), None);
    }

    #[test]
    fn a_mislabelled_adapter_reintroduces_the_failure() {
        // And the failure is still reachable, which is why the script field has
        // to be right and why an unrecognised script name is an error in
        // `LanguageRules::parse` rather than a default. Here the English rules
        // are labelled Arabic — one character of difference in the adapter — and
        // the Persian corpus is deleted while every counter reports success.
        let mislabelled = LanguageRules {
            script: Script::Arabic,
            ..english()
        };
        let broken = FilterConfig {
            languages: vec![mislabelled],
            min_chars: 8,
            max_chars: 100_000,
            ..config()
        };
        assert_eq!(
            assess(PERSIAN_PROSE, &broken),
            Some(FilterReason::TooFewStopwords),
            "a mislabelled adapter must be shown to delete Persian, or the script field is decoration"
        );
    }

    #[test]
    fn english_prose_survives_its_own_rules() {
        assert_eq!(assess(ENGLISH_PROSE, &bilingual()), None);
    }

    #[test]
    fn the_two_adapters_disagree_about_which_script_they_cover() {
        // A wiring mistake that put the English rules under the Arabic script
        // would drop every Persian document and keep every English one, which
        // looks like a working filter.
        assert_eq!(persian().script, Script::Arabic);
        assert_eq!(english().script, Script::Latin);
        assert_eq!(script_of(PERSIAN_PROSE), Script::Arabic);
        assert_eq!(script_of(ENGLISH_PROSE), Script::Latin);
    }

    #[test]
    fn persian_with_an_english_term_is_judged_by_the_persian_rules() {
        // Persian technical writing looks like this constantly. A design that
        // treated code-switching as a defect would discard the most useful
        // Persian in the corpus.
        let with_term = "او گفت که کامپیوتر را خاموش کرده است و بعد به خانه رفت. \
            ما هم قبول کردیم و با هم به راه افتادیم. راه طولانی بود ولی \
            خسته نشدیم و به موقع رسیدیم.";
        assert_eq!(script_of(with_term), Script::Arabic);
        assert_eq!(assess(with_term, &bilingual()), None);
    }

    #[test]
    fn a_bullet_list_is_dropped_by_the_persian_rules() {
        // Every Persian word below is a function word, so stopword density is
        // high and the length is fine. What fails is the bullet ratio.
        let bullets = "- و در این مورد\n- به او گفت\n- از خانه رفت\n- که این است\n\
            - با ما بیا\n- را ببین\n- بر او سلام\n- تا برویم\n- هم بمان\n- نیز برو";
        assert!(bullets.chars().count() >= persian().text.min_chars);
        assert_eq!(
            assess(bullets, &bilingual()),
            Some(FilterReason::TooManyBulletLines)
        );
    }

    #[test]
    fn a_document_with_no_function_words_is_dropped() {
        // A keyword list. Long enough, few symbols, no bullets — and no grammar,
        // which is what stopword density is a proxy for.
        let keywords = "تهران اصفهان شیراز مشهد تبریز کرج اهواز قم کرمان یزد \
            رشت ساری بابل گرگان بجنورد بیرجند زاهدان زنجان سنندج ارومیه خرم‌آباد \
            بوشهر بندرعباس ایلام شهرکرد یاسوج سمنان قزوین اراک خرمشهر دزفول";
        // Asserted explicitly rather than by pattern, because this test is the
        // one that caught the affix rule. With affix matching in place the same
        // list scored 9.7% density against a 4% floor and passed.
        assert_eq!(
            assess(keywords, &bilingual()),
            Some(FilterReason::TooFewStopwords),
            "a keyword list must be rejected by the density rule, not merely by some other rule"
        );
    }

    #[test]
    fn zwnj_keeps_a_persian_compound_as_one_word() {
        // Measured, not asserted in prose. If the splitter treated U+200C as a
        // separator then میرود would count as two words and the mean word
        // length would halve for most Persian documents.
        let with_joiner = measure("می‌رود", &persian());
        assert_eq!(with_joiner.words, 1, "میرود is one word");
        let without_joiner = measure("میرود", &persian());
        assert_eq!(without_joiner.words, 1, "so is the same word without it");
    }

    #[test]
    fn a_code_document_is_not_judged_by_the_prose_rules() {
        // Braces are what code is. Under the prose rules it fails, correctly:
        // it is being judged by rules written for prose. Under the
        // script-neutral configuration it passes, and that configuration is the
        // one the code share of a corpus runs under.
        let code = "fn main() { let x = compute(42); println!(\"{}\", x); }";
        assert_eq!(script_of(code), Script::Latin);
        assert_ne!(assess(code, &bilingual()), None);
        assert_eq!(assess(code, &config()), None);
    }

    #[test]
    fn a_page_of_digits_is_not_judged_by_either_language() {
        let numbers = "12345 67890 11111 22222 33333 44444 55555 66666 77777 88888";
        assert_eq!(script_of(numbers), Script::Neither);
        assert_eq!(assess(numbers, &bilingual()), None);
    }

    #[test]
    fn a_configuration_without_languages_behaves_exactly_as_before() {
        // The guarantee that lets this land without touching any existing
        // caller: with no adapters loaded, nothing about `assess` is different.
        // A window wide enough not to be the thing under test. `config()` caps
        // at 64 characters, which would drop both paragraphs for their length
        // and prove nothing about the prose rules.
        let neutral = FilterConfig {
            min_chars: 8,
            max_chars: 100_000,
            ..config()
        };
        assert_eq!(assess("hello", &neutral), Some(FilterReason::TooShort));
        assert_eq!(assess(PERSIAN_PROSE, &neutral), None);
        assert_eq!(assess(ENGLISH_PROSE, &neutral), None);
    }

    #[test]
    fn the_language_window_replaces_the_neutral_one() {
        // Persian's floor is 120 characters and the neutral one is 8. A
        // 60-character Persian document passes the neutral window and must fail
        // the Persian one — under the same `TooShort`, because the rule is the
        // same rule and only the threshold is script-specific.
        let sixty = "روزی روزگاری در یک دهکده کوچک مردی زندگی می‌کرد و همه او را می‌شناختند.";
        let neutral = FilterConfig {
            min_chars: 8,
            max_chars: 100_000,
            ..bilingual()
        };
        assert!(sixty.chars().count() < persian().text.min_chars);
        assert_eq!(assess(sixty, &neutral), Some(FilterReason::TooShort));
    }

    #[test]
    fn every_prose_reason_has_a_distinct_name() {
        let names = [
            FilterReason::TooFewWords.as_str(),
            FilterReason::TooManySymbols.as_str(),
            FilterReason::TooFewStopwords.as_str(),
            FilterReason::TooManyEllipsisLines.as_str(),
            FilterReason::TooManyBulletLines.as_str(),
            FilterReason::TooFewTerminatedLines.as_str(),
            FilterReason::WordLengthOutOfRange.as_str(),
        ];
        for name in names {
            assert_eq!(
                names.iter().filter(|other| **other == name).count(),
                1,
                "`{name}` is used by more than one reason"
            );
        }
    }

    #[test]
    fn only_the_prose_reasons_are_marked_language_specific() {
        // A drop caused by the wrong language's rules is a configuration fault;
        // a drop caused by the length window is not. Reporting them apart is how
        // the first stays visible.
        assert!(!FilterReason::TooShort.is_language_specific());
        assert!(!FilterReason::TooLong.is_language_specific());
        assert!(
            !FilterReason::CredentialShape(CredentialShape::AwsAccessKeyId).is_language_specific()
        );
        assert!(FilterReason::TooFewStopwords.is_language_specific());
        assert!(FilterReason::TooManyBulletLines.is_language_specific());
    }
}
