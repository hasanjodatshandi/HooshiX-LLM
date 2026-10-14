//! Which writing system a document is in, and why the answer is not a language.
//!
//! # The failure this exists to prevent
//!
//! The Gopher and C4 quality rules are the standard heuristic filter for web
//! text, and they are English-shaped. Applied to Persian they do not merely
//! perform badly, they delete the corpus:
//!
//! - their stopword list is `the, be, to, of, and, …`, so **every Persian
//!   document scores zero stopword density** and fails the rule;
//! - their sentence-terminator rule expects `. ! ? "`, while Persian uses `؟`,
//!   `؛` and `»`, so **most Persian sentences read as unterminated**;
//! - their symbol rule counts characters Persian uses as ordinary punctuation.
//!
//! Every counter still reports success. Nothing errors. The Persian half of the
//! corpus is simply gone, and the run continues with English.
//!
//! So the rules have to be selected by script before they are applied, and that
//! is all this module does. **It answers "which ruleset", not "which language".**
//! For choosing between a Persian ruleset and an English one, counting letters
//! is the right tool and a language model would be overkill.

/// Which writing system a document is mostly written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    /// Latin letters dominate. English, and most code.
    Latin,
    /// Arabic-script letters dominate. Persian, and Arabic it is not told apart
    /// from — deliberately, because the rules that follow are about the writing
    /// system and apply to both.
    Arabic,
    /// Both are substantial.
    ///
    /// **A first-class outcome, not an error.** Persian technical writing
    /// routinely puts an English term inside a Persian sentence — `میرود`,
    /// `کامپیوتر`, a URL, a hashtag — and a filter that treated that as a defect
    /// would discard the most useful Persian in the corpus.
    Mixed,
    /// Neither dominates: digits, punctuation, symbols, or text in a third
    /// script. Such a document is not evidence of anything and the prose rules
    /// should not judge it.
    Neither,
}

/// How many characters of each kind a document holds.
///
/// Counted in characters, not bytes. A byte count would flatter Persian by a
/// factor of about two, because the Arabic script costs two bytes per character
/// in UTF-8, and every ratio computed from it would hide the thing it is meant
/// to show.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScriptProfile {
    /// Characters in the Latin blocks.
    pub latin: usize,
    /// Characters in the Arabic blocks.
    pub arabic: usize,
    /// Zero-width non-joiner, U+200C.
    ///
    /// Counted apart from both letters and other characters, and this is the
    /// point of having a field for it. ZWNJ joins letters **inside** a Persian
    /// word — `میرود` is one word, not two — so it is not a separator and must
    /// never be treated as one. It is also not a letter, so counting it as one
    /// would inflate the script share of short documents. It is a third thing
    /// and it gets its own counter so that neither mistake is available.
    pub joiners: usize,
    /// Everything else: spaces, digits, punctuation, and any other script.
    pub other: usize,
}

impl ScriptProfile {
    /// Measures a document's characters.
    #[must_use]
    pub fn measure(text: &str) -> Self {
        let mut profile = Self::default();
        for character in text.chars() {
            match classify_character(character) {
                CharacterClass::Latin => profile.latin += 1,
                CharacterClass::Arabic => profile.arabic += 1,
                CharacterClass::Joiner => profile.joiners += 1,
                CharacterClass::Other => profile.other += 1,
            }
        }
        profile
    }

    /// Total letters counted, excluding joiners and everything else.
    #[must_use]
    pub const fn letters(&self) -> usize {
        self.latin + self.arabic
    }

    /// Decides the script.
    ///
    /// `mixed_share` is the share of letters the minority script must reach
    /// before the document counts as mixed rather than as the majority script.
    /// A document with a single English word in a Persian sentence stays
    /// `Arabic`; a document that is a third English is `Mixed` and is judged by
    /// the rules that are true of both.
    #[must_use]
    pub fn classify(&self, mixed_share: f64) -> Script {
        let letters = self.letters();
        if letters == 0 {
            return Script::Neither;
        }

        let latin_share = self.latin as f64 / letters as f64;

        // A document must be mostly letters to be judged as prose at all. A page
        // of digits and punctuation is neither language and the prose rules would
        // be measuring noise.
        if letters * 2 < self.other {
            return Script::Neither;
        }

        if latin_share >= 1.0 - mixed_share {
            Script::Latin
        } else if latin_share <= mixed_share {
            Script::Arabic
        } else {
            Script::Mixed
        }
    }

    /// Whether the document carries both scripts substantially.
    #[must_use]
    pub fn is_mixed(&self, mixed_share: f64) -> bool {
        self.classify(mixed_share) == Script::Mixed
    }
}

/// One character's class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharacterClass {
    Latin,
    Arabic,
    Joiner,
    Other,
}

/// Classifies one character by the Unicode block it falls in.
///
/// Written out rather than taken from a crate because the set is small, fixed,
/// and worth being able to read: these four ranges are the whole of it, and a
/// reviewer can check them against the Unicode chart in a minute.
fn classify_character(character: char) -> CharacterClass {
    let code = character as u32;
    match code {
        // Basic Latin letters, Latin-1 Supplement letters, Latin Extended-A and
        // -B. Enough for English, French, German, Turkish and the rest of the
        // Latin-script languages this corpus is expected to carry.
        0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F => CharacterClass::Latin,

        // Arabic, Arabic Supplement, Arabic Extended-A.
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => CharacterClass::Arabic,
        // Arabic Presentation Forms-A and -B, which web text does contain.
        0xFB50..=0xFDFF | 0xFE70..=0xFEFF => CharacterClass::Arabic,

        // ZWNJ. See the field comment on `ScriptProfile::joiners`.
        0x200C => CharacterClass::Joiner,

        _ => CharacterClass::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::{Script, ScriptProfile};

    /// The share the adapters use. A tenth is generous, and generosity is the
    /// safe direction: a document wrongly called `Mixed` is judged by rules that
    /// hold for both scripts, while a document wrongly called `Arabic` is judged
    /// only by the Persian rules and may fail one of them for no good reason.
    const MIXED: f64 = 0.10;

    fn script(text: &str) -> Script {
        ScriptProfile::measure(text).classify(MIXED)
    }

    #[test]
    fn english_is_latin() {
        assert_eq!(
            script("The quick brown fox jumps over the lazy dog."),
            Script::Latin
        );
    }

    #[test]
    fn persian_is_arabic() {
        assert_eq!(
            script("روزی روزگاری در یک دهکده کوچک، مردی زندگی می‌کرد."),
            Script::Arabic
        );
    }

    #[test]
    fn a_persian_sentence_with_one_english_word_is_still_arabic() {
        // The case the whole design turns on. Persian technical writing looks
        // like this constantly, and punishing it would discard the most useful
        // Persian in the corpus.
        assert_eq!(
            script("او گفت که کامپیوتر را خاموش کرده است و به خانه رفت."),
            Script::Arabic
        );
    }

    #[test]
    fn a_genuinely_mixed_document_is_mixed() {
        let mixed = "این یک متن فارسی است and this is a long English clause in the middle of it.";
        assert_eq!(script(mixed), Script::Mixed);
    }

    #[test]
    fn zwnj_is_counted_apart_and_never_as_a_separator() {
        // میرود is one word. The joiner is not a letter and not a separator, and
        // the profile has a field for it so neither miscount is available.
        //
        // Six characters — م ی ‌ ر و د — which is five Arabic letters and one
        // joiner. The assertion that matters is the five: a normalizer that
        // treated ZWNJ as whitespace would report four and the word would be
        // counted as two.
        let profile = ScriptProfile::measure("می‌رود");
        assert_eq!(profile.joiners, 1);
        assert_eq!(
            profile.arabic, 5,
            "م ی ر و د is five letters, and the joiner is counted apart from them"
        );
        assert_eq!(profile.latin, 0);
        assert_eq!(profile.other, 0);
    }

    #[test]
    fn zwnj_does_not_change_the_script() {
        let with = script("می‌رود و می‌آید و می‌بیند و می‌شنود و می‌گوید");
        let without = script("میرود و میآید و میبیند و میشنود و میگوید");
        assert_eq!(with, Script::Arabic);
        assert_eq!(without, Script::Arabic);
    }

    #[test]
    fn digits_and_punctuation_alone_are_neither() {
        assert_eq!(
            script("12345 ... --- ??? 67890 !!! ### 42"),
            Script::Neither
        );
    }

    #[test]
    fn an_empty_document_is_neither() {
        assert_eq!(script(""), Script::Neither);
    }

    #[test]
    fn a_third_script_is_neither_rather_than_a_guess() {
        // Greek. Not Latin and not Arabic, so it is not judged by either prose
        // ruleset — which is the honest answer, since neither was written for it.
        assert_eq!(script("αβγδε ζηθικ λμνξο"), Script::Neither);
    }

    #[test]
    fn code_with_braces_and_symbols_is_latin() {
        let code = "fn main() { let x = compute(42); println!(\"{}\", x); }";
        assert_eq!(script(code), Script::Latin);
    }
}
