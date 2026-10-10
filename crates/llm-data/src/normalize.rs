//! Text normalization.
//!
//! The rules are deliberately conservative, because aggressive normalization
//! destroys linguistic information. Nothing here lowercases, strips accents,
//! folds width, or removes format characters: combining marks, zero-width
//! joiners and bidirectional marks all survive untouched.

use std::borrow::Cow;

/// Identifies the normalization ruleset in a dataset manifest.
///
/// A prepared dataset records this string, so changing a rule without changing
/// the version would silently reinterpret previously prepared shards.
pub const NORMALIZATION_VERSION: &str = "line-endings-and-controls-v1";

/// How a corpus that is not valid UTF-8 is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvalidUtf8Policy {
    /// Fail the run, naming the path and the byte offset.
    ///
    /// This is the default: silently altering bytes changes the corpus, and a
    /// corpus that cannot be read as declared is a data problem worth seeing.
    #[default]
    Reject,
    /// Replace each invalid sequence with U+FFFD and continue.
    Replace,
}

/// Applies the [`NORMALIZATION_VERSION`] rules to one document.
///
/// A leading byte-order mark is removed, line endings become `\n`, and control
/// characters that cannot carry linguistic information are dropped:
///
/// - kept: `\t` and `\n`;
/// - `\r\n` and a lone `\r` become `\n`;
/// - dropped: the other C0 controls, `U+007F`, and the C1 block.
///
/// The normalized text is built in one pass, so the returned [`Cow`] is borrowed
/// exactly when no rule applied. Callers can use that to detect a no-op.
#[must_use]
pub fn normalize(input: &str) -> Cow<'_, str> {
    let mut changed = false;
    let mut out = String::with_capacity(input.len());
    let mut at_start = true;
    let mut chars = input.chars().peekable();

    while let Some(character) = chars.next() {
        if at_start {
            at_start = false;
            if character == '\u{feff}' {
                changed = true;
                continue;
            }
        }

        match character {
            '\r' => {
                // A CRLF pair is one line ending, not two.
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push('\n');
                changed = true;
            }
            '\n' | '\t' => out.push(character),
            dropped if is_droppable_control(dropped) => changed = true,
            kept => out.push(kept),
        }
    }

    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(input)
    }
}

/// Whether a character is a control code with no linguistic content.
///
/// `\t`, `\n` and `\r` are excluded because they carry layout, and `\r` is
/// handled separately by the line-ending rule.
const fn is_droppable_control(character: char) -> bool {
    matches!(
        character,
        '\u{0000}'..='\u{0008}'
            | '\u{000b}'
            | '\u{000c}'
            | '\u{000e}'..='\u{001f}'
            | '\u{007f}'..='\u{009f}'
    )
}

#[cfg(test)]
mod tests {
    use super::{InvalidUtf8Policy, normalize};

    #[test]
    fn plain_text_is_returned_borrowed() {
        let text = "The quick brown fox.";
        assert!(matches!(normalize(text), std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn line_endings_are_normalized_to_lf() {
        assert_eq!(normalize("a\r\nb"), "a\nb");
        assert_eq!(normalize("a\rb"), "a\nb");
        assert_eq!(normalize("a\r\r\nb"), "a\n\nb");
        assert_eq!(normalize("a\nb"), "a\nb");
    }

    #[test]
    fn a_leading_byte_order_mark_is_removed_but_an_inner_one_is_kept() {
        assert_eq!(normalize("\u{feff}hello"), "hello");
        assert_eq!(normalize("hello\u{feff}"), "hello\u{feff}");
    }

    #[test]
    fn structural_whitespace_survives() {
        assert_eq!(normalize("a\tb"), "a\tb");
        assert_eq!(normalize("a\nb"), "a\nb");
    }

    #[test]
    fn control_characters_are_dropped() {
        assert_eq!(normalize("a\u{0000}b"), "ab");
        assert_eq!(normalize("a\u{0007}b"), "ab");
        assert_eq!(normalize("a\u{007f}b"), "ab");
        assert_eq!(normalize("a\u{009f}b"), "ab");
    }

    #[test]
    fn linguistic_characters_are_never_altered() {
        // Combining marks, zero-width joiners and non-joiner, bidirectional
        // controls, right-to-left text and emoji must all survive verbatim.
        for text in [
            "e\u{0301}a\u{0300}n\u{0303}",
            "می\u{200c}روم",
            "abc\u{200d}def",
            "\u{200f}مریم\u{200e}",
            "🌍🚀",
            "混合文字列：中文",
        ] {
            assert_eq!(normalize(text), text, "normalization altered {text:?}");
        }
    }

    #[test]
    fn the_default_policy_rejects_invalid_utf8() {
        assert_eq!(InvalidUtf8Policy::default(), InvalidUtf8Policy::Reject);
    }
}
