//! Document filtering.
//!
//! The baseline here is deliberately small and explicit: a length window and a
//! detector for a handful of unambiguous credential shapes. It is not a general
//! PII solution, and it must not be described as one. What it does guarantee is
//! that every drop is counted and attributed to a named reason.

/// Identifies the ruleset in a dataset manifest.
pub const FILTER_VERSION: &str = "length-and-credential-shapes-v1";

/// Length window and credential policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterConfig {
    /// Documents shorter than this many characters are dropped.
    pub min_chars: usize,
    /// Documents longer than this many characters are dropped.
    pub max_chars: usize,
    /// Drop documents containing a recognised credential shape.
    pub reject_credential_shapes: bool,
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
        }
    }
}

/// Decides whether a normalized document is kept.
///
/// Returns `None` when the document is kept.
#[must_use]
pub fn assess(text: &str, config: &FilterConfig) -> Option<FilterReason> {
    // Counted in characters, not bytes: a byte threshold would penalise every
    // non-Latin script.
    let characters = text.chars().count();

    if characters < config.min_chars {
        return Some(FilterReason::TooShort);
    }
    if characters > config.max_chars {
        return Some(FilterReason::TooLong);
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
    use super::{CredentialShape, FilterConfig, FilterReason, assess};

    fn config() -> FilterConfig {
        FilterConfig {
            min_chars: 8,
            max_chars: 64,
            reject_credential_shapes: true,
        }
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
}
