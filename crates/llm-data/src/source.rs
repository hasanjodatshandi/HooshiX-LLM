//! Corpus sources: provenance and bounded-memory streaming.
//!
//! Raw corpora are untrusted input. Provenance is therefore a rejection
//! condition rather than documentation: a source whose licence or origin cannot
//! be stated is refused before a single document is read.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::DataError;
use crate::normalize::InvalidUtf8Policy;

/// Licences that are declarations of absence rather than permission.
///
/// A corpus whose licence is one of these is treated as having no licence at
/// all, because "unknown" is not a grant.
const PLACEHOLDER_LICENCES: [&str; 7] = [
    "unknown",
    "unlicensed",
    "none",
    "n/a",
    "na",
    "tbd",
    "proprietary",
];

/// How a source file is parsed into documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentType {
    /// One document per line, including empty lines.
    PlainText,
    /// One JSON object per line; the named field holds the document text.
    JsonLines {
        /// Object field that carries the document text.
        text_field: String,
    },
}

impl ContentType {
    /// Stable name used in manifests.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PlainText => "plain_text",
            Self::JsonLines { .. } => "json_lines",
        }
    }
}

/// Declared provenance of one corpus source.
///
/// This is what a human asserts before preparation. Measured facts — byte
/// counts, document counts, digests — are recorded separately in the dataset
/// manifest, because a measurement is not a declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceManifest {
    /// Stable identifier used to refer to this source.
    pub source_id: String,
    /// Human-readable name.
    pub source_name: String,
    /// Where the corpus came from: a URL, a dataset name, a publisher.
    pub origin: String,
    /// Licence the corpus is distributed under.
    pub license: String,
    /// Acquisition date, as stated by the provider.
    pub acquisition_date: String,
    /// How the file is parsed.
    pub content_type: ContentType,
    /// Language, when known.
    pub language: Option<String>,
    /// Free-form notes.
    pub notes: Option<String>,
}

impl SourceManifest {
    /// Rejects a declaration this project cannot accept.
    ///
    /// Unknown licensing and unknown provenance are rejection conditions, not
    /// documentation gaps.
    pub fn validate(&self) -> Result<(), DataError> {
        for (field, value) in [
            ("source_id", &self.source_id),
            ("source_name", &self.source_name),
            ("origin", &self.origin),
            ("license", &self.license),
            ("acquisition_date", &self.acquisition_date),
        ] {
            if value.trim().is_empty() {
                return Err(DataError::UnusableProvenance {
                    source_id: self.source_id.clone(),
                    reason: missing_reason(field),
                });
            }
        }

        let licence = self.license.trim().to_ascii_lowercase();
        if PLACEHOLDER_LICENCES.contains(&licence.as_str()) {
            return Err(DataError::UnusableProvenance {
                source_id: self.source_id.clone(),
                reason: "licence is a placeholder rather than a grant",
            });
        }

        Ok(())
    }
}

/// Maps a missing field to a stable reason string.
const fn missing_reason(field: &'static str) -> &'static str {
    match field.as_bytes() {
        b"source_id" => "source_id is empty",
        b"source_name" => "source_name is empty",
        b"origin" => "origin is empty",
        b"license" => "license is empty",
        _ => "acquisition_date is empty",
    }
}

/// One document as it arrived from a source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDocument {
    /// One-based position of the document within its source.
    pub index: u64,
    /// Raw document text, before normalization.
    pub text: String,
}

/// A declared source together with the file it lives in.
///
/// This is the input a preparation run consumes: provenance is asserted by a
/// human in a document, and the path says where the bytes are.
///
/// Unknown fields are not denied here: `serde` does not support
/// `deny_unknown_fields` together with `flatten`, and the flattened provenance
/// struct denies them itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpec {
    /// Declared provenance.
    #[serde(flatten)]
    pub manifest: SourceManifest,
    /// File holding the corpus.
    pub path: PathBuf,
}

/// What was read from one source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceReadStats {
    /// Identifier declared by the source.
    pub source_id: String,
    /// Path that was read.
    pub path: PathBuf,
    /// Size of the file in bytes.
    pub raw_bytes: u64,
    /// Hexadecimal SHA-256 of the raw file bytes.
    pub raw_digest: String,
    /// Number of documents the file yielded.
    pub documents: u64,
}

/// Streams documents from one source file.
///
/// Memory use is one line plus the hash state, so a source larger than memory is
/// read without difficulty. The raw bytes are hashed as they are read, which is
/// why [`SourceReader::finish`] must be called after iteration to obtain the
/// source digest.
pub struct SourceReader {
    path: PathBuf,
    source_id: String,
    content_type: ContentType,
    policy: InvalidUtf8Policy,
    reader: BufReader<File>,
    buffer: Vec<u8>,
    pending: VecDeque<(u64, Vec<u8>)>,
    hasher: Sha256,
    byte_offset: u64,
    line: u64,
}

impl SourceReader {
    /// Opens a source file.
    ///
    /// The declaration is validated first, so an unusable source fails before
    /// any of its bytes are read.
    pub fn open(
        path: impl AsRef<Path>,
        manifest: &SourceManifest,
        policy: InvalidUtf8Policy,
    ) -> Result<Self, DataError> {
        manifest.validate()?;

        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| DataError::SourceOpen {
            path: path.clone(),
            source,
        })?;

        Ok(Self {
            path,
            source_id: manifest.source_id.clone(),
            content_type: manifest.content_type.clone(),
            policy,
            reader: BufReader::new(file),
            buffer: Vec::with_capacity(4096),
            pending: VecDeque::new(),
            hasher: Sha256::new(),
            byte_offset: 0,
            line: 0,
        })
    }

    /// Completes the source and reports what was read.
    #[must_use]
    pub fn finish(self) -> SourceReadStats {
        let digest = self.hasher.finalize();
        SourceReadStats {
            source_id: self.source_id,
            path: self.path,
            raw_bytes: self.byte_offset,
            raw_digest: hex(&digest),
            documents: self.line,
        }
    }

    /// Reads the next raw line, decoded according to the UTF-8 policy.
    ///
    /// Lines are separated by LF, CRLF or a lone CR. Reading only up to LF would
    /// make a carriage-return-separated file a single enormous document, which
    /// is a segmentation error rather than a text error. One physical read can
    /// therefore yield several documents, which is what `pending` holds.
    fn next_line(&mut self) -> Result<Option<String>, DataError> {
        if !self.refill()? {
            return Ok(None);
        }

        let Some((offset, raw)) = self.pending.pop_front() else {
            return Ok(None);
        };

        self.line += 1;

        match std::str::from_utf8(&raw) {
            Ok(text) => Ok(Some(text.to_owned())),
            Err(error) => match self.policy {
                InvalidUtf8Policy::Reject => Err(DataError::InvalidUtf8 {
                    path: self.path.clone(),
                    offset: offset + error.valid_up_to() as u64,
                }),
                InvalidUtf8Policy::Replace => Ok(Some(String::from_utf8_lossy(&raw).into_owned())),
            },
        }
    }

    /// Reads one physical line and splits it into logical documents.
    ///
    /// Returns `false` once the file is exhausted and nothing is pending.
    fn refill(&mut self) -> Result<bool, DataError> {
        if !self.pending.is_empty() {
            return Ok(true);
        }

        let read_start = self.byte_offset;
        self.buffer.clear();
        let read = self
            .reader
            .read_until(b'\n', &mut self.buffer)
            .map_err(|source| DataError::SourceRead {
                path: self.path.clone(),
                source,
            })?;

        if read == 0 {
            return Ok(false);
        }

        self.byte_offset += read as u64;
        self.hasher.update(&self.buffer);

        // A trailing LF, and a CR immediately before it, are the separator.
        let mut end = self.buffer.len();
        if end > 0 && self.buffer[end - 1] == b'\n' {
            end -= 1;
        }
        if end > 0 && self.buffer[end - 1] == b'\r' {
            end -= 1;
        }

        // Any remaining CR is a separator in its own right, so one physical line
        // can carry several documents. Each document records where it started,
        // which is what makes an invalid-UTF-8 report point at the right byte
        // rather than at the start of the read.
        let mut start = 0;
        loop {
            let stop = self.buffer[start..end]
                .iter()
                .position(|byte| *byte == b'\r')
                .map_or(end, |offset| start + offset);

            let piece = self.buffer[start..stop].to_vec();
            self.pending.push_back((read_start + start as u64, piece));

            if stop == end {
                break;
            }
            start = stop + 1;
        }

        Ok(true)
    }

    /// Extracts the text field from a JSON Lines document.
    fn extract_json_text(&self, line: &str) -> Result<String, DataError> {
        let ContentType::JsonLines { text_field } = &self.content_type else {
            return Ok(line.to_owned());
        };

        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|source| DataError::MalformedJsonLine {
                path: self.path.clone(),
                line: self.line,
                source,
            })?;

        match value.get(text_field).and_then(serde_json::Value::as_str) {
            Some(text) if !text.is_empty() => Ok(text.to_owned()),
            _ => Err(DataError::MissingTextField {
                path: self.path.clone(),
                line: self.line,
                field: text_field.clone(),
            }),
        }
    }
}

impl Iterator for SourceReader {
    type Item = Result<SourceDocument, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = match self.next_line() {
            Ok(Some(line)) => line,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
        };

        match self.extract_json_text(&line) {
            Ok(text) => Some(Ok(SourceDocument {
                index: self.line,
                text,
            })),
            Err(error) => Some(Err(error)),
        }
    }
}

/// Renders a digest as lowercase hexadecimal.
fn hex(bytes: &[u8]) -> String {
    const DIGITS: [u8; 16] = *b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{ContentType, SourceManifest, SourceReader};
    use crate::error::DataError;
    use crate::normalize::InvalidUtf8Policy;

    fn scratch_dir() -> PathBuf {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("tmp")
            .join("llm-data-source-tests");
        std::fs::create_dir_all(&directory).expect("scratch directory");
        directory
    }

    fn write_fixture(name: &str, contents: &[u8]) -> PathBuf {
        let path = scratch_dir().join(name);
        std::fs::write(&path, contents).expect("fixture write");
        path
    }

    fn manifest(content_type: ContentType) -> SourceManifest {
        SourceManifest {
            source_id: "test".to_owned(),
            source_name: "Test corpus".to_owned(),
            origin: "written by the test suite".to_owned(),
            license: "CC0-1.0".to_owned(),
            acquisition_date: "2026-10-10".to_owned(),
            content_type,
            language: None,
            notes: None,
        }
    }

    #[test]
    fn plain_text_yields_one_document_per_line() {
        let path = write_fixture("plain.txt", b"first\nsecond\n\nfourth");
        let reader = SourceReader::open(
            &path,
            &manifest(ContentType::PlainText),
            InvalidUtf8Policy::Reject,
        )
        .expect("readable");

        let documents: Vec<String> = reader.map(|item| item.expect("valid").text).collect();
        assert_eq!(documents, ["first", "second", "", "fourth"]);
    }

    #[test]
    fn crlf_and_lone_cr_do_not_leak_into_the_document() {
        let path = write_fixture("crlf.txt", b"first\r\nsecond\rthird\n");
        let reader = SourceReader::open(
            &path,
            &manifest(ContentType::PlainText),
            InvalidUtf8Policy::Reject,
        )
        .expect("readable");

        let documents: Vec<String> = reader.map(|item| item.expect("valid").text).collect();
        assert_eq!(documents, ["first", "second", "third"]);
    }

    #[test]
    fn json_lines_extracts_the_declared_field() {
        let path = write_fixture(
            "lines.jsonl",
            b"{\"text\":\"alpha\",\"id\":1}\n{\"text\":\"beta\",\"id\":2}\n",
        );
        let reader = SourceReader::open(
            &path,
            &manifest(ContentType::JsonLines {
                text_field: "text".to_owned(),
            }),
            InvalidUtf8Policy::Reject,
        )
        .expect("readable");

        let documents: Vec<String> = reader.map(|item| item.expect("valid").text).collect();
        assert_eq!(documents, ["alpha", "beta"]);
    }

    #[test]
    fn a_json_line_without_the_field_names_the_line() {
        let path = write_fixture("missing.jsonl", b"{\"body\":\"alpha\"}\n");
        let mut reader = SourceReader::open(
            &path,
            &manifest(ContentType::JsonLines {
                text_field: "text".to_owned(),
            }),
            InvalidUtf8Policy::Reject,
        )
        .expect("readable");

        match reader.next().expect("one item").expect_err("must fail") {
            DataError::MissingTextField { line, field, .. } => {
                assert_eq!(line, 1);
                assert_eq!(field, "text");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn malformed_json_is_rejected_with_its_line() {
        let path = write_fixture("bad.jsonl", b"not json\n");
        let mut reader = SourceReader::open(
            &path,
            &manifest(ContentType::JsonLines {
                text_field: "text".to_owned(),
            }),
            InvalidUtf8Policy::Reject,
        )
        .expect("readable");

        assert!(matches!(
            reader.next().expect("one item"),
            Err(DataError::MalformedJsonLine { line: 1, .. })
        ));
    }

    #[test]
    fn invalid_utf8_is_rejected_by_default_and_replaced_on_request() {
        let path = write_fixture("invalid.txt", b"ok\n\xff\xfe bad\n");

        let reader = SourceReader::open(
            &path,
            &manifest(ContentType::PlainText),
            InvalidUtf8Policy::Reject,
        )
        .expect("readable");
        let outcome: Vec<_> = reader.collect();
        assert!(matches!(
            outcome[1],
            Err(DataError::InvalidUtf8 { offset: 3, .. })
        ));

        let reader = SourceReader::open(
            &path,
            &manifest(ContentType::PlainText),
            InvalidUtf8Policy::Replace,
        )
        .expect("readable");
        let documents: Vec<String> = reader.map(|item| item.expect("valid").text).collect();
        assert_eq!(documents.len(), 2);
        assert!(documents[1].contains('\u{fffd}'));
    }

    #[test]
    fn the_source_digest_covers_the_raw_bytes() {
        let path = write_fixture("digest.txt", b"alpha\nbeta\n");
        let mut reader = SourceReader::open(
            &path,
            &manifest(ContentType::PlainText),
            InvalidUtf8Policy::Reject,
        )
        .expect("readable");
        for item in reader.by_ref() {
            item.expect("valid");
        }
        let stats = reader.finish();

        assert_eq!(stats.raw_bytes, 11);
        assert_eq!(stats.documents, 2);
        assert_eq!(stats.raw_digest.len(), 64);
    }

    #[test]
    fn provenance_without_a_licence_is_rejected_before_reading() {
        let path = write_fixture("licensed.txt", b"content\n");
        let mut declared = manifest(ContentType::PlainText);
        declared.license = "unknown".to_owned();

        assert!(matches!(
            SourceReader::open(&path, &declared, InvalidUtf8Policy::Reject),
            Err(DataError::UnusableProvenance { .. })
        ));
    }

    #[test]
    fn provenance_with_an_empty_field_is_rejected() {
        for empty in ["source_id", "source_name", "origin", "license"] {
            let mut declared = manifest(ContentType::PlainText);
            match empty {
                "source_id" => declared.source_id = "  ".to_owned(),
                "source_name" => declared.source_name = String::new(),
                "origin" => declared.origin = String::new(),
                _ => declared.license = String::new(),
            }
            assert!(
                matches!(
                    declared.validate(),
                    Err(DataError::UnusableProvenance { .. })
                ),
                "empty {empty} was accepted"
            );
        }
    }
}
