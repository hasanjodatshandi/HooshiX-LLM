//! Token-shard format: writing, checksumming and validation.
//!
//! A shard is a fixed 128-byte header, a payload of little-endian `u32` token
//! ids, and a trailing SHA-256 of everything before it.
//!
//! ```text
//! offset  size  field
//!      0     8  magic, "LLMSHRD1"
//!      8     4  format version
//!     12     4  sequence length
//!     16     1  split tag
//!     17    47  reserved, zero
//!     64    64  tokenizer identity, ASCII hexadecimal
//!    128   ...  payload: token ids, little-endian u32
//!    end    32  SHA-256 of all preceding bytes
//! ```
//!
//! Counts are not in the header because a writer streams and does not know them
//! until it finishes. The format specification allows a shard to *accompany* its
//! metadata, so the counts live in a JSON sidecar written next to the shard.
//!
//! The tokenizer identity is in the header on purpose: token ids only mean
//! something relative to the tokenizer that produced them, so a shard read with
//! the wrong tokenizer fails immediately rather than training on noise.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::DataError;
use crate::split::Split;

/// First eight bytes of every shard.
pub const SHARD_MAGIC: [u8; 8] = *b"LLMSHRD1";
/// Shard format this build reads and writes.
pub const SHARD_FORMAT_VERSION: u32 = 1;
/// Size of the fixed header.
pub const SHARD_HEADER_LEN: u64 = 128;
/// Size of the trailing digest.
pub const SHARD_TRAILER_LEN: u64 = 32;
/// Offset of the tokenizer identity within the header.
const IDENTITY_OFFSET: usize = 64;
/// Length of the stored tokenizer identity.
const IDENTITY_LEN: usize = 64;

/// What a shard contains, as recorded beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardSummary {
    /// File name of the shard, relative to the dataset root.
    pub file_name: String,
    /// Split the shard belongs to.
    pub split: Split,
    /// Tokens per sequence.
    pub sequence_length: usize,
    /// Number of sequences stored.
    pub sequence_count: u64,
    /// Total tokens stored, equal to `sequence_count * sequence_length`.
    pub token_count: u64,
    /// Hexadecimal SHA-256 of the shard's bytes.
    pub checksum: String,
    /// Tokenizer identity the shard was prepared with.
    pub tokenizer_identity: String,
}

/// What a reader requires a shard to be.
#[derive(Debug, Clone, Copy)]
pub struct ShardExpectation<'a> {
    /// Tokenizer identity the caller is using.
    pub tokenizer_identity: &'a str,
    /// Sequence length the caller expects.
    pub sequence_length: usize,
}

/// Streams sequences into one shard file.
pub struct ShardWriter {
    path: PathBuf,
    file: BufWriter<File>,
    hasher: Sha256,
    split: Split,
    sequence_length: usize,
    tokenizer_identity: String,
    sequence_count: u64,
    token_count: u64,
}

impl ShardWriter {
    /// Creates a shard and writes its header.
    pub fn create(
        path: impl AsRef<Path>,
        split: Split,
        sequence_length: usize,
        tokenizer_identity: &str,
    ) -> Result<Self, DataError> {
        let path = path.as_ref().to_path_buf();
        let sequence_length_u32 =
            u32::try_from(sequence_length).map_err(|_| DataError::ShardNotAligned {
                path: path.clone(),
                tokens: sequence_length as u64,
                sequence_length,
            })?;

        let file = File::create(&path).map_err(|source| DataError::Write {
            path: path.clone(),
            source,
        })?;

        let mut writer = Self {
            path,
            file: BufWriter::new(file),
            hasher: Sha256::new(),
            split,
            sequence_length,
            tokenizer_identity: tokenizer_identity.to_owned(),
            sequence_count: 0,
            token_count: 0,
        };

        let mut header = [0_u8; SHARD_HEADER_LEN as usize];
        header[..8].copy_from_slice(&SHARD_MAGIC);
        header[8..12].copy_from_slice(&SHARD_FORMAT_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&sequence_length_u32.to_le_bytes());
        header[16] = split.tag();
        let identity = tokenizer_identity.as_bytes();
        if identity.len() != IDENTITY_LEN {
            // The identity is a SHA-256 rendered as hexadecimal. Anything else
            // would not fit the fixed header, and a truncated identity is worse
            // than no identity at all.
            return Err(DataError::TokenizerMismatch {
                path: writer.path.clone(),
                expected: format!("{IDENTITY_LEN} hexadecimal characters"),
                found: identity.len().to_string(),
            });
        }
        header[IDENTITY_OFFSET..IDENTITY_OFFSET + IDENTITY_LEN].copy_from_slice(identity);

        writer.write_hashed(&header)?;
        Ok(writer)
    }

    /// Appends one sequence of exactly the declared length.
    pub fn push_sequence(&mut self, tokens: &[u32]) -> Result<(), DataError> {
        if tokens.len() != self.sequence_length {
            return Err(DataError::ShardSequenceLength {
                path: self.path.clone(),
                expected: self.sequence_length,
                found: tokens.len(),
            });
        }

        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        self.write_hashed(&bytes)?;

        self.sequence_count += 1;
        self.token_count += tokens.len() as u64;
        Ok(())
    }

    /// Flushes the shard, appends its digest, and writes the sidecar.
    pub fn finish(mut self) -> Result<ShardSummary, DataError> {
        self.file.flush().map_err(|source| DataError::Write {
            path: self.path.clone(),
            source,
        })?;

        let checksum = hex(&self.hasher.clone().finalize());
        let digest = self.hasher.finalize();
        let mut file = self.file.into_inner().map_err(|error| DataError::Write {
            path: self.path.clone(),
            source: error.into_error(),
        })?;
        file.write_all(&digest).map_err(|source| DataError::Write {
            path: self.path.clone(),
            source,
        })?;
        file.flush().map_err(|source| DataError::Write {
            path: self.path.clone(),
            source,
        })?;

        let summary = ShardSummary {
            file_name: file_name(&self.path),
            split: self.split,
            sequence_length: self.sequence_length,
            sequence_count: self.sequence_count,
            token_count: self.token_count,
            checksum,
            tokenizer_identity: self.tokenizer_identity,
        };

        write_sidecar(&self.path, &summary)?;
        Ok(summary)
    }

    /// Writes bytes to the shard and folds them into the checksum.
    fn write_hashed(&mut self, bytes: &[u8]) -> Result<(), DataError> {
        self.hasher.update(bytes);
        self.file
            .write_all(bytes)
            .map_err(|source| DataError::Write {
                path: self.path.clone(),
                source,
            })
    }
}

/// Reads a shard and proves it is intact and usable.
///
/// Every check that can be made without the model is made here: the magic, the
/// format version, the tokenizer identity, the sequence alignment and the
/// checksum.
pub fn validate_shard(
    path: impl AsRef<Path>,
    expectation: &ShardExpectation<'_>,
) -> Result<ShardSummary, DataError> {
    let path = path.as_ref().to_path_buf();
    let mut file = File::open(&path).map_err(|source| DataError::Read {
        path: path.clone(),
        source,
    })?;

    let length = file
        .metadata()
        .map_err(|source| DataError::Read {
            path: path.clone(),
            source,
        })?
        .len();

    if length < SHARD_HEADER_LEN + SHARD_TRAILER_LEN {
        return Err(DataError::ShardTruncated {
            path,
            expected: SHARD_HEADER_LEN + SHARD_TRAILER_LEN,
            found: length,
        });
    }

    let mut header = [0_u8; SHARD_HEADER_LEN as usize];
    file.read_exact(&mut header)
        .map_err(|source| DataError::Read {
            path: path.clone(),
            source,
        })?;

    if header[..8] != SHARD_MAGIC {
        return Err(DataError::NotAShard { path });
    }

    let format_version = u32::from_le_bytes(read_u32(&header, 8));
    if format_version != SHARD_FORMAT_VERSION {
        return Err(DataError::ShardVersion {
            path,
            found: format_version,
            supported: SHARD_FORMAT_VERSION,
        });
    }

    let sequence_length = u32::from_le_bytes(read_u32(&header, 12)) as usize;
    let split_tag = header[16];
    let Some(split) = Split::from_tag(split_tag) else {
        return Err(DataError::NotAShard { path });
    };

    let identity_bytes = &header[IDENTITY_OFFSET..IDENTITY_OFFSET + IDENTITY_LEN];
    let tokenizer_identity = String::from_utf8_lossy(identity_bytes).into_owned();
    if tokenizer_identity != expectation.tokenizer_identity {
        return Err(DataError::TokenizerMismatch {
            path,
            expected: expectation.tokenizer_identity.to_owned(),
            found: tokenizer_identity,
        });
    }
    if sequence_length != expectation.sequence_length {
        return Err(DataError::ShardSequenceLength {
            path,
            expected: expectation.sequence_length,
            found: sequence_length,
        });
    }

    let payload_bytes = length - SHARD_HEADER_LEN - SHARD_TRAILER_LEN;
    if sequence_length == 0 || !payload_bytes.is_multiple_of(sequence_length as u64 * 4) {
        return Err(DataError::ShardNotAligned {
            path,
            tokens: payload_bytes / 4,
            sequence_length,
        });
    }

    // One streaming pass hashes the header and payload; the stored digest must
    // match, which is what proves the shard was not truncated or altered.
    let mut hasher = Sha256::new();
    hasher.update(header);

    let mut remaining = payload_bytes;
    let mut buffer = vec![0_u8; 64 * 1024];
    while remaining > 0 {
        let want = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        file.read_exact(&mut buffer[..want])
            .map_err(|source| DataError::Read {
                path: path.clone(),
                source,
            })?;
        hasher.update(&buffer[..want]);
        remaining -= want as u64;
    }

    let mut stored = [0_u8; SHARD_TRAILER_LEN as usize];
    file.read_exact(&mut stored)
        .map_err(|source| DataError::Read {
            path: path.clone(),
            source,
        })?;

    let computed = hasher.finalize();
    if computed[..] != stored[..] {
        return Err(DataError::ShardChecksum { path });
    }

    let token_count = payload_bytes / 4;
    Ok(ShardSummary {
        file_name: file_name(&path),
        split,
        sequence_length,
        sequence_count: token_count / sequence_length as u64,
        token_count,
        checksum: hex(&computed),
        tokenizer_identity,
    })
}

/// Reads four bytes as a little-endian integer.
fn read_u32(bytes: &[u8], offset: usize) -> [u8; 4] {
    let mut value = [0_u8; 4];
    value.copy_from_slice(&bytes[offset..offset + 4]);
    value
}

/// File name of a path, or the whole path when it has none.
fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Path of the sidecar holding a shard's counts.
fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".json");
    PathBuf::from(name)
}

/// Writes the counts and checksum beside the shard.
fn write_sidecar(path: &Path, summary: &ShardSummary) -> Result<(), DataError> {
    let sidecar = sidecar_path(path);
    let text = serde_json::to_string_pretty(summary).map_err(DataError::ManifestSerialize)?;
    std::fs::write(&sidecar, text).map_err(|source| DataError::Write {
        path: sidecar,
        source,
    })
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

/// Streams sequences out of one shard file.
///
/// # What opening checks, and what it does not
///
/// The header is checked: magic, format version, sequence length, split tag and
/// tokenizer identity. Those are cheap and they are the dangerous ones — a shard
/// read with the wrong tokenizer trains on noise, and the header is where that
/// is caught.
///
/// The trailing digest is **not** verified. That is a full pass over the payload,
/// and a training run opens a shard once and then reads it many times over many
/// epochs, so paying for it on every epoch is the wrong default.
/// [`validate_shard`] performs it as an explicit integrity step.
pub struct ShardReader {
    path: PathBuf,
    file: BufReader<File>,
    split: Split,
    sequence_length: usize,
    sequence_count: u64,
    read: u64,
}

impl ShardReader {
    /// Opens a shard, checking its header against `expectation`.
    pub fn open(
        path: impl AsRef<Path>,
        expectation: &ShardExpectation<'_>,
    ) -> Result<Self, DataError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| DataError::Read {
            path: path.clone(),
            source,
        })?;
        let length = file
            .metadata()
            .map_err(|source| DataError::Read {
                path: path.clone(),
                source,
            })?
            .len();

        if length < SHARD_HEADER_LEN + SHARD_TRAILER_LEN {
            return Err(DataError::ShardTruncated {
                path,
                expected: SHARD_HEADER_LEN + SHARD_TRAILER_LEN,
                found: length,
            });
        }

        let mut reader = BufReader::new(file);
        let mut header = [0_u8; SHARD_HEADER_LEN as usize];
        reader
            .read_exact(&mut header)
            .map_err(|source| DataError::Read {
                path: path.clone(),
                source,
            })?;

        if header[..8] != SHARD_MAGIC {
            return Err(DataError::NotAShard { path });
        }

        let format_version = u32::from_le_bytes(read_u32(&header, 8));
        if format_version != SHARD_FORMAT_VERSION {
            return Err(DataError::ShardVersion {
                path,
                found: format_version,
                supported: SHARD_FORMAT_VERSION,
            });
        }

        let sequence_length = u32::from_le_bytes(read_u32(&header, 12)) as usize;
        let Some(split) = Split::from_tag(header[16]) else {
            return Err(DataError::NotAShard { path });
        };

        let identity_bytes = &header[IDENTITY_OFFSET..IDENTITY_OFFSET + IDENTITY_LEN];
        let tokenizer_identity = String::from_utf8_lossy(identity_bytes).into_owned();
        if tokenizer_identity != expectation.tokenizer_identity {
            return Err(DataError::TokenizerMismatch {
                path,
                expected: expectation.tokenizer_identity.to_owned(),
                found: tokenizer_identity,
            });
        }
        if sequence_length != expectation.sequence_length {
            return Err(DataError::ShardSequenceLength {
                path,
                expected: expectation.sequence_length,
                found: sequence_length,
            });
        }

        // The payload is everything between the header and the trailing digest,
        // so the counts follow from the file size without reading it.
        let payload_bytes = length - SHARD_HEADER_LEN - SHARD_TRAILER_LEN;
        if sequence_length == 0 || !payload_bytes.is_multiple_of(sequence_length as u64 * 4) {
            return Err(DataError::ShardNotAligned {
                path,
                tokens: payload_bytes / 4,
                sequence_length,
            });
        }

        Ok(Self {
            path,
            file: reader,
            split,
            sequence_length,
            sequence_count: payload_bytes / (sequence_length as u64 * 4),
            read: 0,
        })
    }

    /// Split the shard holds.
    #[must_use]
    pub const fn split(&self) -> Split {
        self.split
    }

    /// Tokens per sequence.
    #[must_use]
    pub const fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Sequences the shard holds.
    #[must_use]
    pub const fn sequence_count(&self) -> u64 {
        self.sequence_count
    }

    /// Sequences handed out so far since the last rewind.
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.read
    }

    /// Whether every sequence has been handed out.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.read >= self.sequence_count
    }

    /// Reads the next sequence, or `None` at the end of the shard.
    pub fn next_sequence(&mut self) -> Result<Option<Vec<u32>>, DataError> {
        if self.is_exhausted() {
            return Ok(None);
        }

        let mut bytes = vec![0_u8; self.sequence_length * 4];
        self.file
            .read_exact(&mut bytes)
            .map_err(|source| DataError::Read {
                path: self.path.clone(),
                source,
            })?;

        let tokens = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();

        self.read += 1;
        Ok(Some(tokens))
    }

    /// Returns to the first sequence, so the shard can be read again.
    ///
    /// A training run visits its shards once per epoch, so this is the normal
    /// path rather than an optimisation.
    pub fn rewind(&mut self) -> Result<(), DataError> {
        use std::io::Seek;

        self.file
            .seek(std::io::SeekFrom::Start(SHARD_HEADER_LEN))
            .map_err(|source| DataError::Read {
                path: self.path.clone(),
                source,
            })?;
        self.read = 0;
        Ok(())
    }

    /// Collects every sequence in the shard.
    ///
    /// For tests and for small shards. A training run streams instead, because a
    /// real shard does not fit in memory beside the model.
    pub fn read_all(&mut self) -> Result<Vec<Vec<u32>>, DataError> {
        let mut sequences = Vec::with_capacity(self.sequence_count as usize);
        self.rewind()?;
        while let Some(sequence) = self.next_sequence()? {
            sequences.push(sequence);
        }
        Ok(sequences)
    }
}

#[cfg(test)]
mod reader_tests {
    use super::{ShardExpectation, ShardReader, ShardWriter};
    use crate::error::DataError;
    use crate::split::Split;
    use std::path::PathBuf;

    const IDENTITY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn scratch_dir() -> PathBuf {
        // `CARGO_TARGET_TMPDIR` is only defined for integration tests, and this
        // is a unit test, so the path is built from the manifest directory.
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("tmp")
            .join("llm-data-reader-tests");
        std::fs::create_dir_all(&directory).expect("scratch directory");
        directory
    }

    fn expectation<'a>(identity: &'a str, sequence_length: usize) -> ShardExpectation<'a> {
        ShardExpectation {
            tokenizer_identity: identity,
            sequence_length,
        }
    }

    /// Writes a shard of `sequences` and returns its path.
    fn write_shard(name: &str, sequences: &[Vec<u32>], sequence_length: usize) -> PathBuf {
        let path = scratch_dir().join(name);
        let mut writer =
            ShardWriter::create(&path, Split::Train, sequence_length, IDENTITY).expect("header");
        for sequence in sequences {
            writer.push_sequence(sequence).expect("sequence");
        }
        writer.finish().expect("completes");
        path
    }

    fn sample() -> Vec<Vec<u32>> {
        vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]]
    }

    #[test]
    fn a_shard_reads_back_exactly_what_was_written() {
        let path = write_shard("round-trip.shard", &sample(), 4);
        let mut reader =
            ShardReader::open(&path, &expectation(IDENTITY, 4)).expect("the header checks out");

        assert_eq!(reader.sequence_count(), 3);
        assert_eq!(reader.sequence_length(), 4);
        assert_eq!(reader.split(), Split::Train);
        assert_eq!(reader.read_all().expect("reads"), sample());
    }

    #[test]
    fn reading_past_the_end_reports_nothing_rather_than_failing() {
        let path = write_shard("exhausted.shard", &sample(), 4);
        let mut reader = ShardReader::open(&path, &expectation(IDENTITY, 4)).expect("opens");

        for expected in sample() {
            assert_eq!(reader.next_sequence().expect("reads"), Some(expected));
        }
        assert!(reader.is_exhausted());
        assert_eq!(reader.next_sequence().expect("reads"), None);
        // Asking twice must stay a no-op rather than becoming an error.
        assert_eq!(reader.next_sequence().expect("reads"), None);
    }

    #[test]
    fn rewinding_starts_the_shard_again() {
        let path = write_shard("rewind.shard", &sample(), 4);
        let mut reader = ShardReader::open(&path, &expectation(IDENTITY, 4)).expect("opens");

        let _ = reader.next_sequence().expect("reads");
        let _ = reader.next_sequence().expect("reads");
        assert_eq!(reader.position(), 2);

        reader.rewind().expect("seeks");
        assert_eq!(reader.position(), 0);
        assert!(!reader.is_exhausted());
        assert_eq!(reader.read_all().expect("reads"), sample());
    }

    #[test]
    fn an_empty_shard_reads_as_no_sequences() {
        let path = write_shard("empty.shard", &[], 4);
        let mut reader = ShardReader::open(&path, &expectation(IDENTITY, 4)).expect("opens");

        assert_eq!(reader.sequence_count(), 0);
        assert!(reader.is_exhausted());
        assert_eq!(reader.read_all().expect("reads"), Vec::<Vec<u32>>::new());
    }

    #[test]
    fn the_wrong_tokenizer_is_refused_at_open() {
        // The whole point of storing the identity in the header: ids only mean
        // something relative to the tokenizer that produced them.
        let path = write_shard("wrong-tokenizer.shard", &sample(), 4);
        let other = "f".repeat(64);

        assert!(matches!(
            ShardReader::open(&path, &expectation(&other, 4)),
            Err(DataError::TokenizerMismatch { .. })
        ));
    }

    #[test]
    fn the_wrong_sequence_length_is_refused_at_open() {
        let path = write_shard("wrong-length.shard", &sample(), 4);

        assert!(matches!(
            ShardReader::open(&path, &expectation(IDENTITY, 8)),
            Err(DataError::ShardSequenceLength { .. })
        ));
    }

    #[test]
    fn a_file_that_is_not_a_shard_is_refused() {
        let path = scratch_dir().join("not-a-shard.bin");
        std::fs::write(&path, vec![0_u8; 512]).expect("writes");

        assert!(matches!(
            ShardReader::open(&path, &expectation(IDENTITY, 4)),
            Err(DataError::NotAShard { .. })
        ));
    }

    #[test]
    fn a_truncated_file_is_refused() {
        let path = scratch_dir().join("short.bin");
        std::fs::write(&path, vec![0_u8; 16]).expect("writes");

        assert!(matches!(
            ShardReader::open(&path, &expectation(IDENTITY, 4)),
            Err(DataError::ShardTruncated { .. })
        ));
    }

    #[test]
    fn a_missing_file_is_reported_as_a_read_failure() {
        let path = scratch_dir().join("does-not-exist.shard");

        assert!(matches!(
            ShardReader::open(&path, &expectation(IDENTITY, 4)),
            Err(DataError::Read { .. })
        ));
    }

    #[test]
    fn a_payload_that_is_not_a_whole_number_of_sequences_is_refused() {
        // Built by hand: a valid header and trailer with a payload that cannot
        // be divided into sequences of the declared length.
        let path = scratch_dir().join("misaligned.shard");
        let mut bytes = vec![0_u8; 128];
        bytes[..8].copy_from_slice(b"LLMSHRD1");
        bytes[8..12].copy_from_slice(&1_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&4_u32.to_le_bytes());
        bytes[16] = Split::Train.tag();
        bytes[64..128].copy_from_slice(IDENTITY.as_bytes());
        bytes.extend_from_slice(&[0_u8; 6]); // one and a half sequences
        bytes.extend_from_slice(&[0_u8; 32]); // trailer
        std::fs::write(&path, bytes).expect("writes");

        assert!(matches!(
            ShardReader::open(&path, &expectation(IDENTITY, 4)),
            Err(DataError::ShardNotAligned { .. })
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{SHARD_FORMAT_VERSION, ShardExpectation, ShardWriter, validate_shard};
    use crate::error::DataError;
    use crate::split::Split;

    const IDENTITY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn scratch_dir() -> PathBuf {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("tmp")
            .join("llm-data-shard-tests");
        std::fs::create_dir_all(&directory).expect("scratch directory");
        directory
    }

    fn expectation<'a>(identity: &'a str, sequence_length: usize) -> ShardExpectation<'a> {
        ShardExpectation {
            tokenizer_identity: identity,
            sequence_length,
        }
    }

    fn write_shard(name: &str, sequences: &[Vec<u32>], sequence_length: usize) -> PathBuf {
        let path = scratch_dir().join(name);
        let mut writer = ShardWriter::create(&path, Split::Train, sequence_length, IDENTITY)
            .expect("header writes");
        for sequence in sequences {
            writer.push_sequence(sequence).expect("sequence writes");
        }
        writer.finish().expect("shard completes");
        path
    }

    #[test]
    fn a_written_shard_validates_and_reports_its_counts() {
        let sequences = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let path = write_shard("valid.shard", &sequences, 4);

        let summary = validate_shard(&path, &expectation(IDENTITY, 4)).expect("valid shard");
        assert_eq!(summary.sequence_count, 2);
        assert_eq!(summary.token_count, 8);
        assert_eq!(summary.split, Split::Train);
        assert_eq!(summary.sequence_length, 4);
        assert_eq!(summary.tokenizer_identity, IDENTITY);
        assert_eq!(summary.checksum.len(), 64);
        assert_eq!(SHARD_FORMAT_VERSION, 1);
    }

    #[test]
    fn the_sidecar_is_written_beside_the_shard() {
        let path = write_shard("sidecar.shard", &[vec![1, 2, 3, 4]], 4);
        let sidecar = PathBuf::from(format!("{}.json", path.display()));
        let text = std::fs::read_to_string(&sidecar).expect("sidecar exists");
        assert!(text.contains("\"sequence_count\": 1"));
    }

    #[test]
    fn a_shard_read_with_the_wrong_tokenizer_fails_fast() {
        let path = write_shard("identity.shard", &[vec![1, 2, 3, 4]], 4);
        let other = "f".repeat(64);

        assert!(matches!(
            validate_shard(&path, &expectation(&other, 4)),
            Err(DataError::TokenizerMismatch { .. })
        ));
    }

    #[test]
    fn a_shard_read_with_the_wrong_sequence_length_fails() {
        let path = write_shard("length.shard", &[vec![1, 2, 3, 4]], 4);
        assert!(matches!(
            validate_shard(&path, &expectation(IDENTITY, 8)),
            Err(DataError::ShardSequenceLength { .. })
        ));
    }

    #[test]
    fn a_flipped_payload_byte_is_caught_by_the_checksum() {
        let path = write_shard("flipped.shard", &[vec![1, 2, 3, 4]], 4);

        let mut bytes = std::fs::read(&path).expect("readable");
        let payload_start = super::SHARD_HEADER_LEN as usize;
        bytes[payload_start] ^= 0xff;
        std::fs::write(&path, &bytes).expect("writable");

        assert!(matches!(
            validate_shard(&path, &expectation(IDENTITY, 4)),
            Err(DataError::ShardChecksum { .. })
        ));
    }

    #[test]
    fn a_file_that_is_not_a_shard_is_rejected() {
        let path = scratch_dir().join("not-a-shard.bin");
        std::fs::write(&path, vec![0_u8; 200]).expect("writable");

        assert!(matches!(
            validate_shard(&path, &expectation(IDENTITY, 4)),
            Err(DataError::NotAShard { .. })
        ));
    }

    #[test]
    fn a_truncated_shard_is_rejected() {
        let path = scratch_dir().join("short.bin");
        std::fs::write(&path, vec![0_u8; 16]).expect("writable");

        assert!(matches!(
            validate_shard(&path, &expectation(IDENTITY, 4)),
            Err(DataError::ShardTruncated { .. })
        ));
    }

    #[test]
    fn a_sequence_of_the_wrong_length_is_refused_at_write_time() {
        let path = scratch_dir().join("bad-length.shard");
        let mut writer = ShardWriter::create(&path, Split::Train, 4, IDENTITY).expect("header");

        assert!(matches!(
            writer.push_sequence(&[1, 2, 3]),
            Err(DataError::ShardSequenceLength { .. })
        ));
    }

    #[test]
    fn an_identity_that_is_not_a_digest_is_refused() {
        let path = scratch_dir().join("bad-identity.shard");
        assert!(matches!(
            ShardWriter::create(&path, Split::Train, 4, "short"),
            Err(DataError::TokenizerMismatch { .. })
        ));
    }

    #[test]
    fn every_split_round_trips_through_the_header() {
        for split in Split::ALL {
            let path = scratch_dir().join(format!("split-{}.shard", split.as_str()));
            let mut writer = ShardWriter::create(&path, split, 2, IDENTITY).expect("header");
            writer.push_sequence(&[1, 2]).expect("sequence");
            writer.finish().expect("completes");

            let summary = validate_shard(&path, &expectation(IDENTITY, 2)).expect("valid");
            assert_eq!(summary.split, split);
        }
    }
}
