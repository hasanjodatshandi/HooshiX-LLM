//! Preparation orchestration.
//!
//! The pipeline runs in one streaming pass per source and holds only bounded
//! state: one line, one token window, one packing buffer per split, and the
//! deduplication seen-set. Sources are consumed in the declared order and a
//! document's split is decided by its content, so a whole run is reproducible.
//!
//! ```text
//! source -> decode -> normalize -> filter -> dedup -> split
//!        -> tokenize -> pack (EOS window) -> shard
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use llm_core::DataConfig;
use llm_tokenizer::{LlmTokenizer, SpecialTokenKind};

use crate::dedup::{ExactDedup, digest};
use crate::error::DataError;
use crate::filter::{FilterConfig, assess};
use crate::manifest::{
    CountedReason, DatasetCounts, DatasetManifest, SourceSummary, SplitPolicySummary, SplitSummary,
};
use crate::normalize::{InvalidUtf8Policy, normalize};
use crate::pack::{SequencePacker, TailPolicy};
use crate::shard::{ShardSummary, ShardWriter};
use crate::source::{SourceReader, SourceSpec};
use crate::split::{Split, SplitPolicy};

/// Everything one preparation run needs.
#[derive(Debug, Clone)]
pub struct PrepareRequest {
    /// Directory the shards and the manifest are written into.
    pub output_root: PathBuf,
    /// Sources, in the order they must be consumed.
    pub sources: Vec<SourceSpec>,
    /// Sequence length and split policy.
    pub data: DataConfig,
    /// Document filtering policy.
    pub filters: FilterConfig,
    /// How a source that is not valid UTF-8 is handled.
    pub invalid_utf8: InvalidUtf8Policy,
    /// What to do with an incomplete final window.
    pub tail: TailPolicy,
}

/// Runs the pipeline and returns what it produced.
///
/// The manifest is written into `output_root` as well as returned, so a prepared
/// dataset is self-describing on disk.
pub fn prepare(
    request: &PrepareRequest,
    tokenizer: &LlmTokenizer,
) -> Result<DatasetManifest, DataError> {
    request.data.validate()?;
    validate_sources(&request.sources)?;

    std::fs::create_dir_all(&request.output_root).map_err(|source| DataError::DirectoryCreate {
        path: request.output_root.clone(),
        source,
    })?;

    let separator = tokenizer.registry().id(SpecialTokenKind::Eos);
    let tokenizer_identity = tokenizer.identity_hash();
    let split_policy = SplitPolicy::from_config(&request.data);

    let mut packers = create_packers(request.data.sequence_length, separator, request.tail);
    let mut writers = create_writers(request, &tokenizer_identity)?;

    let mut dedup = ExactDedup::new();
    let mut source_summaries = Vec::with_capacity(request.sources.len());
    let mut totals = DatasetCounts {
        documents_read: 0,
        documents_kept: 0,
        documents_dropped: 0,
        documents_duplicate: 0,
        tokens: 0,
    };
    let mut split_documents = [0_u64; 3];

    for spec in &request.sources {
        let mut reader = SourceReader::open(&spec.path, &spec.manifest, request.invalid_utf8)?;
        let mut kept = 0_u64;
        let mut duplicate = 0_u64;
        let mut dropped: BTreeMap<String, u64> = BTreeMap::new();

        for document in reader.by_ref() {
            let document = document?;
            totals.documents_read += 1;

            let normalized = normalize(&document.text);

            if let Some(reason) = assess(&normalized, &request.filters) {
                totals.documents_dropped += 1;
                *dropped.entry(reason.as_str().to_owned()).or_insert(0) += 1;
                continue;
            }

            let fingerprint = digest(&normalized);
            if dedup.observe(fingerprint) {
                totals.documents_duplicate += 1;
                duplicate += 1;
                continue;
            }

            let split = split_policy.assign(&fingerprint);
            let tokens = tokenizer.encode(&normalized)?;

            let slot = split_slot(split);
            packers[slot].push_document(&tokens, &mut |sequence| {
                writers[slot].push_sequence(sequence)
            })?;

            kept += 1;
            split_documents[slot] += 1;
            totals.documents_kept += 1;
        }

        let stats = reader.finish();
        source_summaries.push(SourceSummary {
            declared: spec.manifest.clone(),
            raw_bytes: stats.raw_bytes,
            raw_digest: stats.raw_digest,
            documents: stats.documents,
            documents_kept: kept,
            documents_dropped: counted(dropped),
            documents_duplicate: duplicate,
        });
    }

    // Flush each packer's final window before its shard is closed.
    for (slot, packer) in packers.iter_mut().enumerate() {
        packer.finish(&mut |sequence| writers[slot].push_sequence(sequence))?;
    }

    // Closing a writer consumes it, so the vector is drained by value.
    let mut splits = Vec::with_capacity(Split::ALL.len());
    let mut shards: Vec<ShardSummary> = Vec::with_capacity(Split::ALL.len());
    for (slot, writer) in writers.into_iter().enumerate() {
        let summary = writer.finish()?;

        splits.push(SplitSummary {
            split: Split::ALL[slot],
            documents: split_documents[slot],
            sequences: summary.sequence_count,
            tokens: summary.token_count,
            dropped_tail_tokens: packers[slot].stats().dropped_tail_tokens,
        });
        totals.tokens += summary.token_count;
        shards.push(summary);
    }

    let mut manifest = DatasetManifest::new(
        created_unix_seconds(),
        tokenizer_identity,
        request.data.sequence_length,
        SplitPolicySummary {
            seed: split_policy.seed,
            validation_basis_points: split_policy.validation_basis_points,
            test_basis_points: split_policy.test_basis_points,
        },
    );
    manifest.sources = source_summaries;
    manifest.counts = totals;
    manifest.splits = splits;
    manifest.shards = shards;

    manifest.write(request.output_root.join(DatasetManifest::file_name()))?;
    Ok(manifest)
}

/// Checks source declarations before any file is opened.
fn validate_sources(sources: &[SourceSpec]) -> Result<(), DataError> {
    if sources.is_empty() {
        return Err(DataError::UnusableProvenance {
            source_id: String::new(),
            reason: "at least one source must be declared",
        });
    }
    for spec in sources {
        spec.manifest.validate()?;
    }
    Ok(())
}

/// Creates one shard writer per split.
fn create_writers(
    request: &PrepareRequest,
    tokenizer_identity: &str,
) -> Result<Vec<ShardWriter>, DataError> {
    Split::ALL
        .iter()
        .map(|split| {
            ShardWriter::create(
                shard_path(&request.output_root, *split),
                *split,
                request.data.sequence_length,
                tokenizer_identity,
            )
        })
        .collect()
}

/// Creates one packer per split.
fn create_packers(sequence_length: usize, separator: u32, tail: TailPolicy) -> Vec<SequencePacker> {
    Split::ALL
        .iter()
        .map(|_| SequencePacker::new(sequence_length, separator, tail))
        .collect()
}

/// Index of a split in the per-split arrays.
const fn split_slot(split: Split) -> usize {
    split.tag() as usize
}

/// Path of a split's shard inside a dataset root.
fn shard_path(root: &Path, split: Split) -> PathBuf {
    root.join(format!("{}.shard", split.as_str()))
}

/// Converts a counted map into sorted entries.
///
/// A `BTreeMap` keeps the order deterministic, so two identical preparations
/// produce byte-identical manifests.
fn counted(map: BTreeMap<String, u64>) -> Vec<CountedReason> {
    map.into_iter()
        .map(|(reason, count)| CountedReason { reason, count })
        .collect()
}

/// Current time in seconds since the Unix epoch.
///
/// A clock failure is reported as zero rather than aborting preparation: the
/// timestamp is metadata, and `content_hash` excludes it precisely so that it
/// cannot affect reproducibility.
fn created_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Reasons a split's shard received no sequences.
///
/// Used by callers that want to explain an empty split rather than report a
/// silent zero.
#[must_use]
pub fn empty_split_reason(summary: &SplitSummary, dropped: &[CountedReason]) -> Option<String> {
    if summary.sequences > 0 {
        return None;
    }
    let dropped_total: u64 = dropped.iter().map(|entry| entry.count).sum();
    Some(if dropped_total == 0 {
        "the split received no documents".to_owned()
    } else {
        format!("the split received no documents; {dropped_total} were filtered or duplicated")
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{PrepareRequest, prepare};
    use crate::filter::FilterConfig;
    use crate::normalize::InvalidUtf8Policy;
    use crate::pack::TailPolicy;
    use crate::shard::{ShardExpectation, ShardWriter, validate_shard};
    use crate::source::{ContentType, SourceManifest, SourceSpec};
    use crate::split::Split;
    use llm_core::{DataConfig, SpecialTokens, TokenizerConfig, TokenizerModel};
    use llm_tokenizer::LlmTokenizer;

    const VOCAB: usize = 320;

    fn scratch(name: &str) -> PathBuf {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("tmp")
            .join("llm-data-pipeline-tests")
            .join(name);
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("scratch directory");
        directory
    }

    fn fixture_corpus() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("tokenizer-corpus.txt");
        std::fs::read_to_string(path).expect("fixture corpus")
    }

    fn tokenizer() -> LlmTokenizer {
        let config = TokenizerConfig {
            vocab_size: VOCAB,
            model: TokenizerModel::Bpe,
            special_tokens: SpecialTokens::default(),
            artifact: None,
        };
        let lines: Vec<String> = fixture_corpus().lines().map(str::to_owned).collect();
        LlmTokenizer::train(lines.into_iter(), &config, 1).expect("tokenizer trains")
    }

    fn request(
        root: &Path,
        corpus: &str,
        sequence_length: usize,
        validation_basis_points: u16,
        test_basis_points: u16,
    ) -> PrepareRequest {
        let path = root.join("corpus.txt");
        std::fs::write(&path, corpus).expect("corpus writes");

        PrepareRequest {
            output_root: root.to_path_buf(),
            sources: vec![SourceSpec {
                manifest: SourceManifest {
                    source_id: "fixture".to_owned(),
                    source_name: "Fixture corpus".to_owned(),
                    origin: "written by the test suite".to_owned(),
                    license: "CC0-1.0".to_owned(),
                    acquisition_date: "2026-10-10".to_owned(),
                    content_type: ContentType::PlainText,
                    language: Some("en".to_owned()),
                    notes: None,
                },
                path,
            }],
            data: DataConfig {
                sequence_length,
                validation_basis_points,
                test_basis_points,
                ..DataConfig::default()
            },
            filters: FilterConfig {
                min_chars: 32,
                ..FilterConfig::default()
            },
            invalid_utf8: InvalidUtf8Policy::Reject,
            tail: TailPolicy::Drop,
        }
    }

    #[test]
    fn preparation_is_reproducible_for_the_same_input_and_configuration() {
        let corpus = fixture_corpus();
        let tokenizer = tokenizer();

        let first_dir = scratch("reproducible-a");
        let second_dir = scratch("reproducible-b");
        let first = prepare(&request(&first_dir, &corpus, 32, 50, 50), &tokenizer)
            .expect("preparation succeeds");
        let second = prepare(&request(&second_dir, &corpus, 32, 50, 50), &tokenizer)
            .expect("preparation succeeds");

        assert_eq!(first.counts, second.counts);
        assert_eq!(first.splits, second.splits);
        assert_eq!(first.shards, second.shards);
        assert_eq!(
            first.content_hash().expect("hashes"),
            second.content_hash().expect("hashes")
        );

        // The shard bytes themselves must agree, not just their checksums.
        for split in Split::ALL {
            let name = format!("{}.shard", split.as_str());
            let left = std::fs::read(first_dir.join(&name)).expect("first shard");
            let right = std::fs::read(second_dir.join(&name)).expect("second shard");
            assert_eq!(left, right, "{name} differs between runs");
        }
    }

    #[test]
    fn every_produced_shard_validates() {
        let root = scratch("validates");
        let tokenizer = tokenizer();
        let manifest = prepare(&request(&root, &fixture_corpus(), 64, 500, 500), &tokenizer)
            .expect("preparation succeeds");

        for summary in &manifest.shards {
            let path = root.join(&summary.file_name);
            let validated = validate_shard(
                &path,
                &ShardExpectation {
                    tokenizer_identity: &manifest.tokenizer_identity,
                    sequence_length: manifest.sequence_length,
                },
            )
            .expect("shard validates");
            assert_eq!(&validated, summary);
        }
    }

    #[test]
    fn documents_reach_every_split_when_the_policy_asks_for_it() {
        let root = scratch("all-splits");
        let tokenizer = tokenizer();
        let manifest = prepare(
            &request(&root, &fixture_corpus(), 16, 3_000, 3_000),
            &tokenizer,
        )
        .expect("preparation succeeds");

        for split in Split::ALL {
            let summary = manifest
                .splits
                .iter()
                .find(|entry| entry.split == split)
                .expect("every split is reported");
            assert!(
                summary.documents > 0,
                "split {} received no documents",
                split.as_str()
            );
        }
        assert_eq!(
            manifest.counts.documents_kept,
            manifest
                .splits
                .iter()
                .map(|entry| entry.documents)
                .sum::<u64>()
        );
    }

    #[test]
    fn a_repeated_document_is_counted_as_a_duplicate() {
        let root = scratch("dedup");
        let tokenizer = tokenizer();
        let line = "A sentence long enough to pass the minimum length filter.";
        let corpus =
            format!("{line}\n{line}\n{line}\nA different sentence of similar length here.\n");

        let manifest =
            prepare(&request(&root, &corpus, 16, 0, 0), &tokenizer).expect("preparation succeeds");

        assert_eq!(manifest.counts.documents_read, 4);
        assert_eq!(manifest.counts.documents_duplicate, 2);
        assert_eq!(manifest.counts.documents_kept, 2);
        assert_eq!(manifest.sources[0].documents_duplicate, 2);
    }

    #[test]
    fn a_credential_shaped_document_is_dropped_and_attributed() {
        let root = scratch("credential");
        let tokenizer = tokenizer();
        let corpus = concat!(
            "A perfectly ordinary sentence that passes every filter.\n",
            "-----BEGIN RSA PRIVATE KEY----- MIIEowIBAAKCAQEA\n",
        );

        let manifest =
            prepare(&request(&root, corpus, 16, 0, 0), &tokenizer).expect("preparation succeeds");

        assert_eq!(manifest.counts.documents_dropped, 1);
        let reasons = &manifest.sources[0].documents_dropped;
        assert_eq!(reasons.len(), 1);
        assert_eq!(reasons[0].reason, "pem_private_key");
        assert_eq!(reasons[0].count, 1);
    }

    #[test]
    fn a_short_document_is_dropped_as_too_short() {
        let root = scratch("short");
        let tokenizer = tokenizer();
        let corpus = "tiny\nA sentence long enough to pass the minimum length filter.\n";

        let manifest =
            prepare(&request(&root, corpus, 16, 0, 0), &tokenizer).expect("preparation succeeds");

        assert_eq!(manifest.sources[0].documents_dropped[0].reason, "too_short");
        assert_eq!(manifest.counts.documents_dropped, 1);
    }

    #[test]
    fn the_manifest_is_written_into_the_dataset_root_and_reloads() {
        let root = scratch("manifest");
        let tokenizer = tokenizer();
        let written = prepare(&request(&root, &fixture_corpus(), 32, 100, 100), &tokenizer)
            .expect("preparation succeeds");

        let path = root.join(crate::manifest::DatasetManifest::file_name());
        let loaded = crate::manifest::DatasetManifest::load(&path).expect("manifest reloads");

        assert_eq!(
            loaded.content_hash().expect("hashes"),
            written.content_hash().expect("hashes")
        );
    }

    #[test]
    fn a_source_without_usable_provenance_is_rejected_before_writing() {
        let root = scratch("provenance");
        let tokenizer = tokenizer();
        let mut request = request(&root, &fixture_corpus(), 32, 0, 0);
        request.sources[0].manifest.license = "unknown".to_owned();

        assert!(prepare(&request, &tokenizer).is_err());
    }

    #[test]
    fn no_declared_source_is_rejected() {
        let root = scratch("no-sources");
        let tokenizer = tokenizer();
        let mut request = request(&root, &fixture_corpus(), 32, 0, 0);
        request.sources.clear();

        assert!(prepare(&request, &tokenizer).is_err());
    }

    #[test]
    fn an_empty_corpus_produces_valid_empty_shards() {
        let root = scratch("empty");
        let tokenizer = tokenizer();
        let manifest =
            prepare(&request(&root, "", 32, 0, 0), &tokenizer).expect("preparation succeeds");

        assert_eq!(manifest.counts.documents_read, 0);
        assert!(
            manifest
                .shards
                .iter()
                .all(|shard| shard.sequence_count == 0)
        );

        for summary in &manifest.shards {
            validate_shard(
                root.join(&summary.file_name),
                &ShardExpectation {
                    tokenizer_identity: &manifest.tokenizer_identity,
                    sequence_length: manifest.sequence_length,
                },
            )
            .expect("an empty shard is still a valid shard");
        }
    }

    #[test]
    fn a_writer_refuses_a_sequence_of_the_wrong_length() {
        let root = scratch("writer-length");
        let mut writer =
            ShardWriter::create(root.join("x.shard"), Split::Train, 4, &"a".repeat(64))
                .expect("header writes");

        assert!(writer.push_sequence(&[1, 2, 3]).is_err());
    }
}
