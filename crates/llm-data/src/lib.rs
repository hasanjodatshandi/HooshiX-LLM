//! Corpus preparation for the Rust-native LLM stack.
//!
//! This crate owns the path from raw corpus files to checksummed token shards:
//! provenance validation, bounded-memory streaming, normalization, filtering,
//! exact deduplication, deterministic splitting, sequence packing and shard I/O.
//!
//! It owns no tokenizer implementation, no model mathematics, no training loop
//! and no serving behaviour. It depends on [`llm_tokenizer`] only to turn text
//! into ids and to obtain the tokenizer identity a shard is bound to.
//!
//! # Contract
//!
//! - Provenance is a rejection condition. A source whose licence or origin
//!   cannot be stated is refused before a single document is read.
//! - Every step that changes the data has a version recorded in the dataset
//!   manifest, so a prepared dataset can be explained after the fact.
//! - A document's split is decided by its content, not its position, so the
//!   assignment survives a growing corpus.
//! - A shard records the tokenizer identity it was prepared with, and reading it
//!   with any other tokenizer fails immediately.
//! - Same input plus same configuration produces byte-identical shards and the
//!   same manifest content hash.

pub mod dedup;
pub mod error;
pub mod filter;
pub mod manifest;
pub mod normalize;
pub mod pack;
pub mod pipeline;
pub mod shard;
pub mod source;
pub mod split;

pub use error::DataError;
pub use filter::{CredentialShape, FilterConfig, FilterReason};
pub use manifest::{DatasetManifest, MANIFEST_VERSION};
pub use normalize::{InvalidUtf8Policy, normalize};
pub use pack::{PACKING_VERSION, PackingStats, SequencePacker, TailPolicy};
pub use pipeline::{PrepareRequest, prepare};
pub use shard::{
    SHARD_FORMAT_VERSION, ShardExpectation, ShardReader, ShardSummary, ShardWriter, validate_shard,
};
pub use source::{ContentType, SourceManifest, SourceReader, SourceSpec};
pub use split::{SPLIT_VERSION, Split, SplitPolicy};
