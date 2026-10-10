# Tokenizer and Data Pipeline

## 1. Tokenizer

Use Hugging Face `tokenizers` from Rust.

Stage-1 default:

- Byte-level BPE or equivalent BPE configuration with robust arbitrary-text handling;
- vocabulary target: 32,768;
- deterministic training seed/order where supported;
- explicit special-token registry;
- tokenizer artifact committed only if licensing/provenance permits.

### Required special tokens

Reserve explicit IDs for at least:

- BOS;
- EOS;
- PAD;
- UNK.

Chat/instruction tokens are not required for base pretraining and should not be introduced until the post-training format is defined.

### Tokenizer invariants

Tests must verify:

- encode/decode does not panic on arbitrary valid UTF-8;
- special-token IDs never change after the model is trained;
- vocabulary size equals model configuration;
- EOS handling is stable;
- empty input is defined;
- long input truncation behavior is explicit;
- Unicode combining marks and non-Latin scripts do not corrupt the pipeline.

## 2. Corpus manifest

Every dataset source must have a manifest entry:

```text
source_id
source_name
source_uri_or_origin
license
acquisition_date
content_type
language_if_known
raw_bytes
documents
sha256_or_equivalent
filter_version
dedup_version
notes
```

Unknown license/provenance is a rejection condition for committed production training data.

## 3. Pipeline

```text
source
  ↓
parse
  ↓
UTF-8 validation / decoding policy
  ↓
normalization
  ↓
quality filtering
  ↓
PII/secrets policy filtering
  ↓
deduplication
  ↓
train/validation/test split
  ↓
tokenization
  ↓
document boundary handling
  ↓
sequence packing
  ↓
sharded token dataset
```

## 4. Normalization

Avoid aggressive text normalization that destroys linguistic information.

Allowed examples:

- normalize line endings;
- reject or replace invalid byte sequences according to a documented rule;
- remove clearly corrupt control-byte runs.

Do not automatically lowercase or strip accents for a general LLM corpus.

## 5. Deduplication

Two levels are required for serious runs:

1. exact document/content hashing;
2. near-duplicate strategy when corpus scale justifies it.

Deduplication must occur before final split creation to reduce train/evaluation leakage.

## 6. Evaluation contamination

Holdout/evaluation datasets must be isolated before training sharding.

For benchmark datasets:

- record exact version/hash;
- search training corpus for exact/near duplicates when feasible;
- report known contamination limitations.

## 7. Sequence packing

Training batches should minimize padding.

Rules:

- preserve explicit EOS/document separators;
- never concatenate documents without a defined boundary token;
- labels shift by one token;
- sequence packer must be deterministic under a fixed seed/config;
- packing statistics are logged.

Metrics:

- usable-token ratio;
- padding ratio;
- average packed sequence length;
- documents per packed sequence.

## 8. Data streaming

Do not load an entire large corpus into memory.

Use bounded-memory streaming with:

- buffered reads;
- bounded channels;
- explicit backpressure;
- shard-level parallelism;
- deterministic shard ordering when reproducibility mode is enabled.

**Burn 0.22 dataset semantics:** dataset access is fallible in 0.22. `Dataset::get` returns `Result`, dataset iterator items are `Result<I, E>`, and dataloader iterator items are `Result<Batch, DatasetError>`. Retrieval failure is therefore an explicit, typed outcome rather than an `Option` — the pipeline must surface it with context instead of treating it as end-of-data. A corrupt or unreadable shard must fail the run, per `12_SECURITY_AND_RELIABILITY.md`.

## 9. Data integrity

Each prepared shard must store or accompany:

- format version;
- tokenizer hash;
- sequence length;
- token count;
- record count;
- checksum.

A shard prepared with a different tokenizer must fail fast.

## 10. Stage-1 token budget

There are two separate runs:

### Smoke/training-system validation

Use a small legal corpus sufficient to:

- overfit a tiny sample;
- validate loss decrease;
- benchmark end-to-end throughput.

### Full Stage-1 pretraining

Token budget is compute-dependent. A rough planning range for a ~110M model is on the order of billions of tokens, not millions, if useful base-model quality is expected.

The exact budget must be derived from available accelerator time and documented before the full run. Do not hardcode it into the architecture.
