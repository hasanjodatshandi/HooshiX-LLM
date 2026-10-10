# Rust-Native LLM

A decoder-only language model stack built entirely in Rust: data pipeline,
tokenizer, training, inference and serving. No Python runtime or Python source is
part of the training or inference path.

The project plans five stages. The current scope is **Stage 1**: a correct
100M–150M dense decoder-only causal language model and its Rust-native
training/inference pipeline.

The full specification lives in [`docs/`](docs/00_README.md). Coding agents must
start at [`docs/AGENTS.md`](docs/AGENTS.md).

## Requirements

- Rust `stable` with a toolchain satisfying MSRV **1.95** (Burn 0.22's minimum).
  The workspace declares `rust-version = "1.95"`, so Cargo rejects older
  toolchains during resolution.
- Windows only: the MSVC target plus the Visual Studio C++ toolset and a Windows
  SDK component. Without the Windows SDK every link fails with
  `LNK1181: cannot open input file 'kernel32.lib'`.
- No execution backend is enabled by default; select one explicitly (below).

## Build and test

```bash
cargo check --workspace
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --doc --workspace
```

A check that did not run is reported as `Not run`, never as `Passed`.

Backend features are mutually exclusive, so do not use `--all-features`.

## Selecting an execution backend

Burn 0.22 chooses its backend at runtime through `Device`, and its default
features enable none. The composition roots in `apps/*` select exactly one:

```bash
# CPU reference path (default): pure-Rust eager Flex backend.
cargo run -p train

# CubeCL CPU runtime (LLVM JIT); a performance path, not the reference path.
cargo run -p train --no-default-features --features cpu

# NVIDIA GPU through CubeCL's CUDA runtime.
cargo run -p train --no-default-features --features cuda

# Portable graphics API through WGPU.
cargo run -p train --no-default-features --features wgpu
```

`flex` and `cpu` are different backends and are not interchangeable. `flex` is
the deterministic reference path used by correctness tests.

Always construct devices explicitly. `Device::default()` chooses from the
backends compiled into the binary rather than from the hardware present.

## Configuration

An experiment is described by one strict, versioned TOML document. Unknown keys
are rejected, so a typo fails instead of being silently ignored.

```bash
# Run the Stage-1 baseline described by a configuration file.
cargo run -p train -- configs/stage1-110m.toml

# Tiny model for pipeline validation.
cargo run -p train -- configs/stage1-smoke.toml

# No argument: built-in Stage-1 defaults.
cargo run -p train
```

Exit codes are distinct per failure class, so a supervisor or CI job can tell a
bad configuration from a broken device:

| code | meaning |
|---|---|
| 0 | success |
| 2 | configuration could not be read, parsed or validated |
| 3 | configuration names a backend family this binary was not built with |
| 4 | the device failed its preflight |

`runtime.backend` and `runtime.precision` must be supported by the binary that
loads them. A configuration is checked in full before any tensor is allocated.

## Tokenizer

The tokenizer is a byte-level BPE artifact built by this project, not a
downloaded file. Reserved-token literals are fixed by the code; their ids come
from the configuration, and an artifact is rejected unless both agree.

```bash
# Train an artifact. --corpus repeats and is consumed in the order given.
cargo run -p tokenizer-cli -- train \
  --config configs/stage1-110m.toml \
  --corpus path/to/corpus.txt \
  --output artifacts/tokenizer.json

# Print the identity hash and check the stored sidecar against it.
cargo run -p tokenizer-cli -- hash \
  --config configs/stage1-110m.toml \
  --artifact artifacts/tokenizer.json
```

`train` writes the artifact plus a `<artifact>.sha256` sidecar. The identity is a
SHA-256 over a canonical, sorted vocabulary listing rather than over the file,
because a hash-map-backed vocabulary has no guaranteed JSON key order. Training
is deterministic: the BPE trainer has no random seed, so the same corpus in the
same order always produces the same identity.

Exit codes are distinct per failure class:

| code | meaning |
|---|---|
| 0 | success |
| 2 | configuration could not be read, parsed or validated |
| 5 | corpus file could not be opened or read |
| 6 | tokenizer training, loading or saving failed |
| 7 | a stored identity hash does not match the artifact |

The vocabulary must be at least 256 (the byte alphabet) plus the reserved
tokens, because a byte-level tokenizer has to represent every byte from the
start; a smaller one is rejected rather than silently producing a tokenizer that
cannot encode some inputs.

## Preparing a dataset

Raw corpus files become checksummed token shards in one streaming pass. Corpus
provenance is a rejection condition, not documentation: a source whose licence or
origin cannot be stated is refused before a single document is read.

Declare the sources in a JSON file:

```json
[
  {
    "source_id": "my-corpus",
    "source_name": "My corpus",
    "origin": "https://example.com/dataset",
    "license": "CC0-1.0",
    "acquisition_date": "2026-10-10",
    "content_type": { "kind": "plain_text" },
    "language": "en",
    "path": "path/to/corpus.txt"
  }
]
```

Then prepare:

```bash
cargo run -p prepare-cli -- \
  --config configs/stage1-smoke.toml \
  --tokenizer artifacts/tokenizer.json \
  --sources tests/fixtures/sources.json \
  --output artifacts/data
```

The output directory receives one shard per split (`train`, `validation`, `test`),
a JSON sidecar per shard, and `dataset-manifest.json`. The manifest records every
ruleset version, the tokenizer identity, the split policy, per-source provenance
and measurements, the counters, and each shard's checksum.

Exit codes are distinct per failure class:

| code | meaning |
|---|---|
| 0 | success |
| 2 | configuration could not be read, parsed or validated |
| 5 | the source declaration file could not be read or parsed |
| 6 | the tokenizer artifact could not be loaded |
| 7 | corpus preparation failed |
| 8 | a produced shard failed validation |

A prepared dataset is reproducible: the same corpus and configuration produce
byte-identical shards and the same manifest content hash. A document's split is
decided by hashing its content together with a split seed rather than by its
position, so the assignment survives a growing corpus.

A shard records the tokenizer identity it was prepared with. Reading it with any
other tokenizer fails immediately, because token ids only mean something relative
to the tokenizer that produced them.

## Workspace layout

```
apps/prepare         prepare-cli: corpus to checksummed shards
apps/tokenizer       tokenizer-cli: training and artifact inspection
apps/train           composition root: configuration, backend selection, device construction
crates/llm-core      shared configuration, value types, typed errors
crates/llm-data      normalization, filtering, dedup, splitting, packing, shard I/O
crates/llm-tokenizer tokenizer training, loading, encoding and artifact identity
configs/             experiment configuration documents
tests/fixtures/      small deterministic fixtures
```

`crates/llm-core` depends on no other workspace crate, so the dependency
direction cannot invert. It owns the configuration schema and its invariants,
and it deliberately does not depend on Burn: backend and precision are
represented as plain value types.

`crates/llm-tokenizer` depends on `llm-core` only. It owns the tokenizer
artifact and its vocabulary contract, and no dataset loading, training loop or
serving behaviour.

`crates/llm-data` depends on `llm-core` and `llm-tokenizer` only. It owns the
path from raw corpus files to token shards, and no tokenizer implementation,
model mathematics or training loop.

Crates are added when they have code that owns a responsibility, not to match a
planned tree. The remaining crates from the target layout arrive with the
Stage 1 checklist sections that implement them.

Dependency direction is enforced by the crate graph.
