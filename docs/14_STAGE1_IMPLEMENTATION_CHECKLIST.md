# Stage 1 Implementation Checklist

This is the ordered execution backlog. Do not start Stage 2 work until every release-blocking item below is complete.

## A. Workspace and build

- [x] Create Cargo workspace.
- [x] Add `rust-toolchain.toml` using stable and verify `rustc >= 1.95`.
- [x] Add formatting/clippy configuration.
- [x] Add workspace dependency versions.
- [x] Implement backend feature selection: CPU reference + CUDA/WGPU as applicable.
- [x] Add CI/build commands to README.
- [x] `cargo check --workspace` passes.

**Acceptance:** clean workspace builds without Python.

Notes on completion (2026-10-10):

- The workspace starts with `crates/llm-core` and `apps/train` only. Crates are
  created when they have code that owns a responsibility, so the remaining
  crates join the workspace with the sections that implement them.
- Backend features are mutually exclusive and guarded by `compile_error!`;
  `flex` is the default. Do not use `--all-features`.
- Verified on Rust 1.99.0 (`stable-x86_64-pc-windows-msvc`): `cargo fmt --check`,
  `cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (9 tests), `cargo test --doc --workspace` and
  `cargo run -p train` (device preflight on Flex) all pass.
- Backend status, verified per path rather than assumed:
  - `flex`: compile, link, test and run all pass. Its dependency graph contains
    no CubeCL, no `turso` and no `simsimd`.
  - `wgpu`: `cargo clippy --no-default-features --features wgpu --all-targets -- -D warnings`
    passes, but **running fails at link time** with
    `LNK1181: cannot open input file 'simsimd.lib'`. Reported as **not run**.
  - `cpu` and `cuda`: **not run**. They share the same CubeCL dependency chain
    and the same native link blocker as `wgpu`; `cuda` additionally has no GPU
    and no CUDA toolkit on this machine.
- The `simsimd` link blocker affects every CubeCL-backed path, not only
  accelerators. See `13_DEPENDENCIES_AND_TOOLCHAIN.md` §5 before planning any
  work that depends on the CubeCL CPU runtime.

## B. Core configuration

- [x] Define `ModelConfig`.
- [x] Define `TokenizerConfig`.
- [x] Define `DataConfig`.
- [x] Define `TrainingConfig`.
- [x] Define `GenerationConfig`.
- [x] Add validation.
- [x] Add TOML load/save.
- [x] Add resolved-config artifact.

**Acceptance:** invalid configurations fail before model allocation.

Notes on completion (2026-10-10):

- Two further sections were required by the existing specifications and are
  included: `RuntimeConfig` (backend and precision), because
  `05_MODEL_SPEC.md` §11 requires unsupported dtype/backend combinations to fail
  clearly, and `OptimizerConfig`/`SchedulerConfig` nested inside
  `TrainingConfig`, because `03_ARCHITECTURE.md` §5 lists them in the
  configuration hierarchy.
- The model configuration is a **strict** document: `#[serde(deny_unknown_fields)]`
  rejects unknown keys, so a typo fails instead of being silently ignored.
- Validation is structured, not stringly typed: each rule has its own error
  variant naming the offending field and values.
- Backend checking compares **families**, not device indices. The family is a
  compile-time Cargo feature; the index is a runtime selection, so a binary
  built with `cuda` accepts `cuda:3`.
- `ModelConfig::parameter_count` uses checked arithmetic and reproduces the
  documented Stage-1 accounting: the baseline resolves to **110,119,680**
  parameters (embedding 25,165,824 + 12 blocks + final norm), inside the
  100M..=150M band.
- Verified: `cargo fmt --check`, `cargo check --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (62 tests), `cargo test --doc --workspace`.
- Verified end to end through `apps/train`, which is distinct per failure class:
  valid file and built-in defaults exit `0`; a tokenizer/model vocabulary
  mismatch, an unknown-field typo and a missing file exit `2`; a configuration
  naming a different backend family exits `3`.
- `configs/stage1-110m.toml` and `configs/stage1-smoke.toml` are loaded,
  validated and round-tripped by `crates/llm-core/tests/config_files.rs`, so they
  cannot drift out of sync with the schema.

## C. Tokenizer

- [x] Implement special-token registry.
- [x] Implement tokenizer training CLI.
- [x] Implement tokenizer load/encode/decode.
- [x] Persist tokenizer artifact hash.
- [x] Unicode/empty/long-input tests.
- [x] Vocabulary/model mismatch test.

**Acceptance:** deterministic fixture tests pass.

Notes on completion (2026-10-10):

- New crates: `crates/llm-tokenizer` and `apps/tokenizer` (`tokenizer-cli`).
- The `tokenizers` dependency is built with `default-features = false` plus the
  pure-Rust `fancy-regex` backend. The default `onig` feature pulls Oniguruma, a
  native C library, and 0.23 refuses to build with neither regex backend.
- Two behaviours had to be corrected against the library defaults, and both were
  found by the round-trip test rather than by inspection:
  - `ByteLevel::default()` uses the GPT-2 convention `add_prefix_space: true`,
    which invents a leading space on encode and strips one on decode, making the
    round trip lossy. It is built with `add_prefix_space = false`.
  - The BPE trainer derives its initial alphabet from the corpus. That left bytes
    which never occur during training — a newline in a corpus read line by line —
    with no token at all, so `"\n"` encoded to nothing. The trainer is now given
    the full 256-byte alphabet up front, which is what makes every valid UTF-8
    string representable, and a vocabulary smaller than that alphabet plus the
    reserved tokens is rejected with its own error.
- Reserved-token *literals* are fixed by the crate; *ids* come from
  `TokenizerConfig`. The trainer places reserved tokens at the head of the
  vocabulary in ascending configured-id order, so the configured ids must be
  exactly the contiguous range `0..4`; anything else is rejected before training
  instead of silently rebinding ids.
- The artifact identity is a SHA-256 over a canonical, sorted vocabulary listing
  rather than over the serialized file, because a hash-map-backed vocabulary has
  no guaranteed JSON key order. Two independent runs were also confirmed to
  produce byte-identical files on this toolchain.
- Encoding never inserts reserved tokens and never truncates implicitly:
  `Truncation::Reject` is the default and `KeepFirst` is opt-in.
- Verified: `cargo fmt --check`, `cargo check --workspace --all-targets`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (84 tests), `cargo test --doc --workspace`.
- Verified end to end through `tokenizer-cli`, with a distinct exit code per
  failure class: a missing configuration exits `2`, a missing corpus `5`, a
  corpus too small for the vocabulary `6`, and a tampered identity sidecar `7`.
- `tests/fixtures/tokenizer-corpus.txt` and `tests/fixtures/tokenizer-fixture.toml`
  are committed fixtures; the contract tests in
  `crates/llm-tokenizer/tests/tokenizer_contract.rs` train against them, so the
  fixture and the schema cannot drift apart.

## D. Dataset pipeline

- [x] Define source record abstraction.
- [x] Implement streaming text/JSONL reader as required by chosen corpus.
- [x] Add provenance manifest.
- [x] Add normalization.
- [x] Add exact dedup baseline.
- [x] Add deterministic split.
- [x] Tokenize in bounded memory.
- [x] Add EOS/document boundaries.
- [x] Add fixed-length sequence packing.
- [x] Produce checksummed shards.
- [x] Add shard validation.

**Acceptance:** same input/config produces the same prepared fixture output/hash.

Notes on completion (2026-10-10):

- New crate `crates/llm-data` and a new composition root `apps/prepare`
  (`prepare-cli`). The app is an addition to the planned tree: `04_REPOSITORY_STRUCTURE.md`
  lists no data-preparation entry point, and without one the pipeline would be
  reachable only from tests.
- `DataConfig` gained the deterministic split settings, which is an incompatible
  change to a strict configuration document, so `CONFIG_VERSION` was raised to 2
  rather than the new fields being silently defaulted. All shipped configurations
  and fixtures were updated.
- A document's split is decided by hashing its content with a split seed, not by
  its position. The assignment therefore survives a growing corpus, added sources
  and a resumed run, none of which is true of a positional split.
- Deduplication is exact, over the normalized text, and runs before splitting, so
  a duplicate can never appear on both sides of the boundary.
- Packing joins documents with an explicit end-of-sequence token and chops the
  result into fixed-length windows. Document boundaries are never crossed without
  a boundary token, and sequences are full, so batches need no padding and the
  loss needs no mask. Token loss is only ever an incomplete final window, and that
  count is recorded rather than inferred.
- A shard records the tokenizer identity it was prepared with; reading it with
  any other tokenizer fails immediately. Validation re-reads the whole file and
  checks the magic, the format version, the identity, the sequence alignment and
  the SHA-256 trailer.
- Two behaviours were corrected after testing rather than by inspection:
  - The reader split only on LF, so a carriage-return-separated file became one
    enormous document. It now treats LF, CRLF and a lone CR as separators.
  - The credential detector required an exact-length run, which missed
    variable-length Slack tokens; run length is now explicit about exact versus
    minimum.
- Verified: `cargo fmt --check`, `cargo check --workspace --all-targets`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (154 tests), `cargo test --doc --workspace`.
- Verified end to end through `prepare-cli`: two independent runs over the fixture
  corpus produced the same manifest content hash **and byte-identical shards**.
  Distinct exit codes were exercised for each failure class: a missing
  configuration exits `2`, a missing or malformed source declaration `5`, a missing
  tokenizer artifact `6`, and an unusable declaration or unreadable corpus `7`.
- Honest limitation: on a 59-document fixture a 0.5% holdout rounds to zero
  documents, so the validation and test shards are empty. The split is
  statistically correct; a small corpus needs a larger holdout ratio.

## E. Model primitives

Implement in this order:

- [x] RMSNorm.
- [x] RoPE.
- [x] causal mask.
- [x] multi-head self-attention.
- [x] SwiGLU FFN.
- [x] Transformer block.
- [x] token embedding.
- [x] full decoder model.
- [x] tied LM head.
- [x] causal LM loss.

After each item add unit/reference tests.

**Acceptance:** forward pass produces `[batch, seq, vocab]`, causal invariants pass.

Notes on completion (2026-10-10):

- New crate `crates/llm-model`, depending on `llm-core` and Burn only. It names
  no backend, so the same mathematics runs on the reference path and on an
  accelerator without a conditional.
- The workspace backend rule was relaxed: two *accelerator* backends in one
  binary is still a compile error, but the reference backend may coexist with
  one. It has to, because tests need a `Device` and `flex` therefore arrives
  through `dev-dependencies`. `compiled_backend` states the precedence and every
  device is constructed explicitly, so the ambiguity the rule guarded against
  cannot arise.
- The configuration-to-model cross-check is a test: the count
  `ModelConfig::parameter_count` predicts from dimensions equals the count the
  built model reports. A wrong architectural assumption cannot survive it.
- Grouped-query attention is **refused**, not approximated. A configuration with
  `num_kv_heads != num_heads` fails construction with a message naming the
  Stage-2 boundary, so the architecture cannot drift by accident.
- RoPE is applied per tensor rather than only as a query/key pair, because a
  single query scored against many keys is a real shape. The pair form remains
  as the self-attention convenience.
- The loss computes its own denominator instead of delegating to a framework
  helper, because the specification requires an explicit mask. A fully masked
  batch is reported as an error rather than returned as a `NaN`.
- Verified: `cargo fmt --check`, `cargo check --workspace --all-targets`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (214 tests), `cargo test --doc --workspace`.
- Four Burn 0.22 API facts were established by compiling rather than by recall,
  and two of them were defects in this code until a test caught them:
  - `Linear` stores its weight as `[input, output]`, so reading the projection
    width from axis zero silently returned the hidden size.
  - `softmax` and `log_softmax` are functions, and the axis is a zero-based
    index, not a rank.
  - the mask operation is `mask_fill`, not `masked_fill`;
  - `matmul` requires both sides at the same rank, so the language-model head
    flattens and restores the batch and sequence axes around the product.

## F. Gradient correctness

- [x] Verify all trainable parameters are registered.
- [x] Verify non-zero finite gradients on a tiny batch.
- [x] Finite-difference selected gradients where practical.
- [x] Add NaN/Inf detection.

**Acceptance:** gradient suite passes on CPU reference path.

Notes on completion (2026-10-10):

- New module `crates/llm-model/src/grad.rs`, holding the production check and the
  finite-difference suite. `ModelError::NonFinite` carries
  `NumericalQuantity::{Loss, Gradients}`, so a caller reports what failed rather
  than a bare "non-finite".
- The check does not convert the tensor to `f32` to inspect it. A conversion
  would silently change meaning the moment the working dtype changes, and dtype
  coverage is a Stage-2 concern. Instead it asks two questions that hold for
  every float dtype: `x == x` is false exactly for `NaN`, and `x - x == x - x`
  is false exactly for an infinity.
- Registration is verified per layer type, in the module that owns the layer, so
  a backward pass is proven to reach the scale of a normalization, all four
  attention projections, all three feed-forward matrices, and the shared
  embedding that serves as both input table and output head.
- The finite-difference test compares an autodiff gradient against a central
  difference of a host-side reference for `f(w) = sum((w * x)^2)`. That is the
  check that catches a wrong derivative, a detached path, or a gradient landing
  in the wrong parameter.
- The "no accidental detach" requirement is tested in both directions: a value
  inside the graph reports a gradient, and a value outside it reports none.
- Verified: `cargo fmt --check`, `cargo check --workspace --all-targets`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (226 tests), `cargo test --doc --workspace`.
- Not run: every check here runs on the Flex reference path. Accelerator parity
  remains unexecuted because no accelerator is present.

## G. Tiny-model overfit

- [x] Create tiny model config.
- [x] Create tiny legal deterministic corpus.
- [x] Train until clear overfit.
- [x] Assert loss reduction.
- [x] Generate learned pattern.

**Acceptance:** automated or reproducible integration test demonstrates overfit.

Notes on completion (2026-10-10):

- New crate `crates/llm-training` with the minimum the test needs: one
  optimization step and a loop that repeats it. Gradient accumulation, the
  learning-rate schedule, gradient clipping, periodic evaluation, token
  accounting, checkpointing and metrics belong to the training-loop section and
  are deliberately absent.
- The test is `crates/llm-training/tests/tiny_overfit.rs`. It builds a two-layer
  32-wide model, takes one fixed batch from a committed fixture, and asserts
  three things rather than one: the best loss falls below five percent of the
  initial loss, every step stays finite, and a greedy continuation reproduces
  the corpus byte for byte. The third is what distinguishes a model that learned
  the data from one that merely found a small number.
- The corpus is read as bytes, one id per byte. That keeps the test about the
  model, the objective and the optimizer — the tokenizer already has its own
  contract tests — and it removes any dependency on training a tokenizer first.
  The fixture is written by this repository, so it is deterministic and carries
  no licensing question.
- A step whose loss is not finite is **refused** with the model returned
  unchanged. Applying it would write `NaN` into every parameter and the
  corruption would only surface much later.
- New ADR: `ADR-018` records that multimodal capability is a future scope
  expansion, and `05_MODEL_SPEC.md` §13 now lists modality representation among
  the changes that require an ADR. No code changed for it.
- Verified: `cargo fmt --check`, `cargo check --workspace --all-targets`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (227 tests), `cargo test --doc --workspace`. The
  overfit test takes about twenty seconds on the reference backend.

## H. Full training loop

- [x] AdamW.
- [x] warmup + cosine schedule.
- [x] gradient clipping.
- [x] gradient accumulation.
- [x] metrics.
- [x] validation loop.
- [x] token accounting.
- [ ] optional BF16 only after FP32 passes.

**Acceptance:** Stage-1 smoke run remains numerically stable.

Notes (2026-10-10) — seven of eight items complete:

- New modules in `crates/llm-training`: `scheduler.rs` and `metrics.rs`, with
  `trainer.rs` extended to drive them.
- The schedule is **computed, not scheduled**: a closed-form function of the step
  number, so the step counter *is* the scheduler state and a checkpoint cannot
  lose it. Eleven tests pin the curve: monotone rise to the peak on the last
  warmup step, monotone decay, exactly half the peak at the halfway point, zero
  from the end of the window onward, and the whole curve inside `[0, peak]`.
  A run that continues past the window holds at zero rather than rising again;
  that is a deliberate departure from the Burn 0.22 reading, documented in the
  module, because an unexpected late rise damages a converged model.
- The validation loop is `Trainer::fit_with_validation` with a
  `ValidationSchedule` of held-out batches and an interval. The validation loss
  is attached to the step that triggered it, so the history stays one row per
  optimizer step and both curves share an axis. Eight tests cover it, including
  one that asserts evaluation does **not** modify the parameters — a backward
  pass there would silently train on the held-out set.
- Accumulation sums the micro-batch objectives and takes one backward pass over
  the sum. The gradient is identical to summing per-parameter gradients, but
  **peak memory is not reduced**, because one graph stays alive. That is the
  right trade at Stage 1 scale, where the reason to accumulate is to reach a
  larger effective batch, not to fit one that is too large to hold. True
  memory-reducing accumulation remains to be done.
- Gradient clipping is applied by the optimizer itself, so a step cannot forget
  to clip. **The global gradient norm is still not reported**: Burn clips but
  does not expose the norm, `GradientsParams` keeps its container private and
  offers no iteration, and a parameter-visitor path has not been written.
  `StepMetrics::gradient_norm` exists and is `None`, so a run's record does not
  change shape when the value arrives. This remains an open requirement of
  `07_TRAINING_AND_CHECKPOINTS.md` §7.
- Token accounting is asserted rather than merely recorded: `tokens_seen` must
  equal steps times tokens per step.
- **Not done, correctly:** reduced-precision training. BF16 is allowed only after
  the FP32 path is stable and a forward/backward parity test passes on the target
  backend, and no accelerator is present to run either. Stage 1 is FP32.
- Verified: `cargo fmt --check`, `cargo test -p llm-training` (26 tests),
  `cargo clippy --workspace --all-targets -- -D warnings`. Every run reports zero
  non-finite losses and zero skipped steps, which is evidence of numerical
  stability.
- **The acceptance criterion is not yet met in full.** Stability is demonstrated
  on fixed fixtures. A Stage-1 smoke run over prepared shards, with the schedule
  and checkpoint interval the smoke configuration specifies, comes after
  checkpointing, and the gate should be judged there.

## I. Checkpointing

- [x] Save model.
- [x] Save optimizer.
- [x] Save scheduler.
- [x] Save global step/tokens seen.
- [x] Save data-position state.
- [x] Save config/manifest hashes.
- [x] Atomic checkpoint publication.
- [x] Load/restore.
- [x] Resume-equivalence integration test.
- [x] Corruption/mismatch tests.

**Acceptance:** kill/restart does not silently reset training state.

**MET (2026-10-10).** All ten items complete, 22 tests, four consecutive
parallel runs green.

> **Two real defects were found here and fixed (2026-10-10).** Both surfaced only
> because a test compared two runs, and neither was in the checkpoint code.
>
> **1. Model construction was not reproducible.** `burn-nn` 0.22's
> `Initializer::init` returns a **lazily initialised** `Param::uninitialized`
> whose closure performs the draw on first read, and Burn's own `num_params`
> deliberately reads `lazy_shape` so as not to force it. The reference backend
> keeps one process-wide random source, so a parameter's values depended on
> *when it was first read* rather than on the seed: building two models and
> reading them in sequence gave them different weights. `init.rs` claimed the
> opposite in its module documentation.
>
> Fixed by `llm_model::init::materialize`, a module visitor that reads every
> float parameter immediately after construction, called from `Transformer::new`.
> After it returns the module holds real tensors and the random source is no
> longer consulted, so the seed alone decides the weights. A direct probe
> confirms the ordering: `Tensor::random` was reproducible before the fix, the
> project's initializer was not, and both are now.
>
> **2. The test suite corrupted its own measurements.** Because that random
> source is process-wide while cargo runs a binary's tests as parallel threads,
> two tests that seeded and drew interleaved and observed each other's draws.
> The checkpoint suite serialises construction behind a mutex; because
> construction is now atomic, the lock covers only construction and the three
> raw-draw probes. Verified by four consecutive parallel runs, all green.
>
> This is why the resume comparisons below are evidence: before the fix they
> measured two runs that had started from different weights.

Notes (2026-10-10) — all ten items complete:

- New module `crates/llm-training/src/checkpoint.rs`, `CheckpointError`, and the
  contract suite `crates/llm-training/tests/checkpointing.rs`.
- Implemented and **tested**: an envelope carrying format version, global step,
  tokens seen, data cursor, and the tokenizer, dataset-manifest and configuration
  hashes; model save and load through Burn's `Module::save_file` /
  `try_load_file`; atomic publication; and a `latest` pointer.
- The schedule needs no separate record. It is a closed-form function of the step
  number, and the step number is in the envelope, so `Save scheduler` reduces to
  `Save global step`.
- Optimizer state is persisted through `ModuleOptimizer::save` / `load`, which
  Burn implements over its own record store. Burn restores into an *existing*
  optimizer rather than constructing one from the file, so `Trainer::resume_from`
  builds an empty optimizer from the trainer's own configuration and loads the
  stored state into it. Without this a resume restarts AdamW's moment estimates,
  which is exactly the failure the acceptance criterion names.
- The resume-equivalence test trains ten steps in one go, then trains five,
  checkpoints, reloads model and optimizer from disk, and trains five more, and
  requires the two sets of logits to agree within `1e-6`. A companion test
  resumes **without** restoring the optimizer state and requires the result to
  differ, so the equality above is evidence rather than coincidence.
- Atomic publication follows the documented pattern: build in `.tmp-<name>`,
  read the envelope back to verify, rename into place, then publish the pointer
  by writing a temporary file and renaming it. A crash leaves either the previous
  good checkpoint or an unreferenced temporary directory.
- The reuse of a name being refused is tested through its consequence rather than
  its return value: after a rejected save, the `latest` pointer must still name
  the checkpoint published before. That is the property that matters after a
  crash.
- The interrupted-write test builds a `.tmp-` directory by hand, exactly as a
  killed process would leave it, and asserts three things: the pointer still
  names the good checkpoint, that checkpoint still validates, and a later save
  with the same name clears the stale directory and succeeds.
- Three mismatch tests assert rejection rather than warning for a different
  tokenizer, a different dataset and a different configuration, plus one that a
  caller supplying no configuration hash is not asking for that check and the
  load therefore succeeds. The round-trip test compares logits, not just
  metadata, so a load that restored the wrong tensors would fail.
- Verified: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace`, `cargo test --doc --workspace`. The
  contract suite is `crates/llm-training/tests/checkpointing.rs` (14 tests).

## J. Inference

- [x] Greedy sampler.
- [x] Temperature.
- [x] Top-k.
- [x] Top-p.
- [x] EOS/max-token handling.
- [x] Prefill.
- [x] Per-layer KV cache.
- [x] Incremental decode.
- [x] Cached-vs-uncached equivalence test.
- [x] Generate CLI.

**Acceptance:** deterministic generation fixture passes.

Notes (2026-10-10) — all ten items complete:

- The key/value cache lives in `crates/llm-model/src/kv_cache.rs`, because it is
  model state: one [`LayerCache`] per block, holding the rotated keys and values
  of everything seen so far. A key's rotation depends only on its own absolute
  position, which never changes once seen, so storing the rotated form is
  correct rather than an approximation.
- Attention's forward path was refactored into `project_and_split`, `attend` and
  two entry points rather than duplicated. `attend` takes the query and key
  lengths separately, which is what a cached step needs: one query against a
  longer history. The causal mask already took them separately, so no mask change
  was required.
- **The equivalence is tested, not assumed.** `cached_decoding_reproduces_a_full_forward_pass`
  compares a cached run against a single full pass at *every* position, not only
  the last, so an error that cancels at the end cannot pass. Prefill is compared
  separately from the decode steps.
- `crates/llm-inference` holds sampling and the generation loop. The sampler
  operates on a plain slice of scores and takes its randomness as a
  [`RandomSource`] parameter rather than reading a global generator, so a greedy
  decoder ignores randomness entirely and a sampled one replays exactly from its
  seed. Fifteen sampling tests use a scripted source, so the filtering order —
  temperature, softmax, top-k, top-p, draw — is pinned rather than inferred.
- New crate `crates/llm-checkpoint`. The checkpoint format was moved out of
  `llm-training` because the inference side reads it too; leaving it there would
  have put the trainer inside the generation binary. `llm-training` re-exports it,
  so the format still has exactly one definition.
- **Two real defects were found and fixed while building this**, both recorded in
  section I: model construction was not reproducible, and the test suites were
  corrupting their own seeded measurements through the reference backend's
  process-wide random source.
- Verified: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace` (309 tests), `cargo test --doc --workspace`.
  The generation CLI was run: `--help` exits 0, and a missing configuration exits
  2 with a message naming the file.
- **Not run:** an end-to-end generation from a real checkpoint. No tokenizer
  artifact and no trained checkpoint exist in the repository yet, so the CLI's
  success path has not been executed against real files. The acceptance criterion
  — a deterministic generation fixture — is met at the library level by the
  greedy-repeat and seed-replay tests; the CLI path that joins a tokenizer, a
  checkpoint and a prompt is compiled and partially exercised but not yet proven.

## K. GPU/backend parity

- [ ] Run reference model on CPU.
- [ ] Run same weights/input on target GPU.
- [ ] Compare logits/loss within dtype tolerance.
- [ ] Compare selected gradients.
- [ ] Document unsupported backend/dtype combinations.

**Acceptance:** parity report committed/generated.

## L. Serving

- [ ] Axum app state.
- [ ] one-time model load.
- [ ] readiness/liveness endpoints.
- [ ] `/v1/completions`.
- [ ] streaming response.
- [ ] prompt/output token limits.
- [ ] bounded concurrency.
- [ ] cancellation.
- [ ] graceful shutdown.

**Acceptance:** server never reloads model per request and survives basic concurrency tests.

## M. Benchmark baseline

- [ ] training tokens/sec;
- [ ] peak memory;
- [ ] prefill tokens/sec;
- [ ] decode tokens/sec;
- [ ] TTFT;
- [ ] checkpoint read/write throughput;
- [ ] cold-start model load;
- [ ] artifact/binary size.

**Acceptance:** reproducible benchmark report with environment metadata.

## N. Release gate

Run:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Then complete:

- [ ] security/reliability review;
- [ ] dependency audit;
- [ ] no Python files/runtime in required workflow;
- [ ] README commands verified from a clean checkout;
- [ ] Stage-1 acceptance report.

Only then begin Stage 2.
