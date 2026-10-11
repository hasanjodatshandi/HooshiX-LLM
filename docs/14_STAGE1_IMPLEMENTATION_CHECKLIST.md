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
- **Acceptance met (2026-10-10).** The smoke run that this criterion names now
  exists and passes. `llm-train` was a composition root that loaded a
  configuration and preflighted a device; it now reads a prepared dataset, builds
  batches, trains, evaluates and checkpoints. Against
  `tests/fixtures/smoke-pipeline.toml` — 2 layers, width 64, 320-token
  vocabulary, 100 steps over a 41-sequence training shard — two consecutive runs
  produce **bit-identical** results: first loss 5.7687, best 3.9223, last
  4.1606, validation 4.3158 at step 100, four checkpoints, **zero refused steps
  and zero abandoned steps**, exit 0.
- Three defects were found by running the chain, none of which any unit test
  could have caught:
  - **The training app never seeded the device.** The configuration has a seed
    and `init.rs` states that the composition root seeds before constructing, but
    the composition root did not, so two runs began from different weights. The
    first loss differed between runs and nothing else did. This is the
    reproducibility rule of section F being enforced in the crate that owns the
    seeding and forgotten in the crate that must call it.
  - **The device built for training was not an autodiff device.** A step cannot
    produce gradients on a device that is not tracking them, so the first run
    panicked inside Burn. `build_device` now returns an autodiff device, which
    makes it a property of construction rather than something each call site has
    to remember.
  - **A second run refused to start twenty-five steps in.** Checkpoint
    publication refuses to replace an existing directory, which is correct, but
    the refusal arrived after the run had already done work. The checkpoint root
    is now checked before the first step.
- `llm-data` had no shard reader. It could write and validate shards and nothing
  could read one back, so the training path had no way to obtain a batch.
  `ShardReader` streams sequences out, checks the header on open — magic, version,
  sequence length, split tag, tokenizer identity — and deliberately does not
  verify the trailing digest, which is a full pass over the payload and belongs
  to `validate_shard` as an explicit integrity step. Ten tests cover it,
  including the case that matters most: a shard opened with the wrong tokenizer
  is refused rather than trained on.
- The fixture corpus is 59 documents, and the 0.5% holdout of
  `configs/stage1-smoke.toml` is zero documents, which leaves the validation
  split empty and the evaluation path untested. `tests/fixtures/smoke-pipeline.toml`
  uses a 20% holdout for this reason, and says so.

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

- [x] Run reference model on CPU.
- [x] Run same weights/input on target GPU.
- [x] Compare logits/loss within dtype tolerance.
- [x] Compare selected gradients.
- [x] Document unsupported backend/dtype combinations.

**Acceptance:** parity report committed/generated.

**MET (2026-10-11).** Five of five items. This section was expected to
be unrunnable because the machine has no NVIDIA card and CUDA is NVIDIA-only.
That expectation was wrong, and the correction is the main result here.

- **The machine has a GPU, and it works.** `AMD Radeon R9 200 Series`
  (device `0x6810`, GCN 1st generation, 4 GB), driver 27.20.20912.1002, reporting
  Vulkan 1.2.170 and enumerating as `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`.
  CUDA does not apply to it, and ROCm does not support GCN 1st generation — but
  `wgpu` does, because it goes through Vulkan/DX12 rather than a vendor runtime.
- **`wgpu` had been written off as blocked and is not.** It was never built,
  because the `simsimd` and LLVM blockers stopped every CubeCL path at the same
  point; section 13 section 5 removed those, and `wgpu` then built in 8.5 minutes
  and ran first time. It was never retested after the blockers fell, and it should
  have been.
- **End to end on the GPU:** `apps/train` ran the full smoke configuration on
  `wgpu` — 100 steps, four checkpoints, zero refused or abandoned steps, exit 0.
- **Parity, same weights and same input.** `crates/llm-training/examples/parity.rs`
  writes the weights once from the reference path and reads them back on both
  devices, because two devices seeded identically do *not* produce the same
  weights — the random source belongs to the device. Comparing two runs that each
  trained from their own seed would measure the seeds, not the backends.
  Result, on the smoke architecture:

  ```text
  elements         = 10240
  worst absolute   = 4.991889e-7
  worst relative   = 1.6346176e-3
  non-finite pairs = 0
  ```

  Five parts in ten million is f32 rounding, not a discrepancy. Build and run
  with `--features flex,wgpu`.
- **Throughput on the GPU**, measured against the reference path in the same
  session:

  | shape | `wgpu` | `flex` |
  |---|---|---|
  | batch 1 × sequence 128 | 0.54 s, 238 tok/s | — |
  | batch 1 × sequence 256 | 0.64 s, 400 tok/s | — |
  | batch 2 × sequence 256 | 1.01 s, **508 tok/s** | 237 tok/s |
  | batch 1 × sequence 512 | **out of memory** | — |

  A 2013-era card is **more than twice** the CPU reference path. It is also
  memory-bound long before the CPU is: 4 GB holds the model, its gradients and
  AdamW's moments at about 1.8 GB, and the score tensor at sequence 512 does not
  fit beside them. Section 7's finding that the configured shape is a memory
  problem applies to the GPU more sharply than to the CPU.
- **A defect this exposed.** The first GPU run failed inside `ensure_finite` with
  a panic, not a returned error, because it read the result back with `to_data`,
  which panics when the device has died. That erases exactly the distinction the
  module was written to preserve. It now uses `try_to_data`, so a device that ran
  out of memory becomes `ModelError::CheckUnavailable` rather than a crash.
- **Gradient parity (2026-10-11).** The harness now runs the objective on both
  devices and collects every parameter's gradient through a `ModuleVisitor`, in
  the module's own field order, so the two sequences are comparable element by
  element:
  ```text
  logits     elements  10240, worst absolute 4.4703484e-7, non-finite 0
  gradients  elements 151872, worst absolute 2.3841858e-7, non-finite 0
  parameters without a gradient: reference 0, accelerated 0
  ```
  All 151872 gradient elements — every parameter in the model — agree to 2.4e-7,
  which is f32 rounding. Logits agreeing proves nothing about the backward pass:
  a wrong derivative or a detached path leaves the forward output untouched and
  moves every parameter the wrong way. This is the check that covers it.
- **Tuning the 4 GB card (2026-10-11).** The card no longer drives the display,
  which returned about 900 MB: dedicated usage fell from 945 MB to 64 MB. What
  that changed, measured with `throughput`:

  | shape | tokens per step | result |
  |---|---|---|
  | 2 × 256 | 512 | **514 tok/s** |
  | 1 × 512 | 512 | 383 tok/s |
  | 4 × 256, 2 × 512, 1 × 1024 | 1024 | out of memory |
  | 8 × 256, 4 × 512, 8 × 512 | 2048 or more | out of memory |

  The ceiling is **512 tokens per micro-batch, whatever the shape**. Freeing the
  display's memory did not raise it — it moved the first row of the second group,
  and before the card was freed even `1 × 512` failed. The binding term is the
  logits tensor at `tokens × vocab_size × 4`: 67 MB per copy at 512 tokens
  against 134 MB at 1024, multiplied by the copies `log_softmax` and its backward
  need, on top of about 1.76 GB of weights, gradients and AdamW moments that do
  not move.
- **Gradient accumulation is not free here**, unlike on the CPU where it was
  measured at no per-token cost. `ACCUM` in `throughput` holds the total fixed
  and shrinks the micro-batch, and shrinking it costs: 2 × 256 gives 514 tok/s
  while 1 × 256 gives 261. The card needs a real micro-batch to be efficient, so
  accumulation buys memory rather than throughput, and here it spends throughput
  to get it.
- **The configured Stage-1 shape cannot run on this GPU at all.** `batch 8 ×
  sequence 2048` is 16384 tokens per step against a ceiling of 512. Sequence 2048
  is not reachable at any batch size — `1 × 1024` is already out of memory — so
  the sequence length is the wall rather than the batch. This is a memory limit
  of the card, not a defect: the same configuration runs on the reference path,
  slowly.
- **Unsupported combinations, documented:** CUDA needs an NVIDIA card. ROCm
  needs GCN 3rd generation or newer, so it excludes this card, and is barely
  supported on Windows regardless. `cpu` (CubeCL LLVM JIT) works but measured
  about three times slower than `flex` and costs nine to eighteen minutes of
  kernel compilation. `wgpu` works and is faster, but is bounded by 4 GB here.

## L. Serving

- [x] Axum app state.
- [x] one-time model load.
- [x] readiness/liveness endpoints.
- [x] `/v1/completions`.
- [x] streaming response.
- [x] prompt/output token limits.
- [x] bounded concurrency.
- [x] cancellation.
- [x] graceful shutdown.

**Acceptance:** server never reloads model per request and survives basic concurrency tests.

**MET (2026-10-11).** All nine items complete; 15 tests in `llm-server` plus a
live run.

- New crate `crates/llm-server` (routing, limits, state, errors) and
  `apps/serve` (the composition root). Generation moved to a streaming entry
  point, `llm_inference::generate_streaming`, so that a token can be emitted as
  it is produced and so that cancellation has somewhere to happen.
- **The model is loaded once.** `AppState` holds it in an `Arc`, built by the
  composition root before the listener exists. There is no code path that
  constructs a `Transformer` inside a handler, which is what the acceptance
  criterion asks for and is enforced by construction rather than by discipline.
- **Readiness is earned, not assumed.** The server starts unready and becomes
  ready only after a real self-test generation through the real path. A model
  that loaded is not a model that works: a shape error or a backend that cannot
  execute an operation appears on the first forward pass, and traffic must not
  arrive before that. Liveness deliberately does *not* depend on readiness — a
  server whose model failed to load is still alive, and reporting it as dead
  would have an orchestrator restart it into the same failure.
- **Generation runs on `spawn_blocking`.** A forward pass holds its thread for
  the whole step; running it inside a Tokio task blocks a runtime worker, and a
  runtime with fewer workers than concurrent requests stops answering health
  checks exactly when they are needed.
- **Cancellation is structural, not a flag.** The slot permit is moved into the
  blocking task, so capacity is released when the work ends — including when it
  is cancelled. In the streaming path the client's receiver *is* the signal: when
  it drops, `blocking_send` fails, the callback returns `false`, and the loop
  stops instead of computing tokens into a socket nobody reads. The contract is
  tested directly: two tokens accepted, the third refused, `StopReason::Cancelled`,
  and the refused token is not reported as produced.
- **Shutdown is a parameter, not a keypress.** `llm_server::serve` takes the
  shutdown future rather than calling `ctrl_c` internally, so stopping is
  something a test can do. The first attempt to verify it live used `taskkill`,
  which does not deliver `Ctrl+C`, so the drain message never appeared — the test
  method was wrong, not the code, and rather than assert it the design was
  changed until it could be tested. `the_server_answers_over_a_socket_and_stops_when_asked`
  binds a real port, makes a real HTTP request over a real socket, sends the
  signal and requires the task to finish.
- **Verified live**, not only in-process: the server started against
  `target/tmp/checkpoints/checkpoint-0000000100`, passed its self-test, answered
  `GET /health/live` and `GET /health/ready` with `200`, returned
  `{"text":" lototot","tokens":[224,79,82,87,82,87,82,87],"stop_reason":"max_new_tokens","prompt_tokens":10}`
  from `POST /v1/completions`, streamed SSE frames with
  `content-type: text/event-stream`, refused a negative temperature and an
  oversized token budget with `400` and a message naming the limit, and answered
  `429` to the second of two concurrent requests against a one-slot server. Two
  runs of the same request produced byte-identical output.
- `GET /health/live`, `GET /health/ready` and `POST /v1/completions` are the
  whole surface. No chat endpoint: a base pretraining model is not an instruction
  follower, and `09_INFERENCE_AND_SERVING.md` section 7 defers that until a chat
  template and a post-trained model exist.
- Unknown fields in a request body are accepted, unlike a configuration
  document. A wire boundary and an operator's file are different trust surfaces:
  an unknown key in a config means the operator believes a setting is in effect
  that is not, while a client sending one extra key is not making that mistake.

## M. Benchmark baseline

- [x] training tokens/sec;
- [x] peak memory;
- [x] prefill tokens/sec;
- [x] decode tokens/sec;
- [x] TTFT;
- [x] checkpoint read/write throughput;
- [x] cold-start model load;
- [x] artifact/binary size.

**Acceptance:** reproducible benchmark report with environment metadata.

**MET (2026-10-11).** All eight figures, with the environment beside them, in
`11_BENCHMARKING.md` section 15. `apps/bench` takes them:

```bash
cargo build --release -p bench
target/release/bench --config configs/stage1-110m.toml --micro-batch 8 --sequence 512
```

- **Peak memory is taken from outside the process.** Reading a peak working set
  needs an operating-system call and this workspace does not permit `unsafe`, so
  the binary does not measure it and does not claim to. The wrapper samples the
  operating system's own `PeakWorkingSet64`. The figure is labelled where it came
  from.
- **The shape is a parameter**, because the configured Stage-1 shape cannot be
  measured here: `batch 8 × sequence 2048` allocates a 1.6 GB attention score
  tensor and fails. A baseline that cannot be taken is not a baseline.
- **The training figure was wrong, and the measurement was at fault.** The first
  run reported 71 tokens/s for a shape `throughput` measures at 195–235. It was
  published as untrusted rather than buried, and the cause was found: the default
  was two timed steps, and the first step after warmup carries a one-time
  allocation — AdamW's moment buffers, 880 MB for Stage-1 — which at two steps was
  half the average. At five timed steps `bench` reports **192.3** against
  `throughput`'s **195** for the same shape, and generation before training makes
  no difference. The default is now five timed steps over two warmup steps, and
  the accepted figure is **210.9 tokens/s**.
- **Checkpoint writing varies by more than 4×** between runs — 292.9 against
  72.7 MiB/s on the same bytes — while reading does not move. That is the
  machine's background load, which `11_BENCHMARKING.md` section 12 already records
  as the reason cross-run comparisons are invalid here. The write figure should be
  quoted as a range until it is taken on a quiet machine or repeated.
- **Prefill and decode are separated**, as `09_INFERENCE_AND_SERVING.md` section 1
  requires and as section 4 of this checklist asks: 675.4 against 28.0 tokens/s.
  One inference figure would have hidden a factor of 24.

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
