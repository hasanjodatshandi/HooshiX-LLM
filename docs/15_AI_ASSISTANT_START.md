# AI Coding Assistant Handoff — Start Here

Use this file as the controlling implementation instruction for an AI coding assistant.

---

## Mission

Implement **Stage 1** of the Rust-native LLM project described in this documentation set.

The end goal of the overall project is a GPT/Qwen/DeepSeek-style LLM stack. Your current scope is **only Stage 1: a correct 100M–150M-class dense decoder-only causal language model and its Rust-native training/inference pipeline**.

Do not start Stage 2 features.

## Non-negotiable constraints

1. All project/application source code must be Rust.
2. Do not introduce Python, PyTorch, TensorFlow, JAX, Java or JVM tooling.
3. Use **Burn 0.22** as the primary tensor/autodiff/training framework.
4. Burn 0.22 uses runtime `Device` selection. Do not write obsolete Burn 0.21-style backend-generic model code.
5. Rust must satisfy Burn's MSRV: `rustc >= 1.95`.
6. Use Rust `tokenizers` for tokenizer functionality.
7. Use Burn's current checkpoint/storage APIs. Do not copy obsolete `Recorder` examples from old Burn versions.
8. Do not add direct CubeCL custom kernels until profiling proves a missing/slow operation and a reference Burn implementation already exists.
9. Do not introduce MoE, GQA, RLHF, DPO, GRPO, speculative decoding, paged attention or multi-node training in Stage 1.
10. Do not redesign the five-stage roadmap.
11. Keep changes minimal and domain-separated according to `04_REPOSITORY_STRUCTURE.md`.
12. Treat dataset/checkpoint/request input as untrusted.
13. Never silently ignore NaN/Inf, corrupt checkpoints, tokenizer mismatch or data corruption.
14. Do not claim model quality comparable to ChatGPT/Claude/DeepSeek. Stage 1 proves the stack.

## Required reading order

Before editing code in a fresh session/context, read:

1. `AGENTS.md`
2. `00_README.md`
3. `01_PROJECT_CHARTER.md`
4. `02_ROADMAP.md`
5. `03_ARCHITECTURE.md`
6. `18_RUST_STANDARDS_ROUTER.md`
7. `19_RUST_CODING_STANDARDS.md`
8. `20_RUST_ARCHITECTURE_AND_DESIGN_PATTERNS.md`
9. `21_RUST_RUNTIME_CONCURRENCY_ERROR_STANDARDS.md`
10. `23_RUST_TESTING_QUALITY_DELIVERY_STANDARDS.md`
11. `05_MODEL_SPEC.md`
12. `06_TOKENIZER_AND_DATA.md`
13. `07_TRAINING_AND_CHECKPOINTS.md`
14. `10_TESTING_AND_VALIDATION.md`
15. `13_DEPENDENCIES_AND_TOOLCHAIN.md`
16. `14_STAGE1_IMPLEMENTATION_CHECKLIST.md`
17. `16_DECISIONS.md`

Read `22_RUST_API_OBSERVABILITY_SECURITY_STANDARDS.md` when server/API/telemetry/security work begins. Use `18_RUST_STANDARDS_ROUTER.md` for later tasks instead of blindly rereading every standard.

`24_JAVA_TO_RUST_STANDARDS_MAPPING.md` explains which Java/Spring concepts were adapted and which infrastructure concerns were intentionally omitted.

If the repository already contains code, inspect the current implementation and test state before modifying anything. Continue from actual state; do not regenerate working code unnecessarily.


## Mandatory Rust engineering discipline

The engineering standards are implementation constraints, not suggestions. In particular:

- use explicit constructor/composition wiring; no DI container or service locator;
- use traits only at real replaceable/open boundaries; no interface-for-every-struct pattern;
- prefer enums for closed strategies/state machines;
- keep HTTP/CLI/filesystem concerns out of model math;
- prefer ownership/borrowing and immutable sharing over `Arc<Mutex<_>>`;
- keep async at I/O/orchestration boundaries; do not make tensor/model math async by convention;
- use typed errors and translate them at crate/interface boundaries;
- production paths must not use panic/unwrap as error handling;
- no database, Redis, Kafka, ORM, migrations, gRPC, or frontend abstractions unless a future explicit requirement adds them;
- no custom kernels/unsafe/performance rewrites without reference tests and benchmark evidence.

## Stage-1 architecture

Implement the baseline model:

```text
decoder-only Transformer
12 layers
hidden size 768
12 attention heads
64 head dimension
MHA
SwiGLU FFN (~2048 intermediate)
RMSNorm
RoPE
2048 context
32768 vocabulary
tied embedding / LM head
causal next-token loss
```

The exact parameter count must be calculated by the program and reported. The target is approximately 110M and must remain within the Stage-1 100M–150M range unless a correctness constraint requires a small adjustment.

## Execution order

Follow `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` in order.

Do not begin the full-size 110M training run first. Use this progression:

```text
compile
→ unit/reference tests
→ tiny forward pass
→ gradient checks
→ tiny-model overfit
→ checkpoint/resume
→ inference/KV-cache
→ CPU/GPU parity
→ Stage-1 smoke training
→ benchmark
→ only then full Stage-1 training configuration
```

## First implementation task

Start with workspace foundation and configuration only, then immediately add tests.

Create the Cargo workspace and minimal crates described in `04_REPOSITORY_STRUCTURE.md`, but do not create empty abstraction crates that have no immediate implementation.

Initial practical subset:

```text
crates/llm-core
crates/llm-tokenizer
crates/llm-data
crates/llm-model
crates/llm-training
crates/llm-inference
apps/train
apps/generate
```

Add server/eval crates when their checklist sections are reached.

## Burn version discipline

Use current Burn **0.22** documentation as authoritative.

Important 0.22 facts:

- model structs no longer need the older backend generic parameter pattern;
- execution backends are selected through `Device`;
- no execution backend is enabled by default;
- old recorder/checkpoint APIs changed;
- Rust 1.95 is the MSRV.

When a Burn API is uncertain, verify the official Burn 0.22 docs/source before implementing. Do not guess APIs from memory.

## Backend strategy

Start with a CPU/reference-capable backend for correctness and one practical development accelerator backend.

Keep backend selection behind Cargo features/runtime wiring. Model code stays backend-agnostic.

For NVIDIA training, add CUDA support after the reference path compiles/tests. Verify the environment with a minimal tensor operation before running the model.

## Testing rule

Every mathematical primitive is implemented with a test in the same change.

Critical release-blocking tests:

- causal masking;
- RoPE offset/cached position behavior;
- RMSNorm reference;
- loss shift;
- tiny-model overfit;
- checkpoint resume;
- KV-cache equivalence;
- CPU/GPU parity.

Do not postpone these until after the model is large.

## Error-handling rule

Use typed errors at library boundaries.

Do not use `unwrap()`/`expect()` on:

- user files;
- dataset records;
- checkpoints;
- server input;
- CLI-provided paths.

`expect()` is acceptable only for a true internal invariant whose violation is a programming error and whose message explains the invariant.

## Performance rule

No speculative optimization.

For any optimization:

1. preserve a reference path;
2. run correctness tests;
3. profile;
4. record benchmark before;
5. implement;
6. record benchmark after;
7. keep only if the target metric improves.

## Checkpoint rule

A “checkpoint” is not complete if it only contains model weights.

Resume must restore:

- model;
- optimizer;
- scheduler;
- global step;
- token count;
- relevant data cursor/sampler state;
- configuration/artifact identity.

Test process restart, not just in-process serialization.

## Data rule

Do not download or assume a massive dataset automatically.

Implement the data interface and deterministic fixture first. For any real corpus, require documented source/license/provenance in the dataset manifest.

## Commands required before completing each coherent implementation slice

At minimum:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Run backend-specific tests when that backend is modified and hardware is available.

## Completion reporting

For each assignment, report only after implementation and validation, using:

- Completed
- Result
- Validation
- Problems Resolved
- Remaining Issues
- Suggestions — only meaningful out-of-scope improvements
- Self-Review — bugs, edge cases, performance and security

Do not claim success when tests were not run. If a hardware-specific test cannot run because the hardware/runtime is unavailable, state exactly which validation remains unexecuted.

## Stop condition

For the current project phase, stop when all Stage-1 acceptance criteria in `10_TESTING_AND_VALIDATION.md` and `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` are met.

Do not proceed into Stage 2 without an explicit instruction.
