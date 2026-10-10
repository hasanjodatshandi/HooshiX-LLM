# Testing and Validation

## 1. Test hierarchy

```text
unit tests
  ↓
property/invariant tests
  ↓
component integration tests
  ↓
tiny-model end-to-end tests
  ↓
GPU parity tests
  ↓
training smoke test
  ↓
resume/recovery test
  ↓
serving/load test
```

## 2. Mandatory unit tests

### Model

- RMSNorm reference comparison;
- RoPE known-position/reference comparison;
- causal mask prevents future attention;
- attention shape tests;
- SwiGLU shape/reference tests;
- residual block shape;
- tied embedding identity/parameter-sharing invariant;
- loss shift correctness;
- padding loss-mask correctness.

### Tokenizer

- special token IDs;
- empty string;
- Unicode;
- encode/decode behavior;
- tokenizer hash mismatch rejection.

### Data

- deterministic packing;
- EOS/document boundaries;
- truncated/corrupt input handling;
- shard checksum verification.

### Sampling

- greedy argmax;
- top-k support restriction;
- top-p support restriction;
- deterministic seeded sampling;
- EOS termination.

## 3. Gradient validation

For a tiny tensor/model, compare selected autodiff gradients with finite differences where practical.

Check:

- gradient exists;
- finite values;
- expected zero/non-zero paths;
- no accidental detach.

## 4. Tiny overfit test

Create a tiny deterministic corpus.

A very small model must overfit it.

Failure indicates a fundamental issue in one or more of:

- target shift;
- masking;
- optimizer;
- gradient flow;
- parameter registration;
- model output.

This test is required before any large training run.

## 5. CPU/GPU parity

Run the same small model, fixed weights and fixed inputs on CPU reference and target GPU backend.

Compare:

- logits;
- loss;
- selected gradients.

Tolerance is dtype-specific and documented. Never require bitwise equality between different accelerator kernels.

## 6. Checkpoint tests

- save/load output equivalence;
- corrupt checkpoint rejection;
- wrong model config rejection;
- wrong tokenizer hash rejection;
- resume step state;
- optimizer state;
- scheduler state;
- interrupted write leaves previous good checkpoint intact.

## 7. KV-cache test

For a prompt plus generated continuation:

- compute logits with full uncached forward;
- compute incrementally with KV cache;
- compare each decode step.

This is a release-blocking inference test.

## 8. Determinism

Determinism modes:

- `strict-test`: deterministic test fixtures and seeded RNG;
- `training`: best-effort reproducibility, with non-deterministic accelerator operations documented.

Do not claim full numerical determinism across hardware/backend versions unless verified.

## 9. Static quality gates

Before merge:

```text
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Backend-specific test jobs should run separately where hardware is available.

## 10. Stage-1 acceptance test

A Stage-1 release candidate must demonstrate in one report:

- clean build;
- test pass;
- tiny overfit;
- real-corpus loss decrease;
- checkpoint resume;
- generation;
- KV-cache parity;
- CPU/GPU parity;
- performance numbers;
- memory numbers;
- dependency/security audit status.

## 11. Self-review checklist for each implementation change

- correctness regression?
- shape edge cases?
- empty/maximum input?
- dtype mismatch?
- device mismatch?
- memory growth?
- panic on user-controlled input?
- numerical instability?
- serialization compatibility?
- concurrency race?
- security/trust-boundary impact?

## Rust quality standard

`23_RUST_TESTING_QUALITY_DELIVERY_STANDARDS.md` is mandatory for implementation and CI behavior, including backend-specific feature-matrix validation, Clippy policy, Git safety, and AI completion rules.
