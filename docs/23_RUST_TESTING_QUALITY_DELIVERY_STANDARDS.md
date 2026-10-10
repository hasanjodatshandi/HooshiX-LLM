# Rust Testing, Quality, Git, and AI Delivery Standards

## 1. Testing is part of implementation

Applicable test categories:

- unit/reference tests;
- invariant/property tests;
- integration tests;
- regression tests;
- negative/error-path tests;
- concurrency tests;
- backend parity tests;
- checkpoint recovery tests;
- performance/load tests.

Tests verify behavior and invariants, not private implementation trivia.

## 2. Minimum local validation

For ordinary code changes:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --doc --workspace
```

Do not force `--all-features` if backend features are mutually exclusive or require unavailable hardware/toolchains. CI should test the supported feature/backend matrix explicitly.

## 3. Backend feature matrix

At minimum maintain a CPU/reference job that can run without accelerator hardware.

Accelerator jobs run when available for the changed backend:

```text
CPU/reference → required general correctness
CUDA         → NVIDIA-specific compile/parity/runtime tests
WGPU         → portable backend compile/parity where supported
```

A CUDA test unavailable because no GPU exists is `Not run`, not `Passed`.

## 4. Clippy policy

Warnings are errors in project code.

Do not enable every `clippy::pedantic` lint globally by default. Use high-signal lints and add targeted lints when they prevent a real defect class.

Do not silence a lint with broad `allow` attributes unless:

- the lint is inapplicable to that exact location;
- the reason is documented when non-obvious;
- the narrowest possible scope is used.

## 5. Format policy

Use rustfmt. Do not maintain a custom formatting dialect unless a concrete repository need exists.

Repository-wide formatting is not mixed into unrelated feature changes.

## 6. Architecture checks

Cargo crate boundaries are the primary architecture enforcement mechanism.

CI/review must reject dependency-direction violations such as:

```text
llm-model -> llm-server
llm-core -> llm-training
llm-training -> axum
```

Do not solve cycles by moving unrelated implementation into `llm-core`.

A dedicated architecture-check tool/script may be added only if Cargo boundaries alone cannot enforce an important rule.

## 7. Mathematical correctness tests

Every new/changed numerical primitive needs a reference/invariant test before optimization.

For optimized/custom backend paths:

```text
reference implementation
+ randomized representative shapes
+ dtype coverage
+ backend equivalence tolerance
+ benchmark
```

Performance is not evidence of correctness.

## 8. Property/invariant testing

Prefer deterministic generated test cases when they are enough.

Add a property-testing dependency only when it materially improves defect detection beyond a small internal deterministic generator.

Useful properties:

- causal mask never exposes future positions;
- sequence packer preserves token order/document boundaries;
- top-k output support never contains excluded tokens;
- checkpoint encode/decode is round-trip compatible;
- KV-cached and uncached decode agree within tolerance.

## 9. Failure-path testing

Cover applicable:

- invalid config;
- empty/oversized input;
- invalid sampling parameters;
- corrupt/mismatched artifact;
- tokenizer hash mismatch;
- cancellation;
- queue saturation;
- process interruption during checkpoint;
- NaN/Inf;
- unsupported dtype/backend;
- OOM/error propagation where testable;
- partial/corrupt dataset shard.

## 10. Performance gates

Benchmark changes to hot paths.

A performance optimization is accepted only if:

- correctness suite remains green;
- benchmark method is reproducible;
- target metric improves on target hardware;
- memory/latency tradeoff is documented;
- complexity is proportional to benefit.

## 11. Dependency changes

Before adding/updating a dependency:

1. identify why existing std/Burn/current crates are insufficient;
2. inspect maintenance/license/security/source;
3. update workspace manifest/lockfile deliberately;
4. compile relevant targets;
5. run tests;
6. run advisory/license/source checks configured for the repository;
7. inspect the final diff for transitive growth/version drift.

Do not upgrade unrelated dependencies in the same change.

## 12. Git safety

Before modifications when Git exists, inspect:

```bash
git status
git branch --show-current
git diff
git log --oneline --decorate -n 10
```

Preserve user/uncommitted work.

Do not run destructive operations such as reset/clean/force push/rebase over user work without explicit authorization.

## 13. Commit/change discipline

One coherent change per commit/PR where practical.

Do not combine:

- feature work;
- unrelated refactor;
- repository-wide formatting;
- dependency upgrade;
- architecture redesign.

Commit messages should state the logical change, not “update/fix stuff/final”.

## 14. AI coding assistant rules

An AI assistant must:

1. inspect existing code and tests first;
2. trace the affected execution path;
3. read the smallest sufficient standards set via `18_RUST_STANDARDS_ROUTER.md`;
4. preserve the five-stage roadmap and ADRs;
5. make the smallest complete safe change;
6. avoid speculative abstraction;
7. preserve user changes;
8. validate assumptions against repository/tool state;
9. use current Burn 0.22 docs when APIs are uncertain;
10. implement success and failure handling;
11. add tests with implementation;
12. run relevant format/lint/build/test/benchmark checks;
13. diagnose and fix task-caused failures rather than stop after first failure;
14. never weaken tests/lints/security to obtain green status;
15. report `Not run` for checks that could not execute;
16. inspect final diff before completion.

## 15. AI prohibited behavior

Do not:

- regenerate working modules unnecessarily;
- create empty speculative crates/directories;
- add a trait for every concrete type;
- add DB/Redis/Kafka abstractions not in scope;
- hide errors with `unwrap`/ignored `Result`;
- use clone/Arc/Mutex as a default ownership workaround;
- expose backend/HTTP types across wrong crate boundaries;
- write direct CubeCL kernels before profiling/reference implementation;
- add `unsafe` for convenience;
- bypass checkpoint integrity/resume semantics;
- claim a test/benchmark passed if it did not run.

## 16. Definition of done

A change is complete only when applicable criteria are satisfied:

```text
requirement satisfied
+ architecture preserved
+ code formatted
+ clippy clean
+ compile/tests pass
+ failure paths tested
+ numerical invariants verified
+ security/resource bounds preserved
+ benchmark run when performance-sensitive
+ final diff reviewed
+ docs/ADR updated when contract/architecture changed
```
