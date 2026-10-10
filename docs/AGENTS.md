# AGENTS.md — Rust-Native LLM Engineering Entry Point

This repository is governed by the Rust-native LLM engineering standards in this documentation set.

These rules are mandatory for design, implementation, refactoring, review, testing, performance work, security, and source organization. They are not optional style advice.

## Start of every fresh coding context

Before modifying code:

1. Read `18_RUST_STANDARDS_ROUTER.md`.
2. Use its router to load the smallest sufficient standards set for the task.
3. Read the project-specific specification for the affected subsystem.
4. Inspect the current repository, Git state, neighboring code, tests, and execution flow.
5. Make the smallest complete change that satisfies the current stage and requirement.

Do not load every document blindly. Do not rely on memory when the exact rule is available in the repository.

## Fixed project constraints

- Project/application/model/training/inference source is Rust.
- Burn 0.22 is the primary ML framework.
- Python/PyTorch/TensorFlow/JAX/Java/JVM are not part of the project runtime or training stack.
- The roadmap has exactly five stages. Do not invent a sixth stage or move later-stage features into Stage 1.
- Stage 1 is a dense decoder-only 100M–150M model. No MoE, GQA, RL, speculative decoding, paged attention, or multi-node training.
- No database, Redis, Kafka, ORM, migration system, or distributed cache exists in the current architecture. Do not add them unless an explicit future requirement and ADR introduces them.
- Do not build a custom tensor framework. Burn owns tensor/autodiff/runtime behavior unless a measured gap requires an isolated extension.

## Mandatory implementation discipline

- Inspect before editing.
- Preserve dependency direction.
- Keep model mathematics out of HTTP/CLI/filesystem orchestration.
- Keep HTTP/CLI/configuration out of the model crate.
- Use explicit constructors/composition. No service locator or DI container.
- Create traits only at real substitution/testing/platform boundaries; do not create “interface for every struct”.
- Prefer enums for closed strategy sets and traits for genuinely open boundaries.
- Prefer ownership/borrowing over shared mutable state.
- Prefer RAII over manual cleanup.
- Validate untrusted input at the boundary.
- Use typed errors and preserve source context.
- Do not use `unwrap`, `expect`, `panic!`, `todo!`, or `unimplemented!` on production paths reachable by untrusted/runtime input.
- Do not add dependencies or abstractions speculatively.
- Do not optimize before correctness and a benchmark baseline.
- Do not introduce `unsafe` into ordinary project crates. Any approved low-level `unsafe` must be isolated, documented, tested, and benchmark-justified.

## Implementation direction

For this project, inside-out means:

```text
scientific/model invariants
→ pure model primitives
→ training/inference application logic
→ data/checkpoint/runtime adapters
→ CLI/HTTP interfaces
→ composition/bootstrap
→ integration/system validation
```

This is the Rust/ML equivalent of clean dependency direction. Do not mechanically copy Java/Spring layer names into the repository.

## Validation before completion

Run applicable checks and report only what actually ran:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --doc --workspace
```

Run backend-specific and performance checks when the changed area requires them. A check that did not run is `Not run`, never `Passed`.

Inspect the final diff for:

- scope creep;
- accidental dependency/version changes;
- architecture boundary violations;
- leaked prompts/data/secrets;
- unnecessary clones/allocations;
- unbounded queues/tasks;
- ignored errors;
- unrelated user changes.

## Scope expansion

If a task starts touching a new concern—e.g. HTTP exposure, custom kernels, distributed execution, artifact format compatibility, authentication, or deployment—stop changing that new area until the corresponding standards/specification is loaded.

Database, Redis, Kafka, ORM and migration standards are intentionally absent because those technologies are not part of the current system.
