# Rust Engineering Standards Router

Use this file to decide which standards must be read before a coding task. Load the smallest sufficient set.

`AGENTS.md` is the mandatory entry point.

## Task router

| Task | Required documents |
|---|---|
| New module/crate/refactor/source move | `19`, `20`, `23` + `04_REPOSITORY_STRUCTURE.md` |
| Transformer/model math | `19`, `20`, `23` + `05_MODEL_SPEC.md` + `10_TESTING_AND_VALIDATION.md` |
| Tokenizer/data pipeline | `19`, `20`, `23` + `06_TOKENIZER_AND_DATA.md` + `12_SECURITY_AND_RELIABILITY.md` |
| Training/optimizer/checkpoint | `19`, `20`, `23` + `07_TRAINING_AND_CHECKPOINTS.md` + `10_TESTING_AND_VALIDATION.md` |
| Distributed/multi-GPU | `19`, `20`, `23` + `08_DISTRIBUTED_AND_SCALING.md` + `11_BENCHMARKING.md` |
| Inference/KV cache/sampling | `19`, `20`, `23` + `09_INFERENCE_AND_SERVING.md` + `10_TESTING_AND_VALIDATION.md` |
| HTTP/Axum serving | `19`, `20`, `21`, `23` + `09_INFERENCE_AND_SERVING.md` + `12_SECURITY_AND_RELIABILITY.md` |
| Async/concurrency/resource ownership | `20`, `23` + affected subsystem specification |
| Logging/tracing/telemetry | `21`, `23` + affected subsystem specification |
| Security/dependency/artifact handling | `21`, `23` + `12_SECURITY_AND_RELIABILITY.md` + `13_DEPENDENCIES_AND_TOOLCHAIN.md` |
| Dependency/toolchain change | `19`, `23` + `13_DEPENDENCIES_AND_TOOLCHAIN.md` |
| Performance/custom kernel | `19`, `20`, `23` + `11_BENCHMARKING.md`; add `03_ARCHITECTURE.md` and current Burn/CubeCL docs |
| Git/delivery/AI coding behavior | `23` |

## Conditional rules

Load `21_RUST_API_OBSERVABILITY_SECURITY_STANDARDS.md` only when the change touches server/API/telemetry/security boundaries.

Load distributed documentation only when multi-device/process behavior is in scope.

Load current official Burn/CubeCL documentation whenever an API or backend behavior is version-sensitive.

## Intentionally omitted standards

The current architecture has no:

- relational database;
- ORM;
- schema migration layer;
- Redis/cache tier;
- Kafka/message broker;
- transactional outbox/inbox;
- frontend/UI.

Do not create standards, abstractions, ports, crates, or dependencies for these absent concerns.

If one is introduced by an explicit future requirement, add a scoped ADR and a dedicated standard at that time.

## Precedence

When guidance conflicts, use this order:

1. Current verified repository/runtime behavior.
2. Current explicit user/project requirement.
3. Architecture decisions in `16_DECISIONS.md`.
4. Stage-specific project specifications.
5. Rust engineering standards (`19`–`23`).
6. Generic preferences.

Do not use a generic clean-architecture rule to break an intentional ML-specific design.
