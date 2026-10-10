# Java/Spring Engineering Standards → Rust LLM Mapping

This document records how the general Java/Spring engineering-standard concepts were translated for this Rust-native LLM project. It prevents future agents from copying irrelevant enterprise-Java infrastructure into the Rust stack.

## Direct conceptual mappings

| Java/Spring concept | Rust LLM equivalent |
|---|---|
| Clean Architecture dependency direction | Cargo crate dependency direction + explicit inner/outer ownership |
| Spring constructor injection | Ordinary Rust constructors + explicit composition in `apps/*` |
| Spring IoC container | **Not used**; no DI container |
| Java interface at external port | Rust trait only at real replaceable/open boundary |
| DTO/request mapping | Axum transport type → internal inference command/options |
| Domain invariants | Model/data/training invariants encoded by Rust types/config validation |
| Java records/value objects | Rust structs/newtypes/enums |
| Service state machine | Rust enum/state machine |
| try/finally resource cleanup | RAII/Drop |
| Java checked/runtime exception model | Typed `Result<T, E>` + `thiserror` |
| Virtual-thread concurrency | Tokio I/O + bounded explicit compute scheduling |
| synchronized/locks | ownership first; narrow Mutex/RwLock only when necessary |
| Gradle wrapper/dependency locks | Cargo workspace + committed `Cargo.lock` |
| Spotless | rustfmt |
| SpotBugs | rustc + Clippy + focused tests |
| ArchUnit | Cargo crate graph + dependency checks/review |
| JUnit | `cargo test` |
| Micrometer structured instrumentation | `tracing`; metrics/export added only when needed |
| Framework configuration properties | Serde/TOML typed config + startup `validate()` |

## Deliberately not mapped because absent from the project

### Database / ORM / Flyway

Not present.

Do not create:

- repository abstractions for SQL;
- persistence entities;
- transactions;
- migrations;
- ORM layer;
- database connection pool.

Checkpoint files/model artifacts are not a relational persistence layer.

### Redis/cache tier

Not present.

KV cache is **Transformer inference state**, not Redis/application cache. Do not conflate them.

No Redis client, distributed cache abstraction, cache-aside pattern, or session store should be added.

### Kafka/messaging/outbox/inbox

Not present.

The current system has no message broker. Internal training/inference queues are in-process bounded concurrency primitives, not Kafka topics/events.

### gRPC/Protobuf

Not part of Stage 1.

Do not add an RPC contract merely because the Java baseline uses gRPC internally.

### Frontend/browser stack

Not present.

### Spring Security/session architecture

Not present in the Stage-1 local inference server. Public authentication/authorization requires a later explicit security design if the server becomes multi-user/public.

## Concepts adapted rather than copied

### DDD

The LLM project has mathematical/scientific domains rather than a conventional business aggregate model.

Use domain thinking for:

- tensor/model invariants;
- token/sequence semantics;
- training state;
- checkpoint identity;
- generation state.

Do not invent entities/aggregates/repositories merely to label the design “DDD”.

### Hexagonal Architecture

Use ports/adapters only around real outer capabilities:

```text
HTTP
filesystem/artifact store
dataset source
backend/runtime initialization
```

Core model math should remain direct and readable, not hidden behind unnecessary trait layers.

### Observability

The Java standard's Micrometer/OpenTelemetry specifics are not copied. The Rust baseline uses `tracing`. Exporter/metrics crates are conditional additions when operational deployment requires them.

### Resilience

No Resilience4j equivalent is added by default. Bounded queues, cancellation, explicit retry ownership and failure semantics are implemented with Rust/Tokio primitives appropriate to the actual dependency.

## Rule for future additions

If a future stage introduces a database, Redis, broker, remote object store, public auth, Kubernetes deployment, or another major infrastructure concern:

1. treat it as scope expansion;
2. add/update an ADR;
3. define a dedicated Rust-specific standard for that concern;
4. add only the dependencies and patterns actually required.

Never preload architecture for infrastructure the project does not have.
