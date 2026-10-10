# Rust Architecture and Design Patterns

## 1. Architecture model for this ML project

Do not mechanically reproduce a Java/Spring DDD package tree. This project is a numerical/ML system, so its clean architecture is expressed by **scientific ownership and runtime boundaries**.

Dependency direction:

```text
apps / CLI / HTTP server
        ↓
training / inference / evaluation orchestration
        ↓
model + tokenizer/data contracts
        ↓
llm-core configuration/value types
        ↓
Burn + foundational Rust crates
```

External/platform concerns enter through adapters at the boundary:

```text
filesystem / HTTP / GPU runtime / artifact format
                ↓
             adapters
                ↓
       application orchestration
                ↓
          model invariants
```

Important exception: `llm-model` legitimately depends on Burn because Burn is the project's tensor/autodiff abstraction, not an HTTP/database infrastructure concern.

## 2. Crate responsibilities

### `llm-core`

Owns:

- shared value/config types;
- validated identifiers/units where cross-crate;
- common error-independent primitives;
- experiment/run manifest types.

Must not become a miscellaneous dumping ground.

### `llm-model`

Owns:

- RMSNorm;
- RoPE;
- causal attention;
- SwiGLU;
- Transformer blocks/model;
- model config invariants;
- causal LM loss/math.

Must not own:

- HTTP;
- CLI parsing;
- filesystem paths/checkpoint directory policy;
- dataset acquisition;
- request queues;
- logging/export infrastructure.

### `llm-data`

Owns corpus parsing, normalization, filtering, tokenized shard reading and packing.

External source formats are adapters. Sequence packing semantics are domain logic for the training pipeline.

### `llm-training`

Owns training use-case orchestration:

```text
batch → forward → loss → backward → optimizer → scheduler → metrics/checkpoint trigger
```

It does not implement HTTP or dataset-specific parsing.

### `llm-inference`

Owns:

- prefill/decode orchestration;
- KV-cache lifecycle;
- sampling;
- generation limits/state.

It does not know Axum request/response types.

### `llm-server`

Inbound HTTP adapter only:

- request validation;
- limit enforcement;
- mapping to inference commands/options;
- streaming/cancellation;
- public error mapping.

No Transformer math or checkpoint implementation lives here.

### `apps/*`

Composition roots:

- parse CLI/config;
- construct `Device` and concrete components;
- wire dependencies;
- start requested use case/server;
- choose exit code.

No reusable domain/model logic belongs in `main.rs`.

## 3. Dependency inversion in Rust

Do not introduce a Java-style DI container.

Use explicit constructors and ordinary ownership:

```rust
let model = Model::new(model_config, &device)?;
let store = BurnCheckpointStore::new(checkpoint_dir)?;
let trainer = Trainer::new(model, store, training_config)?;
```

Use traits only where the inner layer must remain independent of a concrete outer capability.

Likely boundaries, only when needed:

- checkpoint storage;
- data/batch source;
- clock/ID source if determinism/testing requires it;
- generation output sink/stream only if multiple transports need it.

Do not create a generic service-locator map or reflection-based DI system.

## 4. Pattern selection

Patterns are tools, not goals.

### Adapter Pattern — use

Use for translating outer representations into internal ones:

- JSONL/text corpus → internal document stream;
- SafeTensors/Burnpack → internal model state;
- Axum request → generation command;
- Burn backend/device details → runtime initialization.

### Strategy Pattern — prefer enum for closed strategies

Sampling is initially a closed set. Prefer:

```rust
pub enum SamplingStrategy {
    Greedy,
    Temperature { temperature: f32 },
    TopK { k: usize, temperature: f32 },
    TopP { p: f32, temperature: f32 },
}
```

Move to a trait only when third-party/open strategies are actually required.

### State Pattern — use explicit enums/state machines

Good candidates:

- model server lifecycle: `Loading → Ready → Draining → Stopped/Failed`;
- checkpoint publication: `Writing → Validating → Published`;
- generation request: `Prefill → Decode → Completed/Cancelled/Failed`.

Do not encode lifecycle in unrelated booleans such as `is_ready`, `is_failed`, `is_stopping` when combinations can become invalid.

### Newtype Pattern — selective

Use for semantically dangerous primitives:

- token IDs;
- global steps;
- sequence length;
- possibly byte/token limits.

Do not wrap every number merely for aesthetics.

### RAII — mandatory default

Resources are owned by values whose drop/lifetime controls cleanup:

- files;
- locks;
- memory mappings when introduced;
- temporary directories;
- tracing guards;
- runtime resources.

Avoid manual paired `open/close`, `lock/unlock`, or setup/cleanup APIs where RAII can express ownership.

### Builder Pattern — conditional

Use only when object construction has many optional ergonomic settings that cannot be represented cleanly by a validated config struct.

For experiment/model configuration, prefer typed config + `validate()` rather than a redundant builder.

### Typestate — exceptional

Typestate is allowed when compile-time lifecycle enforcement materially prevents a severe class of misuse. Do not use it for ordinary request/training state when an enum/state machine is clearer.

## 5. Patterns intentionally not adopted

Do not introduce without explicit need:

- Repository Pattern for a database — there is no database;
- Unit of Work — no transactional DB;
- Active Record/ORM models — no ORM;
- Cache-aside/Redis abstractions — no Redis;
- Kafka producer/consumer/outbox/inbox — no message broker;
- Spring-style service/repository/controller stereotypes;
- Abstract Factory hierarchies;
- custom event bus;
- plugin system;
- global service locator;
- singleton mutable registries.

## 6. Composition over inheritance

Rust has no class inheritance. Prefer composition of focused structs/traits.

Example:

```text
GenerationEngine
├── model runtime
├── KV cache
├── sampler config
└── limits
```

Do not simulate inheritance with deep trait-supertrait hierarchies unless the type relationship is real and necessary.

## 7. Model/backend separation

Model source should not branch everywhere on backend:

```rust
if cuda { ... }
else if wgpu { ... }
```

Backend-specific initialization belongs at composition/runtime boundaries. Use Burn's backend-agnostic model APIs where possible.

A genuinely backend-specific optimized path must:

- live behind a narrow boundary;
- retain a reference implementation;
- capability-check at startup/runtime;
- fall back or fail explicitly;
- have equivalence tests and benchmarks.

## 8. Cross-crate boundaries as architecture enforcement

Use Cargo crate dependencies to enforce architecture.

Prohibited examples:

- `llm-model -> llm-server`;
- `llm-model -> axum`;
- `llm-training -> llm-server`;
- `llm-core -> llm-training`;
- `llm-inference -> apps/*`.

Because Rust's crate graph is compile-time enforced, prefer real crate boundaries over documentation-only layering.

## 9. No circular dependencies

Cargo cycles are prohibited and usually impossible directly. Do not work around them by moving unrelated types into `llm-core`.

If two crates need each other, re-evaluate ownership and move only the genuinely shared inward contract to the correct owner.

## 10. Generated code

Generated backend/kernel artifacts are owned by the generating toolchain and are not hand-edited.

Handwritten wrappers/adapters live separately.

## 11. Architecture decision trigger

Create/update an ADR before changing:

- model normalization/order;
- attention family;
- position encoding;
- checkpoint/storage format contract;
- public HTTP contract;
- core crate dependency direction;
- async execution model;
- custom kernel/unsafe policy;
- adding a database/cache/broker;
- changing Burn major/minor architecture assumptions.
