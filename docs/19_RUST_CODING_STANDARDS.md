# Rust Coding Standards

## 1. Purpose

These rules define how Rust code in the LLM project should be written: readable, explicit, testable, resource-safe, performance-aware, and idiomatic without cleverness.

The objective is not “maximum abstraction”. The objective is the smallest correct design whose invariants are visible.

## 2. Core principles

Apply:

- high cohesion / low coupling;
- separation of concerns;
- explicit dependencies;
- dependency inversion at real external/substitution boundaries;
- composition over inheritance-style designs;
- DRY for duplicated knowledge/invariants, not duplicated syntax;
- KISS;
- YAGNI;
- fail fast on invalid configuration/artifacts;
- bounded resource use;
- deterministic behavior where practical;
- measure before optimizing.

Rust-specific principles:

- ownership expresses resource ownership;
- borrowing expresses temporary access;
- RAII owns cleanup;
- types should make invalid states harder to construct;
- `Result`/`Option` express fallibility/absence; avoid sentinels;
- module visibility is part of the architecture.

## 3. Visibility and public API

Default to the narrowest visibility:

```text
private → pub(crate) → pub
```

Use `pub` only for an intentional cross-crate API.

Rules:

- keep internal implementation private;
- do not expose Burn/backend-specific types from a crate unless the API genuinely represents tensor/model operations;
- do not expose Axum/Tokio HTTP types from model/training crates;
- avoid public fields when invariants require controlled construction/mutation;
- constructors validate invariants or return `Result` when construction can fail.

## 4. File and module responsibility

One file/module has one primary responsibility and reason to change.

Prefer precise names:

```text
rope.rs
rms_norm.rs
causal_mask.rs
checkpoint_manifest.rs
top_p.rs
sequence_packer.rs
```

Avoid dumping grounds:

```text
utils.rs
helpers.rs
common.rs
manager.rs
misc.rs
stuff.rs
```

A small private helper stays with its owner when it has no independent responsibility.

## 5. Naming

Follow Rust conventions:

- modules/functions/variables: `snake_case`;
- types/traits/enums: `UpperCamelCase`;
- constants/statics: `SCREAMING_SNAKE_CASE`;
- lifetime names: short only when obvious (`'a`), semantic when multiple lifetimes need distinction.

Names must state intent and units.

Prefer:

```rust
max_sequence_length
warmup_steps
tokens_per_step
checkpoint_interval_steps
```

For values where unit confusion can cause correctness bugs, prefer a newtype over suffix-only conventions.

## 6. Functions

A function should:

- perform one conceptual operation;
- use explicit inputs/outputs;
- keep side effects visible;
- stay at a consistent abstraction level;
- avoid boolean flags that switch unrelated behavior;
- avoid large unrelated parameter lists.

Do not split functions solely to meet a line-count target.

Prefer domain/config structs when a function has a coherent parameter group.

## 7. Structs and invariants

Make invalid states difficult to represent.

Use:

- private fields + validated constructor when values have invariants;
- enums for mutually exclusive states;
- newtypes when raw primitives can be confused;
- dedicated config validation before expensive allocation.

Example candidates:

```rust
struct TokenId(u32);
struct GlobalStep(u64);
struct SequenceLength(usize);
```

Do not wrap every primitive mechanically. Use newtypes where they prevent real category/unit errors.

## 8. Enums vs traits

Use an `enum` when the strategy set is closed and owned by this project.

Examples:

```text
SamplingMethod = Greedy | Temperature | TopK | TopP
BackendChoice = Cpu | Cuda | Wgpu
```

Use a `trait` when:

- multiple implementations are expected;
- the implementation belongs to an outer adapter;
- tests need a meaningful substitute;
- consumers should not know the concrete provider;
- the set is intentionally open.

Do not create a trait solely to wrap one struct “for clean architecture”.

## 9. Trait design

Traits are capability-focused and small.

Prefer:

```rust
trait CheckpointStore {
    fn save(&self, ... ) -> Result<..., CheckpointError>;
    fn load(&self, ... ) -> Result<..., CheckpointError>;
}
```

Avoid broad god traits such as:

```text
LlmManager
ModelService
SystemProvider
EverythingRepository
```

Define the trait in or near the consuming abstraction when practical, not automatically next to the implementation.

Avoid unnecessary trait-object dispatch in hot token/tensor loops. Static dispatch is appropriate when it materially helps hot-path performance and does not explode complexity.

## 10. Ownership and borrowing

Prefer:

- `&T` for shared read access;
- `&mut T` for exclusive mutation;
- owned `T` when ownership transfer is semantically correct;
- `Arc<T>` for genuinely shared cross-task/process-lifetime immutable or synchronized state.

Avoid using `clone()` to escape ownership design.

A clone of a large tensor, buffer, token sequence, string, or model state must be intentional. If cloning is non-obvious in cost, add a short comment or restructure ownership.

Prefer slices over owned containers when ownership is unnecessary:

```rust
&[TokenId]
&mut [f32]
&str
```

## 11. Interior mutability

Do not reach first for:

```text
Arc<Mutex<T>>
Arc<RwLock<T>>
Rc<RefCell<T>>
```

First ask whether ownership can be simpler.

Shared mutable state requires:

- one clear owner;
- documented synchronization strategy;
- bounded critical sections;
- concurrency tests when correctness depends on it.

## 12. Error propagation

Use `Result<T, E>` for recoverable failures.

Use `?` to preserve the normal error path.

Project library crates define typed errors, normally with `thiserror`.

Do not use strings as the only error taxonomy for meaningful failures.

Errors should identify the failed operation without embedding sensitive/raw large payloads.

## 13. Panic policy

Production paths reachable by user input, dataset content, artifact content, I/O, backend failure, or request parameters must not rely on:

```text
unwrap()
expect()
panic!()
todo!()
unimplemented!()
```

`expect()` is acceptable only for an internal invariant that is proven by construction and whose message states that invariant.

Tests may use `unwrap`/`expect` when failure should abort the test and the context is obvious.

## 14. Comments and documentation

Comments explain:

- why a non-obvious choice exists;
- mathematical invariants;
- ownership/synchronization invariants;
- numerical stability assumptions;
- backend limitations;
- safety conditions;
- benchmark-motivated tradeoffs.

Do not comment syntax.

Public APIs and non-obvious mathematical modules need Rustdoc describing inputs, outputs, invariants, errors, and examples where useful.

## 15. `unsafe`

Ordinary project crates should use:

```rust
#![deny(unsafe_code)]
```

when practical.

If a future low-level kernel/FFI path genuinely requires `unsafe`, isolate it in a narrow crate/module and require:

- explicit ADR/rationale;
- documented safety invariants;
- smallest possible unsafe block;
- tests around the safe wrapper;
- Miri/other applicable validation where possible;
- benchmark proof that the unsafe path is necessary.

Never introduce `unsafe` to save a small amount of boilerplate.

## 16. Macros and metaprogramming

Derive macros from established crates are acceptable.

Do not use custom macros to hide ordinary control flow, ownership, or error handling.

A custom macro must eliminate meaningful repeated structure without making debugging/type errors harder.

## 17. Generic programming

Use generics where they express a real type relationship or zero-cost reusable algorithm.

Avoid generic abstraction layers that exist only for hypothetical future backends/features.

Be aware that excessive monomorphization can increase compile time and binary size.

## 18. Allocation discipline

Especially in decode/training hot paths:

- reuse buffers where ownership permits;
- preallocate known capacities;
- avoid per-token `String` creation;
- avoid repeated collect/clone chains;
- avoid converting between `Vec<T>` and slices unnecessarily;
- measure before introducing pools/custom allocators.

Correctness remains higher priority than micro-optimization.

## 19. Serialization/config

Configuration is typed and startup-validated.

Do not read environment variables throughout model/training logic.

Persisted artifact formats require explicit version fields and compatibility rules.

For strict user-authored config, unknown fields should normally fail rather than silently ignore likely typos, unless forward-compatibility requirements explicitly require otherwise.

## 20. Dependency rule

Before adding a crate, verify that std/Burn/Tokio/current dependencies cannot reasonably solve the need.

A new dependency needs a specific responsibility; do not add “convenience crates” that duplicate a few maintainable lines of code.
