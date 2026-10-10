# Rust Runtime, Concurrency, Error, and Resource Standards

## 1. Runtime principle

Tokio orchestrates I/O and request lifecycles. It is **not** the tensor-compute abstraction.

Model execution remains a synchronous/explicit computational boundary unless Burn's backend API itself requires async behavior.

Do not make every model method `async` just because the server is async.

## 2. Async boundary

Async is appropriate for:

- HTTP server I/O;
- streaming responses;
- async file/network operations where supported and beneficial;
- cancellation/deadline orchestration;
- bounded request queues.

Heavy CPU/GPU work must not monopolize Tokio core worker threads.

Use a bounded model-execution scheduler/worker boundary appropriate to the backend. Do not spawn one blocking worker per generated token.

## 3. Task ownership

Every spawned task has:

- an owner;
- start condition;
- cancellation behavior;
- timeout/deadline if applicable;
- error propagation path;
- shutdown behavior.

Detached fire-and-forget tasks are prohibited unless loss is explicitly harmless and documented.

Join/background task failures must become observable.

## 4. Bounded concurrency

Unbounded concurrency is prohibited.

The server defines:

- maximum in-flight generation requests;
- queue capacity;
- saturation behavior;
- per-request token/context bounds;
- cancellation handling.

Do not use unbounded channels for request or work queues.

## 5. Locks

Prefer ownership/message passing over locks.

When a lock is necessary:

- keep the protected state small;
- keep critical sections short;
- never perform long model compute while holding a general-purpose mutex;
- never hold a synchronous mutex guard across `.await`;
- use an async mutex only when the resource truly requires an async-held guard;
- define lock ordering if more than one lock can be held.

Avoid global locks around the entire inference engine.

## 6. `Arc`

Use `Arc<T>` for shared ownership across tasks/threads when lifetime semantics require it.

`Arc` does not make `T` internally thread-safe; `T` must satisfy the required `Send`/`Sync` semantics.

Prefer `Arc<ImmutableModelState>` over `Arc<Mutex<ModelState>>` when model state can be immutable after load.

## 7. Atomics

Use atomics only for genuinely atomic state/counters where locks would be inappropriate.

Document the correctness reason for memory ordering when it is not obvious. Do not use atomics to build a custom lock-free subsystem without a demonstrated requirement and focused concurrency tests.

## 8. Cancellation

Cancellation is a normal control path, not an error to hide.

Inference cancellation must:

- stop future decode iterations as soon as practical;
- release per-request KV/cache/buffer state;
- not corrupt shared model state;
- distinguish cancelled from failed in telemetry.

Training interruption must checkpoint only at a consistency boundary defined by the training/checkpoint specification.

## 9. Timeouts/deadlines

External/HTTP operations must have finite bounds where waiting indefinitely would consume resources.

Do not place arbitrary micro-timeouts around GPU kernels merely to “have a timeout”; backend cancellation semantics must be understood first.

## 10. Error taxonomy

Each library crate owns a focused error enum.

Examples:

```text
ModelError
DataError
CheckpointError
TrainingError
InferenceError
ServerError
```

Errors are translated at boundaries:

```text
Burn/backend error
→ Model/Inference error with context
→ application error
→ HTTP/CLI representation
```

Do not leak backend driver details or filesystem internals to public HTTP clients.

## 11. Error context

A good internal error states:

- operation attempted;
- stable identifier/path category where safe;
- source error;
- relevant non-sensitive configuration dimension.

Do not include:

- full prompts;
- raw dataset records;
- model tensors;
- API secrets;
- arbitrary request bodies.

## 12. Retry

Retry only if the failure is plausibly transient and the operation is safe to repeat.

Examples potentially retryable only after analysis:

- remote artifact fetch in a future feature;
- transient file/object-store operation.

Examples not automatically retryable:

- invalid checkpoint;
- tokenizer mismatch;
- NaN/Inf loss;
- invalid config;
- unsupported backend/dtype;
- deterministic OOM with unchanged workload.

One layer owns retry. No nested retry multiplication.

## 13. Resource ownership

Every resource has an explicit owner and lifetime:

- model weights/device;
- tokenizer;
- dataset file handles;
- checkpoint temp files;
- per-request KV cache;
- queue permits;
- tracing spans/guards.

RAII cleanup must work on success, error, and cancellation.

## 14. Memory behavior

For LLM code, memory is a correctness constraint.

Required practices:

- validate sizes before allocation;
- use checked arithmetic for externally influenced size calculations;
- bound vectors/queues/caches;
- avoid retaining prompt/token buffers beyond request lifetime;
- release request KV state on completion/cancellation;
- treat allocator/GPU OOM as explicit failure, not an infinite retry condition.

## 15. Arithmetic safety

Dimensions and byte-size calculations influenced by config/input should use checked conversions/arithmetic when overflow is possible.

Do not silently cast from larger signed/unsigned types when truncation could alter tensor shape or allocation size.

## 16. Global state

No hidden mutable global state.

Allowed globals are limited to immutable constants or carefully initialized immutable process-level objects when their lifecycle is explicit.

The server model runtime should normally be constructed in the composition root and passed/shared explicitly.
