# Security and Reliability

## 1. Trust boundaries

Treat as untrusted:

- raw datasets;
- tokenizer files not built by this project;
- model/checkpoint files from external sources;
- HTTP request bodies;
- generation parameters;
- file paths from CLI/user input;
- metadata embedded in artifacts.

## 2. Deserialization

Avoid executable object serialization.

For portable tensors prefer SafeTensors-style non-executable storage.

For Burn-native checkpoint storage:

- validate file structure;
- bind artifacts to model/tokenizer/config hashes;
- reject unexpected versions;
- surface errors instead of panicking.

## 3. Resource exhaustion

Before allocation validate:

- sequence length;
- requested output length;
- batch size;
- file size where possible;
- shard count;
- concurrent requests.

The server must implement bounded queues/backpressure.

## 4. Path safety

Never allow user-provided path traversal to escape configured data/checkpoint roots.

Canonicalize or constrain paths at CLI/server boundaries as appropriate.

## 5. Supply-chain policy

Before release:

- commit `Cargo.lock`;
- run dependency audit tooling;
- inspect newly introduced crates;
- avoid abandoned/high-risk dependencies where alternatives exist;
- minimize features/dependencies;
- document native/system dependencies.

## 6. Secrets

No API key, dataset credential or cloud secret in:

- source;
- config committed to Git;
- checkpoint metadata;
- logs.

Read secrets from environment/secret provider only where required.

## 7. Logs

Do not log raw private training records or full prompts by default.

Telemetry should prefer:

- lengths;
- hashes/IDs;
- timings;
- status codes;
- aggregate metrics.

## 8. Denial of service

Serving controls:

- maximum body size;
- token limits;
- concurrency limit;
- request timeout;
- cancellation;
- bounded streaming buffers.

## 9. Reliability

Training must fail loudly on:

- NaN/Inf loss;
- corrupt shard;
- tokenizer mismatch;
- checkpoint mismatch;
- persistent I/O error;
- unsupported accelerator operation.

It must not silently skip corrupt data unless a documented policy explicitly allows it and reports counts.

## 10. Data integrity

Every prepared dataset and model artifact must be hashable and versioned.

Before resuming, verify:

- model config;
- tokenizer;
- dataset identity;
- checkpoint version.

## 11. Unsafe code

Project-level `unsafe` requires a documented invariant and focused review. Prefer safe Burn/CubeCL APIs.

## 12. External model artifacts

Loading arbitrary externally downloaded weights is a separate trust decision. Hash and provenance must be recorded before use.

## Current infrastructure boundary

`22_RUST_API_OBSERVABILITY_SECURITY_STANDARDS.md` defines the Rust/Axum/tracing-specific rules.

The current project has no database, Redis, Kafka/message broker, ORM, migration layer, or browser authentication/session subsystem. Do not add security/persistence abstractions for technologies that do not exist. If any becomes a real requirement, add a scoped ADR and standard before implementation.
