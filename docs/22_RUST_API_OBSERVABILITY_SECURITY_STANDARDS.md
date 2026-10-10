# Rust API, Observability, and Security Standards

This document covers only concerns present in the current LLM system. Database, Redis, Kafka, ORM, migration, frontend, and browser-session rules are intentionally omitted.

## 1. HTTP adapter boundary

Axum handlers are inbound adapters.

They may:

- parse the request;
- validate transport fields and size limits;
- derive trusted server-side request context;
- map to an inference command/options type;
- invoke the inference service/engine;
- map the result/error to HTTP;
- stream output.

They must not:

- implement attention/sampling math;
- manipulate model weights;
- implement checkpoint storage;
- contain cross-request scheduling policy beyond adapter wiring;
- contain duplicated inference business rules.

## 2. API contract

Stage 1 exposes only the endpoints in `09_INFERENCE_AND_SERVING.md`.

Do not create chat-completions semantics before a chat template/post-trained model exists.

Every public request contract defines:

- required/optional fields;
- type/range constraints;
- prompt/token limits;
- output-token limit;
- timeout/cancellation behavior;
- stable error categories.

## 3. Default exposure

For local development, bind to loopback by default unless deployment configuration explicitly chooses another interface.

Do not expose an unauthenticated inference server publicly by accident.

Authentication/authorization is not part of the current Stage-1 local server. If public/multi-user deployment is introduced, that is a scope expansion requiring an explicit security/API design.

## 4. Public errors

Public responses must not expose:

- stack traces;
- local filesystem paths;
- GPU driver internals;
- checkpoint internals;
- secrets;
- raw dataset records;
- other users' prompts/output.

Map internal errors to stable public categories.

## 5. Input validation

Validate before expensive allocation/model execution:

- request body size;
- UTF-8/JSON validity;
- prompt byte/token length;
- `max_new_tokens`;
- sampling parameters;
- total context limit;
- concurrency admission.

Use checked arithmetic when combining prompt + output limits.

## 6. Path and artifact safety

CLI/server-provided file paths must not escape configured roots where a restricted root exists.

Model/tokenizer/checkpoint artifacts are untrusted until:

- format/version is recognized;
- hash/provenance requirements pass;
- dimensions/config match;
- deserialization completes safely.

Prefer non-executable tensor formats for portable artifacts.

## 7. Secrets

No secrets in:

- source;
- committed config;
- model/checkpoint metadata;
- dataset manifests intended for sharing;
- logs/traces/metrics;
- HTTP error responses.

## 8. `tracing` baseline

Use Rust `tracing` for structured application instrumentation.

Useful spans/events include:

```text
model_load
checkpoint_save
checkpoint_load
training_step (sampled/aggregated, not excessively verbose)
evaluation
inference_request
prefill
decode
```

Do not emit a log/span per generated token by default; it can create severe volume and overhead.

## 9. Logging

Structured logs should describe operations and identifiers, not content payloads.

Do not log raw:

- prompts;
- generated completions by default;
- training records;
- tokenizer text samples;
- secrets/tokens;
- giant error debug representations containing tensor data.

Log an error once at the layer with enough semantic context. Avoid every layer logging the same error.

## 10. Metrics/cardinality

When metrics are introduced, use low-cardinality dimensions.

Good dimensions:

- operation;
- outcome class;
- backend;
- dtype;
- model configuration name/version where bounded.

Bad labels:

- request ID;
- prompt hash if unbounded;
- raw URL;
- user ID;
- exception message;
- token text;
- arbitrary file path.

## 11. Performance telemetry

Measure separately:

- queue wait;
- prefill latency/tokens per second;
- time-to-first-token;
- decode inter-token latency/tokens per second;
- request total latency;
- training step time;
- checkpoint I/O;
- peak/allocated memory where available.

Instrumentation must not materially distort the measured hot path. Benchmark with telemetry configuration documented.

## 12. Dependency/supply-chain security

- commit `Cargo.lock`;
- centralize workspace dependencies;
- avoid git dependencies unless pinned to an immutable revision and justified;
- review licenses and source provenance;
- run Rust advisory scanning for releases/CI where available;
- scan repository/history for secrets;
- do not disable TLS verification in future remote-fetch code.

Use one tool per distinct failure class; do not add overlapping scanners only for tool count.

## 13. Denial-of-service controls

The server must have finite:

- body size;
- prompt/context length;
- generated tokens;
- in-flight request count;
- queue length;
- per-request lifetime according to the supported cancellation/deadline model.

Resource exhaustion must fail predictably, not grow memory without bound.
