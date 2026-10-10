# Inference and Serving

## 1. Inference phases

Autoregressive generation has two distinct phases:

```text
prompt
  ↓
prefill: process all prompt tokens
  ↓
initialize KV cache
  ↓
decode: one/few new token positions per iteration
  ↓
sampling
  ↓
stream output
```

Benchmarks must report prefill and decode separately.

## 2. KV cache

Required by Stage 1.

For each layer cache keys/values from prior positions.

Correctness invariant:

> Logits from cached incremental decoding must match logits from equivalent uncached decoding within the selected dtype/backend tolerance.

Cache must enforce:

- maximum context length;
- batch/request ownership;
- position offsets;
- reset/reuse semantics;
- bounded allocation.

## 3. Sampling

Stage 1 implements:

- greedy;
- temperature;
- top-k;
- top-p.

Sampling API must accept an explicit RNG seed for reproducible tests.

Order of transformations must be defined and tested.

Invalid settings fail before generation, for example:

- temperature < 0;
- top-k greater than vocabulary if unsupported;
- top-p outside valid range;
- max-new-tokens overflow/context violation.

## 4. Stop conditions

Support:

- EOS token;
- max new tokens;
- maximum context length;
- explicit cancellation;
- optional stop-token sequences in later work.

## 5. Server architecture

```text
HTTP
 ↓
Axum routing
 ↓
validation / limits
 ↓
request queue
 ↓
inference scheduler
 ↓
model state
 ↓
token stream
 ↓
SSE/streaming response
```

Tokio handles I/O orchestration. GPU execution must not be accidentally serialized or blocked by unrelated HTTP work.

## 6. Model lifecycle

Model loading is process initialization, not per request.

```text
process start
→ validate config/artifact hash
→ initialize Device
→ load tokenizer
→ load weights
→ warmup/self-test
→ mark READY
→ accept traffic
```

Do not expose readiness until the self-test passes.

## 7. API

Stage-1 minimal API:

```text
GET  /health/live
GET  /health/ready
POST /v1/completions
```

OpenAI-compatible chat endpoints are deferred until a chat template and post-trained model exist.

Do not pretend a base pretraining model is an instruction-following assistant.

## 8. Request limits

Validate before large allocation:

- prompt byte length;
- tokenized prompt length;
- max new tokens;
- batch size;
- timeout/deadline;
- concurrent requests.

## 9. Batching roadmap

Stage 1 may start with bounded simple batching.

Stage 3 introduces continuous batching if benchmark evidence supports it.

Correctness must precede scheduler complexity.

## 10. Quantization

Quantized inference is not required to prove Stage-1 correctness.

Stage 2/3 may use Burn post-training quantization on supported backends or a verified Rust-native path.

Every quantized artifact requires:

- accuracy/perplexity delta;
- memory delta;
- throughput/latency delta;
- exact quantization configuration.

## 11. Shutdown

Graceful shutdown:

- stop accepting new requests;
- allow/cancel in-flight work according to policy;
- release model/device resources;
- flush telemetry where applicable.
