# Architecture

## 1. System overview

```text
Raw corpus
   │
   ▼
Data ingestion ──> validation/provenance/dedup
   │
   ▼
Tokenizer training/loading
   │
   ▼
Token stream
   │
   ▼
Sequence packing / batches
   │
   ▼
┌─────────────────────────────────────────┐
│ Decoder-only Transformer                │
│                                         │
│ Embedding                               │
│   ↓                                     │
│ TransformerBlock × N                    │
│   ├─ RMSNorm                            │
│   ├─ RoPE + causal self-attention       │
│   ├─ residual                           │
│   ├─ RMSNorm                            │
│   ├─ SwiGLU FFN                         │
│   └─ residual                           │
│   ↓                                     │
│ RMSNorm                                 │
│   ↓                                     │
│ tied LM head                            │
└─────────────────────────────────────────┘
   │
   ├──────────── Training ────────────────┐
   │                                      │
   │       cross entropy → autodiff       │
   │       → AdamW → scheduler            │
   │       → checkpoint/eval              │
   │                                      │
   └──────────── Inference ───────────────┤
                                          │
                          KV cache → sampler
                                          │
                                          ▼
                                     generated text
```

## 2. Core stack

```text
Application / CLI / Server
        │
        ├── Tokio / Axum / Serde / tracing
        │
LLM domain crates
        │
        ├── model
        ├── data
        ├── training
        ├── inference
        └── evaluation
        │
Burn 0.22
        │
        ├── Tensor API
        ├── Autodiff
        ├── Optimizers
        ├── Training primitives
        └── runtime Device dispatch
        │
CubeCL-backed runtimes where applicable
        │
        ├── CUDA
        ├── ROCm
        ├── Metal
        ├── Vulkan/WebGPU
        └── CPU
        │
Hardware / vendor runtime
```

## 3. Dependency direction

Domain code must depend inward, not outward.

Preferred dependency direction:

```text
apps
 ↓
training / inference / evaluation
 ↓
model + data + tokenizer
 ↓
shared core/config
 ↓
Burn + foundational Rust crates
```

The HTTP server must not own model logic. The model crate must not know about HTTP.

## 4. Runtime backend policy

Burn 0.22 selects execution backends at runtime through `Device`. Model code should therefore be backend-agnostic.

Required target modes:

- `flex` (pure-Rust eager CPU backend, `Device::flex()`) as the reference path for deterministic correctness tests;
- `cuda` (`Device::cuda(0)`) for NVIDIA training;
- `wgpu`/`vulkan` for portable development where useful.

`cpu` (`Device::cpu()`, CubeCL CPU runtime with LLVM JIT) is a distinct backend from `flex` and must not be used as the reference path. NdArray and LibTorch are deprecated in 0.22 and the Candle backend was removed.

Devices must be constructed explicitly. `Device::default()` selects from the backends compiled into the binary rather than from the hardware present, so enabling an extra backend can silently change it.

Do not write direct CubeCL kernels in the first implementation unless a required primitive is missing. First build with Burn operators, profile, and only then introduce direct custom kernels.

When a target accelerator backend cannot be exercised because the hardware is unavailable, its backend-specific validation is reported as `Not run`, never as `Passed`.

## 5. Configuration

Configuration must be explicit and serializable.

Suggested hierarchy:

```text
ExperimentConfig
├── ModelConfig
├── TokenizerConfig
├── DataConfig
├── TrainingConfig
├── OptimizerConfig
├── SchedulerConfig
├── CheckpointConfig
├── EvaluationConfig
├── GenerationConfig
└── RuntimeConfig
```

Every training run must persist the resolved configuration next to its checkpoint.

## 6. Reproducibility manifest

Every run produces a manifest containing at least:

- UTC start time;
- git commit SHA;
- dirty-worktree flag;
- Rust version;
- dependency lockfile hash;
- Burn version;
- backend;
- device names;
- model config hash;
- tokenizer artifact hash;
- dataset manifest hash;
- seed;
- dtype;
- batch/sequence configuration;
- optimizer and scheduler configuration.

## 7. Error model

Library crates return typed errors. `panic!` is reserved for violated internal invariants, not malformed user data or expected I/O failures.

At application boundaries use a unified error type with context.

Examples:

- invalid model configuration;
- tokenizer/model vocabulary mismatch;
- corrupt checkpoint;
- unsupported dtype/backend operation;
- out-of-memory reported by runtime;
- dataset record corruption;
- request cancellation;
- model-not-ready state.

## 8. Unsafe policy

`unsafe` is prohibited in project code by default.

An `unsafe` block is allowed only when:

1. a measurable requirement cannot be met safely;
2. the invariants are documented adjacent to the block;
3. focused tests cover the boundary;
4. code review explicitly approves it.

Transitive dependencies may use `unsafe`; supply-chain review is separate.

## 9. Custom-kernel boundary

If direct CubeCL is later required, isolate it:

```text
crates/kernels/
├── attention/
├── rope/
├── norm/
└── sampling/
```

Every custom kernel must have:

- a reference implementation using ordinary Burn ops;
- randomized equivalence tests;
- dtype coverage;
- shape coverage;
- benchmark evidence;
- backend capability checks;
- fallback behavior.
