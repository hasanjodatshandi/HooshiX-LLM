# Five-Stage Roadmap

The stage count is fixed at five.

---

## Stage 1 — 100M–150M dense decoder LLM

### Goal

Produce a real, trainable, resumable Rust-native causal language model. This stage proves correctness of the entire pipeline.

### Baseline model

Approximate target:

- parameters: ~110M;
- layers: 12;
- hidden size: 768;
- attention heads: 12;
- head dimension: 64;
- attention: standard MHA;
- FFN: SwiGLU, intermediate size ~2048;
- context: 2048 tokens;
- vocabulary: 32,768;
- normalization: RMSNorm;
- position: RoPE;
- tied input/output embeddings.

Approximate parameter accounting:

- token embedding: ~25.2M;
- attention per block: ~2.36M;
- SwiGLU per block: ~4.72M;
- 12 blocks: ~85M;
- norms and small tensors: minor;
- total: approximately 110M, subject to exact implementation details.

### Required deliverables

- Cargo workspace;
- config schema;
- tokenizer pipeline;
- dataset streaming/packing;
- Transformer implementation;
- causal loss;
- AdamW training;
- warmup + cosine schedule;
- gradient clipping;
- checkpoint + resume;
- eval loop;
- greedy/top-k/top-p sampling;
- KV cache;
- CLI train/eval/generate;
- Axum inference server;
- validation suite;
- performance baseline.

### Stage gate

Do not proceed until `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` is complete and `10_TESTING_AND_VALIDATION.md` passes.

---

## Stage 2 — 300M–500M optimized dense LLM

### Goal

Increase model capacity while introducing the first performance-oriented architecture and runtime changes.

### Planned changes

- parameter target: 300M–500M;
- GQA;
- BF16-first training on supported hardware;
- optimized/fused attention path;
- improved sequence packing;
- larger tokenizer/corpus only if justified by data analysis;
- gradient accumulation;
- multi-GPU single-node training;
- distributed validation;
- more efficient checkpoint I/O;
- quantized inference experiments;
- structured benchmark comparison against Stage 1.

### Gate

Required before Stage 3:

- single-node multi-GPU correctness;
- scale efficiency measurement;
- no regression in checkpoint resume;
- no silent loss divergence;
- stable BF16 behavior;
- KV-cache correctness under GQA;
- documented memory budget.

---

## Stage 3 — ~1B production-grade LLM

### Goal

Build a 1B-class model with operationally mature training and inference.

### Planned work

- ~1B parameters;
- context target increased only when memory/throughput evidence supports it;
- distributed data parallel training as baseline;
- robust sharded dataset ingestion;
- deterministic experiment manifests;
- portable SafeTensors model export;
- quantized inference;
- continuous batching;
- backpressure and request cancellation;
- model load/startup optimization;
- production metrics;
- failure-injection tests;
- long-running stability tests.

### Gate

Required before Stage 4:

- multi-GPU training is reproducible and restartable;
- loss curves are stable;
- checkpoint restore is verified after process termination;
- serving load test passes;
- no unbounded memory growth;
- model artifacts are integrity checked.

---

## Stage 4 — ~7B distributed LLM

### Goal

Scale the same architecture to a model that requires serious distributed systems engineering.

### Planned work

- ~7B dense model;
- multi-node capability if hardware requires it;
- sharded checkpointing;
- communication/computation overlap;
- memory-aware optimizer strategy;
- activation checkpointing/rematerialization where available/necessary;
- topology-aware process placement;
- tensor/sequence/model parallelism only if DDP + memory optimizations are insufficient;
- fault-tolerant job restart;
- throughput-per-GPU and tokens-per-dollar accounting;
- production-grade inference scheduler.

### Gate

Stage 4 is not considered successful solely because the model starts training. Required evidence:

- distributed correctness;
- sustained training without numerical instability;
- acceptable scaling efficiency;
- restart after node/process failure;
- validated checkpoint shards;
- documented infrastructure requirements.

---

## Stage 5 — DeepSeek/Qwen-class research architecture and post-training

### Goal

Move from a conventional dense LLM to research-grade features used by modern frontier/open LLM families.

### Candidate work, only after Stage 4

- Mixture-of-Experts;
- router and top-k expert selection;
- expert parallelism;
- load-balancing strategy;
- long-context techniques;
- advanced attention variants;
- SFT;
- preference optimization such as DPO;
- reward/verifiable-reward pipelines;
- GRPO-style or other RL research;
- reasoning trajectory generation/evaluation;
- tool/function-calling formats;
- safety/post-training evaluation.

### Important boundary

Stage 5 means the **software architecture can express these capabilities**. Matching the quality of current DeepSeek/Qwen/Claude/GPT systems additionally requires massive compute, high-quality data, sophisticated post-training and extensive evaluation.

---

# Cross-stage rule

A later-stage feature may be prototyped behind an experimental feature flag, but it must not alter the previous stage's stable path until it passes its own correctness and benchmark gates.
