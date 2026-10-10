# Project Charter

## 1. Objective

Build a modern decoder-only large language model and its end-to-end runtime in Rust, progressing from a small but real pretrained language model to larger distributed models.

The target is **architectural capability similar to modern GPT/Qwen/DeepSeek-family systems**, not an immediate claim of frontier-model quality.

## 2. Primary requirements

1. Application source code is Rust.
2. Model source code is Rust.
3. Training code is Rust.
4. Tokenization and data-pipeline code is Rust.
5. Inference and serving code is Rust.
6. The model can train from random initialization.
7. The model can save, restore and resume deterministically enough for production training.
8. A trained checkpoint can generate text without Python.
9. The same model definition is used for training and inference unless an optimized inference representation is explicitly introduced later.
10. Every scaling step is gated by correctness and performance evidence.

## 3. Definition of “Rust-only”

### Required

The repository MUST NOT require:

- Python scripts;
- Python virtual environments;
- PyTorch;
- TensorFlow;
- JAX;
- Java/JVM;
- application-level C/C++ source maintained by this project.

### Allowed external dependencies

The following do not violate the requirement:

- GPU drivers;
- CUDA/ROCm/Metal/Vulkan/WebGPU;
- system libraries required by the hardware/runtime;
- code generated internally by Burn/CubeCL;
- external Rust crates and their transitive implementation details, subject to dependency review.

If a future dependency introduces a non-Rust runtime as a material project dependency, it requires an explicit architecture decision before adoption.

## 4. Initial architecture class

The initial model is:

- decoder-only Transformer;
- causal next-token prediction;
- pre-normalization;
- RMSNorm;
- RoPE;
- multi-head self-attention in Stage 1;
- SwiGLU feed-forward network;
- tied token embedding / language-model head;
- no bias in major linear layers unless measurement justifies it.

GQA is intentionally deferred to Stage 2 so Stage 1 remains a correctness-focused baseline.

## 5. Non-goals for Stage 1

Stage 1 will NOT:

- implement Mixture-of-Experts;
- implement RLHF/GRPO;
- implement multi-node training;
- invent a new tensor framework;
- replace Burn;
- optimize every kernel;
- implement speculative decoding;
- implement paged attention;
- claim ChatGPT/Claude/DeepSeek quality;
- train on unlicensed or provenance-unknown data.

## 6. Success criteria

Stage 1 is complete only when all of the following are true:

- tokenizer round-trip and invariants pass;
- forward pass shape and mask tests pass;
- gradients are finite and non-zero where expected;
- a tiny model can overfit a tiny deterministic corpus;
- training loss falls on a real dataset sample;
- checkpoint save/load reproduces model outputs within the defined tolerance;
- interrupted training can resume without silently resetting model/optimizer/scheduler/global-step state;
- greedy generation works;
- sampled generation works with deterministic seed behavior;
- KV-cached decoding matches uncached decoding within tolerance;
- CPU/reference backend and target GPU backend agree within dtype tolerance;
- no Python is required to build, train, evaluate or serve;
- benchmark report exists for throughput, latency, memory and checkpoint I/O;
- `cargo fmt`, `cargo clippy`, unit tests and integration tests pass.

## 7. Quality principle

Scaling is forbidden until the previous scale is correct.

Performance optimization is evidence-driven:

> Profile → identify bottleneck → change one subsystem → re-benchmark → retain only measurable improvements.

## 8. Key feasibility constraint

Rust is not the limiting factor for frontier scale. At large parameter counts the dominant constraints become:

- accelerator count and VRAM;
- high-speed interconnect;
- distributed communication;
- dataset quality and token volume;
- checkpoint bandwidth;
- training stability;
- fault tolerance;
- post-training;
- evaluation.

Therefore the project must separate **software capability** from **available compute** in all reports.
