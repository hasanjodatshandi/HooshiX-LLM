# Rust-Native LLM — Documentation Pack

**Date:** 2026-10-07  
**Project goal:** Build a GPT/Qwen/DeepSeek-style decoder-only language model stack with the application/model/training/inference codebase written in Rust.

## Scope

This repository plan deliberately targets a **Rust-native LLM stack**, not a Python application wrapping a Rust library.

The project will own, in Rust:

- dataset ingestion and preprocessing;
- tokenizer training/loading/encoding;
- model definition;
- causal language-model objective;
- automatic differentiation through Burn;
- optimizer and learning-rate scheduling;
- training loop and checkpoint lifecycle;
- evaluation;
- generation and sampling;
- KV-cache based inference;
- HTTP/streaming serving;
- benchmark and validation tooling.

External platform components are allowed where unavoidable:

- NVIDIA/AMD/Apple/GPU drivers;
- CUDA/ROCm/Metal/Vulkan/WebGPU toolchains/runtime components;
- operating-system facilities;
- generated backend code produced by Burn/CubeCL.

**No Python runtime or Python source is part of the planned production/training stack.**

## Fixed 5-stage roadmap

1. **Stage 1 — 100M–150M dense decoder LLM**
2. **Stage 2 — 300M–500M optimized dense LLM**
3. **Stage 3 — ~1B production-grade LLM**
4. **Stage 4 — ~7B distributed LLM**
5. **Stage 5 — DeepSeek/Qwen-class research architecture and post-training**

The stage count is fixed. Internal tasks inside a stage are not additional stages.

## Current technical baseline

- Rust stable, **minimum Rust version 1.95** because Burn 0.22 requires it.
- Burn **0.22** as the core tensor/autodiff/training framework.
- Burn **Flex** (`flex` feature, `Device::flex()`) as the deterministic CPU/reference backend. Note that Burn's `cpu` feature is a different backend (CubeCL CPU runtime with LLVM JIT) and is not the reference path.
- CubeCL as Burn's heterogeneous compute/kernel layer; direct CubeCL code only when profiling proves a custom kernel is necessary.
- Hugging Face `tokenizers` **0.23.2** for Rust-native tokenizer implementation.
- Burnpack for internal Burn checkpoints; SafeTensors for portable model-weight export when required.
- Tokio + Axum for serving.
- Serde for configuration and manifests.
- `tracing` for structured telemetry.

## Documents

| File | Purpose |
|---|---|
| `01_PROJECT_CHARTER.md` | Project objective, scope, non-goals, success criteria |
| `02_ROADMAP.md` | Fixed five-stage execution plan and gates |
| `03_ARCHITECTURE.md` | Logical/runtime architecture and boundaries |
| `04_REPOSITORY_STRUCTURE.md` | Planned Cargo workspace layout |
| `05_MODEL_SPEC.md` | Transformer design and Stage-1 model specification |
| `06_TOKENIZER_AND_DATA.md` | Tokenizer, corpus, provenance and data pipeline |
| `07_TRAINING_AND_CHECKPOINTS.md` | Training algorithm, precision, checkpoints, resume |
| `08_DISTRIBUTED_AND_SCALING.md` | Multi-GPU/multi-node scaling strategy |
| `09_INFERENCE_AND_SERVING.md` | Decode loop, KV cache, batching and API serving |
| `10_TESTING_AND_VALIDATION.md` | Correctness tests and acceptance gates |
| `11_BENCHMARKING.md` | Performance methodology and metrics |
| `12_SECURITY_AND_RELIABILITY.md` | Trust boundaries, input safety, failure handling |
| `13_DEPENDENCIES_AND_TOOLCHAIN.md` | Version policy, crates and platform requirements |
| `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` | Ordered implementation backlog for Stage 1 |
| `15_AI_ASSISTANT_START.md` | Handoff prompt/instructions for a coding AI |
| `16_DECISIONS.md` | Architecture decision record |
| `17_REFERENCES.md` | Primary official technical references |
| `18_RUST_STANDARDS_ROUTER.md` | Task-based router for engineering rules |
| `19_RUST_CODING_STANDARDS.md` | Rust clean-code, ownership, API and module rules |
| `20_RUST_ARCHITECTURE_AND_DESIGN_PATTERNS.md` | Rust/ML architecture and approved design patterns |
| `21_RUST_RUNTIME_CONCURRENCY_ERROR_STANDARDS.md` | Async, concurrency, resources and typed errors |
| `22_RUST_API_OBSERVABILITY_SECURITY_STANDARDS.md` | Axum boundary, tracing and security rules |
| `23_RUST_TESTING_QUALITY_DELIVERY_STANDARDS.md` | Testing, Clippy, Git and AI delivery rules |
| `24_JAVA_TO_RUST_STANDARDS_MAPPING.md` | Explicit mapping/omissions from Java/Spring standards |
| `25_STAGE1_ACCEPTANCE_REPORT.md` | What Stage 1 delivered, what was measured, and what is not true |
| `AGENTS.md` | Mandatory coding-agent entry point |

## Start here

For a human: read `01_PROJECT_CHARTER.md` → `02_ROADMAP.md` → `03_ARCHITECTURE.md` → `18_RUST_STANDARDS_ROUTER.md` → `20_RUST_ARCHITECTURE_AND_DESIGN_PATTERNS.md` → `14_STAGE1_IMPLEMENTATION_CHECKLIST.md`.

For an AI coding assistant: place `AGENTS.md` at the repository root, provide this whole documentation set, then instruct the assistant to follow `15_AI_ASSISTANT_START.md`. The assistant must route each task through `18_RUST_STANDARDS_ROUTER.md` before coding.
