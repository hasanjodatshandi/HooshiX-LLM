# Architecture Decisions

## ADR-001 — Rust-only project source

**Status:** Accepted

All maintained application/model/training/inference source code is Rust.

External GPU drivers/toolchains and generated backend code are platform dependencies, not project languages.

**Reason:** The project exists specifically to validate and build a Rust-native LLM stack.

---

## ADR-002 — Burn 0.22 is the primary ML framework

**Status:** Accepted

Use Burn 0.22 for tensors, autodiff, modules, optimization/training facilities and backend dispatch.

**Reason:** It provides a Rust-native end-to-end training/inference path and current heterogeneous backend support.

**Constraint:** Use 0.22 APIs, not 0.21 examples.

**Constraint (verified 2026-10-07):** 0.22 selects backends at runtime through `Device`; model and tensor types no longer carry a backend type parameter. The `default` feature set enables no execution backend, `default-features = false` requires `optim` to be enabled explicitly, and `safetensors`/`pytorch` imply `store`. NdArray and LibTorch are deprecated and Candle was removed. `cpu` and `flex` are separate CPU backends; only `flex` is the project reference path (see ADR-010). Devices are constructed explicitly because `Device::default()` selects from compiled-in backends rather than available hardware.

---

## ADR-003 — Do not build a tensor framework

**Status:** Accepted

The project will build an LLM, not reimplement Burn.

Custom backend/kernel work is introduced only when a measured requirement cannot be satisfied through Burn.

---

## ADR-004 — CubeCL direct use is deferred

**Status:** Accepted

CubeCL is part of the underlying compute path, but Stage 1 does not directly depend on it unless profiling or missing primitives require custom kernels.

Every direct custom kernel requires a reference implementation and equivalence tests.

---

## ADR-005 — Dense decoder-only baseline

**Status:** Accepted

Stage 1 uses a dense decoder-only Transformer.

- RMSNorm
- RoPE
- MHA
- SwiGLU
- pre-norm
- tied LM head
- causal language modeling

MoE is deferred to Stage 5.

---

## ADR-006 — MHA in Stage 1, GQA in Stage 2

**Status:** Accepted

Stage 1 optimizes for correctness and simplicity. GQA becomes a Stage-2 controlled architecture change.

The configuration schema should support future `num_kv_heads` without forcing a rewrite.

---

## ADR-007 — Rust Hugging Face Tokenizers

**Status:** Accepted

Use the Rust `tokenizers` crate for tokenizer training/encoding.

No Python tokenizer pipeline is permitted.

---

## ADR-008 — Internal checkpoint vs portable weight export

**Status:** Accepted

Use Burn's current native checkpoint/storage path for full training resume.

Use SafeTensors for portable tensor/model-weight export where required.

Do not confuse model-weight export with a full resumable training checkpoint.

---

## ADR-009 — Tokio + Axum serving

**Status:** Accepted

Use Tokio for async orchestration and Axum for HTTP.

Serving remains isolated from model/training code.

---

## ADR-010 — CPU/reference path is mandatory

**Status:** Accepted

Maintain a reference-capable path for correctness tests independent of high-performance GPU kernels.

GPU acceleration is not allowed to eliminate testability.

---

## ADR-011 — Scale only after gates

**Status:** Accepted

No transition to the next parameter scale until the prior stage meets correctness, checkpoint, inference and benchmark gates.

---

## ADR-012 — Dataset provenance is mandatory

**Status:** Accepted

Production training data without a documented source/license/provenance is rejected.

This is a data-integrity and legal-risk boundary, not an optional documentation task.


---

## ADR-013 — Explicit Rust composition; no DI container

**Status:** Accepted

Dependencies are created explicitly in `apps/*` composition roots and passed through constructors/ownership.

No service locator, reflection registry, or Spring-style DI container is introduced.

---

## ADR-014 — Traits only at real boundaries

**Status:** Accepted

Do not create an interface/trait for every concrete type.

Prefer concrete types for internal model primitives and enums for closed strategy sets. Use traits for genuine substitution, testing, or outer-adapter boundaries.

---

## ADR-015 — Tokio is orchestration, not model compute

**Status:** Accepted

Tokio owns async I/O/request lifecycle. Transformer/tensor computation remains an explicit compute boundary. Heavy compute is bounded and must not monopolize general Tokio worker threads.

---

## ADR-016 — Typed error boundaries

**Status:** Accepted

Library crates expose focused typed errors; outer layers translate them to CLI/HTTP errors. Runtime/untrusted-input failures are not handled with panic/unwrap.

---

## ADR-017 — No speculative enterprise infrastructure

**Status:** Accepted

The current system has no database, ORM, Flyway/migrations, Redis, Kafka/message broker, gRPC, or frontend stack. Do not create abstractions or dependencies for them unless an explicit future requirement adds that concern.

---

## ADR-018 — Multimodal capability is a future scope expansion

**Status:** Accepted (deferred; nothing implemented)

The end goal includes a multimodal model: one that accepts images, audio, or
another non-text modality alongside text. This was not part of the original
brief and is **not** one of the five stages. It is therefore treated as a scope
expansion, not as a Stage-1 feature.

**Decision**

1. Stage 1 remains a text-only dense decoder. No encoder, no adapter, no image
   or audio path, and no modality configuration is added.
2. Any modality work requires, before the first line of code: this ADR updated
   with the chosen architecture, plus a roadmap extension that places the work
   relative to Stages 1–5. Removing modality support from a trained model is not
   possible; adding it to a model that was never designed for it is possible but
   expensive, which is why the decision is recorded rather than assumed.
3. Nothing in Stage 1 may make the conversion structurally hard. The audit below
   is what "structurally hard" was checked against.

**Reversibility audit — 2026-10-10**

Every Stage-1 decision that a multimodal model would touch, and why it is an
addition rather than a rewrite:

| Decision | Where | Cost of adding a modality later |
|---|---|---|
| Sequence is `u32` token ids | `Transformer::forward` | One call site: embedding happens in a single expression, so injecting encoder vectors at reserved positions is a new method beside it, not a change to attention, normalization or the feed-forward path. |
| Rotary positions cover every position uniformly | `Rope`, `MultiHeadAttention` | A per-position policy is a parameter of the attention call. Contained; no other layer reads positions. |
| Attention is multi-head only | `MultiHeadAttention` | Grouped-query attention is already a scheduled Stage-2 change, so the layout a vision-language model usually wants arrives on its own. |
| Loss mask is explicit | `causal_lm_loss` | **Already sufficient.** Positions carrying image or audio tokens are excluded by the existing mask, which is the single most important requirement for cross-modal training. |
| Vocabulary can reserve ids | `SpecialTokenRegistry` | Reserved literals already exist and extend without touching the model. A modality placeholder is one more reserved id. |
| Configuration is a typed document with a version | `llm-core` | A modality section is an additive change with a version bump, exactly as the split settings were. |
| Crate boundaries | workspace | An encoder belongs in its own crate depending on `llm-core` and Burn, feeding the model through an adapter. No existing dependency direction changes. |
| Shard format stores `u32` tokens and a format version | `llm-data` | **The most expensive item.** Non-text samples need side data, so a shard carrying images is a new format version. It is already versioned and readers reject unknown versions, so evolution is supported by design rather than blocked. |

**Why nothing is implemented now**

- The roadmap has exactly five stages and the project forbids moving later-stage
  features forward.
- An embed-injection path with no caller would be an untested abstraction.
- The cheapest thing that actually protects the future — an explicit loss mask
  and a versioned, centralized configuration — is already in place.

**Consequence for the specification**

`05_MODEL_SPEC.md` §13 lists the changes that require an ADR before they may
happen. Adding a modality to the model belongs on that list, and any future
implementation must also state how a non-text token is represented, how it is
positioned, and how it is excluded from the text objective.
