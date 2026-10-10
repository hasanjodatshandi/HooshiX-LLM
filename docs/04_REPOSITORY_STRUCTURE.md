# Cargo Workspace Structure

## Planned repository

```text
rust-llm/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── rustfmt.toml
├── clippy.toml
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   ├── stage1-smoke.toml
│   ├── stage1-110m.toml
│   └── generation.toml
├── docs/
│   └── ...
├── crates/
│   ├── llm-core/
│   │   ├── src/
│   │   │   ├── config.rs
│   │   │   ├── dtype.rs
│   │   │   ├── error.rs
│   │   │   ├── manifest.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── llm-tokenizer/
│   │   ├── src/
│   │   │   ├── train.rs
│   │   │   ├── encode.rs
│   │   │   ├── special_tokens.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── llm-data/
│   │   ├── src/
│   │   │   ├── source.rs
│   │   │   ├── normalize.rs
│   │   │   ├── filter.rs
│   │   │   ├── dedup.rs
│   │   │   ├── pack.rs
│   │   │   ├── batch.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── llm-model/
│   │   ├── src/
│   │   │   ├── embedding.rs
│   │   │   ├── rms_norm.rs
│   │   │   ├── rope.rs
│   │   │   ├── attention.rs
│   │   │   ├── swiglu.rs
│   │   │   ├── block.rs
│   │   │   ├── transformer.rs
│   │   │   ├── loss.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── llm-training/
│   │   ├── src/
│   │   │   ├── trainer.rs
│   │   │   ├── optimizer.rs
│   │   │   ├── scheduler.rs
│   │   │   ├── checkpoint.rs
│   │   │   ├── evaluation.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── llm-inference/
│   │   ├── src/
│   │   │   ├── cache.rs
│   │   │   ├── prefill.rs
│   │   │   ├── decode.rs
│   │   │   ├── sampler.rs
│   │   │   ├── generation.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── llm-eval/
│   │   ├── src/
│   │   │   ├── perplexity.rs
│   │   │   ├── regression.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   └── llm-server/
│       ├── src/
│       │   ├── api.rs
│       │   ├── state.rs
│       │   ├── streaming.rs
│       │   ├── limits.rs
│       │   └── lib.rs
│       └── Cargo.toml
├── apps/
│   ├── train/
│   │   ├── src/main.rs
│   │   └── Cargo.toml
│   ├── eval/
│   │   ├── src/main.rs
│   │   └── Cargo.toml
│   ├── generate/
│   │   ├── src/main.rs
│   │   └── Cargo.toml
│   ├── tokenizer/
│   │   ├── src/main.rs
│   │   └── Cargo.toml
│   └── serve/
│       ├── src/main.rs
│       └── Cargo.toml
├── tests/
│   ├── fixtures/
│   └── integration/
├── benches/
├── scripts/
│   └── README.md
└── artifacts/
    └── .gitkeep
```

## Rules

### Workspace boundaries

- `llm-model` must not depend on `llm-training` or `llm-server`.
- `llm-data` must not depend on serving code.
- `llm-inference` depends on `llm-model`, not vice versa.
- Apps contain wiring only.
- Shared configuration types belong in `llm-core`.
- Avoid a generic `utils` crate. Put behavior in the domain that owns it.

### File-size discipline

Split modules by responsibility when a file becomes difficult to review. Do not create abstraction-only files without a current need.

### Feature flags

Backend features should be centralized so the same source tree can build for:

- CPU reference;
- CUDA;
- WGPU/Vulkan.

Do not compile all accelerator backends by default if it materially increases build complexity.

### Artifacts

Training data and checkpoints are never committed to Git.

Commit only:

- small deterministic fixtures;
- tiny tokenizer fixtures;
- tiny golden checkpoints when license and size permit;
- manifests/hashes;
- benchmark summaries.

## Rust architecture placement rules

The detailed source-placement and design-pattern rules are mandatory in:

- `19_RUST_CODING_STANDARDS.md`;
- `20_RUST_ARCHITECTURE_AND_DESIGN_PATTERNS.md`;
- `21_RUST_RUNTIME_CONCURRENCY_ERROR_STANDARDS.md`.

Do not mechanically add Java/Spring-style `domain/application/infrastructure/interfaces` directories. The existing crate boundaries are the architecture. Use adapters/traits inside those crates only where a real boundary exists.

Do not create planned directories merely to match the tree diagram. Create a directory/crate when the current implementation has code that owns that responsibility.
