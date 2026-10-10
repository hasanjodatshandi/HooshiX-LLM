# Primary Technical References

These references were checked while preparing this plan on 2026-10-07. Before implementation, use the current upstream documentation if a newer minor/patch release has appeared.

## Burn

- Burn website: https://burn.dev/
- Burn Book: https://burn.dev/books/burn/
- Burn API docs: https://burn.dev/docs/burn/
- Burn 0.22 migration guide: https://burn.dev/books/burn/migrating-to-0.22.html
- Burn backend guide: https://burn.dev/books/burn/basic-workflow/backend.html
- Burn backend and device table: https://burn.dev/books/burn/building-blocks/backend.html
- Burn GitHub: https://github.com/tracel-ai/burn
- Burn crate metadata: https://crates.io/api/v1/crates/burn

Key facts used by this plan:

- Burn 0.22 is the current documented line.
- MSRV is Rust 1.95.
- Runtime backend selection uses `Device`.
- No execution backend is enabled by default.
- Backends include CUDA, ROCm, WGPU, Metal, Vulkan, WebGPU, CubeCL CPU and Flex through features.
- Burn's checkpoint/storage APIs changed materially from 0.21.

### Verification record — 2026-10-07

`burn` 0.22.0 was published 2026-10-06 and declares `rust_version = "1.95"`, `edition = "2024"`; it is `max_stable_version`. The following were confirmed against the Burn Book pages listed above:

- The `default` feature set contains no execution backend.
- `Device::cpu()` requires the `cpu` feature and runs on CubeCL's CPU runtime (LLVM JIT); `Device::flex()` requires the `flex` feature and is the pure-Rust eager backend. They are not the same backend.
- NdArray and LibTorch are deprecated; Candle was removed.
- `default-features = false` requires `optim` for `burn::optim`/`burn::lr_scheduler`; `train` also enables it.
- `safetensors` and `pytorch` imply `store`; `linalg` is a separate feature.
- The `distributed` and `collective` flags were removed; collectives use `burn::tensor::distributed`, and CubeCL all-reduce currently requires CUDA.
- `Device::default()` selects from compiled-in backends, not from available hardware.
- Records use burnpack and `ModuleRecord`; `Recorder`, `PrecisionSettings` and `#[derive(Record)]` were removed.
- Cross-entropy excludes padded targets from its denominator and returns `NaN` for an entirely padded batch.
- Cosine annealing no longer performs abrupt warm restarts.
- Dataset access is fallible: `Dataset::get` returns `Result` and dataloader items are `Result<Batch, DatasetError>`.
- `ExecutionError::DevicePoisoned` exists and `Device::flush()` returns `Result<(), ExecutionError>`.

Other pins verified the same day: `tokenizers` `max_stable_version` = 0.23.2 (a `1.0.0-rc.2` pre-release exists and is not the target); `safetensors` `max_stable_version` = 0.8.0.

### Build verification — 2026-10-07

The toolchain and the reference backend were verified by compiling and running a minimal program, not only by reading documentation.

Environment: Rust 1.99.0 (`stable-x86_64-pc-windows-msvc`), MSVC 14.51.36231 (Visual Studio 18 Community), Windows SDK 10.0.28000.0.

```toml
[dependencies]
burn = { version = "0.22", features = ["flex", "autodiff"] }
```

```rust
use burn::tensor::{Device, Tensor};

let device = Device::flex();
let c = Tensor::<2>::ones([2, 3], &device) + Tensor::<2>::ones([2, 3], &device);
assert_eq!(c.to_data().shape().dims, [2, 3]);
assert!(Device::flex().autodiff().is_autodiff());
```

Result: `cargo run` exited 0 in about 3 minutes. This confirms the `flex` feature name, the `Device::flex()` constructor, runtime autodiff device state, and that Rust 1.99 satisfies the Burn 0.22 MSRV of 1.95.

Host prerequisites established by this verification, required for any MSVC Rust build on this machine:

- Rust stable via rustup with the MSVC host toolchain;
- Visual Studio including `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`;
- a Windows SDK component (`Microsoft.VisualStudio.Component.Windows11SDK.28000`).

The Windows SDK was initially absent, which blocked every Rust link step with `LNK1181: cannot open input file 'kernel32.lib'` and also prevented rustc from discovering the MSVC toolchain. Once the SDK component was installed, rustc discovered the toolchain on its own; building from a Visual Studio developer prompt is not required.

The dependency graph for Burn 0.22 with only these two features locks 756 packages, and crates.io downloads from this network failed repeatedly with `transfer too slow`. Reliable builds therefore depend on tolerant Cargo network settings or a warmed cache.

## Burn distributed evolution

- Burn 0.21 release note:
  https://burn.dev/blog/release-0.21.0/

Used only as background for the distributed roadmap. Current 0.22 behavior must be verified from 0.22 docs during implementation.

## CubeCL

- CubeCL GitHub: https://github.com/tracel-ai/cubecl
- CubeCL Book: https://burn.dev/books/cubecl/

Key facts used by this plan:

- Rust-oriented compute-kernel system used by Burn.
- Targets include CUDA, HIP/ROCm, Metal, SPIR-V/WGPU and CPU.
- Public API is described as alpha/evolving, so direct usage should be pinned and isolated.

## Hugging Face Tokenizers

- Rust crate docs: https://docs.rs/tokenizers/latest/tokenizers/
- Hugging Face docs: https://huggingface.co/docs/tokenizers/
- GitHub: https://github.com/huggingface/tokenizers

Version observed while preparing this plan: `0.23.2`.

## SafeTensors

- Rust crate docs: https://docs.rs/safetensors/latest/safetensors/
- Hugging Face docs: https://huggingface.co/docs/safetensors/
- GitHub: https://github.com/huggingface/safetensors

Version observed while preparing this plan: `0.8.0`.

## Rust / Cargo

- Rust Book: https://doc.rust-lang.org/book/
- Cargo Book: https://doc.rust-lang.org/cargo/
- Clippy: https://doc.rust-lang.org/clippy/
