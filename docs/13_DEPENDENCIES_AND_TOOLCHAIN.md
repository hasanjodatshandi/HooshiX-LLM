# Dependencies and Toolchain

## 1. Rust

- Toolchain: stable.
- Burn 0.22 minimum supported Rust version: **1.95**.
- Burn 0.22 is published with `edition = "2024"`; the workspace edition must be compatible with the toolchain that satisfies the Burn MSRV.
- CI must enforce a toolchain that satisfies the current Burn requirement.
- `Cargo.lock` is committed for applications/workspace reproducibility.

Windows toolchain prerequisites, verified by an actual build on the development machine:

- rustup with the MSVC host toolchain (`stable-x86_64-pc-windows-msvc`);
- Visual Studio with `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`;
- a Windows SDK component (`Microsoft.VisualStudio.Component.Windows11SDK.28000`).

The Windows SDK is a real prerequisite, not an optional extra: without it every link fails with `LNK1181: cannot open input file 'kernel32.lib'`, and rustc cannot discover the MSVC toolchain at all. A GNU/MinGW toolchain present on `PATH` must not be allowed to select itself implicitly.

Do not assume the local developer toolchain is sufficient; preflight must check `rustc --version`.

**Verified 2026-10-07 against crates.io:** `burn` 0.22.0 declares `rust_version = "1.95"`, confirming the MSRV above.

## 2. Core ML stack

### Burn

Target line:

```toml
burn = { version = "0.22", features = ["train", "...backend..."] }
```

Burn 0.22:

- uses runtime `Device` backend selection;
- has no execution backend enabled by default;
- supports CUDA, ROCm, WGPU, Metal, Vulkan, WebGPU, CubeCL CPU and Flex according to enabled features;
- requires Rust 1.95;
- changed record/checkpoint APIs from 0.21.

Do not copy Burn 0.21 generic-backend examples into 0.22 code without adapting them.

#### 0.22 feature semantics

Verified 2026-10-07 against the published Burn 0.22 crate metadata and the official Burn Book:

- The `default` feature set contains no execution backend. `Device::wgpu`/`Device::cuda`/`Device::cpu`/`Device::flex` only exist when the matching feature is enabled.
- With `default-features = false`, `optim` must be enabled explicitly for `burn::optim` or `burn::lr_scheduler`. The `train` feature also enables `optim`.
- `safetensors` and `pytorch` imply `store`. `burn-store` is not a direct dependency of this project.
- `linalg` is its own feature; linear algebra lives in `burn::linalg` extension traits.
- NdArray (`ndarray`), LibTorch (`tch`) are deprecated and the Candle backend was removed. None of them may be used as the reference path.
- The `distributed` and `collective` feature flags were removed; collective operations are reached through `burn::tensor::distributed`.
- Out-of-tree backends are not supported in 0.22. `#[backend_extension]` extends Burn's own backends; it does not register new ones.
- An extra backend enabled by Cargo feature unification silently changes `Device::default()`, which selects from compiled-in backends rather than from the hardware present.

#### 0.22 record/checkpoint API

Burn 0.22 removed `Recorder`, `PrecisionSettings`, generated record types and `#[derive(Record)]`. Internal checkpoints use burnpack and `ModuleRecord`:

- `model.try_load_file(path)?` replaces `model.load_file(path, &recorder, &device)?`;
- `model.save_file(path)?` replaces `model.save_file(path, &recorder)?`;
- `load_file` panics on failure; `try_load_file` is the fallible form required by the project panic policy;
- pre-0.22 `.mpk`, `.bin` and JSON checkpoints cannot be loaded directly, and renaming them to `.bpk` does not convert them.

Since this project starts fresh on 0.22, legacy checkpoint migration is not a project concern; it only matters if externally produced artifacts are ever imported.

### CubeCL

Use indirectly through Burn initially.

Direct dependency is allowed only for a justified custom-kernel crate after profiling.

Current CubeCL documentation describes it as an alpha high-performance compute language/runtime with CUDA/HIP/Metal/SPIR-V/WGSL/CPU targets. Its public API may break between minor versions. Therefore pin it if direct use begins.

### Tokenizers

Target:

```toml
tokenizers = "0.23.2"
```

Rust-native core supports BPE, Unigram, WordPiece and related pipeline components.

Avoid enabling unnecessary features.

### SafeTensors

Current Rust crate line observed:

```toml
safetensors = "0.8"
```

Prefer Burn's current store integration when that satisfies export requirements; do not duplicate serialization paths unnecessarily.

## 3. Application crates

Expected candidates:

```text
serde
serde_json
toml
thiserror
tracing
tracing-subscriber
tokio
axum
clap
sha2
rand
rand_chacha
```

Add only when the code needs them. Exact versions belong in Cargo.lock.

## 4. Development tools

Expected:

- rustfmt;
- clippy;
- cargo test;
- cargo bench or Criterion only when benchmark harness needs statistical tooling;
- cargo-audit or equivalent supply-chain scan.

Do not add Criterion until benchmark structure is implemented; standard timing may be enough for early smoke measurements.

## 5. Backend build policy

Recommended workspace features:

```text
flex  -> pure-Rust CPU reference/eager backend   (Device::flex())
cpu   -> CubeCL CPU runtime, LLVM JIT            (Device::cpu())
cuda  -> NVIDIA GPU through CubeCL CUDA runtime  (Device::cuda(0))
wgpu  -> portable graphics API through WGPU      (Device::wgpu(..))
```

`cpu` and `flex` are **two different CPU backends** in Burn 0.22 and are not interchangeable:

- `flex` is the pure-Rust eager backend. Use it for the mandatory deterministic reference path of `03_ARCHITECTURE.md` and `16_DECISIONS.md` (ADR-010).
- `cpu` runs through CubeCL's CPU runtime with LLVM JIT and supports fusion. It is a performance path, not the reference path.

NdArray and LibTorch are deprecated and Candle was removed, so none of them may be selected as the reference path.

A developer should enable the backend they need rather than compile all backends unconditionally.

Always construct devices explicitly (`Device::flex()`, `Device::cuda(0)`). `Device::default()` chooses from the backends compiled into the binary, not from the hardware present, so enabling an additional backend through Cargo feature unification can silently change it. This also affects implicit device selection by `Tensor::from(...)` and dataloaders without `set_device(...)`.

### CubeCL-backed backends: the link blocker, and its resolution

Verified 2026-10-10 by building each path separately.

**Original symptom.** Every CubeCL-backed path (`cpu`, `wgpu`, `cuda`) pulls
`cubecl-environment`, which depends on `turso`, whose `turso_core` depends on the
native `simsimd` crate. Linking `turso_sdk_kit` failed with:

```text
LINK : fatal error LNK1181: cannot open input file 'simsimd.lib'
```

**`simsimd` was not the cause.** It was the last error to be printed, not the
first failure. Two earlier blockers had to be cleared first, and clearing them
made the `simsimd` error disappear without any change to that crate:

1. **`liblzma-sys` could not link.** Its build script runs `lib.exe`, which
   writes a temporary response file under `%TEMP%`; the build failed with
   `LNK1104: cannot open file '...\Temp\lnk{GUID}.tmp'`. Redirecting `TEMP` and
   `TMP` to a directory inside the workspace fixed it. Nothing in the project
   depends on this; it is an environment property of the machine.
2. **`tracel-llvm-bundler` could not install its toolchain.** The CubeCL CPU
   backend needs a bundled LLVM, which the bundler downloads and unpacks into
   `%LOCALAPPDATA%\tracel\tracel-llvm-23.1.0-3`. From inside a build script that
   failed with `Access is denied. (os error 5)` on
   `create_dir_all(%USERPROFILE%\.cache\tracel)`. Overriding `LOCALAPPDATA` does
   not help, because `dirs::data_local_dir()` on Windows calls
   `SHGetKnownFolderPath` and ignores the environment; overriding `USERPROFILE`
   moves the cache but then `data_local_dir()` returns `None` and the bundler
   reports `Unsupported system`.

   The bundler skips installation entirely when its sentinel file exists, so the
   bundle was installed outside the build: download
   `windows-x64.tar.xz` from the `tracel-ai/tracel-llvm` `v23.1.0-3` release,
   verify its SHA-256 against the published `windows-x64.checksums.json`,
   extract it into `%LOCALAPPDATA%\tracel\`, and create the
   `.tracel-llvm-installed` sentinel inside the extracted directory. The archive
   hash matched exactly (`d0a548ee…35ee`, 56.9 MB, 680.6 MB extracted).

   The bundler's own `content_sha256` tree check does not run when the sentinel
   short-circuits it, so that verification was not performed.

**Consequences for planning:**

- `cpu` now builds, links, preflights and runs. It is no longer blocked.
- `wgpu` and `cuda` share the same CubeCL base and are expected to build, but
  neither has been built, because neither has hardware here. They remain
  **not run**.
- The workaround is per-machine, not per-repository. A fresh machine needs the
  same three steps, or a writable `%LOCALAPPDATA%` for the build script. This
  belongs in the environment setup notes, not in the crate graph.
- `simsimd` is a transitive native dependency of Burn/CubeCL, not a project
  dependency, and needed no change.

**A separate defect this uncovered.** With `cpu` working, training refused every
step with "the loss is not finite". The loss was finite; the finiteness check was
wrong. See `crates/llm-model/src/grad.rs`: the original implementation combined
boolean comparisons and read a boolean tensor back, and folded a read-back
failure into "not finite". Reading a boolean tensor is not uniformly supported,
so a backend that could not be *checked* was reported as one that had *failed*.
The check is now arithmetic — the sum of `x * 0` is zero exactly when every
element is finite — and an unevaluable check is its own error. This is recorded
because the shape of the bug is general: a diagnostic that cannot run must never
be indistinguishable from a diagnostic that ran and failed.

## 6. NVIDIA environment

For CUDA builds:

- NVIDIA GPU/driver must be detected;
- required CUDA/toolchain components must match Burn/CubeCL expectations;
- `nvidia-smi` and a minimal Burn CUDA tensor operation should pass before launching training.

Exact driver/toolkit versions are environment-specific and must be recorded in the run manifest.

## 7. Dependency update policy

Updates are deliberate:

1. read upstream migration/release notes;
2. create dependency-update branch;
3. update lockfile;
4. compile all supported targets;
5. run correctness suite;
6. run representative benchmark;
7. inspect checkpoint compatibility;
8. merge only after regressions are explained.

Never combine a major framework update with unrelated model changes.

## 8. Rust code-quality and supply-chain baseline

Required language/toolchain checks:

```text
rustfmt
Clippy
cargo test
rustdoc/doctests where applicable
```

For CI/release supply-chain checks, use focused Rust ecosystem tooling where available, with distinct responsibilities:

- RustSec advisory scanning (`cargo audit` or the repository-approved equivalent);
- license/source/banned-dependency policy (`cargo deny` when configured);
- repository/history secret scanning;
- committed `Cargo.lock`.

Do not add runtime crates solely to implement CI checks. Tool versions/configuration must be pinned by the repository when introduced.

`thiserror` is the default candidate for typed library errors. Do not introduce a second general-purpose error framework unless a concrete application-boundary need justifies it.
