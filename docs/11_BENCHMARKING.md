# Benchmarking Plan

## 1. Rule

A performance claim without a reproducible benchmark is not accepted.

## 2. Record environment

Every benchmark report stores:

- CPU model;
- RAM;
- GPU model(s);
- VRAM;
- driver version;
- accelerator/runtime version;
- OS/kernel;
- Rust version;
- Burn version;
- git commit;
- backend;
- dtype;
- model config;
- batch size;
- sequence length;
- warmup count;
- measurement count.

## 3. Training metrics

Report:

- tokens/sec;
- sequences/sec;
- step latency p50/p95 when relevant;
- peak memory;
- data-loader wait time if measurable;
- host→device transfer time if measurable;
- utilization if available;
- checkpoint write/read throughput.

## 4. Inference metrics

Separate:

### Prefill

- prompt tokens/sec;
- time-to-first-token.

### Decode

- generated tokens/sec;
- inter-token latency.

### Server

- request throughput;
- p50/p95/p99 latency;
- queue time;
- cancellation behavior;
- memory under concurrency.

## 5. Benchmark shapes

At minimum Stage 1:

- batch 1, seq 128;
- batch 1, seq 512;
- batch 1, seq 2048;
- representative training micro-batch at seq 2048;
- generation prompt 128 + 128 new tokens;
- generation prompt 1024 + 128 new tokens.

## 6. Comparison discipline

When comparing implementations:

- same weights;
- same dtype where possible;
- same hardware;
- same batch/sequence;
- warmed execution;
- exclude one-time compile/JIT cost unless measuring cold start;
- report cold start separately.

## 7. Cold-start benchmark

For serverless/embedded goals report:

```text
process start
→ runtime/device init
→ model load
→ first successful inference
```

Record binary size and model artifact size separately.

## 8. Optimization acceptance

An optimization stays only if:

- correctness tests pass;
- it improves the intended metric on target hardware;
- it does not cause unacceptable regression elsewhere;
- complexity is justified by the measured gain.

## 9. Regression thresholds

After obtaining the first stable baseline, CI benchmark thresholds may be introduced. Do not invent thresholds before baseline variance is measured.

## 10. First baseline: Stage-1 training step on the reference backend

Measured 2026-10-10 on the machine the project is developed on. This is the
baseline every later optimization is judged against, and it is the first entry
required by section 1.

**Environment**

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700K, 12 physical / 20 logical cores |
| RAM | 63.8 GB |
| GPU | none present |
| OS | Windows NT 10.0.28000, AMD64 |
| Rust | 1.99.0, `stable-x86_64-pc-windows-msvc` |
| Burn | 0.22.0 |
| Backend | `flex` (pure-Rust eager reference path), `f32` |
| Commit | `f9385b3` |
| Reproduction | `cargo run --release -p llm-training --example throughput` |

**Model:** `configs/stage1-110m.toml`, 110,119,680 parameters.

**Measured steps**, one optimizer step each, AdamW with norm clipping, on a
synthetic batch:

| batch | sequence | seconds/step | tokens/s | peak memory |
|---|---|---|---|---|
| 2 | 512 | 27.2 | 38 | fits |
| 8 | 512 | 80.8 | 51 | fits |
| 1 | 2048 | 39.9 | 51 | fits |
| 2 | 2048 | 77.7 | 53 | fits |
| 4 | 2048 | 151.5 | 54 | fits |
| 8 | 2048 | — | — | **aborted** (`0xC0000409`, allocation failure) |

**Findings**

- Throughput is essentially flat at **51–54 tokens/s** once the batch holds
  enough work to occupy the cores; below that it is under-utilised. Batch scaling
  is linear in time, so increasing the batch does not improve total throughput —
  it only shortens the run in steps.
- **The configured Stage-1 shape does not fit.** Batch 8 at sequence 2048 aborts
  on allocation, while batch 4 at the same sequence completes. The configured
  `micro_batch_size = 8` therefore cannot be run as written on this machine, and
  a real run needs either a smaller batch or gradient accumulation.
- The configured run is 100,000 steps × 8 × 2048 = **1.64B tokens**. At 52
  tokens/s that is **about 365 days**. Stage 1 as configured is not reachable on
  this CPU path at this speed.

**What this does not say**

- It says nothing about any accelerator, because none is present.
- No optimization may be justified by these numbers alone. They are a baseline,
  and section 8 still governs whether any change is kept.

## 11. Second baseline: the `cpu` (CubeCL) backend

Measured 2026-10-10 on the same machine, after the `cpu` backend was unblocked
(see `13_DEPENDENCIES_AND_TOOLCHAIN.md` section 5). Same model, same
configuration, same commit line; only the backend differs.

| batch | sequence | flex tokens/s | cpu tokens/s | ratio |
|---|---|---|---|---|
| 8 | 512 | 51 | **70** | 1.37× |
| 4 | 2048 | 54 | **64** | 1.19× |
| 1 | 2048 | 51 | 41 | 0.80× |

**Findings**

- `cpu` is faster where there is enough work per step to keep the cores busy and
  slower where there is not. At batch 1 it is 20% behind the reference path.
- **The first step costs 9 to 18 minutes.** CubeCL compiles its kernels with LLVM
  on first use: 735 s at batch 8 × sequence 512, 1082 s at batch 1 × 2048. This is
  a one-time cost per shape and per process, and it dominates anything short. A
  throughput figure taken from the first step would be wrong by two orders of
  magnitude, which is why the benchmark reports warmup separately.
- Both backends are within 1.4× of each other. The CPU path is a real but modest
  improvement, not the order-of-magnitude change that would make the configured
  Stage-1 run reachable: 1.64B tokens at 64 tokens/s is still about 296 days.

**What this does not say**

- `wgpu` and `cuda` are unmeasured. Neither has been built, because neither has
  hardware here.
- The batch 8 × sequence 2048 shape, which the reference path could not allocate,
  was **not** attempted on `cpu`. Whether the CubeCL path fits where `flex` did
  not is unknown and must not be assumed.

## 12. Third baseline: the compiler flags were the largest single cost

Measured 2026-10-10. Same model, same backend (`flex`), same commit. Only the
build changed.

Rust's default target for `x86_64-pc-windows-msvc` is the `x86-64` baseline,
which means **SSE2 only** — no AVX2, no FMA. The matrix kernels in `gemm`/`pulp`
select their implementation from the target features the crate was compiled
with, so the default build cannot use them.

| build | batch 8 × seq 512 | batch 1 × seq 2048 | batch 4 × seq 2048 |
|---|---|---|---|
| default release | 80.8 s, 51 tok/s | 39.9 s, 51 tok/s | 151.5 s, 54 tok/s |
| `target-cpu=native` | 33.2 s, 123 tok/s | — | — |
| `+ lto=fat, codegen-units=1, panic=abort` | **18.2 s, 225 tok/s** | **16.6 s, 123 tok/s** | **67.8 s, 121 tok/s** |

**Result: 4.4× on the same hardware and the same code.** The configured run of
1.64B tokens falls from about 365 days to about 67 days at the measured shapes.

Notes:

- Repeats are stable under the final flags (18.02–18.31 s over five steps).
  The `target-cpu=native`-only run varied between 27 s and 39 s across steps, so
  the split between "native" and "LTO" is not measured cleanly; only the
  combination is. Treat 4.4× as the result and the intermediate row as
  indicative.
- `panic = "abort"` changes behaviour, not just speed: a panic no longer unwinds,
  so a test harness cannot catch it. It is acceptable for a training binary and
  is **not** acceptable for the test profile.
- These flags make the binary non-portable. They belong in the release profile of
  the applications, documented, with the default kept for CI and for anything
  distributed — not in the workspace defaults.
- `batch 8 × sequence 2048` still aborts, and the message now names the cause:
  `memory allocation of 1610612736 bytes failed`. That is exactly one layer's
  attention score tensor, `8 × 12 × 2048 × 2048 × 4` bytes = 1.61 GB, which
  autodiff then retains for all twelve layers. The configured shape is a
  **memory** problem, not a speed problem, and it is fixed by tiling attention
  or by accumulating gradients over a smaller micro-batch.

## 13. Fourth baseline: where the time goes

Measured with the final flags above, by ablating the configuration. The
configuration is read at run time, so no rebuild was needed.

At batch 8 × sequence 512, where the whole model costs 18.40 s per step:

| ablation | parameters | step | tokens/s |
|---|---|---|---|
| as configured | 110.1M | 18.40 s | 223 |
| vocabulary 32768 → 8192 | 91.2M | 15.86 s | 258 |
| layers 12 → 6 | 67.6M | 10.56 s | 388 |

Fitting `step = fixed + layers × per_layer` to the first and third rows gives
about **2.7 s fixed** and **1.31 s per layer**: the stack of twelve blocks is
about 85% of the step and everything outside it — the token embedding, the
language-model head, the objective and the optimizer — is about 15%.

Cutting the vocabulary fourfold saves 14% of the step. The vocabulary is 32768
against a hidden size of 768, so the head alone is 25.2M of the 110.1M
parameters, and it is the single largest tensor in the step: the score tensor at
batch 8 × sequence 512 is `8 × 512 × 32768 × 4` bytes = 537 MB per pass.

At sequence 2048 the same model reaches only 123 tokens/s against 225 at
sequence 512, so the quadratic term in attention costs about 45% of the step at
the configured length. That is the second lever, and the first one is memory:
that term is also what makes the configured shape impossible to allocate.

**What has not been tried**

- ~~A real BLAS.~~ Attempted; see section 14.
- ~~`burn-fusion`~~ Attempted; see section 14.
- ~~A tiled attention that never materialises the score matrix.~~ Attempted,
  measured and **reverted**; see section 14.
- ~~Any alternative allocator.~~ Attempted; see section 14.

## 14. Six candidate optimizations, measured

Every row below was measured on the same machine, at batch 8 × sequence 512
unless stated, against the 18.2 s / 225 tokens/s baseline of section 12. An
optimization is kept only if it improves the intended metric, per section 8.

| # | Candidate | Result | Kept |
|---|---|---|---|
| 1 | A real BLAS, via the `ndarray` backend | `burn/ndarray` builds, but the step is **30.9 s, 132 tok/s — 1.7× slower** than `flex`. Burn 0.22 marks `Device::ndarray` deprecated in favour of `flex`. The BLAS-backed variants (`blas-openblas`, `blas-openblas-system`, `blas-netlib`, `blas-accelerate`) were **not** tried: each needs a system or bundled BLAS and a Fortran toolchain, which is its own project. | **No** |
| 2 | Tiled attention, so the score matrix never exists whole | Works, and is **correct** — a test compares a 3000-token tiled run against a 64-token whole run and requires the shared prefix to match. But where it applies it is **101.7 s against 67.8 s, 50% slower**, and it does **not** unblock the configured shape: the peak single allocation falls from 1.6 GB to 48 MB while the process still runs out, because autodiff retains every tile for the backward pass. | **No — reverted** |
| 3 | `burn-fusion` | Burn 0.22's `fusion` feature enables it for `burn-cpu`, `burn-cuda`, `burn-rocm` and `burn-wgpu` — **not for `burn-flex`**, and `burn-flex` contains no fusion code at all. Measured on `cpu`, where it does apply: 56.9 s / 72 tok/s against 58.3 s / 70 tok/s without the tuned build. **No measurable gain**, and `cpu` remains 3× behind `flex`. | **No** |
| 4 | A smaller vocabulary | Measured as a curve: 4096 → 15.49 s, 8192 → 15.64 s, 16384 → 16.66 s, 32768 → 18.23 s. Cutting to 4096 saves **15%**, and the curve is flat below 8192. A vocabulary change alters the tokenizer contract and needs its own decision, so nothing was changed here. | **Measured only** |
| 5 | Gradient accumulation over micro-batches | **This is the one that worked.** The old implementation summed the micro-batch objectives and took a single backward pass, which gives the correct gradient but keeps every graph alive, so peak memory was the whole group. It now backwards each micro-batch and sums the gradients through a parameter visitor — `GradientsParams` has no addition operator, so the sum is built by visiting the model. Batch 8 × sequence 2048, which previously aborted on a 1.6 GB allocation, now runs at **136.9 s per step, 120 tokens/s**. The per-token rate is identical to the un-accumulated run at the same shape (121 tokens/s), so accumulation costs nothing measurable. | **Yes** |
| 6 | An alternative allocator (mimalloc) | **No measurable difference**: 18.23 s against 18.20 s, within noise. The first run after the build reported 61.6 s, which was contamination from a concurrent process rather than the allocator; the same binary reproduced 18.23 s immediately afterwards. | **No** |

**Where this leaves the configured run**

The configured Stage-1 shape now runs, which it did not before this work. At
batch 8 × sequence 2048 it produces 120 tokens/s, so 1.64B tokens is about
**158 days** on this machine — against about 365 days for the original build at
the shapes it could actually run, and against "cannot run at all" for the
configured shape.

The remaining gap is not in the framework. It is that a 110M model at sequence
2048 spends about 45% of its step on the quadratic term in attention, and that
the step runs at roughly 150–230 GFLOP/s against a machine peak near
1.5 TFLOP/s. Closing either gap needs a real GEMM — a BLAS, or attention that
uses one — and neither was reached here.
