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
