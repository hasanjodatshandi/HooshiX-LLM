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

> **Corrected 2026-10-11.** This section's `flex` column is the baseline that
> section 12 **withdrew**, and all three of its ratios are computed against that
> column, so **all three ratios are void**. The `cpu` figures are sound and are
> kept; the comparison they were drawn against is not. The corrected conclusion is
> below the table, and the arithmetic is in section 4 of
> `26_OPTIMIZATION_RESEARCH_REVIEW.md`.
>
> Section 12 states the rule that decides this: only measurements taken within one
> run, or alternating between binaries, mean anything. Section 11's `flex` column
> came from a different session and is not reproducible.

Measured 2026-10-10 on the same machine, after the `cpu` backend was unblocked
(see `13_DEPENDENCIES_AND_TOOLCHAIN.md` section 5). Same model, same
configuration, same commit line; only the backend differs.

| batch | sequence | flex tokens/s | cpu tokens/s | ratio |
|---|---|---|---|---|
| 8 | 512 | ~~51~~ | **70** | ~~1.37×~~ **void** |
| 4 | 2048 | ~~54~~ | **64** | ~~1.19×~~ **void** |
| 1 | 2048 | ~~51~~ | 41 | ~~0.80×~~ **void** |

**The corrected comparison, from one session**

Section 14 measured both backends in the same session at batch 8 × sequence 512,
which is 4,096 tokens per step:

| | s/step | tokens/s | session |
|---|---|---|---|
| `cpu` | 56.9 and 58.3 | 72 and 70 | section 14 |
| `flex` | 17.2–17.27 | 238 | section 12 |
| `flex` | 18.40 | 223 | section 13 |

`4096 / 56.9 = 72.0`, `4096 / 18.40 = 222.6`, `4096 / 17.20 = 238.1`.

**`cpu` is 3.1× to 3.3× slower than `flex`, not 1.37× faster.** Its own figure
reproduces across two sessions (70 and 72 tokens/s in sections 11 and 14), and the
`flex` figure it was compared against does not.

**Findings**

- **`cpu` is slower than the reference path at every shape where both have been
  measured in the same session.** It is not a speed path. It remains selectable
  and correct, and removing it would remove the ability to measure it.
- **The first step costs 9 to 18 minutes.** CubeCL compiles its kernels with LLVM
  on first use: 735 s at batch 8 × sequence 512, 1082 s at batch 1 × 2048. This is
  a one-time cost per shape and per process, and it dominates anything short. A
  throughput figure taken from the first step would be wrong by two orders of
  magnitude, which is why the benchmark reports warmup separately. This finding is
  unaffected by the correction and remains the most useful thing in this section.
- At 139 tokens/s for the configured Stage-1 shape on `flex`, 1.64B tokens is about
  136 days; on `cpu` at roughly a third of that rate it would be several hundred
  days.

**What this does not say**

- `wgpu` and `cuda` were unmeasured when this section was written. `wgpu` has
  since been measured: it works, and it is 4.3×–8.9× the reference path at the
  shapes it fits on a 4 GB card (section 17 and `14_STAGE1_IMPLEMENTATION_CHECKLIST.md`
  section K).
- The batch 8 × sequence 2048 shape, which the reference path could not allocate,
  was **not** attempted on `cpu`. Whether the CubeCL path fits where `flex` did
  not is unknown and must not be assumed.

## 12. A withdrawn claim: the compiler flags changed nothing

**This section replaces an earlier one that reported a 4.4× speedup from build
flags. That claim does not survive a back-to-back measurement and is withdrawn.
It is kept here, in corrected form, because the failure is instructive.**

The original sequence was measured across several hours, with each row built and
run at a different time:

| build | batch 8 × seq 512 | reported |
|---|---|---|
| default release | 80.8 s | 51 tok/s |
| `-C target-cpu=native` | 33.2 s | 123 tok/s |
| `+ lto=fat, codegen-units=1, panic=abort` | 18.2 s | 225 tok/s |

That reads as a clean 4.4×. It is not. Rebuilt and re-measured **back to back, in
one session, alternating between the two binaries**:

| build | batch 8 × seq 512 | tokens/s |
|---|---|---|
| default profile, current code | 17.27 s | **237** |
| `lto=fat`, one codegen unit, current code | 17.04 s | **240** |
| default profile, **code as it was before any of this work** (`5204457`) | 17.19 s | **238** |
| current code, immediately after | 17.23 s | **238** |

Three conclusions, in order of how uncomfortable they are:

1. **The build flags are worth about one percent**, which is inside the noise, and
   cost a release rebuild going from ninety seconds to seven minutes.
   `docs/11` section 8 says an optimization stays only if it improves the metric,
   so no profile override was added.
2. **The "before" number of 51 tokens/s was wrong.** The same code path, on the
   same machine, measures 238 today. Whatever produced 51 was a property of that
   moment — the machine, a background load, a build state — and not of the code.
   It was never reproducible and should not have been published as a baseline.
3. **The performance work in this session improved nothing.** A worktree of the
   code as it stood before any of it measures the same 238 tokens/s as the code
   now. The gains claimed for `target-cpu=native`, LTO and the finiteness-check
   rewrite are all withdrawn.

The machine carries a background load throughout — a VPN client, two editor
instances and a device-sync service, together holding roughly 30% of the CPU. Any
cross-run comparison on this machine is therefore unreliable, and only
measurements taken **within one run** or **alternating between binaries** mean
anything. Section 6 already said to hold everything but the variable constant;
what this shows is that "the same machine" is not sufficient when the machine is
shared.

What survives:

- `batch 8 × sequence 2048` aborts on allocation, and the message names the
  cause: `memory allocation of 1610612736 bytes failed`. That is exactly one
  layer's attention score tensor, `8 × 12 × 2048 × 2048 × 4` bytes = 1.61 GB,
  which autodiff then retains for all twelve layers. The configured shape is a
  **memory** problem, not a speed problem, and it is fixed by accumulating
  gradients over a smaller micro-batch — which is verified in section 14.
- `-C target-cpu=native` is a real lever in principle, because Rust's default
  x86-64 target enables only SSE2 and the kernels in `gemm`/`pulp` select their
  implementation at compile time. It did not reproduce as a speedup here, and the
  honest position is that it is untested rather than that it works.

## 13. Fourth baseline: where the time goes

Measured by ablating the configuration. The configuration is read at run time, so
no rebuild is needed and **every row below comes from one binary in one run**,
which is why the ratios can be trusted even though section 12 shows that
absolute figures on this machine cannot be compared across runs.

At batch 8 × sequence 512, where the whole model cost 18.40 s per step in that
run (the same shape measures 17.2 s today, so treat the absolute column as
indicative and the ratios as the result):

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
parameters, and it is the single largest tensor in the step: the **logits**
tensor at batch 8 × sequence 512 is `8 × 512 × 32768 × 4` bytes = 537 MB per
pass, and the objective's `log_softmax` holds a copy of it at the same size. That
figure is not the attention score tensor, which at this shape is
`8 × 12 × 512 × 512 × 4` = 100 MB; the two happen to be the same order only
because this sequence is short. Section 17 confirms by measurement that the
logits tensor, not attention, is what sets the GPU's ceiling.

At sequence 2048 the same model reaches **139 tokens/s** against **235** at
sequence 512 — both measured back to back, so this comparison does hold — which
puts the quadratic term in attention at roughly **40% of a step** at the
configured length. That is the second lever, and the first one is memory: that
term is also what makes the configured shape impossible to allocate.

**What has not been tried**

- ~~A real BLAS.~~ **Attempted properly; Burn 0.22 has no path to one.** See
  section 16. The earlier entry here was wrong: nothing had been built when it
  was written.
- ~~`burn-fusion`~~ Attempted; see section 14.
- ~~A tiled attention that never materialises the score matrix.~~ **Delivered,
  measured, and removed.** It was correct and it lost on every comparable shape;
  see section 17. The useful part of that result is negative and is recorded
  there: the 1024-token ceiling is the logits tensor, not attention.
- ~~Any alternative allocator.~~ Attempted; see section 14.
- ~~Using Burn's own attention instead of hand-rolling one.~~ **Delivered**; see
  section 17. It reproduces this repository's logits exactly and does not move
  the GPU ceiling, so it is an equivalence result and not a speedup.

## 14. Six candidate optimizations, measured

Every row below was measured at batch 8 × sequence 512 unless stated, against the
baseline of section 12. Some rows were measured against a baseline taken in a
different run, so for those the ratio is the result and the absolute column is
indicative — section 12 explains why. An optimization is kept only if it improves
the intended metric, per section 8.

| # | Candidate | Result | Kept |
|---|---|---|---|
| 1 | A real BLAS, via the `ndarray` backend | `burn/ndarray` builds, but the step is **30.9 s, 132 tok/s — 1.7× slower** than `flex`. Burn 0.22 marks `Device::ndarray` deprecated in favour of `flex`. The BLAS-backed variants (`blas-openblas`, `blas-openblas-system`, `blas-netlib`, `blas-accelerate`) were **not** tried: each needs a system or bundled BLAS and a Fortran toolchain, which is its own project. | **No** |
| 2 | Tiled attention, so the score matrix never exists whole | Works, and is **correct** — a test compares a 3000-token tiled run against a 64-token whole run and requires the shared prefix to match. But where it applies it is **101.7 s against 67.8 s, 50% slower**, and it does **not** unblock the configured shape: the peak single allocation falls from 1.6 GB to 48 MB while the process still runs out, because autodiff retains every tile for the backward pass. | **No — reverted** |
| 3 | `burn-fusion` | Burn 0.22's `fusion` feature enables it for `burn-cpu`, `burn-cuda`, `burn-rocm` and `burn-wgpu` — **not for `burn-flex`**, and `burn-flex` contains no fusion code at all. Measured on `cpu`, where it does apply: 56.9 s / 72 tok/s against 58.3 s / 70 tok/s without the tuned build. **No measurable gain**, and `cpu` is **3.1×–3.3× behind `flex`** in this same session — see section 11 for the corrected comparison and `26_OPTIMIZATION_RESEARCH_REVIEW.md` §4 for the arithmetic. | **No** |
| 4 | A smaller vocabulary | Measured as a curve: 4096 → 15.49 s, 8192 → 15.64 s, 16384 → 16.66 s, 32768 → 18.23 s. Cutting to 4096 saves **15%**, and the curve is flat below 8192. A vocabulary change alters the tokenizer contract and needs its own decision, so nothing was changed here. | **Measured only** |
| 5 | Gradient accumulation over micro-batches | **This is the one that worked.** The old implementation summed the micro-batch objectives and took a single backward pass, which gives the correct gradient but keeps every graph alive, so peak memory was the whole group. It now backwards each micro-batch and sums the gradients through a parameter visitor — `GradientsParams` has no addition operator, so the sum is built by visiting the model. Batch 8 × sequence 2048, which previously aborted on a 1.6 GB allocation, now runs at **136.9 s per step, 120 tokens/s**. The per-token rate is identical to the un-accumulated run at the same shape (121 tokens/s), so accumulation costs nothing measurable. | **Yes** |
| 6 | An alternative allocator (mimalloc) | **No measurable difference**: 18.23 s against 18.20 s, within noise. The first run after the build reported 61.6 s, which was contamination from a concurrent process rather than the allocator; the same binary reproduced 18.23 s immediately afterwards. | **No** |

**Where this leaves the configured run**

The configured Stage-1 shape now runs, which it did not before this work. At
batch 8 × sequence 2048 it produces **139 tokens/s**, so 1.64B tokens is about
**136 days** on this machine, against "cannot run at all" before.

That is the substantive result of this session, and it is a **correctness** result
rather than a speed one. No optimization tried here made the model faster; see
section 12. The remaining gap is that a 110M model at sequence 2048 spends
roughly 40% of its step on the quadratic term in attention, and that the step
runs at a small percentage of what the machine can do in principle. Closing either
needs a real GEMM — a BLAS, or attention built on one — and neither was reached
here.

## 15. The Stage-1 baseline report

Checklist section M asks for eight figures and the environment beside them.
`apps/bench` takes all of them. Build and run:

```bash
cargo build --release -p bench
target/release/bench --config configs/stage1-110m.toml --micro-batch 8 --sequence 512
```

Peak memory is not taken by the binary. Reading a process's peak working set
needs an operating-system call, and this workspace does not permit `unsafe`; the
figure below is the operating system's own `PeakWorkingSet64`, sampled from
outside the process. Reporting it as if the binary measured it would be a lie, so
it is labelled where it comes from.

### Environment

| | |
|---|---|
| operating system / architecture | windows / x86_64 |
| logical CPUs | 20 |
| backend, compiled and configured | `flex`, f32 |
| model | 12 layers, hidden 768, 12 heads, vocab 32768, 110119680 parameters |
| measured shape | micro-batch 8 × sequence 512, accumulation 1 |
| `bench.exe` | 14.73 MiB (release) |

### Measurements

| figure | value |
|---|---|
| cold start | 0.888 s |
| prefill | 716.6 tokens/s (128 tokens in 0.179 s) |
| TTFT | 0.179 s |
| decode | 31.3 tokens/s (0.0319 s/token mean, 0.0335 s worst) |
| training | 210.9 tokens/s over 5 steps |
| peak memory | 9681.4 MiB |
| checkpoint write | 72.7 MiB/s (1321484283 bytes in 17.344 s) |
| checkpoint read | 1432.2 MiB/s (1321484283 bytes in 0.880 s) |

### A first attempt reported 71 tokens/s, and the measurement was at fault

The first run of this benchmark printed **71 tokens/s** for a shape that
`throughput` measures at 195–235. It was published as untrusted rather than
buried, and the reconciliation is worth recording because the cause was the
measurement, not the code:

**The default was two timed steps, and the first one carries a one-time cost.**
AdamW's moment buffers are allocated on first use — 880 MB for the Stage-1 model
— and a second allocation cost appears on the first step after warmup. At two
steps that fixed cost was half the average. Measured back to back at the same
shape:

| run | tokens/s |
|---|---|
| `throughput`, 8 × 512, 5 timed steps | 195 |
| `bench`, 8 × 512, 5 timed steps, minimal prompt | 192.3 |
| `bench`, 8 × 512, 5 timed steps, real prompt | 199.3 |
| `bench`, 8 × 512, **2 timed steps** | **71** |

Generation before training makes no difference — rows two and three agree — so
the only variable was the step count. The default is now five timed steps over
two warmup steps, and the flag carries the reason.

This is section 1's rule applied to the benchmark itself: a figure taken over too
few steps measures the allocator, not the model, and it is not a measurement
until it can be reproduced by the thing it claims to measure.

### Checkpoint writing varies by more than 4× between runs

The same write took 4.303 s in one run and 17.344 s in another — 292.9 against
72.7 MiB/s — on the same bytes. Reading did not move (1367 against 1432 MiB/s).
Writing is the operation that touches the filesystem, and this machine carries
about 30% background load, which section 12 already records as the reason
cross-run comparisons are invalid here. The reading figure is usable; the writing
figure needs either a quiet machine or several repetitions, and until then it
should be quoted as a range rather than a number.

### What the numbers say

Prefill and decode differ by a factor of 23, which is the shape of the problem
rather than a surprise: prefill processes 128 tokens in one parallel pass, and
decode processes one token at a time with a full weight read per token. It is
also why TTFT is dominated by prefill and why a long generation is dominated by
decode, and why one "tokens/s" figure for inference would have hidden both.

Checkpoint reading is about 20× writing in this run, though that gap is mostly
the write variance above rather than a property of the format.

### Not measured

- The configured Stage-1 shape, `batch 8 × sequence 2048`. It cannot be measured
  on this machine: the attention score tensor is 1.6 GB and the allocation fails.
  The shape is a parameter of the benchmark for exactly this reason.
- `prefill` is reported as TTFT rather than as a separate forward pass. They
  differ by one sampling step, which is microseconds against 179 ms, but the
  report should say that rather than let a reader assume two measurements were
  taken.
- No accelerator figures. `throughput` measured the GPU at 514 tokens/s at
  `2 × 256`; `bench` was not built for `wgpu` in this pass.
- Binary size is for `bench.exe` alone, at 11.83 MiB in release. The other
  binaries were not measured.

## 16. A real BLAS: attempted properly, and Burn 0.22 has no path to one

Section 14 listed "a real BLAS" as attempted. That entry was wrong — nothing had
been built. It has now been built, and the answer is a well-evidenced no.

**The route that exists.** `burn-flex` has **no BLAS feature at all**: its
features are `std`, `simd`, `rayon`, `x86-v4`, `apple-amx`, `fft`, `tracing`,
`critical-section`, and its GEMM is the pure-Rust `gemm` crate. The only backend
with a BLAS feature is `burn-ndarray`, which offers `blas-openblas`,
`blas-openblas-system`, `blas-netlib` and `blas-accelerate`.

**The blocker, and the way around it.** On Windows MSVC,
`openblas-src`'s `system` mode calls `vcpkg::find_package("openblas")` and is the
**only** route it offers — `OPENBLAS_PATH` is ignored, and the evidence is the
panic: `VcpkgNotFound("No vcpkg installation found")`. Rather than install vcpkg,
prebuilt `OpenBLAS-0.3.34-x64` was downloaded, and because `blas-src`'s build
script panics only when *more than one* provider is enabled — zero providers
compiles cleanly and emits no link flags — the library was linked directly with
`RUSTFLAGS="-L native=<openblas>/lib -l openblas"`. The import library needed
renaming from `libopenblas.lib` to `openblas.lib`, which is the name rustc asks
MSVC for. It built, and the example ran.

**The result, and why it is not a BLAS result:**

| shape | `flex` + `gemm` | `ndarray` + "BLAS" |
|---|---|---|
| 8 × 512 | **186 tok/s** | 141 tok/s |
| 2 × 256 | **120 tok/s** | 98 tok/s |
| 1 × 512 | 43 tok/s | **94 tok/s** |

These are not BLAS figures, because **BLAS was never called**:

- `libopenblas.dll` is not in the running process's module list.
- The executable's dependent-DLL table contains no `libopenblas.dll` at all —
  only system libraries. The import was never referenced, so the linker dropped
  it.
- `OPENBLAS_NUM_THREADS=1` and `=16` produced 137 and 142 tokens/s, which is
  noise. If BLAS were computing the GEMMs, thread count would matter.
- `burn-ndarray`'s source imports `ndarray` for containers — `Array2`, `ArrayD`,
  `ArrayView`, `Zip` — and never `ndarray::linalg::Dot`. Its matmul is its own.

So enabling `ndarray/blas` compiles, links and changes nothing: **Burn's ndarray
backend does not route its GEMMs through ndarray's BLAS-backed path.** The
measurement above is "ndarray's own GEMM against `flex`'s `gemm` crate", and in
that comparison `flex` wins at large batches and loses at small ones.

**What this means.** There is no configuration of Burn 0.22 that puts a tuned
BLAS under the tensor matmul. Reaching one means either a Burn release that wires
ndarray's `Dot` in, or a custom kernel — and section 13's finding stands unchanged:
attention and the GEMMs beneath it are the cost, and the reference path is what
this project has.

The experiment's code was reverted. Keeping a feature named `ndarray-blas-system`
that provably does not engage a BLAS would be worse than not having it.

**Corrected:** the line in section 13 that read "needs a real GEMM — a BLAS, or
attention built on one — and neither was reached here" is now precise: a BLAS was
reached, linked, and is unreachable from Burn's tensor operations.


## 17. Attention: a tiled path that was correct, measured, and removed; and Burn's own

Section 13 and section 6 both name attention as roughly 40% of a step, and
attention is what forces the `[batch, heads, sequence, sequence]` score tensor
into memory — 1.6 GB at the configured shape, which is the allocation that fails
on both the processor and the 4 GB GPU. Two attempts followed from that. **One
was removed after measuring it. The other is kept and is an exact-equivalence
result, not a speedup.**

### The tiled path: correct, measured, removed

A block-tiled attention was written for the second time and this time it was
**correct**: it never materialises the whole score block, and its logits and its
parameter gradients match the whole-matrix path element by element. It was then
measured, and it lost on every shape where the two can be compared:

| shape | backend | whole | tiled | tiled is |
|---|---|---|---|---|
| 2 × 256 | `flex` | 4.41 s/step, 116 tok/s | 4.68 s/step, 109 tok/s | 6% slower |
| 1 × 512 | `flex` | 4.35 s/step, 118 tok/s | 6.20 s/step, 83 tok/s | 42% slower |
| 8 × 512 | `flex` | 22.20 s/step, 185 tok/s | 40.03 s/step, 102 tok/s | 80% slower |
| 2 × 512 | `wgpu` | 0 timed steps (OOM) | 3 timed steps, 244–277 tok/s | — |

The one row that favours it is not worth having. Tiled does not unlock a shape
that is otherwise unreachable **and worth reaching**: at 2 × 512 on the GPU it
completes 244–277 tokens/s, against **457 tokens/s at 1 × 512 on the same GPU,
which the whole path already runs**. It unlocks a shape that is slower per token
than one that already works. The 13% peak-memory saving it does deliver at
8 × 512 — 8385 MB against 9680 MB — does not change whether anything runs.

**It was therefore deleted**, along with `TILE_QUERY_ROWS` and its tests. A
second attention implementation is a permanent maintenance cost and this one
bought nothing measurable. The measurement is recorded here rather than the code
kept, because the result is the valuable part.

### The four faults found on the way, two of which are Burn traps

The first attempt at this was wrong and was reverted. The second attempt found
three faults, each of which kept the failure in the same place — the first row of
every query tile — until all three were fixed:

1. **The mask offset must be anchored to the tile's first query row**, as
   `key_start + shift - start` where `shift` is the absolute position of that
   row. Anchor it anywhere else and the causal diagonal lands in the wrong place,
   and only the leading row of each tile shows it.
2. **The online-softmax shift must be the *larger* of the block maximum and the
   running maximum.** A block whose maximum is below the running one must be
   scaled against the running one, or its best entry gets a factor of one and
   takes a share of the row it has not earned. But the running maximum alone is
   not safe either: a block that masks a row entirely has `-inf` for both, and
   `-inf - -inf` is a NaN that the zero factor meant to erase it does not clear.
3. **`Tensor::max_pair` is not a maximum.** Its mask is `self.lower(other)`, so it
   keeps `self` wherever the two are equal **and wherever `self` is NaN**. A
   running maximum built from it silently kept a stale value: no error, no NaN,
   just a wrong answer. This is a trap in the framework, and it is why
   `attention.rs` no longer calls it.
4. **`Tensor::tril_mask` masks nothing when the block is taller than it is
   wide.** `tril_mask([2, 1], 0)` masks nothing, where the causal answer masks the
   second row's only entry. That is exactly the shape of a query tile's last key
   block, which is why the fault lands on the tile's leading row. The removed
   path built its mask by comparing absolute positions instead; a test compared
   the two constructions at the shapes where they can differ.

### Burn's own attention: exact equivalence, not a speedup

`burn-tensor` 0.22 exposes `burn::tensor::module::attention` — a **public,
backend-dispatched** scaled-dot-product attention taking `(query, key, value,
mask, attn_bias, options)`. `burn-flex` auto-selects between `attention_naive`
and `attention_flash` (KV-tiled, online softmax, `TILE_KV` scratch per row) by
the size of the score block; `burn-cubecl`, which is what `wgpu` and `cpu` use,
calls `cubek::attention::forward`, a fused kernel. The model now reaches it
through `AttentionMode::Burn`, selectable with `ATTENTION=burn`.

**It reproduces this repository's logits exactly — a difference of `0.0`, not
merely within tolerance.** Burn defaults the scale to `1 / sqrt(head_dim)`, the
same value used here, and its mask convention is the same one (`true` means "set
to `-inf` before the softmax"), so `causal_mask` is handed over unchanged. The
KV-cache path is untouched: `forward_cached` builds on the whole path, and the
existing test that compares cached logits against uncached ones is green.

**Do not cite this as a win.** It is an equivalence result. It measured **1%
slower on `flex` at 2 × 256** and equal at the shapes where both completed, and
on the GPU it does not raise the ceiling either — at 2 × 512 it completes zero
timed steps, exactly like the whole path. On a machine that drifted 53% during
the session that measured it, 1% is noise, so **`Whole` remains the default**.
Changing the default on this measurement would repeat the mistake section 12 had
to retract. The decision needs a re-measurement on a quiet machine; if Burn's
attention then measures equal or faster it should obviously win, because it is
less code here and is covered by Burn's own backend test suite rather than by
this repository's.

### The 1024-token ceiling is not about attention

This is the most useful negative result in the section, and it **confirms the
prediction in section 13**.

`4 × 256`, `2 × 512` and `1 × 1024` — every 1024-token micro-batch — run out of
memory on the **first** step on **both** paths, including the block-tiled path
that never materialises the score matrix at all. Attention is therefore not what
sets the ceiling. The allocation that fails is the **logits tensor**: at
`batch × sequence × vocab × 4` bytes, `1 × 1024 × 32768 × 4` is 128 MB per pass
and `8 × 2048 × 32768 × 4` is 2.1 GB, and the objective's `log_softmax` holds a
copy of it at the same size. Section 13 named that tensor as the single largest
one in the step and measured the vocabulary as the lever that moves it; this
section confirms it by removing attention from the list of suspects.

**Where not to look:** tiling attention, replacing attention with a kernel, or
any other attention optimization will not move this ceiling. The next lever is
the vocabulary (section 14 measured that curve) or the logits and loss path
itself.

### Measurements

**`flex` (processor)**, 5 timed steps after 2 warmup.

| shape | path | s/step | tokens/s | peak RSS |
|---|---|---|---|---|
| 2 × 256 | whole | 4.41 | 116 | 2089 MB |
| 2 × 256 | whole | 4.37 | 117 | 2111 MB |
| 2 × 256 | burn | 6.77 | 76 | 2119 MB |
| 1 × 512 | whole | 4.35 | 118 | 2337 MB |
| 1 × 512 | burn | 7.78 | 66 | 2351 MB |
| 8 × 512 | whole | 22.20 | 185 | 9680 MB |

**The `flex` absolutes are not trustworthy, and this session is the third time
that has bitten this document.** The same binary measured `2 × 256` at
4.41 s/step early in the session and 6.77 s/step an hour later — a **53% drift**
with nothing else running, on a machine whose power plan is a third-party
"Driver Booster" scheme rather than a Windows one. Only the **alternating
ratios** are usable: whole and burn were run alternately from one binary at each
shape, so a drift in the machine lands on both. The absolute column is
indicative. Against the 120 tokens/s section 14 records for 2 × 256, the `whole`
row reproduces.

**`wgpu` (GPU, AMD Radeon R9 200 Series, 4 GB)**, 5 timed steps after 3 warmup.
The first step of a wgpu run compiles shaders and takes 87–400 s; it is never
counted.

| shape | path | timed steps completed | tokens/s |
|---|---|---|---|
| 2 × 256 | whole | 5 | 414 |
| 2 × 256 | burn | 5 | 411 |
| 1 × 512 | whole | 3 | 457 |
| 1 × 512 | burn | 1 | 394 |
| 2 × 512 | whole | 0 | — |
| 2 × 512 | burn | 0 | — |
| 4 × 256 / 2 × 512 / 1 × 1024 | both | 0 | — |

"Timed steps completed" is its own column because an out-of-memory condition on
this backend is **not immediately fatal and not reliably fatal at the same
place**. The device runner thread panics with `wgpu error: Out of Memory`, the
process prints whatever steps it finished and then stops, and the count varies
between runs of the same binary at the same shape: `1 × 512` on the whole path
completed 1, 2 and 3 timed steps in three runs. The binary's own `at batch …`
summary line averages only the steps that survived and therefore is not
comparable across rows; the table does not use it.

The recorded GPU figures hold: section 14 records 382 tokens/s at `1 × 512` and
479 at `2 × 256`; this session measures 457 and 414.

### A note on RMSNorm, which was checked and not changed

**Read from source, not measured.** Burn 0.22 does expose `burn::nn::RmsNorm` /
`RmsNormConfig`, and its forward pass is the same decomposition into separate
elementwise operations this repository hand-wrote:

```rust
let rms = (x.clone().cast(DType::F32).square().mean_dim(D - 1) + self.epsilon).sqrt();
(x / rms.cast(dtype)) * self.gamma.val().unsqueeze()
```

The one difference does not help: Burn casts to f32 and back, this repository
computes in the model dtype.

**There is no fused RMSNorm hook in Burn 0.22 at all.** `ModuleOps` has hooks for
`attention` and `layer_norm` and nothing for RMS normalisation, and a search for
`rms_norm` across the vendored `burn-flex`, `burn-backend` and `burn-cubecl`
sources finds no implementation. The fused kernel at
`burn-flex/src/ops/activation.rs` is `layer_norm` — a mean/variance
normalisation, reachable as `burn::tensor::module::layer_norm` — and it is **not
reachable from an RMS normalisation**. The lever is closed: replacing this
repository's `RmsNorm` with Burn's would be a rewrite of identical arithmetic
with no backend kernel behind it.

### What is still not answered

- Whether `wgpu` ever reaches `cubek::attention::forward` at these shapes, or
  whether the mask argument routes it down a fallback. Not established.
- Whether an allocator, a smaller vocabulary, or gradient checkpointing moves the
  logits-tensor ceiling. Section 14 measured two of those and this section does
  not revisit them.
- Whether Burn's attention beats the whole path on a quiet machine. It is 1%
  slower here, which is inside the drift, and the default decision waits on that
  re-measurement.

## 18. The TinyStories configuration, measured

Section 13's ablations led to an estimate of 600–800 tokens/s for the
configuration that is actually going to be trained, and `26_OPTIMIZATION_RESEARCH_REVIEW.md`
§F3 recorded that estimate as an estimate. This section replaces it with a
measurement. **The estimate was close and the measured figure is slightly above
it.**

### Environment

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700K, 12 cores, 20 logical, 3600 MHz |
| RAM | 63.7 GB total, 50.5 GB free at the time of the run |
| operating system | Windows 11 Pro, build 28000 |
| toolchain | rustc 1.99.0 (b940084d7 2026-09-28) |
| backend | `flex`, f32 |
| model | `configs/tinystories-small.toml`: 6 layers, hidden 512, 8 heads, vocab 8192, sequence 512 |
| parameters | **24,648,192 (24.6 M)**, as printed by the binary |
| harness | `cargo run --release -p llm-training --example throughput` |
| warmup | 2 steps excluded; 5 timed steps |
| concurrency | none — verified no `train`, `prepare` or `cargo` process was running |

### Configured shape, 8 × 512

```powershell
$env:CONFIG="configs/tinystories-small.toml"
$env:BATCH="8"; $env:SEQUENCE="512"; $env:STEPS="5"; $env:WARMUP="2"; $env:ACCUM="1"
cargo run --release -p llm-training --example throughput
```

```text
step 2: 5.0890s  loss 9.1049  805 tokens/s
step 3: 7.1423s  loss 9.0898  573 tokens/s
step 4: 5.1702s  loss 9.0697  792 tokens/s
step 5: 5.1246s  loss 9.0450  799 tokens/s
step 6: 5.1130s  loss 9.0160  801 tokens/s

at batch 8, sequence 512: 5.5278s per step, 741 tokens/s
```

**741 tokens/s is the harness figure and it is dragged down by one outlier.**
Four of the five timed steps ran at 5.09–5.17 s; step 3 took 7.14 s for reasons
not identified, and the mean over five includes it. Excluding it the rate is
**~799 tokens/s**. Both are reported because the harness prints the first and the
steady state is the second, and quoting only the favourable one would be the
mistake section 12 records.

### Shape curve

Same config, same warmup policy, 5 timed steps each, run in one session:

| shape | tokens/step | s/step | tokens/s |
|---|---|---|---|
| 8 × 512 (configured) | 4,096 | 5.53 | **741** (799 without the outlier) |
| 4 × 512 | 2,048 | 2.61 | **784** |
| 2 × 512 | 1,024 | 1.50 | 681 |
| 1 × 512 | 512 | 0.90 | 570 |
| 4 × 1024 | 4,096 | 6.83 | 600 |
| 2 × 2048 | 4,096 | 9.45 | 433 |
| 1 × 2048 | 2,048 | 4.88 | 419 |

**4 × 512 is the fastest shape measured, at 784 tokens/s**, marginally above the
configured 8 × 512. Throughput falls as sequence length grows and rises with
batch size at a fixed token count.

Note that `4 × 1024` and `2 × 2048` run at sequence lengths **above** the model
configuration's `max_sequence_length = 512`. The harness builds the batch
directly and does not consult that field; RoPE computes positions beyond it
without complaint. These are throughput measurements, not evidence that a
sequence of 2,048 is meaningful for a model trained at 512, and they should not
be read as such.

### Against the 110 M configuration, alternating in one session

Section 12 permits this comparison only within one session. Two rounds, both
configurations, same shape, same warmup:

| round | TinyStories (24.6 M) | Stage-1 (110.1 M) | ratio |
|---|---|---|---|
| 1 | 795 tokens/s | 229 tokens/s | 3.47× |
| 2 | 813 tokens/s | 254 tokens/s | 3.20× |

**The TinyStories model is 3.2×–3.5× faster than the Stage-1 model at 8 × 512.**
The two 110 M figures, 229 and 254, sit inside the 186–238 range recorded
elsewhere in this document.

### Wall clock

The configured run is **20,000 steps × 8 × 512 = 81,920,000 tokens**, at
741 tokens/s:

```text
81,920,000 / 741 / 86,400 = 1.28 days
```

**About 1.28 days** — 30.7 hours — for the run as configured, at the measured
rate, **excluding validation passes, checkpoint writes and the cost of loading
the shard**; those are not in the throughput harness.

At the steady-state 799 tokens/s the same run is 1.19 days. The honest statement
is **1.2 to 1.3 days**.

**That budget is 15.7% of one epoch.** The prepared training shard holds
**520,848,384 tokens**, so one full pass at the measured rate is
`520,848,384 / 741 / 86,400 = 8.1 days`, or 7.5 days at 799 tokens/s. The
configured `max_steps` does not cover the corpus, which is a property of the
configuration rather than of the machine.

### The estimate, judged

`26_OPTIMIZATION_RESEARCH_REVIEW.md` §F3 recorded two anchors: 388 tokens/s from
extrapolating the layer-count ablation, and a 600–800 tokens/s estimate.

- **The measurement is 741 tokens/s** at the configured shape, 799 at steady
  state, and 784 at the best shape measured.
- **The 600–800 estimate was close** — the measurement sits at its upper edge and
  the best shape is just above it. It is not wrong enough to correct.
- **The 388 anchor was wrong**, as §F3 said it might be. It came from scaling the
  layer-count ablation linearly, and the real configuration also halves the hidden
  size and quarters the vocabulary, which the linear scaling did not carry.

The lesson is section 1's: the estimate pointed in the right direction and the
measurement is what may be quoted.

### Peak memory

**Not measured.** The throughput harness does not sample it, and section 15
records that the benchmark binary deliberately does not either — reading a peak
working set needs an operating-system call this workspace does not make from Rust.

---

## 19. The unidentified bottleneck, identified: it is parallel scaling, and the run decays

Section 13 left an open question. The CPU ran at about a sixth of its FP32 peak,
a fifth of its cores and a fraction of a percent of its memory bandwidth, and the
cause was not known. A first attempt to answer it with a thread sweep was made
while training was active and produced nothing but noise; section 14 row 6 is the
standing rule that forbids benchmarking under load.

The TinyStories run finished, the machine went idle, and the sweep was repeated.

### 19.1 The measurement

24.6M model, `flex`, f32, 8×512, 15 timed steps after 3 warmup, on an otherwise
idle machine. **Every configuration below was run in one session, back to back, on
the same binary.** One row is missing: `RAYON_NUM_THREADS=20` produced no result
and was not retried, because 12 and 16 already showed the plateau.

| `RAYON_NUM_THREADS` | tokens/s | s per step | speedup |
|---|---|---|---|
| 1 | 453 | 9.032 | 1.00× |
| 2 | 648 | 6.319 | 1.43× |
| 4 | 808 | 5.070 | 1.78× |
| 8 | 904 | 4.533 | **2.00×** |
| 12 | 895 | 4.577 | 1.98× |
| 16 | 905 | 4.528 | **2.00×** |

**Sixteen times the threads buys twice the throughput, and nothing beyond eight
threads buys anything at all.**

### 19.2 What that implies

Amdahl's law with a speedup of 2.00 at N=16 gives a **serial fraction of 46.7 %**
and a hard ceiling of **2.14×**. The implementation is at 2.00× of a possible
2.14×, which is to say **it is already against its own parallel ceiling** and no
amount of thread tuning will move it.

### 19.3 And what it does *not* imply

The earlier guess in section 13 — per-tensor-op overhead — is **withdrawn as the
explanation**, and the arithmetic is why:

| | |
|---|---|
| FLOPs per step | about 465 GFLOP |
| at 1 thread | 51 GFLOP/s = **44 % of one core's peak** |
| at 16 threads | 103 GFLOP/s = about **10 % of the machine's peak** |

**Single-threaded, this implementation reaches 44 % of one core's peak, which is
good.** The deficit is entirely in scaling, not in arithmetic efficiency. A
per-op-overhead story would show up as poor single-threaded throughput, and it
does not.

The remaining candidate is the autodiff bookkeeping and the dispatch cost of the
many small elementwise operations the backward pass performs, which are serial
with respect to the matmuls they sit between. **That is a hypothesis and is
labelled as one.** No measurement in this section separates it from the
alternatives.

### 19.4 The finding that matters more than the sweep

The TinyStories run began at **912 tokens/s** and ended at **480 tokens/s**,
a 47 % decline over 47.3 hours. The run was predicted at 1.28 days and took
**47.3 hours**, so the decline is the whole of the estimate's error.

**A fresh process with the same binary, same configuration and same shape
produces 904–905 tokens/s.** That is the starting figure, not the ending one.

> **The decay is a property of a long-lived process, not of the configuration.**

Rayon's default is the logical processor count, 20, and the run therefore used
the same setting the sweep measures at 904. The sweep result and the early-run
figure agree, which is what makes the comparison meaningful.

**The cause is not identified.** What has been ruled out:

| Ruled out | Evidence |
|---|---|
| thermal throttling | the CPU performance counter reads 116 % of nominal |
| memory pressure | working set held at 2873 MB for the whole run |
| thread configuration | this section |
| the dataset reader | the run logged **0 passes** over the shard, so it never rewound |

**The leading hypothesis is allocator fragmentation** — Burn allocates and frees
tens of tensors per step and 20,000 steps is a great many — **and it is a
hypothesis, not a measurement.** Settling it needs one more 47-hour run, which is
not worth the machine time for a configuration this project has finished with.

**The practical consequence is available now and costs nothing to use:** the run
already writes a checkpoint every 2,000 steps and already resumes from one. A long
run that restarts from its latest checkpoint periodically would pay the decay once
per restart instead of once per run. **This was not tried and is not claimed to
work**; it is the obvious next experiment and it is recorded here as untried.

### 19.5 What this changes about the Stage-1 plan

Nothing about the 110M model's arithmetic, and **everything about its schedule.**
The 136-day estimate for Stage 1 was computed from a rate measured on a fresh
process. A long run does not hold that rate. **Any Stage-1 timeline that assumes a
constant rate is wrong by up to a factor of two**, and the TinyStories run is the
evidence: predicted 1.28 days, actual 47.3 hours.

**The GPU conclusion is unaffected and is strengthened.** On a GPU the arithmetic
is not the constraint and neither is this decay, because the 4 GB card already
fits only 512 tokens per micro-batch and the work per step is far smaller.

---

## 20. The 24.6M model on the GPU: 3.1x to 3.3x, and the ceiling is four times higher

The last outstanding measurement in `28_BILINGUAL_PILOT_STATUS.md`. The GPU was
4.3×–8.9× the CPU for the 110M model at the shapes that fitted in 4 GB; whether a
model a quarter the size gains anything was unknown. A small model is the
pessimistic case, because there is less arithmetic per byte moved and the
transfer cost is a larger share of each step.

**Both sides were measured in one session on the same machine, the same build of
the same model, immediately after each other.**

| batch × sequence | tokens/step | flex, RAYON=8 | wgpu | speedup |
|---|---|---|---|---|
| 1×256 | 256 | 466 | 1341 | **2.9×** |
| 2×256 | 512 | 587 | 1950 | **3.3×** |
| 1×512 | 512 | 574 | 1845 | **3.2×** |
| 2×512 | 1024 | 698 | 2169 | **3.1×** |
| 4×512 | 2048 | 745 | **2414** | **3.2×** |
| 8×512 | 4096 | 729 | **OOM** | — |
| 1×1024 | 1024 | not measured | 1843 | — |

**The GPU is 3.1×–3.3× the CPU, and it is consistent across every shape that
fits.** The pessimistic expectation — that a quarter-size model would gain less —
is wrong; the ratio is flat.

### 20.1 The ceiling moved from 512 tokens to between 2048 and 4096

The 110M model could not exceed **512 tokens per micro-batch** on this card.
The 24.6M model completes at **2048** (4×512) and fails at **4096** (8×512), so
its ceiling is between the two. It was not narrowed further because the shape
above 4×512 that would separate them is not one the run would use.

**This is worth noting against section 16.** There, the 1024-token ceiling on the
110M was shown not to be attention, because every path failed including one that
never materialises the score matrix, and the raw logits at that shape were only
128 MiB. Whatever the constraint is, it scales with the model: a quarter-size
model reaches four to eight times the tokens per micro-batch on the same card.

The failure is a CubeCL panic, not a clean out-of-memory:

```text
thread 'DSD-4-0' panicked at cubecl-wgpu-0.11.0\src\compute\device_poison.rs:42:17
```

A poisoned device aborts the process rather than reporting a shape that will not
fit, so the ceiling has to be found by trying shapes, which is what this table is.

### 20.2 An error caught before publication

The first version of the sweep script compared the GPU against **43 tokens/s at
1×512 and 120 at 2×256**, figures carried from earlier notes. **Both were wrong
for this model** — the measured CPU at those shapes is **574** and **587**. The
remembered numbers belonged to a different measurement.

Had they been published, the headline would have read "43× faster on the GPU",
which is the kind of number that changes a hardware decision and would have been
false. **Every figure in the table above was measured in this session.** The
lesson is the one this document keeps relearning: a number recalled is not a
number measured, and it has to be taken again even when it feels settled.


