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
parameters, and it is the single largest tensor in the step: the score tensor at
batch 8 × sequence 512 is `8 × 512 × 32768 × 4` bytes = 537 MB per pass.

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
- ~~A tiled attention that never materialises the score matrix.~~ Attempted,
  measured and **reverted**; see section 14.
- ~~Any alternative allocator.~~ Attempted; see section 14.

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
| 3 | `burn-fusion` | Burn 0.22's `fusion` feature enables it for `burn-cpu`, `burn-cuda`, `burn-rocm` and `burn-wgpu` — **not for `burn-flex`**, and `burn-flex` contains no fusion code at all. Measured on `cpu`, where it does apply: 56.9 s / 72 tok/s against 58.3 s / 70 tok/s without the tuned build. **No measurable gain**, and `cpu` remains 3× behind `flex`. | **No** |
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

## 17. Block-tiled attention: attempted, not correct, reverted, and the pitfalls recorded

Section 13 and section 6 both name attention as roughly 40% of a step, and
attention is what forces the `[batch, heads, sequence, sequence]` score tensor
into memory — 1.6 GB at the configured shape, which is the allocation that fails
on both the CPU and the 4 GB GPU. A block-tiled attention that never materialises
that tensor is therefore the obvious next move, and it was attempted.

**It was not delivered.** The implementation was written — query-dimension
blocking, online softmax with a running max, sum and output accumulator,
`TILE_QUERY_ROWS = 128`, a feature threaded through `llm-training`, and six
correctness tests — and it **failed its own correctness test**, so it was never
measured and every change was reverted. `git status --porcelain -uall` is empty.

**This is not a "slower everywhere" result.** Nothing about its speed was
established. It is "not implemented correctly yet", and the two must not be
confused, because the failing version would have produced plausible-looking
throughput figures for wrong logits.

### The failure

Wrong logits for the **first row of every query tile**, in the multi-tile case
only. With `tile_rows = sequence` — one tile — the tiled path matched the
reference exactly and the parameter gradients matched to `0.0`, which is what
localises the fault to the tile boundary rather than to the arithmetic.

### The five pitfalls, which are the expensive part and are worth keeping

Each of these was found by fixing it and watching the failure move:

1. **The mask offset must be anchored to the tile's first query row**, as
   `key_start + shift - start`. Variants that anchor it anywhere else put the
   causal diagonal in the wrong place, and only for the leading row.
2. **The online-softmax rescale factor must use the *running* maximum, not the
   current block's maximum.** Using the block maximum scales the accumulated
   output by the wrong constant and the result comes out about 3× too large.
3. **`f32::MIN - finite` overflows to `-inf`.** Clamping a block maximum with
   `f32::MIN` therefore poisons an entire row. The clamp belongs on the *shifted*
   score: `(scores - block_max).clamp_min(f32::MIN).exp()`.
4. **An all-masked row makes `exp(-inf - -inf)` produce `NaN`**, which needs an
   explicit all-masked zero factor rather than relying on the arithmetic.
5. **`last_index` must be a key *index*, not a key *position*.** The two differ
   by `key_length - sequence`, and the difference is invisible in the
   single-tile case.

### A measurement correction, and a caution about measuring while building

A concurrent session recorded `1 × 512` on `wgpu` failing with
`wgpu error: Out of memory` and concluded the GPU ceiling had fallen since
section 14. That conclusion was wrong, and the cause was the measuring
environment: another process was deleting a 42 GB build directory at the same
time. Re-measured on a quiet machine, the same binary gives **382 tokens/s** at
`1 × 512` and **479 tokens/s** at `2 × 256`, which reproduces the recorded 383
and is consistent with 514. The section 14 figures stand.

The lesson is section 12's, one level down: a measurement taken while the machine
is busy is not a measurement. This is the second time in this document that a
concurrent workload produced a false result, and the first time it produced a
false *regression* — a number that was wrong in the pessimistic direction is no
more trustworthy than one that was wrong in the optimistic direction.

### What a follow-up needs

Either finish it — the reproduction is
`cargo test -p llm-model --features tiled-attention` failing on
`tiled_attention_reproduces_the_whole_matrix_logits` — or abandon it explicitly
in an ADR. Do not report a `wgpu` throughput figure for the tiled path until that
command is green.
