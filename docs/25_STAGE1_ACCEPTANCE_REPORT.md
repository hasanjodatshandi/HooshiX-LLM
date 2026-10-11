# Stage-1 Acceptance Report

This is the report section N of `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` asks for.
It states what was built, what was measured, what was verified, and what is not
true. Anything not run is marked **Not run**, never "passed".

## 1. Scope

Stage 1 is a correct 100M–150M dense decoder-only causal language model with a
Rust-native data, training, inference and serving pipeline. No Python runtime or
Python source is part of any required path.

The model is dense, decoder-only, with RMSNorm, RoPE, grouped-query-capable
multi-head attention, SwiGLU feed-forward blocks and tied embeddings. Grouped
query attention is representable in the configuration but **Stage 1 refuses
`num_kv_heads != num_heads`** rather than running an untested path.

## 2. What exists

| Crate or app | Responsibility |
|---|---|
| `llm-core` | Configuration, backend selection, typed errors |
| `llm-tokenizer` | Byte-level BPE, artifact with identity hash |
| `llm-data` | Source records, dedup, split, packing, checksummed shards, reader |
| `llm-model` | RMSNorm, RoPE, attention, SwiGLU, blocks, decoder, KV cache, loss |
| `llm-training` | AdamW, warmup+cosine, clipping, accumulation, metrics, validation |
| `llm-checkpoint` | Atomic publication, envelopes, model and optimizer state |
| `llm-inference` | Sampling, stop conditions, prefill, incremental decode |
| `llm-server` | Routing, limits, readiness, streaming, cancellation, shutdown |
| `apps/tokenizer-cli` | Train and inspect tokenizer artifacts |
| `apps/prepare` | Build shards from declared sources |
| `apps/train` | The training composition root |
| `apps/generate` | Generate from a checkpoint |
| `apps/serve` | Serve a checkpoint over HTTP |
| `apps/bench` | The benchmark baseline |

137 tracked files, 12 commits.

## 3. Release gate

Run from a **clean clone** of `https://github.com/hasanjodatshandi/HooshiX-LLM.git`
at `c6f811f`, into a fresh directory, with no prior build:

| Command | Result |
|---|---|
| `cargo fmt --all --check` | exit 0 |
| `cargo check --workspace` | exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| `cargo test --workspace` | exit 0, **337 tests passed, 0 failed** |
| `cargo test --doc --workspace` | exit 0 |

Zero files under `target/` are tracked. `.gitignore` matches `/target*/`, which
covers the alternate target directories the backend work created; an earlier
version matched only `/target` and 1500 build artifacts were committed once, which
is why the rule is written the way it is.

The clean-clone verification was **run twice**. The first attempt assigned the
clone path without a `$`, so the gates ran in the working directory instead of the
clone and the result proved nothing. The table above is from the corrected run,
whose log shows `pwd` inside the clone.

## 4. Dependency audit

`cargo audit` against the locked graph:

- **Vulnerabilities: none.**
- **Warnings: one.** `paste 1.0.15`, unmaintained since 2024-10-07
  (RUSTSEC-2024-0436).

`paste` reaches the graph through `burn-dispatch`, from Burn itself; no crate in
this workspace depends on it directly and no direct dependency can remove it. It
is a proc-macro used at compile time and it has no known vulnerability — the
advisory is that it is no longer maintained. It cannot be resolved here and will
resolve when Burn replaces it.

319 crates in the resolved graph. Direct dependencies are Burn, serde, clap,
axum, tokio, tokio-stream and the workspace's own crates. No database, cache,
queue, ORM or migration dependency exists, and none was added.

## 5. No Python in the required workflow

- **No `.py`, `.pyi`, `.ipynb` or `.pyx` file exists anywhere in the tree.**
- No tracked file outside `docs/` references `import torch`, `import numpy`,
  `python3`, `pyproject.toml`, `requirements.txt`, `conda` or `venv`.
- `docs/` mentions Python only to state that it is excluded.

Verified by searching the working tree and by `git grep` over tracked files.

## 6. Security and reliability review

| Concern | Finding |
|---|---|
| `unsafe` | **Denied at the crate level** in `llm-training`, `llm-server` and `apps/bench`; no `unsafe` block exists anywhere in the workspace. |
| Panics on production paths | None. Every `unwrap`/`expect`/`panic!` in the workspace is inside a `#[cfg(test)]` module, a `tests/` file or an `examples/` file, which is what `clippy.toml` relaxes the lint for. Verified by searching all `.rs` files. |
| Untrusted input | Validated at the boundary and before allocation: request byte length, tokenized length, token budget and sampling settings are all checked before any large allocation. Sampling settings are rejected before the prompt is tokenized. |
| Typed errors | Every crate has its own error enum with `thiserror`, preserving source context. No error is discarded silently. |
| Secret or prompt leakage | No credentials, tokens or keys exist in the tree. The server's error bodies carry the error's own message, which is written for a caller and contains no internal paths or stack detail. |
| Path handling | No path is computed from untrusted input. The server takes no path from a request. |
| Unbounded work | The server has no queue: a request that cannot be served is refused with `429`, not held. Concurrency is bounded by a semaphore whose permit travels with the work. |
| File writes | Checkpoint publication is atomic and refuses to replace an existing directory, so two runs cannot interleave into one directory. The training app checks the output directory before its first step rather than discovering the conflict at the first checkpoint. |
| Device memory | The server does not build an autodiff device; training does. `ensure_finite` reads results with `try_to_data`, so a device that has died becomes `CheckUnavailable` rather than a panic. |

**A known and deliberate gap:** the checkpoint suite's `content_sha256` tree
verification has never executed, because the bundler's sentinel short-circuits it.
The envelope and per-file checks do run. This is recorded rather than closed.

## 7. Measured baseline

Full report in `11_BENCHMARKING.md` section 15. Stage-1, `flex`, f32,
micro-batch 8 × sequence 512, on the reference machine:

| Figure | Value |
|---|---|
| cold start | 0.888 s |
| prefill | 716.6 tokens/s |
| TTFT | 0.179 s |
| decode | 31.3 tokens/s |
| training | 210.9 tokens/s |
| peak memory | 9681.4 MiB |
| checkpoint write | 72.7–292.9 MiB/s (varies; see below) |
| checkpoint read | 1432.2 MiB/s |
| `bench.exe` | 11.83 MiB |

The configured Stage-1 shape, `batch 8 × sequence 2048`, runs and produces
**139 tokens/s**, which is about **136 days** for the configured 1.64B tokens.
That is the honest number for this machine.

## 8. Backend parity

`flex` (reference) against `wgpu` (AMD Radeon R9 200, Vulkan), same weights, same
input:

| | elements | worst absolute | non-finite |
|---|---|---|---|
| logits | 10240 | 4.47e-7 | 0 |
| gradients | 151872 | 2.38e-7 | 0 |

Both are f32 rounding. `parameters without a gradient: reference 0, accelerated 0`.

**The GPU is 4.3× to 8.9× the reference path at the same shape.** Measured back
to back, same model, same tokens per step:

| shape | `flex` (CPU) | `wgpu` (GPU) | |
|---|---|---|---|
| 2 × 256 | 120 tok/s | **514 tok/s** | 4.3× |
| 1 × 512 | 43 tok/s | **383 tok/s** | 8.9× |
| 8 × 512 | **186 tok/s** | out of memory | — |

An earlier version of this report compared 514 tokens/s on the GPU against 237 on
the reference path and called the GPU "also faster". That comparison was between
**different shapes** — 512 tokens per step against 4096 — and it understated the
result by a wide margin. At equal work the GPU wins by 4.3× to 8.9×, and it is the
CPU that is poor at small batches rather than the GPU being merely competitive.

**The limit is memory, not speed.** The card caps at 512 tokens per micro-batch
and cannot run sequence 1024 at any batch size, so it cannot run the configured
Stage-1 shape at all. The 136-day figure is a **reference-path** figure for that
shape, and this GPU cannot improve it. On a card with enough memory the same code
would be several times faster; on this one it is faster only where it fits.

**CUDA and ROCm are Not run.** No NVIDIA card is present, and ROCm does not
support this card's generation. `wgpu` was built and measured; `cuda` and `wgpu`
were never built for any other machine.

## 9. What is not true

- **The performance work did not make the model faster.** The "51 → 225 tokens/s,
  4.4×" claim was withdrawn in `11_BENCHMARKING.md` section 12: measured back to
  back, the code before the work and after it are both about 237–238 tokens/s.
  What the work produced was correctness and capability, not speed: real gradient
  accumulation, a finiteness check that works, reproducible initialisation, an
  unblocked `cpu` backend, and a working `wgpu` path.
- **`cuda` has never been built or run.** No NVIDIA hardware exists here.
- **`wgpu` has been run only on one machine**, with one 4 GB card.
- **The `cpu` backend is about three times slower than `flex`** (72 against 235
  tokens/s) and costs 9–18 minutes of LLVM kernel compilation before its first
  step. It is correct, and it is slower.
- **Checkpoint write throughput varies by more than 4× between runs** on the same
  bytes, because the machine carries about 30% background load. It should be
  quoted as a range.
- **The server has never been load-tested.** It has been exercised over a socket
  for correctness: health endpoints, a completion, streaming, limit refusals,
  `429` under concurrency, and clean shutdown.
- **BF16 is not implemented.** Checklist F's optional item is unchecked; FP32 was
  not finished first in the sense that item asks for.
- **One shard per split.** There is no shard rotation and no data parallelism
  beyond the configuration field.
- **No chat endpoint exists**, deliberately: a base pretraining model is not an
  instruction follower.
- **There is no git history before the first commit.** All work is in 12 commits.

## 10. Verdict

Stage 1 is **complete against the checklist**, with the exceptions recorded in
section 9 and the two open items inside it: BF16 (F, optional) and the
`content_sha256` tree check.

The model trains, checkpoints, resumes, generates, serves over HTTP, and agrees
with itself across two backends to f32 rounding. It is **not fast**: about 136
days for the configured Stage-1 run on this hardware, and the largest single
cause is that attention and the GEMMs beneath it are not built on a real BLAS.

Stage 2 should not begin until the two open items are either closed or explicitly
deferred by an ADR.
