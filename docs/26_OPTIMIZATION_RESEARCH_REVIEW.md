# 26 — Optimization research review, and three work orders

**Date:** 2026-10-11
**Status:** review complete. WO-1, WO-2 and WO-3 were executed in `c62f899` — the TinyStories path measured 741 tokens/s (about 1.28 days for the configured run) and the `cpu` backend record is reconciled in every document that carried it.
**Scope of this document:** analysis and documentation only. It proposes no source change.

---

## 1. Why this document exists

Two external research reports were commissioned to answer one question: **can the
Stage-1 training run be made materially faster in software, on this machine?**

- **Report A** ("Rust LLM Training Optimization Research — Burn 0.22"). Ranked
  fused linear + Cut Cross-Entropy first, the CubeCL `cpu` backend second, and a
  modern CUDA GPU third.
- **Report B** (a second, independent review). Ranked the CubeCL `cpu` backend
  first, chunked linear + cross-entropy with a custom backward second, and Burn's
  built-in gradient checkpointing third; concluded that no software path is
  demonstrated to give a multi-fold win.

This document records what survived contact with the source and with this
repository's own measurements, and states three work orders. Both reports were
written from the same brief, and both were written **without seeing the
repository**, which is why several of their recommendations are already closed
here — and why one of them exposed a real contradiction inside our own
documentation.

## 2. What claims were checked against

1. **Burn 0.22.0 as vendored by Cargo on this machine:**
   `C:\Users\Coder\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\`
   — `burn-0.22.0`, `burn-autodiff`, `burn-backend`, `burn-backend-extension`,
   `burn-cpu`, `burn-cubecl`, `burn-dispatch`, `burn-flex`, `burn-nn`,
   `burn-optim`, `burn-tensor`, `burn-train`, `burn-wgpu`, `cubecl-*-0.11.0`.
2. **The repository** at `d4da033`, plus the uncommitted TinyStories configuration.
3. **Source, not documentation.** Two of the claims reviewed were documentation-level
   claims, and one of them did not survive: see §3, rows 10 and 11.

Every claim below carries either a `file:line` into the vendored source, a
reference into this repository's own documents, or an explicit note that it is
arithmetic performed here.

## 3. Verification results

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | `Device::cpu()` is a CubeCL/LLVM-JIT CPU backend, distinct from `flex` | **True** | `burn-cpu-0.22.0`, `cubecl-cpu-0.11.0`; already wired at `apps/train/src/device.rs:79` |
| 2 | `device.autodiff().gradient_checkpointing()` exists in 0.22 | **True** | `burn-tensor-0.22.0/src/device.rs:540`; panics if autodiff is off (line 535) |
| 3 | `BalancedCheckpointing` classifies operations as memory-bound vs compute-bound and uses `RetroForward`; it is **not** whole-block rematerialisation | **True** | `burn-autodiff-0.22.0/src/checkpoint/strategy.rs:36-95` (`ComputingProperty::MemoryBound { retro_forward }`) |
| 4 | `#[backend_extension(...)]` is a real Burn 0.22 mechanism for a custom op with custom backward | **True** | `burn-backend-extension-0.22.0/src/lib.rs:113,190` |
| 5 | `burn-flex` has **no** BLAS/MKL feature; its GEMM is the pure-Rust `gemm` crate | **True** | `burn-flex-0.22.0/Cargo.toml`; `burn-flex-0.22.0/src/ops/matmul.rs:83-104`; already proved empirically in `11_BENCHMARKING.md` §16 |
| 6 | `gemm` is a SIMD/FMA micro-kernel GEMM, not a naive triple loop, and flex has parallel paths | **True**, and it corrects the brief | `burn-flex` features include `simd` and `rayon`; `Cargo.toml` (workspace) already notes `gemm`/`pulp` select wider SIMD kernels |
| 7 | `Adafactor` exists in 0.22 with factored second moments and **no** first moment | **True** | `burn-optim-0.22.0/src/optim/adafactor.rs:53-55` (doc comment), `:76-80` (state) |
| 8 | `AdafactorConfig` defaults `relative_step = true` and `scale_parameter = true` | **True** | `burn-optim-0.22.0/src/optim/adafactor.rs:39-43`; the semantics are documented at `:18-22` |
| 9 | BF16 is dropped on the `cpu` backend because the LLVM dialect has no bfloat type | **True, verbatim** | `burn-cpu-0.22.0/src/lib.rs:58-59`: "BF16 is dropped: the LLVM dialect has no bfloat type to compute with." |
| 10 | `CUBECL_WGPU_MAX_TASKS` exists and the default is 32 | **True** | `cubecl-wgpu-0.11.0/src/runtime.rs:352-368` |
| 11 | Burn 0.22 exposes **used vs reserved** device-memory reporting | **Not found** | No such API in `burn-tensor`, `burn-backend`, `burn-cubecl`. The only memory metric is `burn-train-0.22.0/src/metric/memory_use.rs`, which reads **process RAM** via sysinfo. Treat this claim as unsupported |
| 12 | `burn-tch` is deprecated as of 0.22.0 | **True** | `burn-tch-0.22.0/README.md` |
| 13 | `blas-src` alone does not attach BLAS to a tensor path | **True** | matches `11_BENCHMARKING.md` §16: OpenBLAS was linked and the DLL never entered the process |
| 14 | Burn has a DDP strategy using gradient collectives | **True, with report A's caveat** | `burn-train-0.22.0/src/learner/supervised/strategies/ddp/worker.rs:13`; the Cube collectives are CUDA-only, so this does not help the WGPU path |
| 15 | 128 MiB of raw logits at 1,024 tokens; 2 GiB at 8 × 2,048 | **True** (arithmetic here) | `1024 × 32768 × 4 = 134,217,728`; `8 × 2048 × 32768 × 4 = 2,147,483,648` |
| 16 | Fixed training state is 1.76 GB (4 × 110 M × f32) | **True** (arithmetic here) | matches `11_BENCHMARKING.md` §15 |
| 17 | Throughput needed for 60/30/14/7 days is 316/632/1,354/2,709 tok/s | **True** (arithmetic here) | `1.6384e9 / days / 86400` |
| 18 | Removing attention entirely bounds the gain at 1.67× (Amdahl, attention = 40 %) | **True** (arithmetic here) | `1 / 0.6` |

Two claims in **Report A** were rejected outright: the used/reserved memory
reporting in row 11, and the framing that Cut Cross-Entropy is the first thing to
build (see F2 below). Report B contained no claim that could be falsified.

## 4. Findings

### F1 — The CubeCL `cpu` backend is already measured, and our own records contradict each other

Both reports rank "benchmark `Device::cpu()`" first, as the cheapest untried path.
It is not untried — and the repository currently states two opposite conclusions
about it:

| Where | Says |
|---|---|
| `11_BENCHMARKING.md` §11 (line 167-198) | `cpu` is **1.37× faster** than `flex` at 8 × 512 (70 against 51 tok/s) |
| `11_BENCHMARKING.md` §14 row 3 (line 326) | "`cpu` remains **3× behind** `flex`" |
| `25_STAGE1_ACCEPTANCE_REPORT.md` line 175 | "`cpu` is about **three times slower** (72 against 235)" |
| `apps/train/src/device.rs` `backend_caveat()` | "about **three times slower** … 72 against 235 tokens/s" |
| `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` line 671 | "about **three times slower**" |

The contradiction resolves against §11, and the arithmetic is unambiguous. All
figures are at batch 8 × sequence 512 = 4,096 tokens per step:

- `cpu`: **56.9 s** and **58.3 s** per step in §14 (72 and 70 tok/s), and **70 tok/s**
  in §11's own table. The CPU figure reproduces across two sessions.
- `flex`: **80.8 s** per step in §11's table (51 tok/s) — the baseline that §12
  **withdrew**: "The 'before' number of 51 tokens/s was wrong. The same code path,
  on the same machine, measures 238 today … It was never reproducible and should
  not have been published as a baseline."
- `flex` in the sessions where it is trusted: **17.2-17.27 s** (§12) and
  **18.40 s** (§13), i.e. 238 and 223 tok/s.

So: `4096 / 56.9 = 72.0` and `4096 / 18.40 = 222.6`, `4096 / 17.20 = 238.1`.
**`cpu` is 3.1× to 3.3× slower than `flex`, not 1.37× faster.** §11's ratio column
is computed against the withdrawn `flex` column, so all three of its ratios are
void. §12's own rule is the one that decides this: "only measurements taken
within one run or alternating between binaries mean anything."

**Consequence:** the first-ranked recommendation in both reports is a closed,
negative result. No further research on it is worth commissioning.

### F2 — Cut Cross-Entropy attacks a bottleneck that is not the binding one here

Both reports rank a fused/chunked linear + cross-entropy highly, on the grounds
that the output head materialises a `[tokens, vocabulary]` tensor. Report B is
explicit that its purpose is to make `sequence = 2048` fit, not to go faster.

Our own measurements put a ceiling on what that can buy:

- **The head is ~15 % of a step.** §13 fits `step = 2.7 s + 12 × 1.31 s`: the twelve
  blocks are ~85 % of a step; embedding, LM head, objective and optimizer together
  are the rest.
- **At 8 × 512, logits + `log_softmax` are ~1 GiB out of a 9,680 MiB peak** — about
  11 %, and that peak is dominated by retained activations, not by the head.
- **The GPU ceiling is not explained by logits.** The measured threshold is between
  512 and 1,024 total tokens (`14_STAGE1_IMPLEMENTATION_CHECKLIST.md` lines 646-647),
  where raw logits are 64-128 MiB. Both reports independently noticed this; §17's
  line "the 1024-token ceiling is the logits tensor, not attention" is therefore
  **directionally plausible but not yet demonstrated**. What is demonstrated is the
  opposite for the CPU path: the allocation that fails at 8 × 2,048 is the 1.5 GiB
  attention score tensor (§12), and gradient accumulation is what fixed it.

**Consequence:** a custom `linear_cross_entropy` op is a **memory** measure whose
expected benefit on the shapes that already run is small, whose cost is 4-10 days
and a custom backward, and whose stated purpose only becomes relevant if a GPU
with enough VRAM arrives. It stays deferred. If it is ever built, it must be
justified by a measured allocation peak at the target shape, not by the full-target
arithmetic.

### F3 — The "136 days" figure no longer describes the active path

Every report was written against `configs/stage1-110m.toml`. The active path is
`configs/tinystories-small.toml` (6 layers, hidden 512, vocab 8192, sequence 512,
20,000 steps), which is uncommitted and unmeasured. It is **81.92 M tokens**, not
1.64 B.

Our own ablations (§13, one binary, one run) bound it: 12 → 6 layers = **388 tok/s**
at 8 × 512, and vocabulary 32768 → 8192 = **258 tok/s**. The TinyStories model takes
both cuts and halves the hidden size as well, so **600-800 tok/s is a plausible
estimate — and it is an estimate, not a measurement.** No report knows about this
path, and it is the only lever anyone has identified that moves the calendar.

## 5. Work orders

Three. They are ordered, and none of them changes model or training source code.

### WO-1 — Measure the TinyStories configuration, and report the real wall-clock

**Goal.** Replace an estimate with a measurement for the configuration that is
actually going to be trained, and state the resulting wall-clock in the same
place every other throughput figure lives.

**Preconditions.**

- No training run is in progress. `Get-Process | Where-Object { $_.ProcessName -match 'train|cargo' }`
  must be empty. §14 row 6 records a measurement contaminated by a concurrent
  process; do not add another.
- Record the environment block required by `11_BENCHMARKING.md` §2 with the run.

**Steps.**

1. Configured shape of the TinyStories run, warm:

   ```powershell
   $env:CONFIG="configs/tinystories-small.toml"
   $env:BATCH="8"; $env:SEQUENCE="512"; $env:STEPS="5"; $env:WARMUP="2"
   cargo run --release -p llm-training --example throughput
   ```

2. The shape curve, so the number is not a single point:
   `BATCH=4 SEQUENCE=1024`, `BATCH=2 SEQUENCE=2048`, `BATCH=1 SEQUENCE=2048`.
   If the training configuration uses gradient accumulation, repeat the configured
   shape with `ACCUM` set to the configured value.
3. **If, and only if, a "the small model is N× faster than the 110 M model" claim
   is to be written**, measure both configurations **alternating in one session**
   — `^CONFIG` between the two, same shapes, same warmup. §12 forbids the
   cross-session form of that comparison.
4. Compute the wall-clock from the measured rate, do not extrapolate it:
   `days = 20,000 × 8 × 512 / tokens_per_second / 86,400`, and state that it
   excludes validation and checkpoint time.
5. Record the result as a new section in `11_BENCHMARKING.md`, with the
   environment, the warmup policy, and the cold-start cost reported separately.

**Acceptance criteria.**

- The number is measured by the existing harness at the configured shape, with
  warmup excluded and at least five timed steps.
- The parameter count printed by the binary is consistent with
  `configs/tinystories-small.toml`.
- The wall-clock is stated with the token budget it assumes.
- If peak memory is reported, the method is named (the binary does not measure it:
  `apps/bench/src/main.rs:21-25`; §15 reports the OS's own `PeakWorkingSet64`).
  Otherwise write "not measured".

**Not acceptable.** Quoting 388 or 600-800 tok/s as the result. Those are the
anchor and the estimate; the measurement replaces them, and if it disagrees with
them, the measurement wins and the estimate is recorded as wrong.

**Rollback.** None needed: documentation and a benchmark run only. If the
configuration fails to load, that is a defect to report, not to work around.

---

### WO-2 — Reconcile the `cpu` backend record

**Goal.** The repository must stop publishing two opposite conclusions about the
same backend. This is a documentation defect with four surviving copies.

**Steps.**

1. In `11_BENCHMARKING.md` §11, mark the `flex` column as the measurement
   withdrawn in §12, mark all three ratios void, and add the valid same-session
   comparison from §14 (56.9 s / 72 tok/s on `cpu` against that session's `flex`
   baseline of ~17-18 s per step, i.e. **3.1×-3.3× slower**). Keep the section:
   the `cpu` figures are sound and the JIT warmup finding (735 s at 8 × 512,
   1,082 s at 1 × 2048) is still the most useful thing in it.
2. Optionally, and preferably, replace the table with one alternating
   back-to-back measurement, which is what §12 requires and what §12 itself did
   for the LTO claim:

   ```powershell
   # flex
   $env:BATCH="8"; $env:SEQUENCE="512"; $env:STEPS="5"; $env:WARMUP="2"
   cargo run --release -p llm-training --example throughput
   # cpu -- a separate binary; the first step costs 9-18 minutes of LLVM JIT,
   # so WARMUP must absorb it before anything is timed.
   cargo run --release -p llm-training --example throughput --no-default-features --features cpu
   ```

   Alternate the two binaries at least twice and report all four numbers. Feature
   names are in `crates/llm-training/Cargo.toml:20-33`.
3. Then check that `11_BENCHMARKING.md` §11 and §14, `25_STAGE1_ACCEPTANCE_REPORT.md`
   line 175, `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` line 671 and the
   `backend_caveat()` text in `apps/train/src/device.rs` all state the same
   conclusion and cite the same measurement.

**Acceptance criteria.** No document in the repository claims `cpu` is faster than
`flex`, and every surviving `cpu` figure names the session it came from.

**Constraint.** `device.rs` is a comment change only. The feature, the device
constructor and the caveat must keep working: `cpu` remains a correct backend that
is slower, and removing the ability to select it would remove the ability to
measure it.

---

### WO-3 — Record the decisions, and close the research brief

**Goal.** Three decisions are currently only in a conversation. Put them where the
next reader finds them, and give the next research request an accurate "already
closed" list.

**Steps.**

1. Add ADRs to `16_DECISIONS.md` (the file currently ends at ADR-018):
   - **The CubeCL `cpu` backend is not a speed path.** Status: Accepted. Evidence:
     WO-2's measurement. It stays selectable and correct; it is not a candidate.
   - **Fused/chunked linear + cross-entropy is deferred until a GPU path exists.**
     Status: Deferred. Reason: F2 — it is a memory measure against a head that is
     ~15 % of a step and ~11 % of peak memory at the shapes that run, and the
     measured GPU ceiling is not explained by the logits tensor alone.
   - **Gradient checkpointing and Adafactor remain untried and are memory-only.**
     Status: Deferred, with the reason for each: `gradient_checkpointing()` is one
     line (`burn-tensor-0.22.0/src/device.rs:540`) but recomputes, so it is a
     memory-for-time trade and cannot improve the priority-one metric; Adafactor
     is a change of optimization algorithm, and with its defaults
     (`relative_step = true`, `scale_parameter = true`) the configured
     `learning_rate` stops being an absolute rate, so it needs a convergence
     decision, not a wiring decision.
2. The research brief lives outside this repository. Its replacement "do not
   suggest" entries are written out ready to paste in Appendix A, in the brief's
   own language. Merge them there:
   - the CubeCL `cpu` backend — measured, 3.1×-3.3× slower than `flex`
     (`11_BENCHMARKING.md` §11/§14), with a 9-18 minute LLVM JIT warmup per shape;
   - "there is no optimised GEMM under Burn" — **wrong as written**: `flex`'s GEMM
     is the `gemm` crate with SIMD/FMA microkernels and `rayon`; what is missing is
     BLAS, which is a different claim;
   - Burn "used vs reserved" device-memory reporting — not present in 0.22;
   - Cut Cross-Entropy as a first step — a memory measure, deferred per F2, and not
     a way to reduce the `cpu`-bound wall-clock;
   - and the correction that the 1,024-token GPU ceiling cannot be attributed to
     the logits tensor alone (128 MiB at that shape).
3. Register this document and the existing `25_STAGE1_ACCEPTANCE_REPORT.md` in
   `docs/manifest.json` — 25 is present in the directory but absent from the
   manifest's `files` list.

**Acceptance criteria.** `16_DECISIONS.md` carries the three decisions with their
evidence; Appendix A carries the five entries that replace the brief's closed list; the manifest
lists every file in `docs/`.

## 6. What is closed, in one table

| Idea | Status | Why |
|---|---|---|
| CubeCL `cpu` backend as a speed path | **Closed, negative** | 3.1×-3.3× slower than `flex`; see F1 |
| A real BLAS under Burn | **Closed, negative** | §16: linked, never called |
| `ndarray` backend | **Closed, negative** | 1.7× slower (§14 row 1) |
| Hand-written tiled attention | **Closed, negative** | correct, 6 %/42 %/80 % slower, reverted (§17) |
| Burn's own attention as a speedup | **Closed, equivalent** | identical logits, ~1 % slower (§17) |
| Fused RMSNorm | **Closed, does not exist** | `burn::nn::RmsNorm` is the same sequence |
| `burn-fusion` | **Closed, wrong target** | not enabled for `flex`; no gain on `cpu` (§14 row 3) |
| Smaller vocabulary as a speed fix | **Closed** | 15 % at 4096 against a tokenizer-contract change (§14 row 4) |
| Gradient accumulation as a speedup | **Closed as a speedup** | it is a memory fix; it is kept for that (139 tok/s at 8 × 2,048) |
| LTO, `codegen-units`, mimalloc | **Closed, noise** | §12, §14 row 6 |
| AVX-512 on the i7-12700K | **Closed** | the hardware does not have it |
| 8-bit optimizer states | **Closed for now** | no validated Burn equivalent |
| Custom CubeCL GEMM | **Closed** | report B's own conclusion: no evidence of a win, weeks of work |
| Fused/chunked linear + cross-entropy | **Deferred** | F2 |
| Gradient checkpointing, Adafactor | **Deferred, memory-only** | WO-3 |
| A newer GPU with more VRAM | **Open, and the only identified large lever** | both reports and `25_STAGE1_ACCEPTANCE_REPORT.md` agree |
| The TinyStories configuration | **Open — WO-1** | the active path; unmeasured |

## 7. Coordination

- `crates/llm-core/tests/config_files.rs` is modified in the working tree and
  `configs/tinystories-small.toml` plus `configs/tinystories-sources.json` are
  untracked. They belong to the concurrent TinyStories work. These work orders
  **read** `configs/tinystories-small.toml` and touch none of the three.
- Do not run any benchmark in WO-1 or WO-2 while a training run is active. The
  machine carries roughly 30 % background load (§12), and §14 row 6 records a
  measurement that had to be discarded for exactly this reason.
- Commit these documents separately from the TinyStories configuration so the
  history stays readable.

## 8. Non-goals

- No change to model, loss, optimizer, tokenizer, data or checkpoint code.
- No custom kernel, no backend work, no dependency addition.
- No purchase recommendation: hardware is a decision for the project owner, and
  the only thing this review adds to it is that the software alternatives are
  measured and closed.

---

## Appendix A — ready-to-paste entries for the research brief

The brief is a prompt that lives **outside** this repository, so its corrections
are recorded here rather than applied to it. The entries below are in the brief's
own language; merge them into its "do not suggest" section, and the three
corrections into its measured-figures section.

<!-- paste from here -->

**افزودنی به فهرست «⛔ چیزهایی که اندازه گرفتیم و بسته‌اند»:**

| راه | نتیجه | چرا |
|---|---|---|
| بک‌اند `cpu` (CubeCL) | ❌ ۳٫۱ تا ۳٫۳ برابر کندتر از `flex` | اندازه‌گیری‌شده: ۵۶٫۹ و ۵۸٫۳ ثانیه در هر گام ۸×۵۱۲ (۷۰ تا ۷۲ توکن/ثانیه) در برابر ۱۷٫۲ تا ۱۸٫۴ ثانیه برای `flex` در همان سشن‌ها. علاوه بر آن، اولین گام ۹ تا ۱۸ دقیقه صرف JIT با LLVM می‌کند. منبع: `docs/11_BENCHMARKING.md` بخش‌های ۱۱ و ۱۴ |
| Cut Cross-Entropy به‌عنوان کار بعدی | ⏸ موکول | یک اقدام حافظه‌محور است، نه سرعت‌محور: سر خروجی حدود ۱۵٪ یک گام است و logits به‌همراه `log_softmax` در ۸×۵۱۲ حدود ۱ گیگ از پیک ۹٫۷ گیگابایتی. اولویت اول ما زمان است، نه حافظه. |
| gradient checkpointing و Adafactor | ⏸ موکول، فقط حافظه | checkpointing یک ترید «حافظه در برابر زمان» است؛ Adafactor الگوریتم بهینه‌سازی را عوض می‌کند و با پیش‌فرض‌هایش (`relative_step` و `scale_parameter` روشن) مقدار `learning_rate` دیگر نرخ مطلق نیست. |

**تصحیح در بخش اعداد اندازه‌گیری‌شده:**

- جملهٔ «هیچ GEMM بهینه‌ای زیر عملیات تنسوری Burn نیست» **غلط است**. GEMM مسیر `flex` همان crate `gemm` با microkernelهای SIMD/FMA و موازی‌سازی `rayon` است. چیزی که وجود ندارد BLAS است، که ادعای دیگری است.
- انتساب سقف ۱۰۲۴ توکنی GPU به «تنسور logits» **اثبات‌نشده** است: در آن شکل، logits خام فقط ۱۲۸ مگابایت است. سقف باید با اندازه‌گیری پیک تخصیص جدا شود، نه با حساب شکل کامل هدف.
- ادعای «Burn 0.22 گزارش حافظهٔ used در برابر reserved دارد» در سورس ۰٫۲۲ وجود ندارد؛ هیچ API‌ای برای آن پیدا نشد.

<!-- to here -->
