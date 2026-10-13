# 28 — Bilingual pilot: status, plan, and what is outstanding

**Created:** 2026-10-12
**Purpose:** this document exists because the conversation that produced it is
compacted and lost. Everything needed to answer "where are we, what was I
supposed to do, what is left, why was each thing done, and what did it give" is
here rather than in a chat log. Update it as work lands.

---

## 1. Where we are, in one paragraph

Stage 1 of the main roadmap is **done and committed**: a Rust-native LLM stack
that trains, checkpoints, resumes, generates, serves over HTTP, and agrees with
itself across two backends to f32 rounding. A TinyStories model (24.6 M
parameters) is **trained and working** — it produces coherent English. The
original Stage-1 target (110 M parameters, 1.64 B tokens) **cannot be trained on
this machine in reasonable time**: 136 days at the measured 139 tokens/s. A new,
separate plan exists for a **bilingual Persian/English/code 110 M model**. Its
Stage 1 must be validated with a small pilot before anything large is attempted.

---

## 2. The bilingual plan, in short

A full plan lives outside this repository. Its essentials:

| | |
|---|---|
| Model | 110 M dense decoder-only, same architecture and code as now |
| Languages | Persian + English + code, trained jointly from the start |
| Stages | **Three**: (1) pretraining, (2) SFT + distillation, (3) advanced post-training |
| Data | Common Pile / Comma v0.1 (English, code), FineWeb2 `fas_Arab` (Persian) |
| Start | Random weights. **No TinyStories weight is ever loaded.** |
| Tokenizer | Shared, byte-level BPE, 24k–32k, frozen before pretraining |

**Do not do Stage 2 or 3 work now.** This repository is at roadmap Phase 1.
SFT belongs to Phase 2; DPO/F-DPO/RLCR/RLVR belong to Phase 5. Respect the phase.

---

## 3. The pilot that was agreed

Train **100 M tokens** of bilingual data end to end. If the model produces
coherent Persian, scale up. If not, the fault is found in days rather than
months.

| Language | Share | Tokens | Source | Size |
|---|---|---|---|---|
| English | 50 % | 50 M | Comma v0.1 `project_gutenberg` | 114 MB per shard |
| Persian | 40 % | 40 M | FineWeb2 `data/fas_Arab` | **4.6 GB per shard** |
| Code | 10 % | 10 M | Comma v0.1 `stackv2_edu` | 949 MB per shard |

**Why these sizes and not everything:** the full Stage-1 run needs about 11 GB of
text. The sources hold about 595 GB. Downloading all of it to use 2 % is waste.
Download proportionally, spread across shards, and expand only after the pipeline
is proven.

---

## 4. What has been done

### 4.1 Stage 1 of the main roadmap — complete

All of `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` sections A–N, with two items open
inside them: BF16 (optional, unchecked) and the checkpoint suite's
`content_sha256` tree check, which has never executed. See
`25_STAGE1_ACCEPTANCE_REPORT.md`.

### 4.2 TinyStories training — running, working

| | |
|---|---|
| Model | 24,648,192 parameters, 6 layers, hidden 512, vocab 8192, sequence 512 |
| Data | TinyStoriesV2 GPT-4, 2,717,495 stories, 520,848,384 tokens prepared |
| Backend | `flex`, f32, on the i7-12700K |
| Measured rate | **741 tokens/s** at 8×512, 799 excluding one outlier |
| Configured run | 20,000 steps = 81.92 M tokens → **1.28 days** at that rate |
| Status | reached step 15,840 while this was written; ~9 hours left |
| Output | **coherent English.** "Once upon a time" → "…there was a little girl named Lily. She had a big, soft, velvet toy that she loved to play with." |

**It cannot hold a conversation, and never will at this stage.** Asked "Hello, how
are you?" it wrote a story in which a fairy answers. Asked "سلام چطوری" it
repeated one token to the budget. A base model continues text; it does not answer.
Chat is Stage 5 work.

### 4.3 Pilot data — downloaded

| File | Size |
|---|---|
| `artifacts/raw/bilingual/fa-fineweb2-000.parquet` | 4618.1 MB |
| `artifacts/raw/bilingual/en-gutenberg-00.jsonl.gz` | 114.3 MB |
| `artifacts/raw/bilingual/code-stackv2edu-00.jsonl.gz` | 948.5 MB |

Downloaded with `curl.exe` against the Hub's resolve URLs. **No Python.** The
pattern is `huggingface.co/datasets/{repo}/resolve/{revision}/{path}`, and the file
tree comes from `huggingface.co/api/datasets/{repo}/tree/{revision}`.

### 4.4 `apps/convert` — written and working

Turns a Parquet file into one JSON object per line so the existing pipeline can
read it. Kept in an app rather than in `llm-data` deliberately: a Parquet reader
brings an Arrow tree, and the crate the model links against should not carry it.

Two build faults were fixed on the way: `parquet` needs its own
`flate2-rust_backend` feature rather than a bare `flate2`, and `arrow-array` must
match the version `parquet` uses (60, not 56).

**Verified:** it read the Persian shard and produced Persian text.

**Finding from that read:** FineWeb2 is raw web text and carries **blog
boilerplate** — `نوشته شده در ... توسط behrouz`, navigation words like `درباره`,
stray `+`. Filtering is not optional, and `29_DATA_CLEANING_RESEARCH.md` is the
research into what filtering to build.

---

## 5. What is outstanding, and why

### 5.1 Blocked on the TinyStories run finishing

| Task | Why it waits |
|---|---|
| Read the final training report | the app prints it only at the end |
| **Read the validation loss** | computed every 500 steps but **printed only in the final report**. The log has zero validation lines, so overfitting cannot be seen during a run. **This is a defect in `apps/train` to fix.** |
| Generate text from the step-20,000 checkpoint | needs the checkpoint |

### 5.2 Blocked on a quiet machine

No benchmark may run while a training process is running. `docs/11` §14 row 6
records a measurement that had to be discarded for exactly this reason, and this
project has since produced a **false regression** the same way.

| Task | Why it matters |
|---|---|
| **`RAYON_NUM_THREADS` sweep** (2, 4, 8, 20) | **the unidentified bottleneck.** The CPU runs at 16 % of FLOP peak, 20 % of cores, 0.4 % of memory bandwidth. Neither compute nor bandwidth nor cores is saturated, and the cause is not known. A first attempt was run while training was active and was pure noise — it must be redone on an idle machine. |
| **Measure the 24.6 M model on `wgpu`** | never done. The GPU is 4.3×–8.9× the CPU for the 110 M model; whether that holds for a 24.6 M model is unknown, and it was a direct question from the project owner. |

### 5.3 Blocked on CPU being free

| Task | Cost |
|---|---|
| Convert the full 4.6 GB Persian Parquet | heavy |
| Train the bilingual tokenizer (24k–32k, balanced across three languages) | heavy |
| Prepare the bilingual shards | heavy — took 89 minutes for a comparable TinyStories run |

### 5.4 Awaiting a decision

| Question | Options |
|---|---|
| Which cleaning filters the pilot uses | see `29_DATA_CLEANING_RESEARCH.md`; the plan needs a chosen, documented set |
| Whether code stays in the pilot | the owner said **yes, it must** |

---

## 6. Known facts that must not be re-derived

Every one of these was measured and is recorded in `11_BENCHMARKING.md`:

| Backend | tok/s, 110 M at 8×512 |
|---|---|
| `flex` (reference) | 223–238 |
| `cpu` (CubeCL) | 70–72 — **3.1×–3.3× slower** |
| `ndarray` | 1.7× slower than `flex` |
| `wgpu` (4 GB R9 200) | 4.3×–8.9× **faster** at shapes that fit; ceiling 512 tokens/micro-batch |

**Closed, do not retry:** BLAS under Burn (linked but never called), AVX-512 on
this CPU (absent), hand-written tiled attention (correct, 6–80 % slower),
Burn's own attention as a speedup (identical, ~1 % slower), fused RMSNorm (does
not exist in Burn 0.22), compiler flags, mimalloc, smaller vocabulary, gradient
accumulation as a speed measure.

**The one open large lever is a GPU with more VRAM — 16 GB minimum, 24 GB safer.**

---

## 7. How to update this document

When a task in section 5 lands, move it into section 4 with its result, and say
what it gave. When a new blocker appears, add it to section 5 with the reason it
is blocked. Keep section 6 to things that were measured; an estimate belongs in
the text where it is made, labelled as an estimate.
