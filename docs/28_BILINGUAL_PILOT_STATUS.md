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
| English | **55 %** | 55 M | Comma v0.1 `project_gutenberg` | 114 MB per shard |
| Persian | **45 %** | 45 M | FineWeb2 `data/fas_Arab` | **4.6 GB per shard** |
| ~~Code~~ | **0 % in the pilot** | — | Comma v0.1 `stackv2_edu` | downloaded, held back |

**Why these sizes and not everything:** the full Stage-1 run needs about 11 GB of
text. The sources hold about 595 GB. Downloading all of it to use 2 % is waste.
Download proportionally, spread across shards, and expand only after the pipeline
is proven.

### 3.1 Decision: code is excluded from the pilot's training data, and kept in the tokenizer

**Decided 2026-10-12.** The plan requires Persian, English **and code** in the
mixture from the start of pretraining. That is right for the full run and wrong
for the pilot, and the difference is scale.

| Run | Total tokens | Code at 10 % | Does code teach anything? |
|---|---|---|---|
| **Pilot** | 100 M | **10 M** | **No.** Spread over Python, JS, Rust and the rest that is 1–2 M tokens per language. A model needs hundreds of millions of tokens of a language before it can do anything with it. |
| **Full Stage-1** | 2–3 B | **200–300 M** | Yes. Enough to matter. |

At the pilot's scale code is three costs and no benefit: a third variable to
debug when Persian comes out wrong, vocabulary taken from Persian in a 24k
tokenizer split three ways, and ten percent of a hundred million tokens spent on
tokens that teach `def`, `{` and `return` and nothing else.

**But the tokenizer is still trained on all three.** This is the part that must
not be got wrong. A tokenizer built without code would have to be **rebuilt** when
code is added to the full run, and rebuilding it changes the embedding shape and
**invalidates every checkpoint trained with the old one**. So:

| Stage | English | Persian | Code |
|---|---|---|---|
| Tokenizer training | yes | yes | **yes** — so it never needs rebuilding |
| Pilot model training | 55 % | 45 % | **no** |
| Full Stage-1 training | 50 % | 40 % | **10 %**, as the plan requires |

The tokenizer is built once, frozen, and used unchanged by both runs.

**Code is not dropped, it is deferred to its own experiment.** Once the bilingual
pilot works, a small run adds code as the single changed variable and measures
what it is worth. That is the one-change-per-run discipline of
`29_DATA_CLEANING_RESEARCH.md` section 5, applied instead of guessing.

**The 949 MB code shard is already downloaded** and stays on disk for that
experiment.

### 3.2 Why 55/45 and not 50/40/10

Removing code leaves 100 % to split between two languages. The 55/45 split keeps
English slightly ahead, which is deliberate: the tokenizer, the evaluation sets
and most of the tooling are English-first, and a Persian share that fails will be
visible sooner against an English share that works. The plan's 50/40/10 is a
three-language ratio and does not survive the removal of the third language
unchanged.

---

## 4. What has been done

### 4.1 Stage 1 of the main roadmap — complete

All of `14_STAGE1_IMPLEMENTATION_CHECKLIST.md` sections A–N, with two items open
inside them: BF16 (optional, unchecked) and the checkpoint suite's
`content_sha256` tree check, which has never executed. See
`25_STAGE1_ACCEPTANCE_REPORT.md`.

### 4.2 TinyStories training — **finished, and it worked**

| | |
|---|---|
| Model | 24,648,192 parameters, 6 layers, hidden 512, vocab 8192, sequence 512 |
| Data | TinyStoriesV2 GPT-4, 2,717,495 stories, 520,848,384 tokens prepared |
| Backend | `flex`, f32, on the i7-12700K |
| Configured run | 20,000 steps = 81,760,000 tokens |
| **Result** | **completed, exit code 0, 47.3 hours** |
| Loss | first 9.0471, best 1.1332, last 1.4902 |
| **Validation** | **3.0262 → 1.4191, monotone, still falling at the final step** |
| Stability | 0 refused steps, 0 abandoned, 0 non-finite losses |
| Checkpoints | 10, steps 2000 through 20000 |

**The model was not overfitting. It was undertrained.** The validation curve fell
at every one of the forty evaluation points, from 3.0262 at step 500 to 1.4191 at
step 20000, with no reversal anywhere. And the run covered **15.7 % of one epoch**
— the training shard holds 1,017,282 sequences and the run consumed 159,687 of
them, so `apps/train` logged **0 passes**. More steps would have kept helping.

**Output is coherent English.** From the final checkpoint:

```text
"Once upon a time"
→ ", there was a little girl named Lily. She loved to play with her toys and
   eat yummy food. One day, she found a big, red apple in"
```

**It cannot hold a conversation, and will not at this stage.** Asked "Hello, how
are you?" it wrote a story in which a fairy answers. Asked "سلام چطوری" it
repeated one token to the budget. A base model continues text; it does not answer.
Chat is Stage 5 work.

**Three findings from the run that change later planning:**

1. **The estimate was wrong by 54 %.** 1.28 days predicted, 47.3 hours actual.
   The rate decayed from 912 to 480 tokens/s across the run. `11_BENCHMARKING.md`
   §19.4 records the measurement: a fresh process with the same configuration
   reproduces 904, so **the decay belongs to the long-lived process, not to the
   configuration.** Any timeline assuming a constant rate is wrong by up to a
   factor of two.
2. **The bottleneck is parallel scaling, not arithmetic.** Sixteen threads buy
   2.00×, which implies a 46.7 % serial fraction and a hard ceiling of 2.14×.
   Single-threaded, the implementation reaches 44 % of one core's peak. §19 has
   the table.
3. **`apps/train` now prints validation loss during the run**, not only in the
   final report. It did not before. The TinyStories run happened to be configured
   for exactly the number of steps it needed, so nothing was lost — **that was
   luck, and it was not a plan.** A run that starts to memorise shows it in this
   line first, and it now appears every `evaluation_interval_steps`:

### 4.3 Pilot data — downloaded, converted, and the tokenizer trained

| File | Size |
|---|---|
| `artifacts/raw/bilingual/fa-fineweb2-000.parquet` | 4618.1 MB |
| `artifacts/raw/bilingual/fa-fineweb2-000.jsonl` | **15,959.6 MB** (converted output) |
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

### 4.5 The bilingual tokenizer — trained, and it is the whole point

**Trained 2026-10-14.** 32768 vocabulary, byte-level BPE, from a balanced sample
of all three languages in one pass.

| Source | Sample | Share |
|---|---|---|
| Persian (FineWeb2) | 140.9 MB | 42.7 % |
| English (Gutenberg) | 145.7 MB | 44.1 % |
| Code (stackv2_edu) | 43.7 MB | 13.2 % |

| | |
|---|---|
| Artifact | `artifacts/tokenizer-bilingual.json` |
| Vocab | 32768, matching `configs/stage1-110m.toml` |
| **Identity** | **`c727beb3c4c31047cf37ff244598bbf2046610b64ed7198155deb12db6d7dbb1`** |
| Training time | 17.9 s |
| Code is in the tokenizer, and **not** in the pilot's training data | decision D1 |

**And the measurement that justifies the whole exercise.** The same three files,
through the bilingual tokenizer and through the English-only TinyStories one:

| Language | English-only, vocab 8192 | **Bilingual, vocab 32768** | Improvement |
|---|---|---|---|
| **Persian** | **0.57** chars/token | **3.70** | **6.5×** |
| English | 3.14 | 4.11 | 1.31× |
| Code | 1.60 | 3.30 | 2.06× |

**0.57 characters per token means 1.75 tokens per character.** The English-only
tokenizer emitted more than one token for every Persian letter, so the same
compute would have bought **6.5× less Persian**. That is not a rounding error and
it is not a preference; it is the difference between a Persian model and no
Persian model.

**This is the failure `29_DATA_CLEANING_RESEARCH.md` warned about, measured
rather than argued.** The warning was that English-shaped tooling deletes Persian
while every counter reports success. Here nothing even reports an error — the
tokenizer trains, loads, encodes and decodes perfectly, and simply costs six and
a half times as much.

**Reproduce it with:**

```powershell
.\target\release\tokenizer-cli.exe stats --config configs\stage1-110m.toml `
    --artifact artifacts\tokenizer-bilingual.json `
    --corpus <persian.txt> --corpus <english.txt> --corpus <code.txt>
```

Characters rather than bytes, deliberately: the Arabic script costs two bytes per
character in UTF-8, so a byte ratio would flatter Persian and hide exactly this.

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

**The machine is idle now** — the TinyStories run finished. Both items below can
run whenever they are picked up.

| Task | Status |
|---|---|
| ~~`RAYON_NUM_THREADS` sweep~~ | **DONE.** See `docs/11` §19. Sixteen threads buy 2.00×, serial fraction 46.7 %, ceiling 2.14×. The bottleneck is parallel scaling; the earlier per-op-overhead explanation is withdrawn. |
| ~~Measure the 24.6 M model on `wgpu`~~ | **DONE.** See `docs/11` §20. **The GPU is 3.1×–3.3× the CPU**, flat across every shape that fits, and its ceiling is between 2048 and 4096 tokens per micro-batch against the 110 M model's 512. |

**Nothing is blocked on a quiet machine any more.** Both measurements are in
`docs/11` with their tables and their conditions.

### 5.2b What is left

| Task | Why |
|---|---|
| **Run the pilot** | the corpus is prepared and the config validates. 26,000 steps over 101.6 M tokens |
| Wire `Boilerplate` into the pipeline | it needs two passes over the sources and `pipeline.rs` reads them once. The module is written and tested; the plumbing is not |
| Wire `Decontamination` into the pipeline | the module and its tool are written. The corpus reports 5 contaminated documents in 31,400, which the pipeline removes on their own merits |
| Save the rest of the pilot corpus | only one Persian shard of twenty-four is downloaded, and the full run needs about 11 GB of text against 595 GB available |

### 5.3 Blocked on CPU being free

| Task | Status |
|---|---|
| Convert the full 4.6 GB Persian Parquet | **DONE** — 2,645,000 rows, 8.3 % dropped by the adapter, 15,959.6 MB of JSON lines carrying all 11 columns |
| Train the bilingual tokenizer | **DONE** — 32768 vocab, identity `c727beb3c4c31047cf37ff244598bbf2046610b64ed7198155deb12db6d7dbb1`, and the measurement that justifies it is in section 4.5 |
| **Prepare the pilot corpus** | **DONE** — see section 4.6 |
| Fix `apps/train` to print validation loss during the run | **DONE, and verified** on a 200-step run |

### 4.6 The pilot corpus — prepared

**2026-10-14.** `configs/bilingual-pilot.toml` and `configs/bilingual-pilot-sources.json`,
prepared with both language adapters active.

| | |
|---|---|
| Documents | 50,300 read, **48,614 kept**, 1,686 filtered, 0 duplicates |
| **Train** | 47,674 documents, 198,527 sequences, **101,645,824 tokens** |
| Validation | 466 documents, 1,697 sequences, 868,864 tokens |
| Test | 474 documents, 1,778 sequences, 910,336 tokens |
| Content hash | `711ee152b1a099f2f7d3a0824c0a17df05d14a040ec4b9ad68b4dac7eea29ffa` |

**And the drop reasons are the point.** With the adapters wired up, the run
reports:

```text
dropped 1669 for too_few_stopwords
dropped   16 for too_long
dropped    1 for too_many_symbols
```

`too_few_stopwords` is the per-script rule doing its job — 3.4 % of documents were
prose-shaped enough to pass a length check and not language-shaped enough to keep.
**Under the English rules that reason would have fired on every Persian document**,
and the corpus would have been 100 % English with every counter reporting success.

**The configuration is the Stage-1 model, not a smaller stand-in.** The test in
`crates/llm-core/tests/config_files.rs` asserts the parameter count is *equal* to
`stage1-110m.toml`'s, because a pilot that measures a different model measures
nothing that transfers. Three things differ, each for a reason: `sequence_length`
512 rather than 2048 because that is the 4 GB card's measured ceiling for this
model; `max_steps` 26,000 because that is one epoch over this corpus rather than
an arbitrary budget; and the paths, which point at `artifacts/pilot-data` and
touch nothing the TinyStories run produced.

### 5.3 Blocked on CPU being free

### 5.4 Awaiting a decision

| Question | Options |
|---|---|
| Which cleaning filters the pilot uses | see `29_DATA_CLEANING_RESEARCH.md` for what exists in the field and `30_CLEANING_PIPELINE_DESIGN.md` for what to build. Four additions: script classification, per-script rules, near-duplicate, boilerplate removal |
| ~~Whether code stays in the pilot~~ | **Decided — excluded from training, kept in the tokenizer. See section 3.1.** |
| The Persian stopword list, the boilerplate frequency threshold, whether dedup is global or per shard, and who writes the Persian evaluation set | all four block the cleaning build; none is answerable from research |

---

## 5b. Decisions taken, and why

Each of these was a real fork. They are recorded with the reason so the next
reader does not have to reopen them.

| # | Decision | Reason | Date |
|---|---|---|---|
| D1 | **Code is out of the pilot's training data, in the tokenizer** | 10 M code tokens teach nothing and cost a third debug variable; a tokenizer built without code would have to be rebuilt later, and rebuilding invalidates every checkpoint | 2026-10-12 |
| D2 | Pilot split is 55/45 English/Persian, not 50/40/10 | the third language is gone, so its share has to go somewhere; English slightly ahead makes a Persian failure visible against an English baseline that works | 2026-10-12 |
| D3 | Parquet is converted by a separate Rust app, not read by `llm-data` | a Parquet reader brings an Arrow tree into the crate the model links against; if Parquet becomes a primary format rather than a source format, this reverses | 2026-10-12 |
| D4 | Download proportionally, not everything | the full run needs ~11 GB of text from ~595 GB available; download for the run plus a margin for rejection, spread across shards | 2026-10-12 |
| D5 | Heuristic filters are written once per script | the Gopher and C4 rule sets are English-shaped and delete Persian if applied unchanged — the stopword list has no Persian words | 2026-10-12 |
| D6 | The quality classifier is skipped for the pilot, and recorded as skipped | it needs Persian human labels that do not exist, and the pilot's job is to prove the pipeline, not to build the best corpus | 2026-10-12 |
| D7 | Stage 2 and Stage 3 of the bilingual plan are **not** started now | the repository is at roadmap Phase 1. SFT is Phase 2 work; DPO/RLCR/RLVR are Phase 5 | 2026-10-12 |

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
