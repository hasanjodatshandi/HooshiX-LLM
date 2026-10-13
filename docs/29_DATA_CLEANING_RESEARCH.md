# 29 — Data cleaning for a Persian/English/code corpus: what exists, and what to build

**Created:** 2026-10-12
**Scope:** research and a proposed filter set. No code changes.
**Question it answers:** what filters exist for making downloaded web text usable,
which of them matter for a Persian + English + code corpus, and what should the
pilot pipeline actually run.

---

## 1. The finding that makes this urgent

`apps/convert` read the first documents of the FineWeb2 Persian shard. The first
one is a blog post and it opens like this:

```text
+
نوشته شده در شنبه بیستم خرداد 1391ساعت 15:13 توسط behrouz

درباره
با اسپیکر روشن وارد شوید!!! ...
```

The `+` is noise. `نوشته شده در … توسط behrouz` is blog boilerplate. `درباره` is
site navigation. The article starts on the fourth line.

**FineWeb2 is filtered, and it still contains this.** FineWeb2 applies language
identification and a quality filter tuned per language; it does not strip every
site's own furniture. Whatever pipeline this project builds has to do more than
trust the download.

---

## 2. The pipeline stages, in the order they must run

Each stage is a **measurable lever**, and each must log its own drop rate. A stage
whose drop rate is unknown cannot be ablated and cannot be defended.

| # | Stage | What it removes | Tooling that exists |
|---|---|---|---|
| 1 | **Extraction** | HTML, scripts, navigation | `trafilatura`, datatrove `HTMLExtractor`, JusText |
| 2 | **Language ID** | wrong-language documents | fastText `lid.176`, **GlotLID** (2000+ languages) |
| 3 | **Heuristic quality** | short, symbol-heavy, list-like, repetitive | Gopher rules, C4 rules |
| 4 | **Classifier quality** | text that is fluent but worthless | FineWeb-Edu edu-score, DCLM fastText classifier |
| 5 | **Near-duplicate** | paraphrases and boilerplate variants | MinHash + LSH |
| 6 | **Exact-substring duplicate** | short repeated blocks MinHash misses | suffix array |
| 7 | **Decontamination** | benchmark text in training data | n-gram overlap against eval sets |
| 8 | **PII / safety** | emails, phones, identifiers | regex, then a classifier |
| 9 | **Tokenize and shard** | — | already built, `apps/prepare` |

**Stage 1 is already done for us.** FineWeb2 and Common Pile ship extracted text,
not WARC. `apps/convert` is what bridges the container format. So the pipeline
starts at stage 2.

---

## 3. The stages that matter most here, and why

### 3.1 Language identification — **different for Persian than for English**

FastText `lid.176` covers 176 languages and is the default everywhere. For
Persian in a mixed corpus it is workable but not ideal, because:

- **Code-switching is normal in Persian web text.** A Persian sentence with an
  English technical term, a URL, or a `#hashtag` can score below threshold.
- **Arabic and Persian share a script.** `ک`/`ك` and `ی`/`ي` differ, and
  undiacritized Arabic text is easy to confuse with Persian for a naive model.
- **FineWeb2 itself standardised on GlotLID**, not `lid.176`, for exactly this.

**Threshold:** ≥ 0.65 keeps recall; ≥ 0.85 kills noisy multilingual text. For a
corpus that must contain real Persian *including* code-switching, start at 0.65
and measure how much Persian is being lost.

**And a trap:** a language filter that drops documents with English words will
remove most real Persian technical writing. The filter must be per-document, not
per-sentence.

### 3.2 Heuristic quality rules — and their Persian problem

The Gopher and C4 rules are English-shaped. Applied unchanged to Persian they
mis-fire:

| Rule | English threshold | Why it breaks on Persian |
|---|---|---|
| stopword density ≥ 2 per 100 words | `the, be, to, of, and, that, have, with` | **there is no Persian stopword list in the rule.** An English list applied to Persian scores every Persian document at zero and deletes the corpus |
| line ends in `.!?"` ≥ 95 % | ASCII terminators | Persian uses `؟` and `؛` and `«»`. The rule must be extended or it deletes Persian |
| no curly braces | proxy for "code leaked in" | **the corpus contains code on purpose.** Braces are expected in 10 % of it |
| symbol-to-word ratio < 0.1 | `#%|…` | Persian text uses `،` `؛` `«` `»` `٪` heavily. The symbol set must be per-script |
| mean word length 3–10 chars | Latin | Persian words are shorter in characters and the script is right-to-left |

**Conclusion: the heuristic rules must be written twice — once per script — and
the code share must bypass the ones aimed at prose.** Copying the English list
is the single fastest way to destroy a Persian corpus while every counter says
the pipeline worked.

### 3.3 Near-duplicate removal — the one that pays most

Web text repeats. The same article appears on many sites, the same boilerplate on
every page of a site, the same legal notice everywhere.

**Method:** MinHash + LSH banding.

```text
shingle size  n = 9 tokens
permutations  k = 128
bands         b = 20, rows r = 6

P(collision) ≈ 1 - (1 - s^r)^b
  at Jaccard 0.8 -> 0.86   most duplicates found
  at Jaccard 0.5 -> 0.28   most near-misses missed (deliberate)
```

**The trap that matters for us:** the plan requires that **all members of a
duplicate cluster land in the same split.** Deduplicating *after* splitting leaks
the same document into train and validation, and the validation number becomes a
memory test rather than a generalisation test. Cluster first, then split.

**A second trap:** boilerplate on Persian sites is often *identical short blocks*
— a footer, a date line, a byline. MinHash with 9-token shingles will not catch a
5-word byline. That is what **exact-substring dedup** is for, and the boilerplate
seen in section 1 is exactly its job.

### 3.4 Decontamination — the failure that is silent

If an evaluation set's text is in the training data, the score is meaningless and
nothing in the pipeline reports it. The standard: **any document sharing a
13-gram or longer with an eval set is removed, and the removal is logged loudly.**

For this project specifically: the eval sets do not exist yet for Persian. They
have to be written, and they must be written **before** the corpus is finalised,
or the corpus will contain them.

### 3.5 The stage nobody wants to do, and the pilot should skip

Training a quality **classifier** (FineWeb-Edu's edu-score, or DCLM's fastText
classifier) beats twenty hand-written rules on recall. But it needs **human
labels** for the target languages, and for Persian those labels do not exist and
would have to be created.

**For the pilot: skip it, and say so.** The pilot's purpose is to prove the
pipeline and the tokenizer, not to produce the best possible corpus. Record it as
deferred with the reason.

---

## 4. What the pilot should actually run

A defensible minimal set, in order, each logging its own counts:

| Stage | Pilot decision | Why |
|---|---|---|
| Language ID | **GlotLID**, threshold 0.65 | Persian/Arabic confusion and code-switching make `lid.176` risky |
| Heuristic quality | **Per-script rules**, separate for Persian and English | an English stopword list deletes Persian |
| Code | **Bypass prose rules**; apply only length and duplicate rules | braces and symbols are expected |
| Near-duplicate | **MinHash + LSH**, before the split | the plan requires cluster-then-split |
| Exact-substring | **Run it** — the byline case needs it | catches what MinHash misses |
| Decontamination | **Write the Persian eval set first**, then match 13-grams | cannot decontaminate against a set that does not exist |
| Classifier quality | **Skip, and record it as skipped** | no Persian labels; the pilot does not need it |
| PII | **Regex only** for the pilot | a classifier cascade is a later refinement |

**Every stage logs: documents in, documents out, and the reason for each drop.**
A count that is not logged cannot be ablated, and the plan will want to ablate.

---

## 5. The ablation method, so the filters can be defended

The plan asks for controlled comparison. The standard protocol:

| Run | Corpus | Change | Signal to expect |
|---|---|---|---|
| A | raw converted text | baseline | noisy |
| B | + language ID | one change | removes the wrong language |
| C | + per-script heuristics | one change | short/symbol-heavy gone |
| D | + both deduplications | one change | the largest single gain |
| E | + decontamination | one change | eval numbers stop lying |

**One change per run.** Two changes make the delta uninterpretable, and this is
the entire discipline of a data ablation.

**And the honest caveat, from the project's own history:** this repository has
already published one speedup claim and had to retract it, because the two runs
were not taken within one session. **Data ablations have the same failure mode.**
Same proxy model, same seed, same step count, same evaluation, runs alternated —
or the numbers mean nothing.

---

## 6. What not to do

| Anti-pattern | Why it is wrong here |
|---|---|
| Copy the Gopher rule set as-is | it is English-shaped and deletes Persian |
| Deduplicate after splitting | leaks documents across splits and inflates validation |
| Skip decontamination because "we have no benchmark yet" | write the benchmark first, then; a contaminated eval set is worse than none |
| Report a filter's effect without its drop rate | an unlogged filter cannot be ablated or defended |
| Train a quality classifier for the pilot | needs Persian human labels that do not exist; the pilot does not need it |
| Trust FineWeb2's filtering | measured: blog boilerplate survives it |
| Run ablation variants in separate sessions | this project has already been burned by exactly this |

---

## 7. Sources

- Pretraining data curation reference, stage-by-stage with thresholds and traps:
  [ai-data-curation-pretraining](https://github.com/vasilyu1983/AI-Agents-public/blob/main/frameworks/shared-skills/skills/ai-data-curation-pretraining/SKILL.md)
- FineWeb / datatrove pipeline and the FineWeb-Edu classifier: the reference
  recipe for English web curation, and the origin of the multilingual extension.
- FineWeb2 — 1000+ languages, per-language threshold tuning, standardised on
  GlotLID: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- Common Pile v0.1 — public-domain and openly licensed text, 30 sources:
  [arXiv:2506.05209](https://arxiv.org/abs/2506.05209)
- DCLM — the controlled data-curation benchmark showing a single trained
  classifier beats heuristic stacks on recall: [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
- Nemotron-CC — classifier ensemble plus synthetic rephrasing to recover tokens
  aggressive filters would discard: [arXiv:2412.02595](https://arxiv.org/abs/2412.02595)
- SemDeDup — embedding-cluster deduplication, complements MinHash:
  [arXiv:2303.09540](https://arxiv.org/abs/2303.09540)

**Not verified here:** the Persian-specific thresholds in section 4 are choices
to be measured, not published numbers. No Persian filter stack was benchmarked
during this research, and none of the counts in section 4 exist yet.
