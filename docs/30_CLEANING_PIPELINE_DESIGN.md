# 30 — The cleaning pipeline: design

**Created:** 2026-10-12
**Status:** design. No code written.
**Read with:** `29_DATA_CLEANING_RESEARCH.md` (what exists in the field) and
`28_BILINGUAL_PILOT_STATUS.md` (where this sits in the work).

---

## 1. The finding that shapes this design

Most of the machinery **already exists** in `crates/llm-data/src/filter.rs`:

| Already there | Where |
|---|---|
| A length window, counted in characters not bytes | `filter.rs` |
| Credential shapes, dropped rather than redacted | `filter.rs` |
| A `FilterReason` enum with a stable `as_str()` per reason | `filter.rs` |
| The rule that **every drop is counted and attributed to a named reason** | `filter.rs` module doc |
| A `FILTER_VERSION` string that travels in the manifest | `filter.rs` |
| Exact document deduplication, versioned `exact-sha256-v1` | `dedup.rs` |
| Normalization, splitting, packing, shard I/O | the rest of the crate |

**So this is not a new pipeline. It is an extension of `assess` and a precise
statement of what runs before it.** Building a parallel cleaner would mean two
places to look when a count is wrong, and the existing filter already has the
right discipline: a drop is never silent.

The one thing the existing design does **not** have is any notion of **script**.
Every threshold in it is script-neutral, and section 3 is about why that cannot
hold for a Persian corpus.

---

## 2. What is missing, and the order it must run in

| # | Stage | Exists | Why this position |
|---|---|---|---|
| 1 | Normalization | ✅ `normalize.rs` | must precede everything; filters judge normalized text |
| 2 | **Script classification** | ❌ | every later rule needs it |
| 3 | Length window | ✅ | cheapest, and bounds the work after it |
| 4 | **Per-script heuristics** | ❌ | the large win, and the one that breaks Persian if written in English |
| 5 | Credential shapes | ✅ | cheap, already correct |
| 6 | **Near-duplicate** | ❌ | before the split, or the split leaks |
| 7 | **Boilerplate lines** | ❌ | catches what MinHash misses |
| 8 | Exact duplicate | ✅ | after near-duplicate; the cheap confirmation |
| 9 | Split | ✅ `split.rs` | **after** dedup, never before |
| 10 | **Decontamination** | ❌ | against eval sets that must be written first |
| 11 | Tokenize and pack | ✅ | unchanged |

Two orderings are load-bearing and are not preferences:

- **Dedup before split.** Deduplicating after splitting puts the same document in
  train and validation, and the validation number then measures memory rather
  than generalisation.
- **Decontamination against an eval set that exists.** It cannot be written
  afterwards; a benchmark finalised after the corpus will be inside it.

---

## 3. Script classification, and why it is the whole design

The research document's central warning: **the Gopher and C4 rules are
English-shaped and delete Persian if applied unchanged.** The stopword list has
no Persian words, so an unchanged rule scores every Persian document at zero
stopword density and drops the lot.

So the filter needs to know which ruleset to apply. That is one new concept:

```rust
/// Which writing system a document is mostly in.
pub enum Script {
    /// Latin letters dominate: English, code, transliterated text.
    Latin,
    /// Arabic-script letters dominate: Persian, and Arabic it must be told apart from.
    Arabic,
    /// Both are substantial. Common in Persian technical writing, where an English
    /// term sits inside a Persian sentence, and it must not be punished for it.
    Mixed,
    /// Neither dominates: digits, punctuation, symbols.
    Neither,
}
```

**Classification is by counting, not by a model.** Count characters in the Arabic
block (U+0600–U+06FF, plus U+200C ZWNJ which Persian uses constantly) against the
Latin block. The majority wins; `Mixed` when the minority exceeds a configured
fraction.

This is deliberately not a language identifier. It answers "which ruleset", not
"which language" — and for choosing between a Persian ruleset and an English one,
counting letters is the right tool and a model is overkill.

**`Mixed` is a first-class outcome, not an error.** Persian technical text is
routinely mixed, and a design that treats mixing as a defect would throw away
the most useful Persian in the corpus.

---

## 4. Per-script heuristics — the rules, stated twice

Every threshold below is a **starting value to be measured**, not a published
number. The pilot's job is to measure them and log the drop rate of each.

| Rule | Latin ruleset | Arabic ruleset | Why they differ |
|---|---|---|---|
| Stopword density | ≥ 2 per 100 words from an English list | ≥ 2 per 100 words from a **Persian list** (`و`, `در`, `به`, `از`, `که`, `این`, `را`, `با`, `است`, `می`) | the whole point: an English list scores Persian at zero |
| Sentence terminators | ≥ 95 % of lines end `. ! ? "` | ≥ 95 % end `. ! ? ؟ ؛ »` | Persian uses `؟`, `؛`, `»` |
| Symbol ratio | `# % \| …` below 0.1 | `، ؛ « » ٪ *` below 0.1 | the symbol sets barely overlap |
| Word length | mean 3–10 characters | mean 2–8 characters | Persian words are shorter in characters |
| Ellipsis lines | < 30 % | < 30 % | same |
| Bullet-start lines | < 90 % | < 90 % | same |
| Curly braces | **forbidden** | forbidden | **both must be disabled for the code share** |
| JavaScript warning | forbidden | forbidden | same |

**The code share bypasses the prose rules entirely.** Curly braces are what code
is; a rule that forbids them is a rule that deletes the corpus's code tenth. Code
gets the length window, the credential check and the deduplication, and nothing
else.

**ZWNJ (U+200C, نیم‌فاصله) must never be treated as punctuation or whitespace.**
It is a letter-level joiner inside Persian words — `می‌رود` is one word, not two.
A tokenizer or filter that splits on it corrupts the language. Normalization must
preserve it and the word counter must not split on it.

---

## 5. Near-duplicate removal, and how to build it without a dependency

Web text repeats: the same article on many sites, the same site furniture on
every page.

**Method:** MinHash with LSH banding.

```text
per document:
  1. shingles  = the set of 9-token n-grams
  2. signature = for each of 128 hash functions, the minimum hash over the shingles

across documents:
  3. split each 128-value signature into 20 bands of 6
  4. hash each band; documents sharing a band hash are candidates
  5. confirm candidates by comparing signatures, drop the later one
```

`P(collision) ≈ 1 − (1 − s^6)^20`, which is 0.86 at Jaccard 0.8 and 0.28 at 0.5.
That is the wanted behaviour: catch near-identical, let genuinely different text
through.

**Why write it rather than add a crate:** it is about 200 lines of arithmetic over
`u32` with no I/O and no unsafe, and it is testable against hand-built pairs. A
dependency here would bring more code than the algorithm.

**The memory cost, stated:** 128 `u32` per document is 512 bytes, so 500,000
documents is 256 MB. That is acceptable for the pilot and must be re-checked
before the full run, where the document count is orders of magnitude larger. If it
becomes the limit, the fix is to dedup per shard and accept that cross-shard
duplicates survive — recorded, not hidden.

**The cluster rule from the plan:** all members of a duplicate cluster must land
in the same split. The signature is what identifies the cluster, so the split
assignment must key on it rather than on the document index.

---

## 6. Boilerplate lines — the case MinHash cannot see

The Persian shard's first document opens with:

```text
نوشته شده در شنبه بیستم خرداد 1391ساعت 15:13 توسط behrouz
```

That is five words. A 9-token shingle never forms from it, so MinHash is blind to
it — and it appears on every page of that site.

**Method, which is simple and effective:** count line frequency across the corpus
per source, then drop lines occurring in more than a configured fraction of
documents. A line on 40 % of a site's pages is furniture.

This needs two passes, or one pass that holds a line counter. The counter is
`HashMap<u64, u32>` of line hash to occurrence count; a few million lines is
tens of MB.

**It must run before the length filter, not after.** A document that is *mostly*
boilerplate passes a length check easily; the point is to remove the lines so the
remaining text is judged on its own.

---

## 7. What the pilot deliberately does not build

| Skipped | Reason | Recorded as |
|---|---|---|
| Quality classifier (FineWeb-Edu style) | needs Persian human labels that do not exist | skipped, with the reason, in the manifest |
| fastText language ID | `Script` answers the question the rules need; a language model is for a different question | not needed at this scale |
| PII classifier cascade | regex shapes are already there; a cascade is a later refinement | deferred |
| Semantic dedup (SemDeDup) | embeddings, and a model to make them, for a gain MinHash mostly captures | deferred |

**A skipped stage is recorded with its reason.** A pipeline whose omissions are
invisible cannot be compared against one that has them.

---

## 8. Where the code goes

```text
crates/llm-data/src/
  filter.rs        extended: Script, ScriptRules, new FilterReason variants
  script.rs        new: script classification, ~80 lines
  minhash.rs       new: signatures and LSH banding, ~200 lines
  boilerplate.rs   new: line frequency and removal, ~120 lines
  dedup.rs         unchanged: exact dedup, still last
  pipeline.rs      the chain, with per-stage counters
```

**It belongs in `llm-data`, not in an app.** Filtering is a property of the
corpus, it is versioned in the manifest, and the training path must be able to
reproduce it. An app that cleaned data before handing it over would put the
rules outside the artifact that records them.

**`FILTER_VERSION` must change** whenever a rule or threshold changes. It travels
in the dataset manifest, so a corpus can be tied to the ruleset that produced it
— and a change of ruleset invalidates comparisons against an earlier corpus.

---

## 9. Testing

The existing filter's tests are the model: each rule gets a case that must be
dropped and a case that must survive.

The tests that matter most here, and why:

| Test | What it would catch |
|---|---|
| A Persian document survives the heuristic filter | **the failure in section 3** — the one that deletes the corpus silently |
| A Persian document with an English term survives as `Mixed` | punishing code-switching, which is most Persian technical text |
| A document containing ZWNJ is not split into two words | the نیم‌فاصله trap |
| `می‌رود` and `میرود` are different documents | normalization that was too aggressive |
| Two documents differing in one paragraph are caught as near-duplicate | MinHash banding set too tight |
| Two documents sharing a topic but no sentences are **kept** | MinHash too aggressive |
| The byline from section 6 is removed; a sentence of the same length is not | boilerplate removal keyed on frequency, not length |
| A code document containing `{ }` survives the heuristic filter | the curly-brace rule leaking out of the prose ruleset |

**And a real-data test.** The Persian shard is on disk. A test that runs the
filter over its first thousand documents and asserts the boilerplate lines are
gone would catch more than any synthetic case.

---

## 10. What has to be decided before building

| Question | Why it blocks |
|---|---|
| The Persian stopword list — write it or take it? | it is the single rule that decides whether Persian survives |
| The boilerplate frequency threshold | 40 % is a guess; too low removes repeated quotations, too high leaves furniture |
| Whether dedup is global or per shard | global is correct and costs memory; per shard is cheap and leaks across shards |
| Who writes the Persian evaluation set | decontamination cannot run without it, and it must exist before the corpus is finalised |

**None of these is answerable from the research.** Each needs either a decision
from the project owner or a measurement from the pilot.

---

## 11. The honest bottom line

The pipeline is **mostly built and well disciplined** — the existing filter
already refuses to drop anything silently, which is more than most corpora
manage. Four additions are needed: script classification, per-script rules,
near-duplicate detection, and boilerplate removal.

**The single highest risk is not any of those four.** It is that the heuristic
rules get copied from an English source and applied unchanged, the drop counters
all report success, and the Persian half of the corpus is simply gone. Section 3
and the first row of section 9 exist to make that failure loud.
