# Persian and English evaluation prompts

**Created:** 2026-10-12
**Purpose:** the fixed prompt list the bilingual pilot is judged against, and the
thing the training corpus is decontaminated against.

---

## Why this file has to exist before the corpus does

An evaluation set written after the corpus is finalised is already inside it.
The model then reproduces the evaluation text from memory and every score is a
measurement of memorisation rather than of language. This is the failure
`29_DATA_CLEANING_RESEARCH.md` calls the field's number one silent one.

So these prompts are written first, frozen, and the corpus is checked against
them: **any training document sharing a 13-gram or longer with anything here is
removed, and the count of removals is reported.**

## Why the test is generation and not multiple choice

The pilot's question is **"does the model speak Persian"**, and at 110 M
parameters there is no benchmark whose score answers that. Multiple-choice facts
measure knowledge; this model has almost none. Coherence measures language, and
coherence is judged by reading.

That makes the evaluation **human and subjective**, and the honest way to run a
subjective evaluation is to fix the prompts and the rubric in advance, so that
two readings of the same outputs can be compared.

## How to use it

1. Generate 128 tokens from each prompt, greedy, seed 1234, no sampling.
2. Read all outputs.
3. Score each on the three axes below, 0 to 2.
4. Report per language: the sum, and **which prompts failed**, not only the total.

**Run it on every checkpoint worth comparing, and on the untrained model as a
control.** An untrained model scores near zero, and if a trained model does not
beat it clearly, the training did not work no matter what the loss curve says.

---

## Rubric

| Axis | 0 | 1 | 2 |
|---|---|---|---|
| **Grammatical** | not a sentence | some agreement, wrong morphology | well-formed sentences |
| **Coherent** | word salad, or one token repeated | on topic but drifts or repeats | a consistent idea carried through |
| **Language-pure** | wrong language, or mixed with no reason | mostly one language, some unjustified mixing | the prompt's language throughout |

**Score 0 on any axis for a repetition loop.** A model that emits one token to
the budget has failed that prompt regardless of what the token was, and this is
the failure mode the Persian run is most likely to show first.

---

## Persian prompts

Chosen to be the openings of ordinary Persian prose, not questions. **A base
model continues text; it does not answer.** A prompt phrased as a question would
test a capability the model does not have and cannot have at this stage, and
would make every checkpoint look equally bad.

| # | Prompt | What it tests | Needle |
|---|---|---|---|
| P1 | `روزی روزگاری` | the canonical story opening — the highest-frequency pattern in any narrative corpus | **no** |
| P2 | `در یکی از روزهای پاییز،` | a descriptive opening; tests whether the model continues rather than restarts | yes |
| P3 | `کتابی که دیروز خواندم` | first person, past tense | yes |
| P4 | `شهر تهران در سال‌های اخیر` | expository register, and **contains ZWNJ** — a model whose tokenizer mangles it fails here and nowhere else | yes |
| P5 | `او گفت: «من نمی‌دانم` | dialogue, quotation marks, and the negative verb prefix | yes |
| P6 | `هوا سرد بود و` | a conjunction ending; tests whether the model completes a clause | yes |
| P7 | `برای یادگیری برنامه‌نویسی` | technical register; **likely to draw English terms**, so it tests code-switching without punishing it | yes |
| P8 | `مادرم همیشه می‌گفت` | reported speech, habitual past | yes |

**P4 and P5 are the two that matter most.** P4 contains `سال‌های` with ZWNJ, and
P5 contains `نمی‌دانم` with both ZWNJ and a negative prefix. If the normalization
or the tokenizer treats ZWNJ as a separator, these degrade and the rest do not.

### Why two rows say `no` in the Needle column

**Measured, on the real corpora, after this table was first written.**

```
persian.txt    14 of 18000 documents matched   P1 x 12
english.txt    56 of  1400 documents matched   E1 x 53
```

`Once upon a time` is in **3.8 %** of a corpus of English books. `روزی روزگاری` is
its Persian equivalent. **This document already said so** — the P1 row described
it as "the highest-frequency pattern in any narrative corpus" — and then used it
as a contamination marker anyway.

**A prompt has two jobs and they pull in opposite directions.** As a *generation*
prompt it should be the most natural opening there is, because that is what a
model is expected to continue. As a *decontamination needle* it must be rare
enough that finding it means something, and a phrase on four percent of documents
means nothing.

So the two roles are separated. P1 and E1 stay as generation prompts — they are
the best ones in the set for that — and are **not** needles. The corpus is
checked against the other fourteen.

## English prompts

The same eight ideas in English, so the two languages can be compared on the same
ground.

| # | Prompt | Needle |
|---|---|---|
| E1 | `Once upon a time` | **no** |
| E2 | `On an autumn day,` | yes |
| E3 | `The book I read yesterday` | yes |
| E4 | `The city of Tehran in recent years` | yes |
| E5 | `She said, "I do not know` | yes |
| E6 | `The air was cold and` | yes |
| E7 | `To learn programming` | yes |
| E8 | `My mother always said` | yes |

---

## What this set does not do

| Not covered | Why, and when it would be added |
|---|---|
| Knowledge and facts | a 110 M pilot has almost none; factual benchmarks become meaningful at a larger scale and with post-training |
| Instruction following | the pilot is a base model. Instruction following is Stage 2 of the bilingual plan and Phase 2 of the roadmap |
| Calibration, sycophancy, hallucination | Stage 3 and Phase 5. Measured then, against a model that can follow an instruction at all |
| Code generation | code is excluded from the pilot's training data by decision D1. It gets its own prompt set when it gets its own experiment |
| Perplexity on a held-out slice | **this one is covered, but not here.** The validation split produced by `apps/prepare` is the perplexity holdout and it is already held out |

---

## Decontamination against this file

The prompts above are short and were written for this document, so nothing in
FineWeb2 or Common Pile should match them beyond coincidence. **That is an
expectation, not a measurement.**

The check to run, once the corpus is prepared:

```text
for each training document:
    for each 13-gram in the document:
        if the 13-gram appears in any prompt above:
            remove the document, and count it
```

Report the count. **A count above a handful means something is wrong with the
prompts or with the corpus, and either way it needs to be seen before training
rather than after.** Eight- and sixteen-word prompts are below a 13-gram
threshold individually, so the practical form of the check is a substring match
of each full prompt against each document, which for nine short strings over a
few hundred thousand documents is cheap.
