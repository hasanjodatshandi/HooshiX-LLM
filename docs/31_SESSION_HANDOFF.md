# Handoff — the audit work order, mid-flight

**Written:** 2026-10-14, at commit `82066b1`
**For:** the next session

This file exists because the work order being executed is long and arrived as a
pasted document rather than a file in the repository. A session that starts
without it will not know what P0-3 means. Read this first, then ask for the work
order itself.

---

## 0. Read these before touching code

`docs/AGENTS.md` governs this repository and is mandatory. It says to start a
fresh coding context by reading `docs/18_RUST_STANDARDS_ROUTER.md` and using its
router to load **the smallest sufficient standards set** for the task, then the
subsystem specification, then the current repository state. Do not load every
document blindly and do not rely on memory when the rule is in the repository.

Fixed constraints worth repeating because they are easy to violate by accident:
Stage 1 is a dense decoder-only 100M–150M model and has no sixth stage; Burn 0.22
owns tensor, autodiff and runtime behaviour; no Python in the runtime or training
stack; no database, cache or queue exists and none may be added; do not build a
custom tensor framework.

`AGENTS.md` also states the validation rule this handoff keeps repeating: **a
check that did not run is `Not run`, never `Passed`.**

---

## 1. What is being executed

A document titled **«برنامه اجرایی دقیق اصلاح Hooshix-LLM — ویژه دستیار کدنویسی»**,
dated 2026-10-14. It has four groups:

| Group | Contents |
|---|---|
| Baseline | record commit, toolchain, gate results before touching anything |
| **P0** | P0-1 optimizer config, P0-2 sequence-length preflight, P0-3 real resume |
| **P1** | P1-1 memory profiling, P1-2 attention mode, P1-3 validation, P1-4 shuffle |
| **P2** | P2-1 head/loss memory, P2-2 BF16, P2-3 kernels, P2-4 server safety |

**Ask the user to paste it again.** The parts that matter most and are easy to get
wrong:

- Stage, architecture and product goal do not change. 12 layers, hidden 768,
  12 heads, vocab 32768, decoder-only.
- Rust and Burn only. No Python runtime substitution.
- No existing data, checkpoint, weight, tokenizer or artifact may be deleted,
  rewritten or implicitly migrated. `artifacts/pilot-run/` is preserved.
- **Test results are `PASS / FAIL / BLOCKED / NOT RUN` and nothing else.** A CUDA
  test on a machine with no GPU is `NOT RUN`. No fabricated success.
- Every change needs a limited diff, a passing test, a failure-path result and a
  stated way to revert.
- **Do not turn validation off and call it a memory fix.** The current
  `evaluation_interval_steps = 100000` over a 13486-step run means validation
  never fires. That is a workaround, not a solution, and the document says so.
- Final deliverable: `artifacts/audit-2026-10-14/implementation-report.md` with
  Completed / Result / Validation / Problems Resolved / Remaining Issues /
  Self-Review / Suggestions.

---

## 2. State at this handoff

```
repository   D:\Projects\Hooshix-LLM
HEAD         82066b1   everything pushed to origin/main
toolchain    rustc 1.99.0, cargo 1.99.0, edition 2024
burn         0.22.0, from Cargo.lock rather than the manifest
gates        fmt 0, clippy 0, test 0 (406 passed, 0 failed), doc 0
```

The gate commands, in this order:

```powershell
Set-Location 'D:\Projects\Hooshix-LLM'
cargo fmt --all
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --doc --workspace
```

`cargo test --workspace` is the right scope. **Do not reach for
`--all-features`**: cuda, wgpu and cpu are deliberately mutually exclusive in the
composition root. The CUDA build needs a card and is `NOT RUN` locally:

```bash
cargo check --locked -p train --no-default-features --features cuda
```

---

## 3. Done, with the evidence each rests on

### P0-1 — the configured AdamW settings now reach the optimizer · `4a36453`

`build_optimizer` called `AdamWConfig::new()` and set only the gradient-clipping
threshold, so `beta1`, `beta2`, `epsilon` and `weight_decay` came from Burn's
defaults no matter what `[training.optimizer]` said. **The A100 bilingual pilot
was trained under those settings**, which is part of why it learned as slowly as
it did.

Fixed by making `&OptimizerConfig` a required argument of `Trainer::new` and
assigning all four fields. There is deliberately no constructor that omits them.

**Proven both ways.** `crates/llm-training/tests/optimizer_config.rs` varies each
setting and compares losses. With the fix, five pass. With `build_optimizer`
temporarily restored to the old behaviour, four fail, each reporting losses
differing by exactly zero.

### P0-2 — actionable message, and length 1 refused · `7e3bca3`

**The preflight the work order asked for already existed**, with a test:
`ShardReader::open` compares the header's length against `ShardExpectation`, and
`open_split` runs before `init::seed` and `Transformer::new`. Nothing was added
there.

What was wrong was the message. Read and write shared one error saying a sequence
was "received", which is true in `push_sequence` and false in `open`. It now
names both lengths and the two ways out. `sequence_length = 1` is also refused at
validation rather than at the first batch, since a one-token sequence has no
next-token pair.

**Not done from P0-2, and not claimed:** `predicted_positions_per_step` in the
run report, and a shard-checksum sweep in preflight.

### P0-3 — three of seven pieces · `db6bb0c`, `afdf9d0`, `82066b1`

| Piece | State |
|---|---|
| `ShardReader::seek(index)` | **done**, tested for every index and both bounds |
| `ShardBatches::cursor()` / `seek_cursor()` | **done**, four tests including a token-for-token round trip |
| the placeholder `sequence: 0` in the checkpoint | **done**, replaced by `train.cursor()` |
| `TRAINING_SEMANTICS_VERSION` guard | **done, NOT TESTED** — see below |
| the `--resume` path | **not started** |
| `config_hash` from the serialised configuration | **not started** |
| checksums over `model.bpk` and `optimizer.bpk` | **not started** |
| checkpoint root refusing only fresh runs | **not started** |

**The untested guard is the most important thing on this list.** It is
implemented, compiles and passes every gate, but nothing proves it fires. The
check lives in `load_model`, which needs a `Transformer`, and the test was not
written. The work order is explicit that an untested guard must not look
finished, so the commit says so. Write that test first.

---

## 4. Immediate next steps, in order

1. **Test `TRAINING_SEMANTICS_MISMATCH`.** Save a checkpoint with
   `training_semantics_version: 0`, load it with
   `llm_checkpoint::TRAINING_SEMANTICS_VERSION`, and assert
   `CheckpointError::TrainingSemanticsMismatch`. `checkpointing.rs` in
   `crates/llm-training/tests/` has the save/load helpers.
2. **The `--resume` path.** `training::run` currently takes
   `(&ExperimentConfig, &Device)`; it needs an `Option<&Path>`. The loop at
   `apps/train/src/training.rs:178` runs `0..max_steps` and must run
   `global_step..max_steps`. The `CheckpointRootNotEmpty` check at line 156 must
   apply to fresh runs only. `Trainer::resume_from` already exists and is the
   loader. **This is what makes `seek_cursor` reachable**, which is currently
   marked `#[allow(dead_code)]` with a comment naming this caller.
3. `config_hash` from `serde_json::to_vec(config)` then SHA-256, so two TOMLs
   differing only in comments hash the same and a changed `beta2` does not.
4. Then P1, then P2.

---

## 5. Environment facts that cost time to learn

- **`CARGO_TARGET_TMPDIR` is undefined in unit tests** inside `src/`. It works in
  `tests/`. Use `std::env::temp_dir()` for a unit-test scratch directory.
- **An autodiff device is required for `backward`.** `Device::default()` panics
  with "requires a tracked autodiff tensor". Use `Device::flex().autodiff()`.
- **`cargo fmt --all` invalidates in-flight edits** — the `edit` tool then
  reports "file changed since it was read". Re-read, then retry. This happened
  several times.
- **PowerShell string interpolation breaks on `"` inside a commit message.** Use
  `git commit -F <file>`.
- **Write JSON and TOML without a BOM**: `Set-Content -Encoding UTF8` adds one in
  PowerShell 5.1 and the parse fails with "expected value at line 1 column 1".
  Use `[System.IO.File]::WriteAllText($p, $t, (New-Object System.Text.UTF8Encoding($false)))`.
- **`cargo fmt` normalises line endings**, so a `.Replace()` on exact text from a
  file read before a format pass will miss. Patched by line content instead.
- The workspace denies `unwrap_used`, `expect_used` and `panic`; integration tests
  need a file-level `#![allow(...)]`.
- **`cargo fmt` may reflow** a `struct` initialiser, so field insertion is safer
  by line match than by literal block.

---

## 6. How the user wants to be worked with

- **Persian prose**, exact evidence: exit codes, byte counts, hashes, timings.
- **`اجرا نشد` / `NOT RUN` for anything unverified.** Never an estimate presented
  as a measurement, and never a success that was not observed.
- **Retract wrong claims explicitly.** Earlier in this work an arithmetic error
  led to a withdrawn memory-bandwidth hypothesis, and a conclusion that "memory
  grows with the step count" was wrong and was corrected in a commit.
- **Long runs, and especially training, go in a visible detached window** that
  survives the editor closing. `Start-Process` children die with the harness;
  `Invoke-CimMethod -ClassName Win32_Process -MethodName Create` does not.
- **Ask only when a decision is genuinely the user's.** Otherwise continue.
- **Findings that contradict the work order are reported, not silently obeyed.**
  Two of its premises were already handled in the code; saying so was correct.

---

## 7. Context the work order does not have

- **The bilingual pilot ran and failed to produce a Persian-speaking model**, and
  the reason is training volume, not the pipeline. 13,486 steps, 110,369,424
  tokens, 118 minutes on an A100, loss 10.5124 to a best of 1.5702. Persian
  prompts continue in English. The corpus is 39.2 percent Persian against a 40
  percent target, so the mixture is fine; a 110M model wants about 2.2 billion
  tokens and saw 110 million, which is five percent. The log is
  `docs/dataset/a100-pilot-run.log`.
- **The A100 was rented and has been returned.** Any CUDA work is `NOT RUN` until
  another machine exists. Do not plan around one.
- **The optimizer fix invalidates the pilot's settings.** Its checkpoints carry
  no semantics version, so under the new guard they are refused for resume by
  design. That is the intended behaviour, not a regression.
