# Training and Checkpoints

## 1. Training objective

Causal language modeling:

```text
loss = mean(cross_entropy(logits[t], target[t]))
```

where target is the input shifted by one token.

Loss masking must exclude padding or intentionally ignored positions.

## 2. Optimizer

Stage 1 baseline:

- AdamW;
- configurable beta1/beta2/epsilon;
- configurable weight decay;
- exclude parameters from weight decay only by an explicit documented rule;
- gradient clipping by global norm.

No optimizer variant is introduced until the baseline is stable.

## 3. Learning-rate schedule

Baseline:

```text
linear warmup
    ↓
cosine decay
```

Configuration must use step counts derived from effective training steps.

Persist scheduler state in resumable checkpoints.

**Burn 0.22 cosine semantics:** `CosineAnnealingLrScheduler` changed behavior in 0.22. Cycles no longer restart abruptly; `num_iters` specifies the descent to the minimum, after which the rate rises along the cosine curve. Any schedule planned from 0.21-era warm-restart expectations must be recomputed for these semantics.

**Burn 0.22 scheduler record API:** a custom `LrScheduler` no longer has an associated `Record` type or backend-generic record methods. It implements `to_record(&self) -> LrSchedulerRecord` and `load_record(&mut self, record: LrSchedulerRecord)`, and loading mutates the scheduler in place. Persisting scheduler state for resume therefore means persisting `LrSchedulerRecord`.

## 4. Effective batch size

Always report:

```text
effective_tokens_per_step =
micro_batch_size
× sequence_length
× gradient_accumulation_steps
× data_parallel_world_size
```

Configuration must never use ambiguous “batch size” without distinguishing micro-batch and global/effective batch.

## 5. Precision progression

### Correctness first

Start smoke tests in FP32.

### Accelerator training

Move to BF16 only after:

- target backend supports it;
- forward/backward parity tests pass;
- loss remains finite;
- no unexplained convergence regression occurs.

FP16 is not the default unless hardware requires it and loss scaling is implemented correctly.

## 6. Training loop responsibilities

Each step:

```text
read batch
→ transfer to device
→ forward
→ calculate masked causal loss
→ backward
→ optional accumulation
→ gradient finite check
→ gradient clipping
→ optimizer step
→ scheduler step
→ zero/reset gradients as required
→ metrics
→ periodic eval/checkpoint
```

## 7. Numerical health metrics

Track at least:

- train loss;
- validation loss;
- perplexity when meaningful;
- learning rate;
- gradient norm;
- tokens/second;
- step time;
- GPU memory usage when available;
- non-finite loss count;
- skipped/failed steps.

A NaN/Inf loss must not be silently ignored.

## 8. Checkpoint strategy

Burn 0.22 uses its current record/checkpoint APIs and burnpack model storage. Use the framework-native path for internal resume unless measurement or compatibility requires otherwise.

For model portability/export, support SafeTensors through Burn's store facilities or the SafeTensors Rust crate as appropriate.

### Full resumable checkpoint must include

- model weights;
- optimizer state;
- scheduler state;
- global step;
- tokens seen;
- epoch/shard/dataset cursor state sufficient to resume correctly;
- RNG state where available/necessary;
- resolved experiment config;
- tokenizer hash;
- dataset manifest hash;
- code/git metadata.

### Atomicity

Never overwrite the only valid checkpoint in place.

Preferred pattern:

```text
write checkpoint.tmp/
fsync/close as applicable
validate contents
rename atomically to checkpoint-N/
update latest pointer/manifest
```

Exact implementation depends on filesystem semantics.

## 9. Resume validation

Automated integration test:

1. train N steps;
2. checkpoint;
3. continue K steps and capture metrics/weights;
4. restart process from checkpoint;
5. run K steps on same deterministic batches;
6. compare within defined tolerance.

Any missing optimizer/scheduler/global-step state is a failure.

## 10. Checkpoint retention

Configurable policy:

- keep latest;
- keep best validation checkpoints;
- keep periodic milestones;
- never delete the only known-good checkpoint;
- deletion runs only after a new checkpoint has been validated.

## 11. Experiment artifacts

Each run directory:

```text
run/
├── resolved-config.toml
├── manifest.json
├── tokenizer-hash.txt
├── dataset-manifest.json
├── metrics.jsonl
├── checkpoints/
├── eval/
└── benchmark/
```
