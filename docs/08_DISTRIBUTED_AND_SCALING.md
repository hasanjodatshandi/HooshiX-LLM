# Distributed Training and Scaling

## 1. Principle

Distributed complexity is introduced only when a single accelerator is no longer sufficient or when time-to-train materially benefits.

Burn's distributed capabilities are improving rapidly, but the project must not assume PyTorch/Megatron-level maturity without direct validation on the target hardware.

## 2. Scaling order

Use the least complex method that works:

```text
single GPU
→ gradient accumulation
→ single-node data parallel
→ multi-node data parallel
→ activation/memory optimization
→ model/tensor/sequence parallelism if required
→ expert parallelism only in Stage 5
```

## 3. Stage mapping

- Stage 1: single GPU baseline; CPU reference tests.
- Stage 2: single-node multi-GPU.
- Stage 3: production-quality multi-GPU and restart.
- Stage 4: multi-node if required by model/hardware.
- Stage 5: expert parallel/distributed MoE research.

## 4. Distributed correctness

Before performance work, verify:

- identical initial weights across replicas where required;
- gradients are reduced as intended;
- optimizer steps remain synchronized;
- global token accounting is correct;
- data shards do not unintentionally overlap;
- validation aggregation is correct;
- checkpoint ownership/writer policy avoids races.

## 5. Scale-efficiency metric

Report:

```text
scaling_efficiency(N) =
throughput_N /
(N × throughput_1)
```

Report both samples/sec and tokens/sec.

Also report:

- communication time;
- device-transfer time;
- compute time;
- idle time where measurable.

## 6. Communication

Burn 0.21 introduced redesigned differentiable collective foundations and major measured improvements in device transfer/all-reduce.

In Burn 0.22 the `distributed` and `collective` feature flags were removed, along with the `burn-collective` dependency. Collective operations are now reached through `burn::tensor::distributed`. `DistributedSession`/`DistributedRuntime` were replaced by `DistributedContext`, and DDP strategies are constructed with `ExecutionStrategy::ddp(devices, DistributedConfig { .. })`.

**Verified 2026-10-07:** CubeCL all-reduce currently requires CUDA, including for remote servers. No other backend can execute a collective reduction today, so any multi-GPU plan on this project must confirm CUDA availability before it is scheduled. This is a hard capability boundary for Stages 2 and 4.

Treat collective operations as a capability boundary. Do not write a custom distributed transport unless Burn's supported primitives are proven insufficient for a stage requirement.

## 7. Data partitioning

Each worker must receive a deterministic, non-overlapping stream unless sampling strategy intentionally differs.

Persist enough sampler/shard state for restart.

## 8. Checkpoint writer

For replicated training:

- designate one authoritative checkpoint writer or use an explicitly sharded checkpoint protocol;
- synchronize around checkpoint boundaries as required;
- include world-size/topology metadata;
- verify that restoration on a changed topology is either supported or explicitly rejected.

## 9. Stage-4 memory planning

Before launching a 7B run, produce a memory model covering:

- weights;
- gradients;
- optimizer states;
- activations;
- attention temporaries;
- communication buffers;
- allocator/runtime overhead;
- checkpoint staging.

Do not infer training feasibility from weight size alone.

## 10. Failure handling

Distributed runs require tests for:

- one process exits;
- communication timeout;
- checkpoint write failure;
- disk full;
- corrupt shard;
- restart from previous checkpoint.

Silent partial progress is unacceptable.
