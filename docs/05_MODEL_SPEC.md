# Model Specification

## 1. Stage-1 model

Name used in documentation: `RustLLM-110M` (working name only).

### Architecture

| Property | Stage-1 baseline |
|---|---:|
| Type | Decoder-only Transformer |
| Objective | Causal next-token prediction |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| KV heads | 12 |
| Head dimension | 64 |
| Attention | MHA |
| FFN | SwiGLU |
| FFN intermediate | ~2048 |
| Vocabulary | 32,768 |
| Max sequence length | 2,048 |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Norm placement | Pre-norm |
| Embedding/LM head | Tied |
| Major linear biases | Disabled |
| Default training dtype | FP32 smoke; BF16 when validated/supported |
| Dropout | 0 by default for initial LLM baseline |

Exact parameter count must be printed by the implementation and stored in every experiment manifest.

## 2. Token flow

For a sequence:

```text
tokens: x0 x1 x2 ... xN
input : x0 x1 x2 ... xN-1
target: x1 x2 x3 ... xN
```

Cross-entropy is calculated over the shifted targets.

Padding tokens, if present, must be excluded from the loss through an explicit loss mask.

**Burn 0.22 behavior to account for:** Burn 0.22's cross-entropy excludes padded targets from its denominator, including the weight sum for weighted loss. A batch that is entirely padding therefore returns `NaN`. That case must be detected and reported as an invalid batch rather than propagated into the optimizer — it is a batching defect, not a numerical-instability event. The explicit loss mask remains required so the intended ignore positions are stated by this project rather than inferred from framework defaults.

## 3. Embedding

Input token IDs map to `hidden_size` vectors.

Requirements:

- validate IDs are within vocabulary;
- preserve dtype/device semantics;
- support weight tying with output projection.

## 4. RMSNorm

Reference formula:

```text
rms(x) = sqrt(mean(x^2) + eps)
y      = x / rms(x) * weight
```

Requirements:

- configurable epsilon;
- computation numerically stable for supported dtype;
- compare against a scalar/reference implementation in tests.

## 5. RoPE

Apply rotary position embedding to query/key channels before attention score computation.

Requirements:

- deterministic position indexing;
- support offset positions for KV-cached decode;
- validate even rotary dimension;
- cached sin/cos tables may be introduced only with bounded memory and equivalence tests.

## 6. Causal self-attention

Stage 1 uses MHA.

Conceptual equations:

```text
Q = X Wq
K = X Wk
V = X Wv

scores = (Q K^T) / sqrt(head_dim)
scores += causal_mask
P = softmax(scores)
A = P V
Y = concat(A_heads) Wo
```

Causal invariant:

A token at position `t` must never observe positions `> t`.

Mask correctness is a critical security/correctness invariant for autoregressive training.

## 7. SwiGLU

Conceptual form:

```text
SwiGLU(x) = SiLU(x W_gate) ⊙ (x W_up)
FFN(x)    = SwiGLU(x) W_down
```

Exact tensor shapes must be documented in code.

## 8. Residual block

```text
h = x + Attention(RMSNorm(x))
y = h + FFN(RMSNorm(h))
```

## 9. Final projection

```text
h = RMSNorm(h)
logits = h × embedding_weight^T
```

Output dimension equals tokenizer vocabulary size.

## 10. Initialization

Use a documented Transformer-appropriate initialization. The initial implementation must:

- centralize initialization policy;
- use deterministic seed control;
- avoid hidden default initializers scattered across modules;
- add tests for finite initialized parameters.

Do not tune initialization before the tiny-model overfit test works.

## 11. Model configuration validation

Reject invalid configurations before allocating large tensors.

At minimum validate:

- `hidden_size % num_heads == 0`;
- vocabulary > special-token maximum ID;
- sequence length > 0;
- number of layers > 0;
- FFN size > hidden size;
- RoPE dimension is valid;
- KV heads divide query heads once GQA is introduced;
- unsupported dtype/backend combinations fail clearly.

## 12. Stage-2 architectural delta

Stage 2 changes the attention layout to GQA and may introduce optimized/fused attention. The public model configuration must be designed so `num_kv_heads` can differ from `num_heads` without redesigning the whole model.

## 13. No hidden architecture drift

Any change to these items requires an ADR update:

- normalization type/order;
- attention type;
- activation;
- positional encoding;
- weight tying;
- loss definition;
- tokenizer vocabulary;
- sequence length semantics;
- whether a position carries a token id or an injected encoder vector, and how
  such a position is excluded from the text objective.

The last item is not implemented. It is listed here so that a future modality
work cannot be introduced as an incidental change: ADR-018 records that
multimodal capability is a scope expansion, and this list is where the
architecture-drift rule meets it.
