# GPT-2 Transformer — Complete Ready Reckoner

> Built from your notes, rated 7/10. Gaps filled, misconceptions corrected, parameter math added.
> This is your single-source reference for Phase 1 pretraining experiments.

---

## 0. Evaluation of Your Notes

| Area | Your Coverage | Score |
|---|---|---|
| Historical context (RNN → Attention → Transformer) | Correct, good narrative | ✅ |
| GPT-2 as decoder-only | Correct | ✅ |
| BPE tokenization | Directionally right, detail missing | ⚠️ |
| Learned embeddings rationale | Correct | ✅ |
| Pre-LN vs Post-LN | Named it but order confused | ⚠️ |
| Multi-head attention mechanics | Correct intent, W_o missing | ⚠️ |
| Causal masking | Correct | ✅ |
| Residual connections | Correct | ✅ |
| FFN + GELU rationale | Correct | ✅ |
| Final LayerNorm before lm_head | Not mentioned | ❌ |
| Weight tying (embedding ↔ lm_head) | Not mentioned | ❌ |
| Parameter counts | Not mentioned | ❌ |
| Dropout placement | Placed before LN — wrong order | ❌ |
| Data leakage question | Raised but unanswered | ⚠️ |

**Overall: 7/10 — Ready for Phase 1 with this document as reference.**

---

## 1. Why Transformers? The Historical Arc

```
Characters → Words → Subwords (BPE)
    ↓
RNN / LSTM   →   captures sequence but suffers vanishing/exploding gradients
    ↓
Seq2Seq (Encoder-Decoder RNN)  →  context vector bottleneck, loses long-range info
    ↓
Bahdanau Attention (2014)  →  encoder hidden states weighted by relevance to decoder step
    ↓
"Attention Is All You Need" — Vaswani et al. (2017)  →  attention replaces recurrence entirely
    ↓
BERT  →  encoder-only, masked LM, bidirectional
GPT   →  decoder-only, causal LM, left-to-right
```

**LSTM gating recap:**
- **Forget gate:** how much long-term memory to erase
- **Input gate:** how much of new input to write into cell state
- **Output gate:** what portion of cell state to expose as hidden state
- Still suffers from gradient problems across long sequences because backprop-through-time multiplies many Jacobians

**Bahdanau attention:** After encoding, a small feedforward net learns a score between each encoder hidden state and the current decoder state → softmax → weighted sum. First demonstration that "attending to all past states" beats a bottleneck vector.

**Vaswani 2017:** Removes recurrence entirely. Attention operates over all positions simultaneously → parallelizable, no sequential bottleneck, gradient path from any output to any input is O(1) in depth.

---

## 2. GPT-2 Architecture — Full Picture

GPT-2 Small hyperparameters (the ones you will experiment with):

| Symbol | Meaning | GPT-2 Small |
|---|---|---|
| V | Vocab size | 50,257 |
| T | Max context length | 1,024 |
| C | d_model (embedding dim) | 768 |
| h | Number of attention heads | 12 |
| d_head | Dim per head = C/h | 64 |
| L | Number of transformer blocks | 12 |
| d_ff | FFN hidden dim = 4×C | 3,072 |

```
INPUT TOKENS  [batch, T]
      │
      ▼
┌─────────────────────────────────┐
│  Token Embedding  (V × C)       │  ← learned lookup table
│  + Positional Embedding (T × C) │  ← learned, not sinusoidal in GPT-2
│  + Dropout                      │
└─────────────────────────────────┘
      │  [batch, T, C]
      ▼
┌─────────────────────────────────┐
│  Transformer Block × L          │  ← repeated 12 times
│                                 │
│   x = x + Attn(LN(x))          │  ← Pre-LN + residual
│   x = x + FFN(LN(x))           │  ← Pre-LN + residual
└─────────────────────────────────┘
      │  [batch, T, C]
      ▼
┌─────────────────────────────────┐
│  Final LayerNorm  (C)           │  ← often missed! Added in GPT-2
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  LM Head  (C → V)               │  ← weight-tied with token embedding
└─────────────────────────────────┘
      │  [batch, T, V]  logits
      ▼
   Cross-Entropy Loss (on next-token targets)
```

---

## 3. Step-by-Step Build with Rationale and Parameters

### 3.1 Data Pipeline

#### BPE Tokenization

Your description was directionally correct but imprecise. Here is the exact algorithm:

```
Start: corpus as individual bytes (characters)
Repeat:
    1. Count all adjacent byte-pair frequencies
    2. Merge the most frequent pair → new token
    3. Update vocabulary with merged token
Until: vocabulary size = target (50,257 for GPT-2)
```

Result: frequent words become single tokens, rare words get split into subwords, very rare patterns stay as bytes. GPT-2 uses byte-level BPE (works on raw bytes, so it can tokenize any Unicode string without OOV).

**Why not character-level?**  
Sequences become very long → attention is O(T²), expensive.

**Why not word-level?**  
OOV problem, huge vocab, morphological variants need separate entries.

**BPE sweet spot:** vocabulary of ~50K covers most common patterns, sequence lengths stay manageable.

#### Sliding Window Dataset

```
text: [t0, t1, t2, t3, t4, t5, t6, t7, ...]

stride = 1, seq_len = 4:
  input:  [t0, t1, t2, t3]   target: [t1, t2, t3, t4]
  input:  [t1, t2, t3, t4]   target: [t2, t3, t4, t5]
  ...

stride = seq_len (non-overlapping):
  input:  [t0, t1, t2, t3]   target: [t1, t2, t3, t4]
  input:  [t4, t5, t6, t7]   target: [t5, t6, t7, t8]
```

**Rationale:** This is self-supervised — labels come from the data itself by shifting one position. Every position in every sequence contributes a training signal. Small stride → more overlap → more data but correlated samples. Large stride → less correlated but fewer samples. For pretraining experiments, stride = seq_len is common.

---

### 3.2 Token Embeddings

```python
token_emb = nn.Embedding(vocab_size, d_model)  # V × C
```

**Parameters:** V × C = 50,257 × 768 = **38,597,376**

**Why learned, not fixed (word2vec/GloVe)?**  
Fixed embeddings are trained on a different objective (co-occurrence stats) and can't be fine-tuned end-to-end. Learned embeddings update to match exactly what the transformer needs — the right geometric space for the downstream task.

---

### 3.3 Positional Embeddings

```python
pos_emb = nn.Embedding(context_length, d_model)  # T × C
```

**Parameters:** T × C = 1,024 × 768 = **786,432**

**GPT-2 uses learned positional embeddings, not sinusoidal.**

| | Sinusoidal (Vaswani 2017) | Learned (GPT-2) |
|---|---|---|
| Parameters | 0 (deterministic) | T × C |
| Can extrapolate to longer T? | Theoretically yes | No (out-of-distribution) |
| Flexibility | Fixed | Adapts during training |
| Empirically | Similar performance | Slightly better on fixed T |

GPT-2 is trained at T=1024 and evaluated at T=1024, so extrapolation is not needed → learned is fine.

---

### 3.4 Input Embedding

```
x = token_emb(tokens) + pos_emb(positions)   # [batch, T, C]
x = dropout(x)                                 # regularization at input
```

**The dropout here is on the summed embedding, not after LayerNorm.** This is an important detail you had out of order.

---

### 3.5 Transformer Block (× L)

Full pseudocode for one block:

```python
# --- Sublayer 1: Multi-Head Causal Self-Attention ---
residual = x
x = layer_norm_1(x)          # Pre-LN: normalize BEFORE attention
x = multi_head_attn(x)       # attention transform
x = dropout(x)               # dropout BEFORE residual add — regularizes only the new info
x = residual + x             # residual add — clean signal unaffected by dropout

# --- Sublayer 2: Feed-Forward Network ---
residual = x
x = layer_norm_2(x)          # Pre-LN: normalize BEFORE FFN
x = ffn(x)                   # FFN transform
x = dropout(x)               # dropout BEFORE residual add — same rationale
x = residual + x             # residual add — clean signal unaffected by dropout
```

**Critical point: Pre-LN order is LN → sublayer → dropout → residual add.**  
Not: sublayer → LN → residual. That is Post-LN (original paper).

**Why dropout goes before the residual add, not after:**  
The residual path carries the clean accumulated signal — it is the gradient highway. Dropout after the add would randomly zero parts of that stream, disrupting gradient flow and undermining the stability residuals provide. Instead, dropout is applied only to the sublayer's *output* — the new information being contributed — so only that delta is noisy. The residual passes through untouched.

```
residual ──────────────────────────────────┐
                                           ▼
input → LN → sublayer → dropout → [ + ] → output
                           ↑
                  regularize only the
                  new information here
```

---

#### 3.5.1 Layer Normalization

```
LayerNorm: for each token vector of dim C,
    μ = mean across C
    σ = std across C
    x̂ = (x - μ) / (σ + ε)
    output = γ * x̂ + β       ← γ, β are learned per-dim scalars
```

**Parameters per LN:** 2 × C = 2 × 768 = **1,536** (γ vector + β vector)

**Why normalize across C (not across batch)?**  
Batch Norm normalizes across the batch dimension → statistics depend on batch size and other samples. Layer Norm normalizes each token's embedding vector independently → no batch dependency, works at any batch size including 1. Critical for autoregressive generation.

**Pre-LN vs Post-LN — Stability Story:**

```
Post-LN (original Vaswani):
    x → sublayer → x + sublayer_out → LN → next

Pre-LN (GPT-2):
    x → LN → sublayer → x + sublayer_out → next
```

In Post-LN, residual streams can grow unbounded across layers before normalization — the LN at the end can't fully compensate for deep stacking. Leads to training instability, requires careful learning rate warmup.

In Pre-LN, each sublayer receives a normalized input regardless of what the residual stream looks like. Gradients flow through the residual path unobstructed. Much more stable, especially at scale.

**Your data leakage question (answered):**  
Pre-LN does NOT leak future tokens. LayerNorm operates on a single token's embedding vector (dimension C) — it computes mean and std across the 768 feature dimensions of *that token's vector*, not across the sequence dimension. Future tokens are still masked in attention. No leakage.

---

#### 3.5.2 Multi-Head Causal Self-Attention

```
          [batch, T, C]
               │
    ┌──────────┼──────────┐
    │          │          │
  W_Q(C×C)  W_K(C×C)  W_V(C×C)   ← learned projection matrices
    │          │          │
    Q          K          V        [batch, T, C] each
    │
    └─── reshape to [batch, h, T, d_head]  (h=12, d_head=64)
```

**QKV projection parameters:**

```
W_Q:  C × C = 768 × 768 = 589,824  +  bias C = 768  →  590,592
W_K:  C × C = 589,824              +  bias C = 768  →  590,592
W_V:  C × C = 589,824              +  bias C = 768  →  590,592

Combined QKV projection: C × 3C + 3C = 768×2304 + 2304 = 1,771,776
(In code: nn.Linear(C, 3*C) — single matrix multiply, then split)
```

**Scaled Dot-Product Attention (per head):**

```
Q, K, V each: [batch, h, T, d_head]

Attention scores:
    A = Q @ K^T / sqrt(d_head)    [batch, h, T, T]

Causal mask (future tokens invisible):
    mask = upper triangle of ones (above diagonal)
    A = A.masked_fill(mask == 1, -inf)

Softmax:
    A = softmax(A, dim=-1)         [batch, h, T, T]
    (each row sums to 1; -inf → 0 after softmax)

Context vector:
    out = A @ V                    [batch, h, T, d_head]
    out = concat heads → [batch, T, C]
```

**Why divide by sqrt(d_head)?**  
With random init, Q@K^T has variance ≈ d_head (sum of d_head products each with variance ~1). Large values push softmax into saturation → near-zero gradients everywhere except the argmax. Dividing by sqrt(d_head) brings variance back to ~1, keeping softmax in its informative regime.

**Why multiple heads?**  
Each head operates on a 64-dim subspace of the 768-dim token vector. Different heads can independently attend to different relational patterns simultaneously — one head might track subject-verb agreement, another syntactic dependencies, another coreference. Forcing all attention through a single head would average these signals and lose resolution.

**What are Q, K, V intuitively?**
```
Query  (Q): "What am I looking for?" — current token's question
Key    (K): "What do I advertise?" — every token broadcasts its identity
Value  (V): "What do I contribute?" — every token's actual content

Attention score = how well Q matches K (relevance)
Context vector  = weighted sum of V (information)
```

**Output Projection (W_o) — the one you missed:**

After concatenating all heads back to [batch, T, C], there is a final linear layer:

```python
out = nn.Linear(C, C)(concat_heads)   # [batch, T, C]
```

**Parameters:** C × C + C = 768 × 768 + 768 = **590,592**

**Why W_o?**  
Concatenating heads just stacks the subspace outputs side by side. W_o mixes information across heads — it learns which combinations of head outputs matter. Without it, heads are independent with no cross-head communication.

**Attention parameter total:**  
1,771,776 (QKV) + 590,592 (W_o) = **2,362,368**

---

#### 3.5.3 Feed-Forward Network

```
x → FC1 (C → 4C) → GELU → FC2 (4C → C)
```

```python
FC1 = nn.Linear(C, 4*C)     # 768 → 3072
FC2 = nn.Linear(4*C, C)     # 3072 → 768
```

**Parameters:**
```
FC1: C × 4C + 4C = 768×3072 + 3072 = 2,362,368
FC2: 4C × C  + C = 3072×768 + 768  = 2,360,064
FFN total: 4,722,432
```

**Why 4× expansion?**  
The bottleneck-then-expand pattern gives the network a "scratchpad" — a high-dimensional space where it can compute feature interactions that would be hard to express in C dimensions. Then it projects back down, forcing a compression/selection of what's useful.

**Why GELU instead of ReLU?**

```
ReLU(x) = max(0, x)     → hard zero for x < 0 → dead neurons
GELU(x) = x · Φ(x)     → Φ is Gaussian CDF
```

GELU is a smooth gate: for very negative x it's ~0 (like ReLU), for positive x it approaches identity. The smooth transition means gradients are never exactly 0 for any neuron — no dead neurons. Also, GELU has a slight dip below 0 for small negative inputs, which acts as a soft regularizer.

**What FFN is actually doing:**  
After attention mixes *which* tokens matter, FFN refines *what* each token means in context. It has been shown empirically that FFN layers store factual associations (e.g., "Paris" → "capital of France"). This is distinct from attention's job of routing information between positions.

---

### 3.6 Final LayerNorm — The One You Missed

```python
x = layer_norm_final(x)   # [batch, T, C]
```

**Parameters:** 2 × C = **1,536**

**Why?**  
After 12 transformer blocks with residual additions, the scale of the residual stream can drift. Without this final LN, the lm_head receives vectors with arbitrary scale. GPT-2 added this (it's not in the original Vaswani architecture). If you look at the original GPT, it had Post-LN inside blocks and no final LN — GPT-2 switched to Pre-LN and added this final normalization.

---

### 3.7 LM Head — and Weight Tying

```python
lm_head = nn.Linear(C, V, bias=False)   # 768 → 50257
```

**Weight tying:** `lm_head.weight = token_embedding.weight`

This means the same matrix (V × C) is used both at the input (to look up token embeddings) and at the output (to project C → V logits). This is not just a trick — it has a semantic justification: both operations map between the same two spaces (token identity space and embedding space), so sharing weights makes sense. It also reduces parameters by 38.6M.

**Without weight tying:** +38,597,376 params  
**With weight tying:** 0 extra params ← GPT-2 uses this

**Output shape:** [batch, T, V] = logits for every vocab token at every position.

---

### 3.8 Loss Computation

```
logits: [batch, T, V]
targets: [batch, T]  (= input shifted right by 1)

For each position t, the model predicts the probability of token t+1.

Cross-Entropy Loss:
    loss = -log(softmax(logits)[target_id])
         = -log(p_correct)
    
    Averaged over all positions and batch.

Perplexity:
    PPL = exp(loss)
```

**Interpreting perplexity:**
```
PPL = 50257 → model is as confused as random guessing (uniform over vocab)
PPL = 1     → model is certain of every next token (perfect)
PPL = 100   → roughly, model is choosing among ~100 equally likely tokens
```

**Why cross-entropy and not MSE?**  
We're predicting a probability distribution over discrete tokens. MSE would treat token IDs as continuous numbers — token 5 being "close" to token 6 is meaningless. Cross-entropy is the natural loss for distributions: it measures bits needed to encode the true label under the model's predicted distribution.

---

## 4. Complete Parameter Count Table — GPT-2 Small

| Component | Formula | Count |
|---|---|---|
| Token Embedding | V × C | 38,597,376 |
| Positional Embedding | T × C | 786,432 |
| **Per Block** | | |
| LayerNorm 1 (γ, β) | 2 × C | 1,536 |
| QKV Projection | C × 3C + 3C | 1,771,776 |
| Output Projection W_o | C × C + C | 590,592 |
| LayerNorm 2 (γ, β) | 2 × C | 1,536 |
| FFN FC1 | C × 4C + 4C | 2,362,368 |
| FFN FC2 | 4C × C + C | 2,360,064 |
| **Block Total** | | **7,087,872** |
| 12 Blocks | 12 × 7,087,872 | 85,054,464 |
| Final LayerNorm | 2 × C | 1,536 |
| LM Head | tied with token emb | 0 |
| **GRAND TOTAL** | | **124,439,808 ≈ 124M** |

**Sanity check:** The known GPT-2 Small parameter count is 124M. ✅

**Parameter distribution insight:**
```
Embedding layers:       ~32% of params  (39.4M)
Transformer blocks:     ~68% of params  (85.1M)
  └── Attention:        ~27%
  └── FFN:              ~38%
  └── LayerNorms:       <1%
```

---

## 5. Architecture Flow — Precise Data Shapes

```
Input token IDs:     [B, T]         B=batch, T=seq_len

Token embedding:     [B, T, C]      C=768
Pos embedding:       [B, T, C]
Sum + dropout:       [B, T, C]

── Block (×12) ──────────────────────────────
LayerNorm:           [B, T, C]
Q, K, V projection:  [B, T, C] each
Reshape to heads:    [B, h, T, d_h]  h=12, d_h=64
Attention scores:    [B, h, T, T]
After causal mask:   [B, h, T, T]   upper triangle = -inf
After softmax:       [B, h, T, T]   rows sum to 1
Context vectors:     [B, h, T, d_h]
Concat heads:        [B, T, C]
W_o projection:      [B, T, C]
Dropout:             [B, T, C]
+ residual:          [B, T, C]

LayerNorm:           [B, T, C]
FC1 (GELU):          [B, T, 4C]    = [B, T, 3072]
FC2:                 [B, T, C]
Dropout:             [B, T, C]
+ residual:          [B, T, C]
─────────────────────────────────────────────

Final LayerNorm:     [B, T, C]
LM Head (C→V):       [B, T, V]     V=50257  ← logits

Loss targets:        [B, T]        (input shifted by 1)
Cross-entropy:       scalar
```

---

## 6. Attention Head Visualization

```
Full attention matrix A [T × T] for one head:

         Keys (all positions)
         t0   t1   t2   t3   t4
    t0 [ a00  -∞   -∞   -∞   -∞  ]
    t1 [ a10  a11  -∞   -∞   -∞  ]  Q
    t2 [ a20  a21  a22  -∞   -∞  ]  u
    t3 [ a30  a31  a32  a33  -∞  ]  e
    t4 [ a40  a41  a42  a43  a44 ]  r
                                    y

After softmax: each row sums to 1.
Token t2 attends to t0, t1, t2 but CANNOT see t3, t4.
This is the causal mask.
```

---

## 7. Common Misconceptions in Your Notes — Corrected

| Your Claim | Correction |
|---|---|
| "Dropout is before LayerNorm" (step 5 before 6) | Dropout is applied *after* attention output and *after* FFN output, before the residual add |
| "BERT uses encoder-decoder" | BERT uses encoder-only |
| "Bahdanou attention" | Bahdanau attention (Dzmitry Bahdanau) |
| "Bhaswani paper" | Vaswani et al. ("Attention Is All You Need") |
| "Softmax on all tokens following the target" | Softmax over all V vocab tokens at each position, not over the sequence |
| Nothing about W_o | Output projection after concatenating heads is essential |
| Nothing about final LayerNorm | GPT-2 has a LayerNorm after all blocks, before lm_head |
| Nothing about weight tying | lm_head shares weights with token embedding — saves 38.6M params |

---

## 8. Experiment Readiness Checklist (Phase 1)

Before running your 8 pretraining experiments, confirm you understand:

- [ ] What changes when you vary `d_model` (C)? → Parameter count scales as C²
- [ ] What changes when you vary `n_heads`? → Each head dim = C/n_heads; compute same, specialization changes
- [ ] What changes when you vary `n_layers`? → Linear param scaling with L
- [ ] What changes when you vary `context_length`? → Attention is O(T²) memory; embedding adds T×C params
- [ ] What does training loss curve shape tell you? → Fast drop = good init/LR; plateau = LR too low; spike = LR too high
- [ ] What does perplexity at init tell you? → Should be ~vocab_size for untrained model (~50257). After 1 step it drops fast
- [ ] What is gradient norm? → Monitor to catch explosions early
- [ ] Pre-LN vs Post-LN: which is more stable? → Pre-LN (GPT-2 choice)

---

## 9. Quick Reference — Key Equations

```
Attention:    Attn(Q,K,V) = softmax(QK^T / √d_head) · V

GELU:         GELU(x) = x · Φ(x)   where Φ is standard Gaussian CDF
              ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])

LayerNorm:    LN(x) = γ · (x - μ)/√(σ²+ε) + β

Cross-Entropy: L = -∑ y_true · log(y_pred)   [per token: -log(p_correct)]

Perplexity:   PPL = exp(L)

Params in block:
    Attn:  C(3C+C) + 3C + C = 4C² + 4C
    FFN:   C(4C) + 4C + 4C·C + C = 8C² + 5C
    LNs:   4C
    Block: 12C² + 13C  ← for C=768: ~7.09M ✓
```

---

*Last updated: 2026-04-07 | Phase 0 → Phase 1 transition document*
