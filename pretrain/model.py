import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        # Single matmul projects input to Q, K, V concatenated
        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=cfg.bias)

    def forward(self, x):
        B, T, C = x.shape
        # B = batch size  |  T = sequence length (tokens)  |  C = n_embd (embedding channels)

        # Single matmul → split: each of Q, K, V is (B, T, n_embd)
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)

        # Break embedding dim into n_head independent heads.
        # transpose(1,2): moves head axis before token axis so attention
        # is computed per-head over the time dimension.
        # Shape: (B, T, n_embd) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash attention: fused CUDA kernel — computes softmax(QKᵀ/√d)V without
        # materialising the full (T×T) attention matrix. is_causal=True applies
        # the causal mask so each token only attends to itself and earlier tokens.
        # Output shape: (B, n_head, T, head_dim)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads back: (B, n_head, T, head_dim) → (B, T, C)
        # contiguous() is required before view() after the transpose.
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear mixes information across heads
        return self.out_proj(attn_out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc_in  = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.act    = nn.GELU()
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        # Expand: project from n_embd → 4×n_embd (richer feature space)
        # Activate: GELU introduces non-linearity
        # Project: compress back from 4×n_embd → n_embd (residual-compatible shape)
        return self.fc_out(self.act(self.fc_in(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp   = MLP(cfg)

    def forward(self, x):
        # Pre-LN residual: normalise → transform → add back to residual stream.
        # Normalising *before* (not after) the sublayer keeps gradient magnitude
        # stable through depth and allows higher learning rates.
        x = x + self.attn(self.norm1(x))   # residual stream updated by attention
        x = x + self.mlp(self.norm2(x))    # residual stream updated by MLP
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg       = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_final  = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: lm_head reuses the token embedding matrix.
        # Rationale: token similarity in input space ≈ token similarity in output space.
        self.token_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        # B = batch size  |  T = sequence length (≤ block_size)

        # Integer positions [0, 1, ..., T-1] — same for every item in the batch
        positions = torch.arange(T, device=token_ids.device)

        # Combine token meaning (what) with position meaning (where).
        # Both embeddings are (B, T, n_embd); addition broadcasts correctly.
        x = self.token_emb(token_ids) + self.pos_emb(positions)

        # Pass through each transformer block in sequence.
        # Each block refines x via attention (context mixing) then MLP (per-token transform).
        for block in self.blocks:
            x = block(x)

        # Final layer norm before the output projection
        x = self.ln_final(x)

        # Project each token's hidden state to a score over the vocabulary.
        # logits shape: (B, T, vocab_size)
        logits = self.lm_head(x)

        # Cross-entropy loss over all token positions when targets are provided.
        # Flatten to (B*T, vocab_size) vs (B*T,) for the loss function.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        tied  = self.token_emb.weight.numel()   # lm_head shares this tensor; subtract once
        return total - tied
