import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# RMSNorm (faster than LayerNorm)
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


# ------------------------------------------------------------
# Multi-Head Self Attention (Flash Attention enabled)
# ------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention (PyTorch optimized)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True
        )

        out = attn.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)


# ------------------------------------------------------------
# SwiGLU FeedForward (better than GELU)
# ------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        hidden_dim = 4 * d_model

        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ------------------------------------------------------------
# Language Model
# ------------------------------------------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_heads, num_layers):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)

        # output head
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # WEIGHT TYING (important)
        self.head.weight = self.token_emb.weight

        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size

        pos = torch.arange(0, T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.head(x)

        return logits