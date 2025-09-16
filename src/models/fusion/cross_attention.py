from __future__ import annotations

from torch import nn, Tensor


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, query: Tensor, key_value: Tensor, key_padding_mask=None) -> Tensor:
        # query: [B, Q, C], key_value: [B, T, C]
        x = query
        attn_out, _ = self.attn(x, key_value, key_value, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


