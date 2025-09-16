from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", max_len: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        enc_dim = self.model.config.hidden_size
        self.proj = nn.Linear(enc_dim, hidden_dim)
        self.max_len = max_len

    def forward(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(self.proj.weight.device) for k, v in tok.items()}
        out = self.model(**tok)
        seq = out.last_hidden_state  # [B, T, H]
        seq = self.proj(seq)  # [B, T, hidden_dim]
        attn_mask = tok["attention_mask"].bool()  # [B, T]
        return {"sequence": seq, "mask": attn_mask}


