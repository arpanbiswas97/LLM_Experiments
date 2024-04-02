import math
from typing import Tuple

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len)

        Returns
        -------
        torch.Tensor
            (batch, seq_len, d_model)
        """

        return self.embedding(x) + math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1000.0)) / d_model
        )  # (d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor
            (batch, seq_len, d_model)
        """

        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))  # [1]
        self.beta = nn.Parameter(torch.zeros(1))  # [0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor
            (batch, seq_len, d_model)
        """

        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.d_k = d_model // heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate masked attention scores and apply dropout

        Parameters
        ----------
        query : torch.Tensor
            (batch, heads, seq_len, d_k)
        key : torch.Tensor
            (batch, heads, seq_len, d_k)
        value : torch.Tensor
            (batch, heads, seq_len, d_k)
        mask : torch.Tensor
            (batch, 1, seq_len, seq_len)
        dropout : nn.Dropout

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (batch, heads, seq_len, d_k), (batch, heads, seq_len, seq_len)
        """
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            # (batch, heads, seq_len, seq_len) <-broadcast (batch, 1, seq_len, seq_len)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores: torch.Tensor = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q : torch.Tensor
            (batch, seq_len, d_model)
        k : torch.Tensor
            (batch, seq_len, d_model)
        v : torch.Tensor
            (batch, seq_len, d_model)
        mask : torch.Tensor
            (batch, 1, seq_len, seq_len)

        Returns
        -------
        torch.Tensor
            (batch, seq_len, d_model)
        """

        query: torch.Tensor = self.w_q(q)
        key: torch.Tensor = self.w_k(k)
        value: torch.Tensor = self.w_v(v)

        # (batch, seq_len, d_model) -> (batch, seq_len, heads, d_k) -> (batch, heads, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.heads, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )  # (batch, heads, seq_len, d_k)

        x = (
            x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        )  # (batch, seq_len, d_model)

        return self.w_o(x)


class Transformer(nn.Module): ...
