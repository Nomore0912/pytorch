#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.,
                 k_dim=None, v_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim

        self.values = nn.Linear(self.head_dim, self.v_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.k_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, values, keys, queries, mask):
        # n = batch size
        n = queries.shape[0]
        # qkv length
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(n, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(n, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(n, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query and keys and scales it
        # n-batch, q-qlen, k-klen, h-head, d-dim
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # dim=3 在nhqk的k维度做softmax -> output "nhqk"
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            n, query_len, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
