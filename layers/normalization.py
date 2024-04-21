#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch import nn
import torch


class RMSNorm(torch.nn.Module):
    # root-mean-square layer normalization
    # RMSNorm = x * weight / (sqrt(mean(x^2) + eps))
    def __init__(self, dim: int, eps: float = 1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = bias
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.bias:
            return output * self.weight + self.offset
        else:
            return output * self.weight
