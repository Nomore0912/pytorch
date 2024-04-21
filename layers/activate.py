#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
# Active Function


def silu(x):
    # Swish激活函数
    # SiLU(x) = x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + e^(-x))
    return x * torch.sigmoid(x)
