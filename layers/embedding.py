#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch


def rotary_embedding(q, k, freq_cis):
    # 旋转位置编码
    xq = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    xk = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freq_cis = reshape_for_broadcast(freq_cis, xq)
    xq_out = torch.view_as_real(xq * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk * freq_cis).flatten(3)
    return xq_out.type_as(q), xk_out.type_as(k)


def reshape_for_broadcast(freq_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freq_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freq_cis.view(*shape)
