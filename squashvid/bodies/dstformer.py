"""MotionBERT's DSTformer, trimmed to the 'stage_st'/'stage_ts' dual-stream blocks the 3D-pose checkpoint uses.

Adapted from https://github.com/Walter0807/MotionBERT (lib/model/DSTformer.py), Apache-2.0,
Copyright (c) 2022 Wentao Zhu et al. Parameter names are kept identical so the released checkpoint loads strictly.
Dropout and stochastic depth are omitted because this is inference only.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Spatial (across joints of one frame) or temporal (across frames of one joint) multi-head attention."""

    def __init__(self, dim: int, num_heads: int, mode: str):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.mode = mode
        self.qkv = nn.Linear(dim, dim * 3, bias=True)

    def forward(self, x, seqlen: int):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.mode == "spatial":
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            def split(t):
                return t.reshape(-1, seqlen, self.num_heads, N, C // self.num_heads).permute(0, 2, 3, 1, 4)

            qt, kt, vt = split(q), split(k), split(v)
            attn = ((qt @ kt.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = (attn @ vt).permute(0, 3, 2, 1, 4).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, st_mode: str, norm_layer):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s, self.norm1_t = norm_layer(dim), norm_layer(dim)
        self.attn_s = Attention(dim, num_heads, "spatial")
        self.attn_t = Attention(dim, num_heads, "temporal")
        self.norm2_s, self.norm2_t = norm_layer(dim), norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_s, self.mlp_t = MLP(dim, hidden), MLP(dim, hidden)

    def forward(self, x, seqlen: int):
        order = ("s", "t") if self.st_mode == "stage_st" else ("t", "s")
        for which in order:
            attn, norm1, mlp, norm2 = (
                (self.attn_s, self.norm1_s, self.mlp_s, self.norm2_s)
                if which == "s"
                else (self.attn_t, self.norm1_t, self.mlp_t, self.norm2_t)
            )
            x = x + attn(norm1(x), seqlen)
            x = x + mlp(norm2(x))
        return x


class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2,
                 num_joints=17, maxlen=243):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.blocks_st = nn.ModuleList([Block(dim_feat, num_heads, mlp_ratio, "stage_st", norm_layer) for _ in range(depth)])
        self.blocks_ts = nn.ModuleList([Block(dim_feat, num_heads, mlp_ratio, "stage_ts", norm_layer) for _ in range(depth)])
        self.norm = norm_layer(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([("fc", nn.Linear(dim_feat, dim_rep)), ("act", nn.Tanh())]))
        self.head = nn.Linear(dim_rep, dim_out)
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for _ in range(depth)])

    def forward(self, x):
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        x = self.joints_embed(x) + self.pos_embed
        x = x.reshape(-1, F, J, self.dim_feat) + self.temp_embed[:, :F]
        x = x.reshape(BF, J, self.dim_feat)
        for idx, (st, ts) in enumerate(zip(self.blocks_st, self.blocks_ts, strict=True)):
            x_st, x_ts = st(x, F), ts(x, F)
            alpha = self.ts_attn[idx](torch.cat([x_st, x_ts], dim=-1)).softmax(dim=-1)
            x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
        x = self.pre_logits(self.norm(x).reshape(B, F, J, -1))
        return self.head(x)
