from .conv import *
import torch
import torch.nn as nn

import itertools
import einops
import math
import numpy as np
from einops import rearrange
from torch import Tensor
from typing import Tuple, Optional, List

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class DSM_SpatialGate(nn.Module):
    def __init__(self, channel):
        super(DSM_SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size, act=False)
        self.dw1 = nn.Sequential(
            Conv(channel, channel, 5, s=1, d=2, g=channel, act=nn.GELU()),
            Conv(channel, channel, 7, s=1, d=3, g=channel, act=nn.GELU())
        )
        self.dw2 = Conv(channel, channel, kernel_size, g=channel, act=nn.GELU())

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out


class DSM_LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel

        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2, 3), keepdim=True)
        return self.a * out * x + self.b * x


class DualDomainSelectionMechanism(nn.Module):
    # https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_Focal_Network_for_Image_Restoration_ICCV_2023_paper.pdf
    # https://github.com/c-yn/FocalNet
    # Dual-DomainSelectionMechanism
    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate = DSM_SpatialGate(channel)
        layers = [DSM_LocalAttention(channel, p=i) for i in range(pyramid - 1, -1, -1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a * out + self.b * x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, resolution, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None, focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        H_sp, W_sp = self.resolution[0], self.resolution[1]
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.conv_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.H_sp * self.W_sp, dim)))

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        # x = x.reshape(-1, self.H_sp * self.W_sp, C).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, self.H_sp * self.W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B C H W
        """
        qkv = self.conv_qkv(qkv)
        q, k, v = torch.chunk(qkv.flatten(2).transpose(1, 2), 3, dim=-1)

        ### Img2Window
        H, W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=self.H_sp, w=self.W_sp)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = windows2img(x, self.H_sp, self.W_sp, H, W).permute(0, 3, 1, 2)
        return x