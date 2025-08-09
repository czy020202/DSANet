import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from termcolor import cprint
from collections import OrderedDict

class PositionalEncoding(nn.Module):
    def __init__(self, n_channels, max_len=128):
        max_len = 512
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, n_channels) # D * C
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_channels, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).unsqueeze(-1).unsqueeze(-1) # D * 1 * C * 1 * 1
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.size()
        pos_enc = self.pe[:D, :].expand(D, B, C, H, W).permute(1, 2, 0, 3, 4) # (D, B, C, H, W) -> (B, C, D, H, W)
        return x + pos_enc

class DeformableSliceGrouped(nn.Module):
    def __init__(self, n_channels, d_model=256, n_heads=2, n_points=3): #d_model=32**2 or 16**2
        super().__init__()
        
        self.n_heads = n_heads
        self.n_points = n_points
        
        self.pe = PositionalEncoding(n_channels)#, max_len=d)
        
        self.sampling_offsets = nn.Linear(n_channels, n_heads * n_points) #up and down
        self.attention_weights = nn.Linear(n_channels, n_heads * n_points)
        
        self._reset_parameters()
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.q = nn.Conv3d(n_channels, n_channels, kernel_size=1, stride=1, bias=False)
        self.v = nn.Conv3d(n_channels, n_channels, kernel_size=1, stride=1, bias=False)
        self.o = nn.Conv3d(n_channels, n_channels, kernel_size=1, stride=1, bias=False)
        
        self.bn = nn.BatchNorm3d(n_channels)
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        direction_init = torch.tensor([-1, 1], dtype=torch.float32)
        direction_init = direction_init.view(2, 1).repeat(1, self.n_points)
        for i in range(self.n_points):
            direction_init[:, i] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(direction_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        
    def forward(self, features):
        residual = features
        b, c, zs, h, w = features.size()
        features_pe = self.pe(features)
        
        q = self.q(features_pe)  # b * c * zs * h * w
        v = self.v(features_pe)  # b * c * zs * h * w
        
        q = q.view(b * c, zs, h * w)
        q = self.pool(q).view(b, c, zs)
        q = q.permute(0, 2, 1)  # b * zs * c

        sampling_offsets = self.sampling_offsets(q).view(b, zs, self.n_heads * self.n_points)  # b * zs * (n_heads * n_points)
        sampling_offsets = sampling_offsets.clamp(0, zs - 1)
        attention_weights = self.attention_weights(q).view(b, zs, self.n_heads * self.n_points)  # b * zs * (n_heads * n_points)
        attention_weights = F.softmax(attention_weights, -1)

        lower_sampling = torch.floor(sampling_offsets).to(torch.int64)  # b * zs * (n_heads * n_points)
        upper_sampling = torch.ceil(sampling_offsets).to(torch.int64)
        frac = sampling_offsets - lower_sampling  # b * zs * (n_heads * n_points)

        b_indices = torch.arange(b)[:, None, None, None]
        c_indices = torch.arange(c)[None, :, None, None]
        zs_indices = torch.arange(zs)[None, None, :, None]
        hp_indices = torch.arange(self.n_heads * self.n_points)[None, None, None, :]

        lower_v = v[b_indices, c_indices, lower_sampling.unsqueeze(1)]  # b * c * zs * (n_heads * n_points) * h * w
        upper_v = v[b_indices, c_indices, upper_sampling.unsqueeze(1)]  # b * c * zs * (n_heads * n_points) * h * w

        frac = frac.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # b * 1 * zs * (n_heads * n_points) * 1 * 1
        sampling_v = (1 - frac) * lower_v + frac * upper_v  # b * c * zs * (n_heads * n_points) * h * w

        sampling_v = torch.sum(attention_weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) * sampling_v, dim=-3)  # b * c * zs * h * w
        sampling_v = sampling_v.view(b, c, zs, h, w)

        output = self.o(sampling_v)
        output = self.bn(output) + residual

        return output
