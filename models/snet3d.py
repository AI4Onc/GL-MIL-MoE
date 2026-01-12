from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm3d(norm: str, num_features: int) -> nn.Module:
    norm = (norm or "instance").lower()
    if norm in ("in", "instance", "instancenorm", "instancenorm3d"):
        return nn.InstanceNorm3d(num_features, affine=True)
    if norm in ("bn", "batch", "batchnorm", "batchnorm3d"):
        return nn.BatchNorm3d(num_features)
    if norm in ("gn", "group", "groupnorm"):
        # 8 groups is a safe default for typical channel widths
        return nn.GroupNorm(num_groups=8, num_channels=num_features)
    raise ValueError(f"Unknown norm type: {norm}")

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "instance",
                 activation: Optional[nn.Module] = None, p_dropout: float = 0.0):
        super().__init__()
        Act = activation if activation is not None else nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = _get_norm3d(norm, out_ch)
        self.act1 = Act
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = _get_norm3d(norm, out_ch)
        self.act2 = Act
        self.drop = nn.Dropout3d(p_dropout) if p_dropout and p_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x

class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "instance",
                 p_dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block = ConvBlock3D(in_ch, out_ch, norm=norm, p_dropout=p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))

class SpatialAttention3D(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        h = max(8, min(hidden, in_ch))
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, h, kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(h, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns sigmoid scores in [0,1]
        return torch.sigmoid(self.net(x))

class SNet3DEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 widths: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
                 feat_dim: int = 256,
                 norm: str = "instance",
                 l2norm: bool = True,
                 attn_strength: float = 1.0,
                 mask_dropout: float = 0.0,
                 roi_only: bool = True, 
                 mask_dilate: int = 0):
        super().__init__()
        if len(widths) != 5:
            raise ValueError("widths must have five elements")
        self.widths = widths
        self.feat_dim = feat_dim
        self.norm = norm
        self.l2norm = l2norm
        self.attn_strength = float(attn_strength)
        self.mask_dropout = float(mask_dropout)
        self.roi_only = roi_only
        self.mask_dilate = int(mask_dilate)

        # Encoder
        w = widths
        self.enc0 = ConvBlock3D(in_channels, w[0], norm=norm)
        self.down1 = DownBlock3D(w[0], w[1], norm=norm)
        self.down2 = DownBlock3D(w[1], w[2], norm=norm)
        self.down3 = DownBlock3D(w[2], w[3], norm=norm)
        self.down4 = DownBlock3D(w[3], w[4], norm=norm)

        # Attention at each scale
        self.att0 = SpatialAttention3D(w[0])
        self.att1 = SpatialAttention3D(w[1])
        self.att2 = SpatialAttention3D(w[2])
        self.att3 = SpatialAttention3D(w[3])
        self.att4 = SpatialAttention3D(w[4])

        # Learnable blend logits: beta = sigmoid(logit) in [0,1]
        self._blend_logit = nn.Parameter(torch.zeros(5))

        # Linear projections to common dim
        self.proj0 = nn.Linear(w[0], feat_dim)
        self.proj1 = nn.Linear(w[1], feat_dim)
        self.proj2 = nn.Linear(w[2], feat_dim)
        self.proj3 = nn.Linear(w[3], feat_dim)
        self.proj4 = nn.Linear(w[4], feat_dim)

    # ---------- helpers ----------
    @staticmethod
    def _interp_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Downsample/upsample mask to feature map spatial size of x."""
        if mask is None:
            return None
        if mask.dtype != x.dtype:
            mask = mask.to(dtype=x.dtype)
        target = x.shape[2:]
        if mask.shape[2:] == target:
            return mask
        #return F.interpolate(mask, size=target, mode="trilinear", align_corners=False)
        return F.interpolate(mask, size=target, mode="nearest")

    def _maybe_dilate(self, m: torch.Tensor) -> torch.Tensor:
        # 可选：给ROI扩一圈，避免ROI太小导致特征太“尖”
        if self.mask_dilate <= 0:
            return m
        k = 2 * self.mask_dilate + 1
        return F.max_pool3d(m, kernel_size=k, stride=1, padding=self.mask_dilate)

    def _weighted_gap(self, feat: torch.Tensor, att_map: torch.Tensor,
                      mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, C, D, H, W = feat.shape

        if mask is not None:
            mask = mask.clamp(0, 1)
            mask = (mask > 0.5).to(feat.dtype)         # 二值化更“严格”
            mask = self._maybe_dilate(mask)

        # --- ROI-only: 只在ROI内做平均 ---
        if self.roi_only:
            if mask is None:
                # 没mask就退化成普通GAP
                return feat.mean(dim=(2,3,4))
            den = mask.sum(dim=(2,3,4)).clamp_min(1e-6)  # [B,1]
            num = (feat * mask).sum(dim=(2,3,4))         # [B,C]
            return num / den

        # --- 下面保留你原来的 ROI-biased 逻辑（非严格ROI） ---
        beta = torch.sigmoid(self._blend_logit.mean())
        blend = att_map if mask is None else (beta * mask + (1.0 - beta) * att_map)
        weight = 1.0 + self.attn_strength * blend
        num = (feat * weight).sum(dim=(2,3,4))
        den = weight.sum(dim=(2,3,4)).clamp_min(1e-6)
        return num / den
    # ---------- forward ----------
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_spatial: bool = False,
                return_per_scale: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

        # Encode
        f0 = self.enc0(x)            # [B,w0,D,   H,   W]
        f1 = self.down1(f0)          # [B,w1,D/2, H/2, W/2]
        f2 = self.down2(f1)          # [B,w2,D/4, H/4, W/4]
        f3 = self.down3(f2)          # [B,w3,D/8, H/8, W/8]
        f4 = self.down4(f3)          # [B,w4,D/16,H/16,W/16]

        # Attention maps
        a0 = self.att0(f0)
        a1 = self.att1(f1)
        a2 = self.att2(f2)
        a3 = self.att3(f3)
        a4 = self.att4(f4)

        # Align mask to each scale
        m0 = self._interp_mask(mask, f0) if mask is not None else None
        m1 = self._interp_mask(mask, f1) if mask is not None else None
        m2 = self._interp_mask(mask, f2) if mask is not None else None
        m3 = self._interp_mask(mask, f3) if mask is not None else None
        m4 = self._interp_mask(mask, f4) if mask is not None else None

        # Weighted GAP and projections
        v0 = self._weighted_gap(f0, a0, m0)
        v1 = self._weighted_gap(f1, a1, m1)
        v2 = self._weighted_gap(f2, a2, m2)
        v3 = self._weighted_gap(f3, a3, m3)
        v4 = self._weighted_gap(f4, a4, m4)

        p0 = self.proj0(v0)
        p1 = self.proj1(v1)
        p2 = self.proj2(v2)
        p3 = self.proj3(v3)
        p4 = self.proj4(v4)

        fused = torch.cat([p0, p1, p2, p3, p4], dim=1)  # fused shape: [B, 256 * 5] = [B, 1280]
        outs: List[torch.Tensor] = [fused]
        if return_spatial:
            outs.append(f4)
        if return_per_scale:
            outs.append({"f0": v0, "f1": v1, "f2": v2, "f3": v3, "f4": v4})
        return tuple(outs) if len(outs) > 1 else fused


__all__ = [
    "ConvBlock3D",
    "DownBlock3D",
    "SpatialAttention3D",
    "SNet3DEncoder",
]
