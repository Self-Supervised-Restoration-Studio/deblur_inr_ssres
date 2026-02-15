"""Deblur-INR internal loss helpers.

These are lightweight loss variants tuned for the deblur optimization loop.
They auto-detect 2D (4D tensor) vs 3D (5D tensor) inputs and have minimal
constructor parameters. Not exported or registered — used only by
DeblurINROptimizer.

These are internal to the deblur optimization loop and not part of the public API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _DeblurSSIMLoss(nn.Module):
    """Structural Similarity loss for deblur pipeline.

    Lightweight SSIM using einsum-based Gaussian window construction.
    Auto-detects 2D/3D from input dimensionality.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, val_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.C1 = (0.01 * val_range) ** 2
        self.C2 = (0.03 * val_range) ** 2

        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        self.register_buffer("g", g)

    def _make_window(self, ndim: int, channels: int, device: torch.device) -> Tensor:
        g = self.g
        if ndim == 4:
            window = torch.einsum("i,j->ij", g, g).unsqueeze(0).unsqueeze(0)
        else:
            window = torch.einsum("i,j,k->ijk", g, g, g).unsqueeze(0).unsqueeze(0)
        return window.expand(channels, 1, *window.shape[2:]).to(device)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        C = pred.shape[1]
        pad = self.window_size // 2
        window = self._make_window(pred.ndim, C, pred.device)

        if pred.ndim == 5:
            conv_fn = F.conv3d
            padding = (pad, pad, pad)
        else:
            conv_fn = F.conv2d
            padding = (pad, pad)

        mu1 = conv_fn(pred, window, groups=C, padding=padding)
        mu2 = conv_fn(target, window, groups=C, padding=padding)

        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(pred**2, window, groups=C, padding=padding) - mu1_sq
        sigma2_sq = conv_fn(target**2, window, groups=C, padding=padding) - mu2_sq
        sigma12 = conv_fn(pred * target, window, groups=C, padding=padding) - mu1_mu2

        ssim = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return 1 - ssim.mean()


class _DeblurFFTLoss(nn.Module):
    """FFT loss with normalization for deblur pipeline.

    Uses rfftn (real FFT) with optional magnitude normalization.
    Uses rfftn over all spatial dims, unlike fft2 + frequency masking approaches.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        spatial_dims = (-3, -2, -1) if pred.ndim == 5 else (-2, -1)

        pred_fft = torch.fft.rfftn(pred.float(), dim=spatial_dims)
        target_fft = torch.fft.rfftn(target.float(), dim=spatial_dims)

        if self.normalize:
            pred_abs = torch.abs(pred_fft)
            target_abs = torch.abs(target_fft)

            pred_max = pred_abs.flatten(1).max(dim=1, keepdim=True)[0]
            target_max = target_abs.flatten(1).max(dim=1, keepdim=True)[0]

            shape = [-1] + [1] * (pred.ndim - 1)
            pred_fft = pred_fft / (pred_max.view(*shape) + 1e-8)
            target_fft = target_fft / (target_max.view(*shape) + 1e-8)

        return torch.abs(pred_fft - target_fft).mean()


class _DeblurTVLoss(nn.Module):
    """Total Variation loss for deblur pipeline. Auto-detects 2D/3D."""

    def forward(self, pred: Tensor, target: Tensor | None = None) -> Tensor:
        if pred.ndim == 5:
            diff_d = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
            diff_h = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
            diff_w = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
            return torch.abs(diff_d).mean() + torch.abs(diff_h).mean() + torch.abs(diff_w).mean()

        diff_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        diff_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return torch.abs(diff_h).mean() + torch.abs(diff_w).mean()
