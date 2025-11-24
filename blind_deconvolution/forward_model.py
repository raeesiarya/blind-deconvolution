"""
forward_model.py

Differentiable forward imaging model for blind deconvolution:

    y = k * x + noise

Implemented in PyTorch so that we can optimize x (or a latent z)
and the kernel parameters via gradient-based methods.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


##############################
# Core forward model
##############################


def forward_convolve(
    x: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Convolve image x with PSF k using 'same' padding.

    Args:
        x: Tensor of shape (B, 1, H, W)   – input image(s)
        k: Tensor of shape (1, 1, Kh, Kw) – PSF kernel
           (you can later generalize to (B,1,Kh,Kw) if needed)

    Returns:
        y: Tensor of shape (B, 1, H, W) – blurred image(s)
    """
    if x.dim() != 4:
        raise ValueError(f"x must be 4D (B,1,H,W), got shape {tuple(x.shape)}")
    if k.dim() != 4:
        raise ValueError(f"k must be 4D (1,1,Kh,Kw), got shape {tuple(k.shape)}")
    if x.shape[1] != 1 or k.shape[1] != 1:
        raise ValueError("Only single-channel images/kernels are supported for now.")

    _, _, Kh, Kw = k.shape
    # 'same' padding for odd-sized kernels
    pad_h = Kh // 2
    pad_w = Kw // 2

    y = F.conv2d(x, k, padding=(pad_h, pad_w))
    return y


def add_gaussian_noise(
    y: torch.Tensor,
    sigma: float = 0.0,
) -> torch.Tensor:
    """
    Add i.i.d. Gaussian noise with std sigma to y.

    Args:
        y: Tensor of shape (B, 1, H, W)
        sigma: noise standard deviation (in same units as y)

    Returns:
        y_noisy: Tensor of same shape as y
    """
    if sigma <= 0.0:
        return y
    noise = torch.randn_like(y) * float(sigma)
    return y + noise


def forward_model(
    x: torch.Tensor,
    k: torch.Tensor,
    noise_sigma: float = 0.0,
) -> torch.Tensor:
    """
    Full forward model: convolution + optional Gaussian noise.

    Args:
        x: (B, 1, H, W) input image(s)
        k: (1, 1, Kh, Kw) PSF kernel
        noise_sigma: standard deviation of additive Gaussian noise

    Returns:
        y: (B, 1, H, W) blurred (and possibly noisy) measurements
    """
    y = forward_convolve(x, k)
    y = add_gaussian_noise(y, noise_sigma)
    return y


if __name__ == "__main__":
    # Tiny sanity check (using a random image + delta kernel)
    H, W = 64, 64
    x = torch.rand(1, 1, H, W)
    k = torch.zeros(1, 1, 15, 15)
    k[0, 0, 7, 7] = 1.0  # delta PSF in the center

    y = forward_model(x, k, noise_sigma=0.01)
    print("x shape:", x.shape)
    print("k shape:", k.shape)
    print("y shape:", y.shape)