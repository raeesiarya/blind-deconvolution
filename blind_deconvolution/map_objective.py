

"""map_objective.py

MAP (Maximum A Posteriori) objective for blind deconvolution.

We model the forward process as

    y = k * x + noise

where
    x : sharp image (B, 1, H, W)
    k : PSF kernel (1, 1, Kh, Kw)
    y : observed blurred image (B, 1, H, W)

The MAP estimator solves

    (x*, k*) = argmin_{x,k}  ||y - k * x||^2
                              + lambda_x * Phi(x)
                              + lambda_k * Psi(k)

This file defines:

- data_fidelity_loss(x, k, y_meas)
- kernel_prior_loss(k, ...)
- kernel_autocorrelation_loss(k)
- image_prior_loss(x, prior_fn=None)
- map_objective(x, k, y_meas, ...)

The image prior is implemented as a *hook* via `prior_fn(x)`, so that you can
later plug in a diffusion-model-based score prior or any other learned prior.
For now, the default prior is zero (i.e., no image prior).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from blind_deconvolution.forward_model import forward_model
from blind_deconvolution.priors.pink_noise import pink_noise_loss
from blind_deconvolution.priors.diffusion import diffusion_prior_loss


##############################
# Data fidelity (likelihood)
##############################


def data_fidelity_loss(
    x: torch.Tensor,
    k: torch.Tensor,
    y_meas: torch.Tensor,
) -> torch.Tensor:
    """Compute the data term || y_meas - k * x ||^2.

    Args:
        x: Sharp image tensor of shape (B, 1, H, W).
        k: PSF kernel tensor of shape (1, 1, Kh, Kw).
        y_meas: Measured blurred image of shape (B, 1, H, W).

    Returns:
        Scalar tensor (0D) with the mean squared error.
    """
    y_pred = forward_model(x, k, noise_sigma=0.0)
    loss = torch.mean((y_pred - y_meas) ** 2)
    return loss


##############################
# Kernel prior (regularizer)
##############################


def kernel_prior_loss(
    k: torch.Tensor,
    l2_weight: float = 0.0,
    center_weight: float = 0.0,
    auto_weight: float = 0.0,
    return_components: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Simple kernel prior / regularizer.

    This is a placeholder for Psi(k) in the MAP objective.

    Current terms:
      - L2 norm of k (encourages small energy kernels)
      - Optional center-of-mass penalty (encourages mass near the center)
      - Optional autocorrelation penalty (encourages k * k to approach delta)

    Args:
        k: PSF kernel tensor of shape (1, 1, Kh, Kw).
        l2_weight: Weight for L2 norm of k.
        center_weight: Weight for center-of-mass penalty.
        auto_weight: Weight for autocorrelation penalty.

    Returns:
        Scalar tensor (0D) representing the kernel prior loss, or a
        (loss, components) tuple when return_components=True.
    """
    loss = k.new_tensor(0.0)
    components = {}

    if l2_weight > 0.0:
        l2_term = l2_weight * torch.mean(k ** 2)
        loss = loss + l2_term
        components["loss_kernel_l2"] = l2_term

    if center_weight > 0.0:
        # Encourage mass near the center of the kernel.
        _, _, Kh, Kw = k.shape
        ys = torch.linspace(-1.0, 1.0, steps=Kh, device=k.device)
        xs = torch.linspace(-1.0, 1.0, steps=Kw, device=k.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r2 = xx**2 + yy**2

        # Weighted average of radius^2 with kernel magnitudes as weights
        weights = torch.abs(k[0, 0])
        weights = weights / (weights.sum() + 1e-8)
        radius2_mean = torch.sum(weights * r2)

        center_term = center_weight * radius2_mean
        loss = loss + center_term
        components["loss_kernel_center"] = center_term

    if auto_weight > 0.0:
        auto_term = auto_weight * kernel_autocorrelation_loss(k)
        loss = loss + auto_term
        components["loss_kernel_auto"] = auto_term

    if return_components:
        return loss, components
    return loss


def kernel_autocorrelation_loss(k: torch.Tensor) -> torch.Tensor:
    """Penalize energy away from the autocorrelation center (k * k ≈ delta).

    We compute the 2D autocorrelation via the Wiener–Khinchin theorem:
    autocorr = ifft2(|FFT(k)|^2). A perfect impulse autocorrelation would have
    all energy at the center; we penalize squared magnitude off the center.

    Args:
        k: PSF kernel tensor of shape (B, 1, Kh, Kw) or (1, 1, Kh, Kw).

    Returns:
        Scalar tensor penalizing off-center autocorrelation energy.
    """
    if k.dim() != 4 or k.shape[1] != 1:
        raise ValueError(f"Expected k shape (B,1,Kh,Kw), got {tuple(k.shape)}")

    # Flatten channel dimension; operate per batch.
    k_hw = k[:, 0]
    fft_k = torch.fft.fftn(k_hw, dim=(-2, -1))
    power_spectrum = torch.abs(fft_k) ** 2
    autocorr = torch.fft.ifftn(power_spectrum, dim=(-2, -1)).real
    # Shift zero-lag to the center for easier masking.
    autocorr = torch.fft.fftshift(autocorr, dim=(-2, -1))

    Kh, Kw = k_hw.shape[-2:]
    cy, cx = Kh // 2, Kw // 2

    # Penalize off-center energy; mean over spatial dims and batch.
    off_center = autocorr.clone()
    off_center[..., cy, cx] = 0.0
    off_energy = (off_center ** 2).mean(dim=(-2, -1))
    return off_energy.mean()


##############################
# Image prior (hook for diffusion, etc.)
##############################


def image_prior_loss(
    x: torch.Tensor,
    prior_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    weight: float = 0.0,
) -> torch.Tensor:
    """Image prior Phi(x) with an optional user-provided function.

    Args:
        x: Image tensor of shape (B, 1, H, W).
        prior_fn: Callable that takes x and returns a scalar tensor.
                  For example, this could be a diffusion-based score
                  objective or TV norm. If None, prior is 0.
        weight: Scalar multiplier for the prior.

    Returns:
        Scalar tensor (0D) representing the image prior.
    """
    if prior_fn is None or weight == 0.0:
        return x.new_tensor(0.0)

    raw_prior = prior_fn(x)
    if raw_prior.dim() != 0:
        # Ensure it's a scalar
        raw_prior = raw_prior.mean()
    return weight * raw_prior


##############################
# Full MAP objective
##############################


def map_objective(
    x: torch.Tensor,
    k: torch.Tensor,
    y_meas: torch.Tensor,
    lambda_x: float = 0.0,
    lambda_k_l2: float = 0.0,
    lambda_k_center: float = 0.0,
    lambda_k_auto: float = 0.0,
    image_prior_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    lambda_pink: float = 0.0,
    lambda_diffusion: float = 0.0,
    return_components: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full MAP objective for blind deconvolution.

    L(x, k) = ||y_meas - k * x||^2
              + lambda_x * Phi(x)
              + lambda_k * Psi(k)

    where:
      - Phi(x) is provided via `image_prior_fn`
      - Psi(k) is implemented in `kernel_prior_loss`

    Args:
        x: Sharp image tensor of shape (B, 1, H, W).
        k: PSF kernel tensor of shape (1, 1, Kh, Kw).
        y_meas: Measured blurred image of shape (B, 1, H, W).
        lambda_x: Weight for the image prior.
        lambda_k_l2: L2 weight for the kernel prior.
        lambda_k_center: Center-of-mass weight for the kernel prior.
        lambda_k_auto: Autocorrelation penalty weight (k * k -> delta).
        image_prior_fn: Optional callable implementing Phi(x).
        return_components: If True, also return a dict of individual loss terms.

    Returns:
        Scalar tensor (0D) representing the total MAP loss, or a tuple of
        (loss, components) if return_components is True.
    """
    loss_data = data_fidelity_loss(x, k, y_meas)
    if return_components:
        loss_k, k_components = kernel_prior_loss(
            k,
            l2_weight=lambda_k_l2,
            center_weight=lambda_k_center,
            auto_weight=lambda_k_auto,
            return_components=True,
        )
    else:
        loss_k = kernel_prior_loss(
            k,
            l2_weight=lambda_k_l2,
            center_weight=lambda_k_center,
            auto_weight=lambda_k_auto,
            return_components=False,
        )
    loss_x = image_prior_loss(x, prior_fn=image_prior_fn, weight=lambda_x)

    loss_pink = x.new_tensor(0.0)
    if lambda_pink > 0.0:
        loss_pink = lambda_pink * pink_noise_loss(x)
    
    loss_diffusion = x.new_tensor(0.0)
    if lambda_diffusion > 0.0:
        loss_diffusion = lambda_diffusion * diffusion_prior_loss(x)
    total = loss_data + loss_x + loss_k + loss_pink + loss_diffusion

    if return_components:
        components = {
            "loss_data": loss_data,
            "loss_kernel": loss_k,
            "loss_image": loss_x,
            "loss_pink": loss_pink,
            "loss_diffusion": loss_diffusion,
        }
        components.update(k_components)
        return total, components

    return total


if __name__ == "__main__":
    # Tiny sanity check: random tensors
    B, H, W = 1, 64, 64
    Kh = Kw = 15

    x = torch.rand(B, 1, H, W)
    k = torch.rand(1, 1, Kh, Kw)
    k = k / (k.sum() + 1e-8)
    y_meas = torch.rand(B, 1, H, W)

    loss = map_objective(
        x,
        k,
        y_meas,
        lambda_x=0.0,
        lambda_k_l2=1e-3,
        lambda_k_center=1e-3,
        image_prior_fn=None,
    )
    print("MAP loss:", float(loss.detach().cpu().item()))
