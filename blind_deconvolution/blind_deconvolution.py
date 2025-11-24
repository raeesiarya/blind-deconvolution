
"""
blind_deconvolution.py

Baseline blind deconvolution solver using the MAP objective defined in
`map_objective.py` and the differentiable forward model in `forward_model.py`.

We solve

    (x*, k*) = argmin_{x,k}  ||y_meas - k * x||^2
                              + lambda_x * Phi(x)
                              + lambda_k * Psi(k)

by directly optimizing over the image x and the PSF kernel k using gradient-based
optimization (Adam). This is a simple but flexible baseline that you can later
extend with a diffusion prior (plugged into Phi(x)) or more sophisticated
kernel models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from blind_deconvolution.forward_model import forward_model
from utils.convertors import numpy_image_to_tensor, numpy_kernel_to_tensor
from blind_deconvolution.map_objective import map_objective
from blind_deconvolution.psf_generator import gaussian_psf, delta_psf


@dataclass
class BlindDeconvConfig:
    # Optimization hyperparameters
    num_iters: int = 500
    lr_x: float = 1e-2
    lr_k: float = 1e-2

    # MAP prior weights
    lambda_x: float = 0.0          # image prior (Phi(x))
    lambda_k_l2: float = 1e-3      # L2 prior on kernel
    lambda_k_center: float = 1e-3  # center-of-mass prior on kernel
    lambda_pink: float = 0.0      # pink-noise prior weight
    lambda_diffusion: float = 0.0 # diffusion prior weight

    # Kernel settings
    kernel_size: int = 15

    # Optional image prior function
    image_prior_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    # Device
    device: str = "cpu"


class BlindDeconvolver(nn.Module):
    """
    Simple blind deconvolution solver that optimizes x and k directly.

    Usage pattern:

        config = BlindDeconvConfig(...)
        solver = BlindDeconvolver(config).to(config.device)

        # y_meas: (1, 1, H, W) tensor
        x_hat, k_hat, losses = solver.run(y_meas)

    """

    def __init__(self, config: BlindDeconvConfig):
        super().__init__()
        self.config = config

        # Placeholders; will be initialized per-observation in `initialize_from_measurement`
        self.x_param: Optional[nn.Parameter] = None
        self.k_param: Optional[nn.Parameter] = None

    def initialize_from_measurement(self, y_meas: torch.Tensor) -> None:
        """
        Initialize x and k given a measurement y_meas.

        Strategy:
          - Initialize x as the observed blurred image (clipped to [0,1]).
          - Initialize k as a delta kernel (no blur), then allow optimizer to
            deviate from that.

        Args:
            y_meas: Tensor of shape (B, 1, H, W). Currently B must be 1.
        """
        if y_meas.dim() != 4 or y_meas.shape[1] != 1:
            raise ValueError(f"Expected y_meas of shape (B,1,H,W), got {tuple(y_meas.shape)}")

        if y_meas.shape[0] != 1:
            raise NotImplementedError("Current implementation supports B=1 only.")

        device = self.config.device
        y_meas = y_meas.to(device)

        B, C, H, W = y_meas.shape

        # Initialize x as the measurement (clipped to [0,1])
        x_init = y_meas.clone().detach()
        x_init = x_init.clamp(0.0, 1.0)

        # Initialize kernel as delta
        k_np = delta_psf(size=self.config.kernel_size)
        k_init = numpy_kernel_to_tensor(k_np)  # (1,1,Kh,Kw)
        k_init = k_init.to(device)

        # Register as learnable parameters
        self.x_param = nn.Parameter(x_init)
        self.k_param = nn.Parameter(k_init)

        # Move module parameters to device
        self.to(device)

    def project_kernel(self) -> None:
        """
        Enforce basic kernel constraints after each optimizer step:
          - non-negativity
          - normalization to sum 1
        """
        if self.k_param is None:
            return

        with torch.no_grad():
            k = self.k_param.data
            k.clamp_(min=0.0)
            k /= (k.sum() + 1e-8)
            self.k_param.data = k

    def project_image(self) -> None:
        """
        Optionally project x into [0,1] after each step (simple box constraint).
        """
        if self.x_param is None:
            return

        with torch.no_grad():
            x = self.x_param.data
            x.clamp_(0.0, 1.0)
            self.x_param.data = x

    def run(
        self,
        y_meas: torch.Tensor,
        verbose: bool = True,
        log_fn: Optional[Callable[[dict, int], None]] = None,
        log_every: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Run blind deconvolution to estimate x and k from y_meas.

        Args:
            y_meas: Observed blurred image, shape (1, 1, H, W).
            verbose: If True, prints loss every 50 iterations.
            log_fn: Optional callback receiving (metrics_dict, step). Used for logging.
            log_every: Log every N iterations when log_fn is provided.

        Returns:
            x_hat: Estimated sharp image, shape (1, 1, H, W).
            k_hat: Estimated PSF kernel, shape (1, 1, Kh, Kw).
            losses: List of loss values over iterations.
        """
        device = self.config.device
        y_meas = y_meas.to(device)

        # Initialize variables
        self.initialize_from_measurement(y_meas)

        # Create separate optimizers for x and k (could also use a single optimizer)
        params = [
            {"params": [self.x_param], "lr": self.config.lr_x},
            {"params": [self.k_param], "lr": self.config.lr_k},
        ]
        optimizer = optim.Adam(params)

        losses: List[float] = []

        iterator = trange(
            self.config.num_iters,
            disable=not verbose,
            desc="Blind deconv",
            leave=False,
        )

        for it in iterator:
            optimizer.zero_grad()

            loss = map_objective(
                self.x_param,
                self.k_param,
                y_meas,
                lambda_x=self.config.lambda_x,
                lambda_k_l2=self.config.lambda_k_l2,
                lambda_k_center=self.config.lambda_k_center,
                lambda_pink=self.config.lambda_pink,
                lambda_diffusion=self.config.lambda_diffusion,
                image_prior_fn=self.config.image_prior_fn,
            )

            loss.backward()
            optimizer.step()

            # Project constraints
            self.project_kernel()
            self.project_image()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)

            if log_fn is not None and log_every > 0:
                if (it % log_every == 0) or (it == self.config.num_iters - 1):
                    log_fn({"loss": loss_value}, it)

            if verbose:
                iterator.set_postfix({"loss": f"{loss_value:.6f}"})

        # Return detached copies
        x_hat = self.x_param.detach().clone()
        k_hat = self.k_param.detach().clone()
        return x_hat, k_hat, losses


# ---------------------------------------------------------------------------
# Convenience demo when running this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick demo:
      1. Generate a synthetic sharp image and PSF (Gaussian).
      2. Blur the image with the forward model to get y_meas.
      3. Run blind deconvolution to recover x_hat and k_hat.
    """

    import numpy as np
    from skimage import io, color, img_as_float

    # Choose device
    device = "cpu"

    # Load a test image (use any of your synthetic images)
    img_path = "images/synthetic/checkerboard.png"  # adjust as needed
    try:
        img_np = io.imread(img_path)
    except FileNotFoundError:
        print(f"Could not find {img_path}. Please adjust the path in __main__.")
        exit(0)

    if img_np.ndim == 3:
        img_np = color.rgb2gray(img_np)
    img_np = img_as_float(img_np).astype(np.float32)

    # Convert to torch tensor
    x_true = numpy_image_to_tensor(img_np).to(device)

    # Generate ground-truth PSF (Gaussian)
    k_true_np = gaussian_psf(size=15, sigma=2.0)
    k_true = numpy_kernel_to_tensor(k_true_np).to(device)

    # Create measurement
    with torch.no_grad():
        y_meas = forward_model(x_true, k_true, noise_sigma=0.01)

    print("x_true shape:", x_true.shape)
    print("k_true shape:", k_true.shape)
    print("y_meas shape:", y_meas.shape)

    # Configure solver
    config = BlindDeconvConfig(
        num_iters=300,
        lr_x=1e-2,
        lr_k=1e-2,
        lambda_x=0.0,          # no image prior yet
        lambda_k_l2=1e-3,
        lambda_k_center=1e-3,
        kernel_size=15,
        image_prior_fn=None,   # hook for diffusion / TV later
        device=device,
    )

    solver = BlindDeconvolver(config).to(device)
    x_hat, k_hat, losses = solver.run(y_meas, verbose=True)

    print("Optimization finished.")
    print("Final loss:", losses[-1])

    # Optionally save or visualize results
    # (left as an exercise: you can use matplotlib to compare x_true, y_meas, x_hat, and kernels)
