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
from blind_deconvolution.psf_generator import gaussian_psf, motion_psf
from utils.cuda_checker import choose_device


@dataclass
class BlindDeconvConfig:
    # Optimization hyperparameters
    num_iters: int = 500
    lr_x: float = 1e-2
    lr_k: float = 1e-2

    # MAP prior weights
    lambda_x: float = 0.0  # image prior (Phi(x))
    lambda_k_l2: float = 1e-3  # L2 prior on kernel
    lambda_k_center: float = 1e-3  # center-of-mass prior on kernel
    lambda_k_auto: float = 0.0  # autocorrelation prior (k * k -> delta)
    lambda_pink: float = 0.0  # pink-noise prior weight
    lambda_diffusion: float = 0.0  # diffusion prior weight

    # Kernel settings
    kernel_size: int = 15

    # Optional image prior function
    image_prior_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
          - Initialize k as a minimal motion kernel (length=1) to approximate
            an identity blur, then allow optimizer to deviate from that.

        Args:
            y_meas: Tensor of shape (B, 1, H, W). Currently B must be 1.
        """
        if y_meas.dim() != 4 or y_meas.shape[1] != 1:
            raise ValueError(
                f"Expected y_meas of shape (B,1,H,W), got {tuple(y_meas.shape)}"
            )

        if y_meas.shape[0] != 1:
            raise NotImplementedError("Current implementation supports B=1 only.")

        device = self.config.device
        y_meas = y_meas.to(device)

        B, C, H, W = y_meas.shape

        # Initialize x as the measurement (clipped to [0,1])
        x_init = y_meas.clone().detach()
        x_init = x_init.clamp(0.0, 1.0)

        # Initialize kernel as a length-1 motion blur (acts like an impulse)
        k_np = motion_psf(size=self.config.kernel_size, length=1, angle=0.0)
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
            k /= k.sum() + 1e-8
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

        # Create separate optimizers for staged training
        opt_x = optim.Adam([self.x_param], lr=self.config.lr_x)
        opt_k = optim.Adam([self.k_param], lr=self.config.lr_k)

        freeze_k_iters = 50  # number of iterations where we only optimize the kernel

        losses: List[float] = []

        iterator = trange(
            self.config.num_iters,
            disable=not verbose,
            desc="Blind deconv",
            leave=False,
        )

        for it in iterator:
            opt_x.zero_grad()
            opt_k.zero_grad()

            need_components = log_fn is not None
            result = map_objective(
                self.x_param,
                self.k_param,
                y_meas,
                lambda_x=self.config.lambda_x,
                lambda_k_l2=self.config.lambda_k_l2,
                lambda_k_center=self.config.lambda_k_center,
                lambda_k_auto=self.config.lambda_k_auto,
                lambda_pink=self.config.lambda_pink,
                lambda_diffusion=self.config.lambda_diffusion,
                image_prior_fn=self.config.image_prior_fn,
                return_components=need_components,
            )

            if need_components:
                loss, loss_components = result
            else:
                loss = result
                loss_components = None

            loss.backward()

            if it < freeze_k_iters:
                # Only update kernel in the first phase
                opt_k.step()
                opt_k.zero_grad()
            else:
                # Update both x and k
                opt_k.step()
                opt_k.zero_grad()
                opt_x.step()
                opt_x.zero_grad()

            # Project constraints
            self.project_kernel()
            self.project_image()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)

            if log_fn is not None and log_every > 0:
                if (it % log_every == 0) or (it == self.config.num_iters - 1):
                    metrics = {"loss": loss_value}
                    if loss_components is not None:
                        metrics.update(
                            {
                                name: float(val.detach().cpu().item())
                                for name, val in loss_components.items()
                            }
                        )
                    log_fn(metrics, it)

            if verbose:
                iterator.set_postfix({"loss": f"{loss_value:.6f}"})

        # Return detached copies
        x_hat = self.x_param.detach().clone()
        k_hat = self.k_param.detach().clone()
        return x_hat, k_hat, losses
