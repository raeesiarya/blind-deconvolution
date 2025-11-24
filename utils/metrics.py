# utils/metrics.py

import torch
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim_fn


def psnr(x_hat: torch.Tensor, x_true: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        x_hat: reconstructed image, shape (1,1,H,W)
        x_true: ground-truth image, shape (1,1,H,W)
        data_range: max value in image (1.0 for normalized)

    Returns:
        PSNR in dB (float)
    """
    x_hat_np = x_hat.detach().cpu().numpy()
    x_true_np = x_true.detach().cpu().numpy()

    mse = ((x_hat_np - x_true_np) ** 2).mean()
    if mse == 0:
        return float("inf")

    return 20 * log10(data_range) - 10 * log10(mse)


def ssim(x_hat: torch.Tensor, x_true: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        x_hat: reconstructed image, shape (1,1,H,W)
        x_true: ground-truth image, shape (1,1,H,W)
        data_range: max pixel value

    Returns:
        SSIM score (float)
    """
    x_hat_np = x_hat.detach().cpu().numpy()[0, 0]
    x_true_np = x_true.detach().cpu().numpy()[0, 0]

    score = ssim_fn(
        x_true_np,
        x_hat_np,
        data_range=data_range,
        win_size=11  # classical SSIM window
    )
    return float(score)