

import torch
import torch.fft as fft

def pink_noise_loss(x: torch.Tensor, alpha: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Pink-noise prior in Fourier domain.

    Encourages image spectrum to follow ~ 1/f^alpha.

    Args:
        x: Tensor of shape (B, 1, H, W)
        alpha: spectral exponent. alpha=1 → pink noise.
        eps: small constant to avoid division by zero.
    Returns:
        Scalar tensor loss.
    """
    # x → (B,1,H,W)
    B, C, H, W = x.shape

    # Compute FFT
    Xf = fft.fft2(x, norm="ortho")
    Xf_shift = fft.fftshift(Xf)

    # Build frequency grid
    fy = torch.linspace(-0.5, 0.5, H, device=x.device)
    fx = torch.linspace(-0.5, 0.5, W, device=x.device)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    f = torch.sqrt(fx**2 + fy**2) + eps

    # Weight = f^alpha
    w = f ** alpha

    # Compute weighted spectral energy
    energy = (torch.abs(Xf_shift) ** 2) * (w[None, None, :, :])

    # Average over batch
    loss = energy.mean()

    return loss