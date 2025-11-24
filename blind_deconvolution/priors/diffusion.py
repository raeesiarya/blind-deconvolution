import torch
from diffusers import DDPMPipeline

# Global cache so we only load the model once
_DDPM_PIPELINE = None

def _get_ddpm_pipeline(device: torch.device) -> DDPMPipeline:
    """
    Lazily load a pretrained DDPM pipeline and move it to the given device.
    """
    global _DDPM_PIPELINE

    if _DDPM_PIPELINE is None:
        # You can swap this for another DDPM, e.g. "google/ddpm-cat-256"
        model_id = "google/ddpm-celebahq-256"
        pipe = DDPMPipeline.from_pretrained(model_id)
        pipe = pipe.to(device)
        pipe.unet.eval()
        _DDPM_PIPELINE = pipe

    # If device changes between calls, you may want to call .to(device) again.
    # For now we assume single device (CPU or a single GPU).
    return _DDPM_PIPELINE


def diffusion_score(
    x: torch.Tensor,
    t_index: int = 200,
) -> torch.Tensor:
    """
    Approximate ∇_x log p(x) using a pretrained DDPM UNet.

    Args:
        x: Tensor of shape (B,1,H,W) in [0,1] (your reconstruction)
        t_index: diffusion timestep index in [0, T-1].
                 Mid-range (~200) encourages 'natural image' statistics.

    Returns:
        score: Tensor of shape (B,1,H,W), approximate score.
    """
    assert x.ndim == 4 and x.shape[1] == 1, "Expected (B,1,H,W) grayscale tensor."

    B, C, H, W = x.shape
    device = x.device

    pipe = _get_ddpm_pipeline(device)
    unet = pipe.unet
    scheduler = pipe.scheduler

    # DDPMs in diffusers expect inputs in [-1, 1] and 3 channels (RGB).
    # We replicate grayscale to 3 channels and rescale.
    x_in = x.repeat(1, 3, 1, 1)          # (B,3,H,W)
    x_in = x_in * 2.0 - 1.0              # [0,1] -> [-1,1]

    # Ensure t_index is in a valid range
    num_train_timesteps = scheduler.config.num_train_timesteps
    t_index = max(0, min(t_index, num_train_timesteps - 1))

    # Create timestep tensor of shape (B,)
    t = torch.full((B,), t_index, device=device, dtype=torch.long)

    with torch.no_grad():
        # UNet predicts the noise ε_θ
        model_output = unet(x_in, t).sample   # (B,3,H,W)

    # Average the RGB channels back to grayscale
    noise_pred_gray = model_output.mean(dim=1, keepdim=True)  # (B,1,H,W)

    # In score-based theory, ∇ log p(x) ≈ -ε / σ_t.
    # Here we ignore exact σ_t and let lambda_diffusion absorb the scale.
    score = -noise_pred_gray

    return score

def diffusion_prior_loss(
    x: torch.Tensor,
    t_index: int = 200,
) -> torch.Tensor:
    """
    Scalar diffusion prior loss based on the DDPM score.

    We define:
        L(x) = 0.5 * || score(x) ||^2

    This is smooth and differentiable, and the overall strength is controlled
    by lambda_diffusion in the MAP objective.
    """
    score = diffusion_score(x, t_index=t_index)
    return 0.5 * (score ** 2).mean()