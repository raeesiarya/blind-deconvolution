from blind_deconvolution.blind_deconvolution import BlindDeconvolver, BlindDeconvConfig
from utils.image_io import load_image
from utils.image_paths import list_image_paths
import torch
from blind_deconvolution.psf_generator import gaussian_psf
from blind_deconvolution.forward_model import forward_model
from utils.convertors import numpy_kernel_to_tensor
from utils.metrics import psnr, ssim
from dataclasses import asdict
import wandb


def tensor_to_wandb_image(tensor, caption: str) -> wandb.Image:
    """Convert a (1,1,H,W) or (1,1,K,K) tensor to a wandb.Image."""
    array = tensor.detach().cpu().squeeze().numpy()
    return wandb.Image(array, caption=caption)


def main():
    device = "cpu"
    image_paths = list_image_paths()

    if not image_paths:
        print("No images found in images directory.")
        return

    config = BlindDeconvConfig(
        num_iters=100,
        lr_x=1e-2,
        lr_k=1e-2,
        lambda_x=0.0,
        lambda_k_l2=1e-3,
        lambda_k_center=1e-3,
        lambda_pink=0.05,
        lambda_diffusion=0.0,
        kernel_size=15,
        device=device,
    )

    wandb_config = asdict(config)
    wandb_config.pop("image_prior_fn", None)  # not serializable

    run = wandb.init(
        project="deconvolution",
        job_type="blind_deconvolution",
        config=wandb_config,
        notes="Blind deconvolution baseline with iterative optimization and PSNR/SSIM logging.",
    )

    psnr_scores = []
    ssim_scores = []

    try:
        for img_idx, img_path in enumerate(image_paths):
            print(f"\n=== Processing {img_path} ===")
            img_label = img_path.stem

            # Load clean image x_true
            x_true = load_image(img_path, mode="torch", grayscale=True, normalize=True).to(device)

            # Generate ground-truth PSF (Gaussian)
            k_np = gaussian_psf(size=config.kernel_size, sigma=2.0)
            k_true = numpy_kernel_to_tensor(k_np).to(device)

            # Create blurred measurement
            with torch.no_grad():
                y_meas = forward_model(x_true, k_true, noise_sigma=0.01)

            solver = BlindDeconvolver(config).to(device)
            step_offset = img_idx * config.num_iters

            def log_fn(metrics: dict, step: int) -> None:
                namespaced = {f"{img_label}/{k}": v for k, v in metrics.items()}
                wandb.log(
                    {"image_name": img_path.name, **namespaced},
                    step=step_offset + step,
                )

            # Log inputs for this image
            wandb.log(
                {
                    "image_name": img_path.name,
                    f"{img_label}/ground_truth": tensor_to_wandb_image(x_true, f"gt_{img_path.name}"),
                    f"{img_label}/measurement": tensor_to_wandb_image(y_meas, f"blurred_{img_path.name}"),
                    f"{img_label}/true_kernel": tensor_to_wandb_image(k_true, f"k_true_{img_path.name}"),
                },
                step=step_offset,
            )

            x_hat, k_hat, losses = solver.run(
                y_meas, verbose=True, log_fn=log_fn, log_every=10
            )

            # Compute evaluation metrics
            p = psnr(x_hat, x_true)
            s = ssim(x_hat, x_true)
            psnr_scores.append(p)
            ssim_scores.append(s)

            wandb.log(
                {
                    "image_name": img_path.name,
                    f"{img_label}/psnr": p,
                    f"{img_label}/ssim": s,
                    f"{img_label}/final_loss": losses[-1],
                    f"{img_label}/reconstruction": tensor_to_wandb_image(x_hat, f"recon_{img_path.name}"),
                    f"{img_label}/estimated_kernel": tensor_to_wandb_image(k_hat, f"k_hat_{img_path.name}"),
                    f"{img_label}/loss_curve": wandb.plot.line_series(
                        xs=list(range(len(losses))),
                        ys=[losses],
                        keys=["loss"],
                        title=f"Loss - {img_path.name}",
                        xname="iter",
                    ),
                },
                step=step_offset + config.num_iters,
            )

            print(f"PSNR: {p:.2f} dB, SSIM: {s:.4f}")
            print(f"Finished. Final loss: {losses[-1]:.6f}")
            print(f"x_hat shape: {tuple(x_hat.shape)}, k_hat shape: {tuple(k_hat.shape)}")
    finally:
        if wandb.run is not None:
            if psnr_scores:
                wandb.run.summary["mean_psnr"] = sum(psnr_scores) / len(psnr_scores)
                wandb.run.summary["mean_ssim"] = sum(ssim_scores) / len(ssim_scores)
            wandb.finish()


if __name__ == "__main__":
    main()
