from blind_deconvolution.blind_deconvolution import BlindDeconvolver, BlindDeconvConfig
from utils.image_io import load_image
from utils.image_paths import list_image_paths
import torch
from blind_deconvolution.psf_generator import get_psf
from blind_deconvolution.forward_model import forward_model
from utils.wandb_logging import tensor_to_wandb_image
from utils.convertors import numpy_kernel_to_tensor
from utils.metrics import psnr, ssim
from utils.cuda_checker import choose_device
from dataclasses import asdict
from collections import defaultdict
import wandb

def main():
    device = choose_device()
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
        lambda_pink=0.0,
        lambda_diffusion=0.01,
        kernel_size=15,
        device=device,
    )

    psf_specs = [
        ("delta", {}),
        ("gaussian", {"sigma": 2.0}),
        ("motion", {"length": config.kernel_size // 2, "angle": 0.0}),
        ("disk", {"radius": config.kernel_size / 4}),
    ]

    wandb_config = asdict(config)
    wandb_config.pop("image_prior_fn", None)  # not serializable
    wandb_config["psf_types"] = [name for name, _ in psf_specs]

    run = wandb.init(
        project="deconvolution",
        job_type="blind_deconvolution",
        config=wandb_config,
        notes="Blind deconvolution baseline with iterative optimization and PSNR/SSIM logging.",
    )

    psnr_scores = defaultdict(list)
    ssim_scores = defaultdict(list)
    num_psfs = len(psf_specs)

    try:
        for img_idx, img_path in enumerate(image_paths):
            print(f"\n=== Processing {img_path} ===")
            img_label = img_path.stem

            # Load clean image x_true
            x_true = load_image(img_path, mode="torch", grayscale=True, normalize=True).to(device)

            for psf_idx, (psf_name, psf_kwargs) in enumerate(psf_specs):
                psf_label = f"{img_label}/{psf_name}"
                combo_idx = img_idx * num_psfs + psf_idx
                step_offset = combo_idx * (config.num_iters + 1)

                # Generate ground-truth PSF
                k_np = get_psf(psf_name, size=config.kernel_size, **psf_kwargs)
                k_true = numpy_kernel_to_tensor(k_np).to(device)

                # Create blurred measurement
                with torch.no_grad():
                    y_meas = forward_model(x_true, k_true, noise_sigma=0.01)

                solver = BlindDeconvolver(config).to(device)

                def log_fn(metrics: dict, step: int) -> None:
                    namespaced = {f"{psf_label}/{k}": v for k, v in metrics.items()}
                    wandb.log(
                        {"image_name": img_path.name, "psf_type": psf_name, **namespaced},
                        step=step_offset + step,
                    )

                # Log inputs for this image/PSF
                wandb.log(
                    {
                        "image_name": img_path.name,
                        "psf_type": psf_name,
                        f"{psf_label}/ground_truth": tensor_to_wandb_image(x_true, f"gt_{img_path.name}"),
                        f"{psf_label}/measurement": tensor_to_wandb_image(y_meas, f"blurred_{img_path.name}"),
                        f"{psf_label}/true_kernel": tensor_to_wandb_image(k_true, f"k_true_{img_path.name}"),
                    },
                    step=step_offset,
                )

                x_hat, k_hat, losses = solver.run(
                    y_meas, verbose=True, log_fn=log_fn, log_every=10
                )

                # Compute evaluation metrics
                p = psnr(x_hat, x_true)
                s = ssim(x_hat, x_true)
                psnr_scores[psf_name].append(p)
                ssim_scores[psf_name].append(s)

                wandb.log(
                    {
                        "image_name": img_path.name,
                        "psf_type": psf_name,
                        f"{psf_label}/psnr": p,
                        f"{psf_label}/ssim": s,
                        f"{psf_label}/final_loss": losses[-1],
                        f"{psf_label}/reconstruction": tensor_to_wandb_image(x_hat, f"recon_{img_path.name}"),
                        f"{psf_label}/estimated_kernel": tensor_to_wandb_image(k_hat, f"k_hat_{img_path.name}"),
                        f"{psf_label}/loss_curve": wandb.plot.line_series(
                            xs=list(range(len(losses))),
                            ys=[losses],
                            keys=["loss"],
                            title=f"Loss - {img_path.name} [{psf_name}]",
                            xname="iter",
                        ),
                    },
                    step=step_offset + config.num_iters,
                )

                print(f"{psf_name} PSF -> PSNR: {p:.2f} dB, SSIM: {s:.4f}")
                print(f"Finished. Final loss: {losses[-1]:.6f}")
                print(f"x_hat shape: {tuple(x_hat.shape)}, k_hat shape: {tuple(k_hat.shape)}")
    finally:
        if wandb.run is not None:
            all_psnr = [score for scores in psnr_scores.values() for score in scores]
            all_ssim = [score for scores in ssim_scores.values() for score in scores]
            if all_psnr:
                wandb.run.summary["mean_psnr"] = sum(all_psnr) / len(all_psnr)
                wandb.run.summary["mean_ssim"] = sum(all_ssim) / len(all_ssim)
                wandb.run.summary["mean_psnr_by_psf"] = {
                    name: sum(scores) / len(scores)
                    for name, scores in psnr_scores.items()
                    if scores
                }
                wandb.run.summary["mean_ssim_by_psf"] = {
                    name: sum(scores) / len(scores)
                    for name, scores in ssim_scores.items()
                    if scores
                }
            wandb.finish()


if __name__ == "__main__":
    main()
