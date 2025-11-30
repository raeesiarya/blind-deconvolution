import torch
import numpy as np
from dataclasses import asdict
from collections import defaultdict
import wandb
from utils.image_io import load_image
from blind_deconvolution.psf_generator import get_psf
from blind_deconvolution.forward_model import forward_model
from utils.wandb_logging import tensor_to_wandb_image
from utils.convertors import numpy_kernel_to_tensor
from utils.metrics import psnr, ssim, kernel_error
from blind_deconvolution.blind_deconvolution import BlindDeconvolver, BlindDeconvConfig
from utils.cuda_checker import choose_device
from utils.image_paths import list_image_paths


def testebench(
    num_iters: int,
    lr_x: float,
    lr_k: float,
    lambda_x: float,
    lambda_k_l2: float,
    lambda_k_center: float,
    lambda_pink: float,
    lambda_diffusion: float,
    kernel_size: int,
    sigma_gaussian: float,
    motion_length: int | None = None,
    angle_motion: float = 0.0,
    fried_parameter_turbulence: float | None = None,
    distortion_strength_turbulence: float = 0.8,
    seed_turbulence: int | None = None,
    bandwidth_rml: float | None = None,
    seed_rml: int | None = None,
    psf_types: list[str] | None = None,
    run_name: str | None = None,
) -> None:
    """Run blind deconvolution experiments across a dataset of images and multiple PSF types,
    logging intermediate results and evaluation metrics to Weights & Biases.

    Args:
        num_iters (int): Number of optimization iterations to perform for each image/PSF.
        lr_x (float): Learning rate for optimizing the latent image.
        lr_k (float): Learning rate for optimizing the kernel/PSF.
        lambda_x (float): Weight for the image prior regularization term.
        lambda_k_l2 (float): L2 regularization weight applied to the kernel to discourage large values.
        lambda_k_center (float): Weight encouraging kernel mass to be centered.
        lambda_pink (float): Weight for a pink-noise (1/f) prior on the kernel, if used.
        lambda_diffusion (float): Weight for diffusion-based regularization on image or kernel updates.
        kernel_size (int): Size (height/width) of the square PSF kernel to generate and estimate.
        sigma_gaussian (float): Standard deviation to use when generating a Gaussian PSF.
        motion_length (int | None): Length of the motion blur in pixels. Defaults to kernel_size // 2.
        angle_motion (float): Angle in degrees to use when generating a motion blur PSF.
        fried_parameter_turbulence (float | None): Effective Fried parameter controlling the
            turbulence PSF width (smaller => stronger blur). Defaults to kernel_size / 10.
        distortion_strength_turbulence (float): Scales random distortions that make the turbulence
            PSF irregular and harder than the motion blur baseline.
        seed_turbulence (int | None): Optional RNG seed to make turbulence PSFs repeatable.
        bandwidth_rml (float | None): Fractional Fourier cutoff for randomized optics PSFs.
        seed_rml (int | None): Optional RNG seed for randomized optics PSFs.
        psf_types (list[str] | None): Which PSF scenarios to run. Supported: ["none", "gaussian",
            "motion", "turbulence", "rml"]. If None, runs the blurred trio (gaussian/motion/turbulence).
        run_name (str | None): Optional W&B run name for deterministic labeling.
    """
    config = BlindDeconvConfig(
        num_iters=num_iters,
        lr_x=lr_x,
        lr_k=lr_k,
        lambda_x=lambda_x,
        lambda_k_l2=lambda_k_l2,
        lambda_k_center=lambda_k_center,
        lambda_pink=lambda_pink,
        lambda_diffusion=lambda_diffusion,
        kernel_size=kernel_size,
        device=choose_device(),
    )

    default_types = ["gaussian", "motion", "turbulence"]
    selected_types = psf_types or default_types
    unknown = [
        t for t in selected_types if t not in ["none", "gaussian", "motion", "turbulence", "rml"]
    ]
    if unknown:
        raise ValueError(
            f"Unsupported psf_types: {unknown}. Supported: ['none', 'gaussian', 'motion', 'turbulence', 'rml']"
        )

    # Only fill defaults for PSFs that are actually requested.
    motion_length_val = (
        motion_length
        if motion_length is not None
        else (max(1, config.kernel_size // 2) if "motion" in selected_types else None)
    )
    sigma_val = (
        sigma_gaussian
        if sigma_gaussian is not None
        else (2.0 if "gaussian" in selected_types else None)
    )
    turbulence_fried = (
        fried_parameter_turbulence
        if fried_parameter_turbulence is not None
        else (max(1.0, config.kernel_size / 10) if "turbulence" in selected_types else None)
    )
    distortion_strength_val = (
        distortion_strength_turbulence
        if distortion_strength_turbulence is not None
        else (0.8 if "turbulence" in selected_types else None)
    )
    rml_bandwidth_val = (
        bandwidth_rml if bandwidth_rml is not None else (0.35 if "rml" in selected_types else None)
    )

    base_specs: dict[str, dict] = {
        "none": {"params": {}},
        "gaussian": {"params": {"sigma": sigma_val}},
        "motion": {"params": {"length": motion_length_val, "angle": angle_motion}},
        "turbulence": {
            "params": {
                "fried_parameter": turbulence_fried,
                "distortion_strength": distortion_strength_val,
                "seed": seed_turbulence,
            }
        },
        "rml": {
            "params": {
                "bandwidth": rml_bandwidth_val,
                "seed": seed_rml,
            }
        },
    }

    psf_specs = [(name, base_specs[name]["params"]) for name in selected_types]

    wandb_config = asdict(config)
    wandb_config.pop("image_prior_fn", None)  # not serializable
    wandb_config["psf_types"] = [name for name, _ in psf_specs]
    wandb_config["psf_params"] = {
        "gaussian_sigma": sigma_val,
        "motion_length": motion_length_val,
        "motion_angle": angle_motion if "motion" in selected_types else None,
        "turbulence_fried_parameter": turbulence_fried,
        "turbulence_distortion_strength": distortion_strength_val,
        "turbulence_seed": seed_turbulence if "turbulence" in selected_types else None,
        "rml_bandwidth": rml_bandwidth_val,
        "rml_seed": seed_rml if "rml" in selected_types else None,
    }

    run = wandb.init(
        project="deconvolution",
        job_type="blind_deconvolution",
        name=run_name,
        config=wandb_config,
        notes="Blind deconvolution baseline with iterative optimization and PSNR/SSIM logging.",
    )

    psnr_scores = defaultdict(list)
    ssim_scores = defaultdict(list)
    num_psfs = len(psf_specs)
    kernel_errors = defaultdict(list)
    device = choose_device()

    try:
        for img_idx, img_path in enumerate(list_image_paths()):
            print(f"\n=== Processing {img_path} ===")
            img_label = img_path.stem

            # Load clean image x_true
            x_true = load_image(
                img_path, mode="torch", grayscale=True, normalize=True
            ).to(device)

            for psf_idx, (psf_name, psf_kwargs) in enumerate(psf_specs):
                psf_label = f"{img_label}/{psf_name}"
                combo_idx = img_idx * num_psfs + psf_idx
                step_offset = combo_idx * (config.num_iters + 1)

                # Generate ground-truth PSF (or identity if psf_name == "none")
                if psf_name == "none":
                    k_np = np.zeros(
                        (config.kernel_size, config.kernel_size), dtype=np.float64
                    )
                    k_np[config.kernel_size // 2, config.kernel_size // 2] = 1.0
                else:
                    k_np = get_psf(psf_name, size=config.kernel_size, **psf_kwargs)

                k_true = numpy_kernel_to_tensor(k_np).to(device)

                # Create blurred measurement
                with torch.no_grad():
                    y_meas = forward_model(x_true, k_true, noise_sigma=0.01)

                solver = BlindDeconvolver(config).to(device)

                def log_fn(metrics: dict, step: int) -> None:
                    namespaced = {f"{psf_label}/{k}": v for k, v in metrics.items()}
                    wandb.log(
                        {
                            "image_name": img_path.name,
                            "psf_type": psf_name,
                            **namespaced,
                        },
                        step=step_offset + step,
                    )

                # Log inputs for this image/PSF
                wandb.log(
                    {
                        "image_name": img_path.name,
                        "psf_type": psf_name,
                        f"{psf_label}/ground_truth": tensor_to_wandb_image(
                            x_true, f"gt_{img_path.name}"
                        ),
                        f"{psf_label}/measurement": tensor_to_wandb_image(
                            y_meas, f"blurred_{img_path.name}"
                        ),
                        f"{psf_label}/true_kernel": tensor_to_wandb_image(
                            k_true, f"k_true_{img_path.name}"
                        ),
                    },
                    step=step_offset,
                )

                x_hat, k_hat, losses = solver.run(
                    y_meas, verbose=True, log_fn=log_fn, log_every=10
                )

                # Compute evaluation metrics
                p = psnr(x_hat, x_true)
                s = ssim(x_hat, x_true)
                k_err = kernel_error(k_hat, k_true)
                psnr_scores[psf_name].append(p)
                ssim_scores[psf_name].append(s)
                kernel_errors[psf_name].append(k_err)

                wandb.log(
                    {
                        "image_name": img_path.name,
                        "psf_type": psf_name,
                        f"{psf_label}/psnr": p,
                        f"{psf_label}/ssim": s,
                        f"{psf_label}/kernel_error": k_err,
                        f"{psf_label}/final_loss": losses[-1],
                        f"{psf_label}/reconstruction": tensor_to_wandb_image(
                            x_hat, f"recon_{img_path.name}"
                        ),
                        f"{psf_label}/estimated_kernel": tensor_to_wandb_image(
                            k_hat, f"k_hat_{img_path.name}"
                        ),
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

                print(
                    f"{psf_name} PSF -> PSNR: {p:.2f} dB, SSIM: {s:.4f}, Kernel Error: {k_err:.4f}"
                )
                print(f"Finished. Final loss: {losses[-1]:.6f}")
                print(
                    f"x_hat shape: {tuple(x_hat.shape)}, k_hat shape: {tuple(k_hat.shape)}"
                )
    finally:
        if wandb.run is not None:
            all_psnr = [score for scores in psnr_scores.values() for score in scores]
            all_ssim = [score for scores in ssim_scores.values() for score in scores]
            all_kernel_errors = [
                score for scores in kernel_errors.values() for score in scores
            ]
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
                wandb.run.summary["mean_kernel_error"] = sum(all_kernel_errors) / len(
                    all_kernel_errors
                )
                wandb.run.summary["mean_kernel_error_by_psf"] = {
                    name: sum(scores) / len(scores)
                    for name, scores in kernel_errors.items()
                    if scores
                }
            wandb.finish()
