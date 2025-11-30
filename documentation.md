Blind Deconvolution – System Notes
- Purpose: single-image blind deconvolution playground that now runs an experiment sweep via `testing/testbench.py` + `testing/testbench_configs.py`. Each config is run across PSF types (gaussian/motion/turbulence/rml) and images, with metrics logged to Weights & Biases.
- Scope: grayscale images only; PSF is single-channel 2D. Assumes batch size 1.
- Entrypoint: `main.py` (loads `.env`, logs into W&B, iterates over `TESTBENCH_CONFIGS`).

Workflow (runtime)
- Discover images under `images/` (recursive; png/jpg/jpeg/tif/tiff/bmp).
- For each config in `TESTBENCH_CONFIGS` and each PSF spec (subset of gaussian `sigma`, motion `length/angle`, turbulence `fried`, `distortion_strength`, optional seeds, RML `bandwidth`, or `none` for identity blur):
  - Load `x_true`, grayscale/normalized torch `(1,1,H,W)`.
  - Generate `k_true` via `get_psf` (or identity when `psf_type="none"`); synthesize measurement `y_meas = k_true * x_true + N(0, 0.01^2)`.
  - Run `BlindDeconvolver` with that config, log every 10 steps; enforce non-negativity + sum-to-one on `k`, clamp `x` to [0,1].
  - Log to W&B: gt, measurement, kernels, reconstructions, loss curves, PSNR, SSIM, kernel error; summary aggregates mean PSNR/SSIM/kernel error overall and by PSF.

Mathematical Model (unchanged core)
- Forward model: `y = k * x + n`, same-padding conv2d (`blind_deconvolution/forward_model.py`), optional Gaussian noise.
- MAP objective (`blind_deconvolution/map_objective.py`):
  - Data: `|| y_meas - k * x ||^2`.
  - Kernel priors: `lambda_k_l2 * mean(k^2)` + center-of-mass penalty `lambda_k_center * E_k[r^2]`.
  - Image prior hook: `lambda_x * prior_fn(x)` (mean-reduced if non-scalar).
  - Pink-noise prior: `lambda_pink * pink_noise_loss(x)` (`priors/pink_noise.py`).
  - Diffusion prior: `lambda_diffusion * diffusion_prior_loss(x)` (DDPM via `priors/diffusion.py`, heavy download/GPU expected).
  - Total: data + kernel + image + pink + diffusion.
- Optimization: separate Adam groups for `x` and `k` (`lr_x`, `lr_k`, `num_iters`). Post-step projection for `k` and clamp for `x`.

Key Modules
- `main.py`: loads WANDB key from `.env` (`WANDB_API_KEY`), logs into W&B, iterates configs, calls `testing/testbench.testebench`.
- `testing/testbench.py`: runs each config across PSF types/images; handles measurement synthesis, logging, metric aggregation.
- `testing/testbench_configs.py`: list of experiment configs (iters, LRs, priors, kernel sizes, PSF params).
- `blind_deconvolution/`: solver (`BlindDeconvolver` + `BlindDeconvConfig`), forward model, MAP objective, PSF generators, priors.
- `utils/`: image I/O/paths, NumPy↔Torch converters, metrics, W&B helpers, device chooser.
- `image_creator/create_synthetic_images.py`: optional synthetic data generator for `images/synthetic/`.

Config Surface
- Testbench configs (`testing/testbench_configs.py`): `num_iters`, `lr_x`, `lr_k`, `lambda_x`, `lambda_k_l2`, `lambda_k_center`, `lambda_pink`, `lambda_diffusion`, `kernel_size`, `sigma_gaussian`, `motion_length`, `angle_motion`, `fried_parameter_turbulence`, `distortion_strength_turbulence`, `seed_turbulence`, `bandwidth_rml`, `seed_rml`, `psf_types` (subset of ["none", "gaussian", "motion", "turbulence", "rml"]), optional `name`. Noise std is fixed at 0.01 inside `testbench.py`.
- Solver config (`BlindDeconvConfig`): same fields plus optional `image_prior_fn` and `device` (from `utils.cuda_checker.choose_device()`).

How to Run (UV kept)
- Install deps: `uv venv && source .venv/bin/activate && uv sync`.
- Set W&B auth: add `WANDB_API_KEY=...` to `.env` or export; use `WANDB_MODE=offline` to avoid uploads.
- (Optional) Generate synthetic images: `python image_creator/create_synthetic_images.py`.
- Execute sweep: `python main.py` (iterates all configs and PSF types over all images).

Extending / Customizing
- Trim or add sweeps in `TESTBENCH_CONFIGS`; adjust PSF list or noise level in `testing/testbench.py`.
- Implement custom priors via `image_prior_fn` or new modules under `blind_deconvolution/priors/` and plug into `map_objective`.
- Add new PSF generators in `blind_deconvolution/psf_generator.py` and register in the testbench.
- Tweak logging payloads or frequency via the `log_fn` in `testing/testbench.py`; disable W&B with `WANDB_MODE=offline` or `wandb.init(..., mode="disabled")`.

Practical Notes / Limitations
- Single-channel, batch size 1 pipeline; extend forward model/solver for RGB or batching if needed.
- Large kernels vs. small images can cause padding artifacts; adjust `kernel_size` accordingly.
- Diffusion prior is optional and resource-heavy; leave `lambda_diffusion=0` if compute or downloads are constrained.
