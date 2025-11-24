Blind Deconvolution – System Notes
- Purpose: single-image blind deconvolution playground that synthesizes blurred measurements, optimizes the sharp image and PSF jointly with a MAP objective, and logs results to Weights & Biases.
- Scope: grayscale images only; kernel treated as a single-channel 2D PSF.
- Entrypoint: `main.py`.

Pipeline at a Glance
- Load images from `images/` (recursive; png/jpg/jpeg/tif/tiff/bmp).
- Choose PSF(s) to simulate blur (delta/gaussian/motion/disk) and add Gaussian noise.
- Run blind deconvolution: optimize both the image `x` and kernel `k` with Adam.
- Enforce PSF non-negativity + sum-to-one; clamp image to [0, 1].
- Log inputs, reconstructions, kernels, losses, PSNR, SSIM to W&B; summarize mean metrics.

Mathematical Model
- Forward model: `y = k * x + n`, implemented via `conv2d` with “same” padding (`blind_deconvolution/forward_model.py`); optional additive i.i.d. Gaussian noise with std `noise_sigma`.
- MAP objective (`blind_deconvolution/map_objective.py`):
  - Data term: `|| y_meas - k * x ||^2`.
  - Kernel prior `Psi(k)`:
    - L2: `lambda_k_l2 * mean(k^2)`.
    - Center-of-mass: `lambda_k_center * E_k[r^2]` with radii on a [-1,1]x[-1,1] grid.
  - Image prior `Phi(x)` hook: `lambda_x * prior_fn(x)` (user-supplied; mean-reduced if non-scalar).
  - Pink-noise prior: `lambda_pink *  mean( |F(x)|^2 * f^alpha )`, alpha=1 default (`priors/pink_noise.py`).
  - Diffusion prior: `lambda_diffusion * 0.5 * ||score(x)||^2`, where `score` is approximated by a pretrained DDPM UNet (`priors/diffusion.py`).
  - Total loss: `L = data + kernel + image + pink + diffusion`.
- Optimization:
  - Separate Adam parameter groups for `x` and `k` with learning rates `lr_x`, `lr_k`; iterations `num_iters`.
  - After each step: project `k` to be non-negative and normalized; clamp `x` to [0,1].

Key Modules
- `main.py`: Orchestrates runs. Loads images, loops over PSF specs, builds measurements, runs solver, logs to W&B. Tracks per-PSF PSNR/SSIM averages in run summary.
- `blind_deconvolution/blind_deconvolution.py`: `BlindDeconvConfig` (hyperparameters + prior hook) and `BlindDeconvolver` (initialization, projections, run loop).
- `blind_deconvolution/forward_model.py`: Convolution + optional Gaussian noise.
- `blind_deconvolution/map_objective.py`: Composes data fidelity + priors into a scalar loss.
- `blind_deconvolution/psf_generator.py`: PSF factories: delta, Gaussian (`sigma`), motion (`length`, `angle`), disk (`radius`); all normalized, non-negative.
- Priors (`blind_deconvolution/priors/`):
  - `pink_noise.py`: Fourier-domain 1/f^alpha prior.
  - `diffusion.py`: DDPM-based score; uses `diffusers.DDPMPipeline` (heavy download, assumes network/GPU if available).
- Utilities (`utils/`):
  - `image_io.py`: Load images (grayscale option, normalization) to numpy or torch `(1,1,H,W)`.
  - `image_paths.py`: Discover images under `images/`.
  - `convertors.py`: NumPy↔Torch helpers for images/kernels.
  - `metrics.py`: PSNR, SSIM (skimage) for `(1,1,H,W)` tensors.
  - `wandb_logging.py`: Safe tensor → `wandb.Image` conversion with normalization.
  - `cuda_checker.py`: Simple CPU/GPU chooser.
- Data generation (`image_creator/create_synthetic_images.py`): Generates patterns (checkerboard, gradients, circle, bars, pink noise) into `images/synthetic/`.

Configuration Surface (BlindDeconvConfig in `main.py` unless overridden)
- `num_iters`: optimization steps (default 100 in `main.py`, 500 in class default).
- `lr_x`, `lr_k`: Adam learning rates for image and kernel.
- `kernel_size`: PSF spatial size (odd recommended for “same” padding symmetry).
- Priors: `lambda_x` (custom image prior hook), `lambda_k_l2`, `lambda_k_center`, `lambda_pink`, `lambda_diffusion`.
- `image_prior_fn`: optional callable `f(x)->scalar`; pass e.g. TV or score function.
- `device`: `"cpu"` or `"cuda"` from `utils.cuda_checker.choose_device()`.

Runtime Flow in `main.py`
- Load W&B API key from `.env` (`WANDB_API_KEY`) and login (respect `WANDB_MODE=offline` to disable uploads).
- Build PSF list: delta, Gaussian, motion, disk (size pulled from config).
- For each image × PSF:
  - Load `x_true` (grayscale torch, normalized).
  - Generate ground-truth PSF (`get_psf`), convert to torch.
  - Form measurement `y_meas = k_true * x_true + N(0, 0.01^2)`.
  - Instantiate `BlindDeconvolver(config)`, run optimization with logging hook every 10 iters.
  - Log inputs (gt, measurement, true kernel) at step offset; log reconstructions, estimated kernel, loss curve, PSNR/SSIM at the end.
- After loop: aggregate mean PSNR/SSIM overall and per-PSF into W&B summary.

Practical Notes / Limitations
- Single-image batches only (`B=1`, single-channel); extend forward model and solver to support RGB or batches if needed.
- Kernel projection ensures sums to 1; if the learned kernel collapses, consider stronger `lambda_k_center` or adjusting learning rates.
- Diffusion prior loads a pretrained DDPM via `diffusers`; expect large downloads and GPU memory needs. Keep `lambda_diffusion=0` if unavailable.
- Pink-noise prior assumes normalized images in [0,1]; scale accordingly if changing data range.
- W&B: set `WANDB_API_KEY` in `.env` or export; use `WANDB_MODE=offline` to avoid network use.

How to Run
- Install deps: `uv venv && uv sync && source .venv/bin/activate` (per `README.md`); ensure PyTorch/torchvision/torchaudio match platform.
- (Optional) Generate synthetic data: `python image_creator/create_synthetic_images.py` to populate `images/synthetic/`.
- Execute the pipeline: `python main.py`. Provide grayscale-friendly images in `images/` (subfolders OK).

Extending / Customizing
- Swap PSFs: edit `psf_specs` in `main.py` or add new generators in `blind_deconvolution/psf_generator.py`.
- Add image priors: pass `image_prior_fn` in `BlindDeconvConfig`, or implement new modules under `blind_deconvolution/priors/` and hook them in `map_objective`.
- Modify logging: adjust `log_fn` or W&B payloads in `main.py`; disable logging by setting `wandb.init(..., mode=\"disabled\")` or env `WANDB_MODE=offline`.
- Different noise model: change `noise_sigma` in measurement creation or modify `forward_model.add_gaussian_noise`.
