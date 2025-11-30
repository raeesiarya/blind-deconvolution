# Blind Deconvolution Playground

Research playground for single-image blind deconvolution. The testbench builds synthetic measurements (blur + noise) for each clean image, then jointly optimizes the latent sharp image and PSF kernel with a MAP objective. Metrics and qualitative outputs are logged to Weights & Biases.

## What’s inside
- `main.py`: orchestrates the sweep of testbench configs and calls the core experiment runner.
- `testing/testbench.py`: runs each config across all images and PSF types, logs W&B artifacts/metrics (PSNR, SSIM, kernel error, loss curves).
- `testing/testbench_configs.py`: list of named experiment configs (iter counts, LRs, priors, kernel sizes, PSF settings).
- `blind_deconvolution/`: core solver (`BlindDeconvolver`), forward model, PSF generators (gaussian/motion/turbulence/rml), and priors (pink-noise + diffusion stub hook).
- `utils/`: image I/O, path helpers, NumPy↔Torch converters, metrics, device chooser, and W&B helpers.
- `images/`: sample `synthetic/` patterns and a `real/` folder for your data. Supported: png/jpg/jpeg/tif/tiff/bmp.

## Setup (UV + Weights & Biases)
1. Install UV if you don’t have it.
2. Create the env and install deps:
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```
3. Put your W&B API key in `.env` (`WANDB_API_KEY=...`) or export it in your shell. Use `WANDB_MODE=offline` if you want no network logging.
4. Authenticate with W&B when online:
   ```bash
   wandb login
   ```

## Running the testbench
1. Drop one or more images under `images/` (subfolders are picked up). Everything is converted to grayscale and normalized.
2. Start the sweep:
   ```bash
   python main.py
   ```
   Each run iterates over every config in `testing/testbench_configs.py` and every PSF type defined in `testing/testbench.py` (gaussian, motion, turbulence; add `psf_types=["rml"]` to configs to try the randomized optics kernel). Measurements add light Gaussian noise.

What gets logged to W&B:
- Per image/PSF: reconstructions, estimated vs. true kernels, loss curves, PSNR, SSIM, kernel error, and the blurred measurement.
- Run summary aggregates: mean PSNR/SSIM/kernel error overall and by PSF type.

## Tuning / extending
- Edit `TESTBENCH_CONFIGS` to add/remove sweeps or tweak hyperparams (`num_iters`, `lr_x`, `lr_k`, priors including `lambda_k_auto`, kernel sizes, PSF params).
- A pre-made `diffusion_rml_only` config runs the randomized PSF generator; copy/tune it to explore different bandwidths or seeds.
- Adjust the PSF list or noise level in `testing/testbench.py`.
- To try a custom prior, implement `image_prior_fn` in `BlindDeconvConfig` and wire it into `blind_deconvolution/priors/diffusion.py`.
- Set `WANDB_MODE=offline` to disable logging, or change device selection in `utils/cuda_checker.py` (defaults to CUDA if available, else CPU).

## Notes
- Pipeline assumes single-channel images; extend the forward model if you need RGB.
- Keep an eye on kernel sizes vs. image dimensions to avoid excessive padding effects.
- Kernel priors include an optional autocorrelation penalty (`lambda_k_auto`) to encourage randomized-optics kernels (k * k ~ delta).
