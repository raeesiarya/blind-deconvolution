from blind_deconvolution.blind_deconvolution import BlindDeconvolver, BlindDeconvConfig
from utils.image_io import load_image
from utils.image_paths import list_image_paths
import torch
from blind_deconvolution.psf_generator import gaussian_psf
from blind_deconvolution.forward_model import forward_model
from utils.convertors import numpy_kernel_to_tensor

def main():
    device = "cpu"
    image_paths = list_image_paths()

    if not image_paths:
        print("No images found in images directory.")
        return

    for img_path in image_paths:
        print(f"\n=== Processing {img_path} ===")

        # Load clean image x_true
        x_true = load_image(img_path, mode="torch", grayscale=True, normalize=True).to(device)

        # Generate ground-truth PSF (Gaussian)
        k_np = gaussian_psf(size=15, sigma=2.0)
        k_true = numpy_kernel_to_tensor(k_np).to(device)

        # Create blurred measurement
        with torch.no_grad():
            y_meas = forward_model(x_true, k_true, noise_sigma=0.01)

        # Configure a lightweight run for quick testing
        config = BlindDeconvConfig(
            num_iters=100,
            lr_x=1e-2,
            lr_k=1e-2,
            lambda_x=0.0,
            lambda_k_l2=1e-3,
            lambda_k_center=1e-3,
            kernel_size=15,
            device=device,
        )

        solver = BlindDeconvolver(config).to(device)
        x_hat, k_hat, losses = solver.run(y_meas, verbose=True)

        print(f"Finished. Final loss: {losses[-1]:.6f}")
        print(f"x_hat shape: {tuple(x_hat.shape)}, k_hat shape: {tuple(k_hat.shape)}")


if __name__ == "__main__":
    main()
