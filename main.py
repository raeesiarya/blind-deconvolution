import numpy as np
from skimage import io

from psf_generator import gaussian_psf
from utils.convertors import numpy_image_to_tensor, numpy_kernel_to_tensor
from forward_model import forward_model

def main():
    x_np = io.imread("images/synthetic/checkerboard.png")  # adjust path
    if x_np.ndim == 3:
        # if saved as RGB, convert to grayscale via mean
        x_np = x_np.mean(axis=-1)
    x_np = x_np.astype(np.float32) / 255.0  # scale to [0,1]

    # 2. Make a Gaussian PSF in numpy
    k_np = gaussian_psf(size=21, sigma=3.0)

    # 3. Convert to torch
    x = numpy_image_to_tensor(x_np)
    k = numpy_kernel_to_tensor(k_np)

    # 4. Run forward model
    y = forward_model(x, k, noise_sigma=0.01)

    print(x.shape, k.shape, y.shape)
    
if __name__ == "__main__":
    main()
