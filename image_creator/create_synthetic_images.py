"""
Generate synthetic images using the helper functions in synthetic_image_gen
and save them into the images directory.
"""
import os
from typing import List
import numpy as np
from skimage import img_as_ubyte, io

DEFAULT_SIZE = 256
DEFAULT_OUTPUT_DIR = "images/synthetic"

def gen_checkerboard(size: int = 256, num_checks: int = 8) -> np.ndarray:
    """
    Generates a checkerboard pattern.

    Args:
        size (int): Image size (size x size).
        num_checks (int): Number of checks along one dimension.

    Returns:
        np.ndarray: Checkerboard image in [0,1].
    """
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    checks = ((xx // (size // num_checks)) +
              (yy // (size // num_checks))) % 2
    return checks.astype(float)


def gen_gradient(size: int = 256, direction: str = "horizontal") -> np.ndarray:
    """
    Generates a simple gradient image.

    Args:
        size (int): Image size.
        direction (str): "horizontal" or "vertical".

    Returns:
        np.ndarray: Gradient image.
    """
    if direction == "horizontal":
        return np.tile(np.linspace(0, 1, size), (size, 1))
    else:
        return np.tile(np.linspace(0, 1, size), (size, 1)).T


def gen_circle(size: int = 256, radius_ratio: float = 0.3) -> np.ndarray:
    """
    Generates an image with a centered filled circle.

    Args:
        size (int): Image size.
        radius_ratio (float): Circle radius relative to size.

    Returns:
        np.ndarray: Image with a white circle on black background.
    """
    radius = size * radius_ratio
    x = np.linspace(-size/2, size/2, size)
    xx, yy = np.meshgrid(x, x)
    mask = (xx**2 + yy**2) <= radius**2
    return mask.astype(float)


def gen_bars(size: int = 256, bar_width: int = 16) -> np.ndarray:
    """
    Generates vertical bar patterns.

    Args:
        size (int): Image size.
        bar_width (int): Width of each bar.

    Returns:
        np.ndarray: Bars image.
    """
    img = np.zeros((size, size))
    for i in range(0, size, 2 * bar_width):
        img[:, i:i + bar_width] = 1.0
    return img


def gen_pink_noise(size: int = 256, beta: float = 1.0) -> np.ndarray:
    """
    Generates 1/f^beta pink noise in the Fourier domain.

    Args:
        size (int): Image size.
        beta (float): Exponent controlling color of noise (1 = pink).

    Returns:
        np.ndarray: Pink noise image in [0,1].
    """
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1  # avoid division by zero

    amplitude = 1 / (f**beta)
    phase = np.exp(1j * 2 * np.pi * np.random.rand(size, size))

    spectrum = amplitude * phase
    noise = np.fft.ifft2(spectrum).real

    noise -= noise.min()
    noise /= noise.max()

    return noise


def generate_images(output_dir: str = DEFAULT_OUTPUT_DIR, size: int = DEFAULT_SIZE) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    images = {
        "checkerboard.png": gen_checkerboard(size=size),
        "gradient_horizontal.png": gen_gradient(size=size, direction="horizontal"),
        "gradient_vertical.png": gen_gradient(size=size, direction="vertical"),
        "circle.png": gen_circle(size=size),
        "bars.png": gen_bars(size=size),
        "pink_noise.png": gen_pink_noise(size=size),
    }

    saved_paths = []
    for filename, image in images.items():
        path = os.path.join(output_dir, filename)

        image = image.clip(0, 1)
        image_uint8 = img_as_ubyte(image)

        io.imsave(path, image_uint8)
        saved_paths.append(path)

    return saved_paths


def main() -> None:
    paths = generate_images()
    print("Saved synthetic images:")
    for path in paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
