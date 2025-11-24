from __future__ import annotations

import numpy as np
from typing import Tuple


##############################
# Utility Functions
##############################


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    """Normalize PSF to have sum 1 and enforce non-negativity.

    Args:
        psf: 2D numpy array.

    Returns:
        Normalized 2D numpy array.
    """
    psf = np.maximum(psf, 0.0)
    s = psf.sum()
    if s <= 0:
        raise ValueError("PSF sum is non-positive; cannot normalize.")
    return psf / s


##############################
# Basic PSF Generators
##############################


def delta_psf(size: int = 15) -> np.ndarray:
    """Delta kernel (identity PSF).

    Args:
        size: Size of the (size x size) PSF.

    Returns:
        2D numpy array with a single 1 at the center.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    psf = np.zeros((size, size), dtype=np.float64)
    psf[size // 2, size // 2] = 1.0
    return psf


def gaussian_psf(size: int = 15, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur PSF.

    Args:
        size: Size of the kernel (size x size).
        sigma: Standard deviation of the Gaussian.

    Returns:
        Normalized 2D Gaussian kernel.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return _normalize_psf(kernel)


def motion_psf(size: int = 15, length: int | None = None, angle: float = 0.0) -> np.ndarray:
    """Simple linear motion blur PSF.

    Args:
        size: Size of the PSF (size x size).
        length: Length of the motion blur in pixels. If None, defaults to size // 2.
        angle: Angle in degrees. 0 = horizontal motion, 90 = vertical.

    Returns:
        Normalized 2D motion blur kernel.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    if length is None:
        length = max(1, size // 2)

    # Create a horizontal line kernel first (center row)
    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    half = length // 2
    start = max(0, center - half)
    end = min(size, center + half + 1)
    psf[center, start:end] = 1.0

    # Rotate by the requested angle using Fourier-based rotation approximation
    # To avoid importing skimage.transform here, we can manually rotate using
    # interpolation via scipy.ndimage if available. For now, implement a
    # simple nearest-neighbor rotation using scipy.ndimage.rotate if present.

    try:
        from scipy.ndimage import rotate

        rotated = rotate(psf, angle=angle, reshape=False, order=1, mode="constant", cval=0.0)
    except Exception:
        # Fallback: no rotation library available, just return horizontal
        rotated = psf

    return _normalize_psf(rotated)


def disk_psf(size: int = 15, radius: float | None = None) -> np.ndarray:
    """Disk-shaped (out-of-focus) blur PSF.

    Args:
        size: Size of the PSF (size x size).
        radius: Radius of the disk in pixels. If None, defaults to size / 4.

    Returns:
        Normalized 2D disk kernel.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    if radius is None:
        radius = size / 4.0

    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    mask = (xx**2 + yy**2) <= radius**2
    psf = mask.astype(np.float64)
    return _normalize_psf(psf)


##############################
# Factory / Convenience
##############################


def get_psf(psf_type: str, size: int = 15, **kwargs) -> np.ndarray:
    """Convenience factory to get a PSF by name.

    Args:
        psf_type: One of ["delta", "gaussian", "motion", "disk"].
        size: Kernel size.
        **kwargs: Extra parameters forwarded to the underlying generator.

    Returns:
        2D PSF as numpy array.
    """
    psf_type = psf_type.lower()

    if psf_type == "delta":
        return delta_psf(size=size)
    if psf_type == "gaussian":
        return gaussian_psf(size=size, sigma=kwargs.get("sigma", 2.0))
    if psf_type == "motion":
        return motion_psf(size=size,
                          length=kwargs.get("length"),
                          angle=kwargs.get("angle", 0.0))
    if psf_type == "disk":
        return disk_psf(size=size, radius=kwargs.get("radius"))

    raise ValueError(f"Unknown psf_type: {psf_type}")


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    for name in ["delta", "gaussian", "motion", "disk"]:
        k = get_psf(name, size=15)
        print(name, k.shape, k.sum(), k.min(), k.max())