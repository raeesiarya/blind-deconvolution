from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate, gaussian_filter


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


def motion_psf(
    size: int = 15, length: int | None = None, angle: float = 0.0
) -> np.ndarray:
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
    rotated = rotate(
        psf, angle=angle, reshape=False, order=1, mode="constant", cval=0.0
    )

    return _normalize_psf(rotated)


def turbulence_psf(
    size: int = 15,
    fried_parameter: float | None = None,
    distortion_strength: float = 0.6,
    seed: int | None = None,
) -> np.ndarray:
    """Atmospheric turbulence-inspired PSF.

    We approximate a Kolmogorov long-exposure PSF with a heavy-tailed radial
    profile and low-frequency distortions, which generally produces a blur
    harder to invert than a single straight-line motion kernel.

    Args:
        size: Size of the PSF (size x size).
        fried_parameter: Controls blur width; smaller => stronger blur. Defaults
            to size / 8.
        distortion_strength: Scales random low-frequency distortions that break
            radial symmetry.
        seed: Optional RNG seed for repeatability.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if distortion_strength < 0:
        raise ValueError("distortion_strength must be non-negative")

    if fried_parameter is None:
        fried_parameter = max(1.0, size / 8)
    if fried_parameter <= 0:
        raise ValueError("fried_parameter must be positive")

    rng = np.random.default_rng(seed)

    # Radial Kolmogorov-like envelope (heavier tails than Gaussian).
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    rho = np.sqrt(xx**2 + yy**2) + 1e-8

    # Add a small random anisotropy to emulate wind-driven shear.
    shear_x = 1.0 + 0.3 * rng.standard_normal()
    shear_y = 1.0 + 0.3 * rng.standard_normal()
    rho_aniso = np.sqrt((xx / shear_x) ** 2 + (yy / shear_y) ** 2) + 1e-8

    base = np.exp(-0.5 * (rho_aniso / fried_parameter) ** (5.0 / 3.0))

    # Low-frequency distortion field; smoothed noise perturbs the envelope.
    noise = rng.standard_normal((size, size))

    distortion = gaussian_filter(noise, sigma=max(1.0, size / 10), mode="reflect")
    distortion = distortion - distortion.mean()
    distortion = distortion / (distortion.std() + 1e-8)

    psf = base * np.exp(distortion_strength * distortion)

    return _normalize_psf(psf)


def rml_psf(
    size: int = 15, bandwidth: float = 0.35, seed: int | None = None
) -> np.ndarray:
    """Randomized optics PSF using band-limited random Fourier phases.

    Args:
        size: Size of the PSF (size x size).
        bandwidth: Fraction of the Nyquist radius to keep in the Fourier domain;
            controls speckle granularity (0 < bandwidth <= 1).
        seed: Optional RNG seed for repeatability.

    Returns:
        Randomized PSF drawn from a low-correlation distribution.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if bandwidth <= 0 or bandwidth > 1:
        raise ValueError("bandwidth must be in (0, 1]")

    rng = np.random.default_rng(seed)

    # Build a circular band-limit mask in normalized frequency units (Nyquist = 0.5).
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    radius = np.sqrt(fx**2 + fy**2)
    cutoff = 0.5 * bandwidth
    mask = (radius <= cutoff).astype(np.float64)
    if not np.any(mask):
        raise ValueError("bandwidth is too small; band-limit mask is empty")

    # Randomize Fourier phases via white noise, then enforce the band-limit.
    spectrum = np.fft.fft2(rng.standard_normal((size, size))) * mask
    field = np.fft.ifft2(spectrum)

    # Intensity of the complex field yields a nonnegative speckle-like PSF.
    psf = np.abs(field) ** 2
    return _normalize_psf(psf)


##############################
# Factory / Convenience
##############################


def get_psf(psf_type: str, size: int = 15, **kwargs) -> np.ndarray:
    """Convenience factory to get a PSF by name.

    Args:
        psf_type: One of ["gaussian", "motion", "turbulence", "rml"].
        size: Kernel size.
        **kwargs: Extra parameters forwarded to the underlying generator.

    Returns:
        2D PSF as numpy array.
    """
    psf_type = psf_type.lower()

    if psf_type == "gaussian":
        return gaussian_psf(size=size, sigma=kwargs.get("sigma", 2.0))
    if psf_type == "motion":
        return motion_psf(
            size=size, length=kwargs.get("length"), angle=kwargs.get("angle", 0.0)
        )
    if psf_type == "turbulence":
        return turbulence_psf(
            size=size,
            fried_parameter=kwargs.get("fried_parameter"),
            distortion_strength=kwargs.get("distortion_strength", 0.6),
            seed=kwargs.get("seed"),
        )
    if psf_type == "rml":
        return rml_psf(
            size=size,
            bandwidth=kwargs.get("bandwidth", 0.35),
            seed=kwargs.get("seed"),
        )

    raise ValueError(f"Unknown psf_type: {psf_type}")


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    for name in ["gaussian", "motion", "turbulence", "rml"]:
        k = get_psf(name, size=15)
        print(name, k.shape, k.sum(), k.min(), k.max())
