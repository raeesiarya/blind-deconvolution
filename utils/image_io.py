# utils/image_io.py
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from skimage import io, img_as_float32, color
import torch

def load_image(
    path: Path,
    mode: Literal["numpy", "torch"] = "numpy",
    grayscale: bool = True,
    normalize: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    Load an image from disk into a numpy array or torch tensor.

    Args:
        path: Path to the image file.
        mode: "numpy" (default) or "torch".
        grayscale: Convert to grayscale if True.
        normalize: Convert to float32 in range [0,1].

    Returns:
        np.ndarray of shape (H, W) or (H, W, C)
        OR a torch.Tensor of shape (1, 1, H, W) if mode="torch".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image via skimage
    img = io.imread(path)

    # Convert to grayscale if needed
    if grayscale and img.ndim == 3:
        img = color.rgb2gray(img)

    # Normalize + convert to float32
    if normalize:
        img = img_as_float32(img)

    if mode == "numpy":
        return img

    elif mode == "torch":
        if img.ndim == 2:
            # H,W → (1,1,H,W)
            tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            # H,W,C → (1,C,H,W)
            tensor = (
                torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
                .unsqueeze(0)
            )
        return tensor

    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'numpy' or 'torch'.")