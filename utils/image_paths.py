from pathlib import Path
from typing import List, Optional

# Supported image file extensions (lowercase)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def get_images_dir() -> Path:
    """Return the repository's images directory."""
    return Path(__file__).resolve().parent.parent / "images"


def list_image_paths(images_dir: Optional[Path] = None, recursive: bool = True) -> List[Path]:
    """Return sorted Paths to images under the images directory.

    Args:
        images_dir: Root directory to search; defaults to the repo's images folder.
        recursive: If True, search through subdirectories.
    """
    root = images_dir or get_images_dir()
    if not root.exists():
        raise FileNotFoundError(f"Images directory not found: {root}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )