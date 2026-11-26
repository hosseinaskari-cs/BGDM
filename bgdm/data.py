"""Read and normalize the reference images used to simulate measurements."""

from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch


def load_array(path: Path, key: str) -> np.ndarray:
    """Read a read-only NPY mapping or a named NPZ array without unpickling."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".npy", ".npz"}:
        raise ValueError(f"Expected an .npy or .npz file, got {path}.")
    try:
        source = np.load(path, allow_pickle=False, mmap_mode="r")
        if isinstance(source, np.ndarray):
            if suffix != ".npy":
                raise ValueError("File content is NPY but its extension is NPZ.")
            return source
        with source as archive:
            if suffix != ".npz":
                raise ValueError("File content is NPZ but its extension is NPY.")
            if key not in archive.files:
                names = ", ".join(archive.files) or "none"
                raise ValueError(f"Missing array {key!r}. Available arrays: {names}.")
            return archive[key]
    except (BadZipFile, EOFError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"Could not read array from {path}: {error}") from error


def _max_magnitude(images: np.ndarray, split_complex: bool) -> float:
    # Process full dataset statistics in chunks to avoid a second full-size array.
    maximum = 0.0
    for start in range(0, len(images), 16):
        block = images[start : start + 16]
        if split_complex:
            magnitudes = np.hypot(block[:, :1], block[:, 1:], dtype=np.float64)
        else:
            magnitudes = np.abs(block, dtype=np.float64)
        block_maximum = float(magnitudes.max())
        if not np.isfinite(block_maximum):
            raise ValueError("Image magnitudes must be finite.")
        maximum = max(maximum, block_maximum)
    return maximum or 1.0


def load_images(
    path: Path,
    *,
    size: int,
    is_complex: bool,
    normalization: str,
    key: str,
    start: int = 0,
    limit: int | None = None,
) -> torch.Tensor:
    """Load [N, 1, H, W] CPU images from a zero-based start index.

    Max-magnitude scaling uses the full input so a limited run has the same
    intensity scale as a full run.
    """
    if normalization not in {"none", "uint8", "unit", "max-magnitude", "clip-unit"}:
        raise ValueError(f"Unknown image normalization: {normalization!r}.")
    if is_complex and normalization in {"uint8", "unit", "clip-unit"}:
        raise ValueError(f"Normalization {normalization!r} requires real-valued images.")
    if size < 1 or start < 0:
        raise ValueError("Image size must be positive and start index must be non-negative.")
    if limit is not None and limit < 1:
        raise ValueError("Image limit must be at least 1.")

    images = load_array(path, key)
    original_shape = images.shape
    if images.ndim == 2:
        images = images[None, None]
    elif images.ndim == 3:
        images = images[:, None]
    if images.ndim != 4 or images.shape[-2:] != (size, size) or not len(images):
        raise ValueError(f"Expected at least one {size} × {size} image in {path}, got shape {original_shape}.")
    if images.dtype.kind not in "fciu":
        raise ValueError(f"Expected numeric images in {path}, got dtype {images.dtype}.")

    native_complex = np.iscomplexobj(images)
    split_complex = is_complex and images.shape[1] == 2 and not native_complex
    if not split_complex and (images.shape[1] != 1 or native_complex != is_complex):
        expected = "one complex channel or two real (real/imaginary) channels" if is_complex else "one real channel"
        raise ValueError(f"Expected {expected} in {path}, got shape {original_shape} and dtype {images.dtype}.")

    if start >= len(images):
        raise ValueError(f"Start index {start} is outside the {len(images)} images in {path}.")
    stop = None if limit is None else start + limit
    selected = images[start:stop]
    if normalization == "max-magnitude":
        selected = selected / _max_magnitude(images, split_complex)
    if not np.isfinite(selected).all():
        raise ValueError(f"Images in {path} must contain only finite values.")
    if normalization in {"uint8", "unit"}:
        upper = 255 if normalization == "uint8" else 1
        if selected.min() < 0 or selected.max() > upper:
            raise ValueError(f"Normalization {normalization!r} requires image values in [0, {upper}] in {path}.")
    if normalization == "clip-unit":
        selected = selected.clip(0, 1)

    dtype = np.complex64 if native_complex else np.float32
    with np.errstate(over="ignore"):
        selected = np.require(selected, dtype=dtype, requirements=["C", "W"])
    if len(selected) < len(images) and np.shares_memory(selected, images):
        # Keep a small sample batch from retaining the entire loaded archive.
        selected = selected.copy()
    if not np.isfinite(selected).all():
        raise ValueError(f"Image values in {path} exceed {np.dtype(dtype)} range.")

    tensor = torch.from_numpy(selected)
    if normalization == "uint8":
        tensor.div_(255)
    if split_complex:
        tensor = torch.complex(tensor[:, :1], tensor[:, 1:])
    return tensor


def synthetic_image(size: int, *, is_complex: bool) -> torch.Tensor:
    """Return a deterministic ellipse phantom, with smooth phase for complex MRI."""
    if size < 1:
        raise ValueError("Phantom size must be at least 1.")
    coordinates = (torch.arange(size, dtype=torch.float32) + 0.5) * (2.0 / size) - 1.0
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = torch.zeros((size, size), dtype=torch.float32)
    # Each ellipse gives intensity, radii, center, and in-plane rotation.
    ellipses = (
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.75, 0.65, 0.86, 0.0, -0.02, 0.0),
        (-0.15, 0.19, 0.31, 0.20, 0.0, -0.31),
        (-0.15, 0.15, 0.39, -0.20, 0.0, 0.31),
        (0.22, 0.22, 0.25, 0.0, 0.28, 0.0),
        (0.20, 0.05, 0.09, -0.12, -0.31, 0.20),
        (0.25, 0.08, 0.06, 0.16, -0.33, -0.20),
    )
    for intensity, radius_x, radius_y, center_x, center_y, angle in ellipses:
        cosine, sine = np.cos(angle), np.sin(angle)
        dx, dy = x - center_x, y - center_y
        rotated_x = cosine * dx + sine * dy
        rotated_y = -sine * dx + cosine * dy
        inside = (rotated_x / radius_x).square() + (rotated_y / radius_y).square() <= 1
        image.add_(intensity * inside)

    image.clamp_(0, 1)
    if is_complex:
        image = torch.polar(image, 0.9 * x - 0.5 * y + 0.35 * x * y)
    return image[None, None]
