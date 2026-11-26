"""Pretrained model architectures and reconstruction defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    image_size: int
    checkpoint: str
    checkpoint_folder: str
    dataset: str
    normalization: str
    channel_mult: tuple[int, ...]
    attention: tuple[int, ...]
    steps: int
    diffusion_steps: int = 1000
    learn_sigma: bool = True
    resblock_updown: bool = False
    is_complex: bool = False
    acceleration: float | None = None
    mask_type: str | None = None
    views: int | None = None
    expansion: int = 4
    iradon_k: float = 1.8
    eta: float = 0.85
    guidance: float = 1.4
    damping: float = 1e-3


TASKS = {
    "brats": Task(
        "brats", "Real-valued undersampled MRI", 240,
        "ema_0.9999_2020000.pt", "BraTS", "BraTS.npz", "uint8",
        (1, 2, 2, 4, 4), (30,), 200, acceleration=8, mask_type="uniform1d",
    ),
    "fastmri-knee": Task(
        "fastmri-knee", "Complex-valued undersampled MRI", 320,
        "model2230000.pt", "fastMRI_knee", "fastMRI_knee.npz", "max-magnitude",
        (1, 2, 4, 4, 6, 6), (40, 20), 30, resblock_updown=True,
        is_complex=True, acceleration=4, mask_type="gaussian1d", guidance=2.0,
        damping=0.0,
    ),
    "fastmri-brain": Task(
        "fastmri-brain", "MRI super-resolution", 320,
        "ema_0.9999_340000.pt", "fastMRI_brain", "fastMRI_brain.npz", "clip-unit",
        (1, 2, 4, 4, 6, 6), (40, 20), 50, resblock_updown=True, acceleration=16,
    ),
    "lidc": Task(
        "lidc", "Sparse-view CT", 320,
        "model4200000.pt", "LIDC", "LIDC_320_vv.npz", "uint8",
        (1, 1, 2, 2, 4, 4), (20, 10), 350, diffusion_steps=2000,
        learn_sigma=False, views=23, eta=0.9, guidance=0.0, damping=0.0,
    ),
}
