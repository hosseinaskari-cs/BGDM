"""Measurement operators for single-slice MRI and parallel-beam CT."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .tasks import Task


SPATIAL_DIMS = (-2, -1)


def _centered_fft(image: Tensor) -> Tensor:
    shifted = torch.fft.ifftshift(image, dim=SPATIAL_DIMS)
    spectrum = torch.fft.fft2(shifted, norm="ortho")
    return torch.fft.fftshift(spectrum, dim=SPATIAL_DIMS)


def _centered_ifft(spectrum: Tensor) -> Tensor:
    shifted = torch.fft.ifftshift(spectrum, dim=SPATIAL_DIMS)
    image = torch.fft.ifft2(shifted, norm="ortho")
    return torch.fft.fftshift(image, dim=SPATIAL_DIMS)


def _opposite_frequencies(spectrum: Tensor) -> Tensor:
    # In centered order, k -> -k flips each axis and moves the even-sized
    # axes' Nyquist bin from the last position back to the first position.
    shifts = tuple(int(size % 2 == 0) for size in spectrum.shape[-2:])
    return spectrum.flip(SPATIAL_DIMS).roll(shifts, dims=SPATIAL_DIMS)


def _prepare_measurement(measurement: Tensor, shape: tuple[int, int]) -> Tensor:
    if measurement.ndim != 4 or measurement.shape[1:] != (1, *shape):
        raise ValueError(f"Expected measurements shaped [N, 1, {shape[0]}, {shape[1]}].")
    if not torch.isfinite(measurement).all():
        raise ValueError("Measurements must be finite.")
    return measurement


@dataclass(frozen=True)
class MRI:
    """Centered orthonormal Cartesian FFT with a shared binary sampling mask."""

    mask: Tensor
    is_complex: bool

    def _image(self, spectrum: Tensor) -> Tensor:
        image = _centered_ifft(spectrum)
        return image if self.is_complex else image.real

    def _real_observations(self, measurement: Tensor) -> tuple[Tensor, Tensor]:
        observed = measurement * self.mask
        summed = observed + _opposite_frequencies(observed).conj()
        counts = self.mask.to(observed.real.dtype) + _opposite_frequencies(self.mask)
        return summed, counts

    def prepare_measurement(self, measurement: Tensor) -> Tensor:
        measurement = _prepare_measurement(measurement, self.mask.shape[-2:])
        if not torch.is_complex(measurement):
            raise ValueError("MRI k-space must use a complex dtype.")
        return measurement.to(device=self.mask.device, dtype=torch.complex64) * self.mask

    def forward(self, image: Tensor) -> Tensor:
        return _centered_fft(image) * self.mask

    def preview(self, measurement: Tensor) -> Tensor:
        """Return a zero-filled preview, inferring conjugate partners for real images."""
        if self.is_complex:
            return self._image(measurement * self.mask)
        summed, counts = self._real_observations(measurement)
        return self._image(summed / counts.clamp_min(1))

    def residual(self, image: Tensor, measurement: Tensor) -> Tensor:
        return self.forward(image) - measurement

    def project(self, image: Tensor, measurement: Tensor, *, damping: float = 0.0) -> Tensor:
        spectrum = _centered_fft(image)
        if self.is_complex:
            measured_bins = (measurement + damping * spectrum) / (1.0 + damping)
            return self._image(torch.where(self.mask, measured_bins, spectrum))

        # A real image has a Hermitian spectrum. One acquired coefficient
        # constrains both partners; two acquisitions are averaged when noisy.
        # The proximal term covers both bins, giving 2*damping per pair.
        summed, counts = self._real_observations(measurement)
        strength = 2 * damping
        measured_bins = (summed + strength * spectrum) / (counts + strength).clamp_min(1)
        return self._image(torch.where(counts > 0, measured_bins, spectrum))


class ParallelBeamCT:
    """Approximate sparse-view CT using discrete Fourier slices.

    The simulation uses equally spaced angles in [0, pi). Each detector row is
    the inverse 1D FFT of one radial slice of a padded Cartesian image spectrum.
    Consistency updates replace the sampled Fourier coefficients, then crop the
    inverse FFT to the model image grid. This is a discrete simulation and is
    suitable for image-derived measurements on this geometry.
    """

    mask = None
    is_complex = False

    def __init__(self, image_size: int, views: int, expansion: float, device: torch.device):
        if views < 1 or not math.isfinite(expansion) or expansion < 1:
            raise ValueError("CT requires at least one view and a finite expansion of at least one.")
        self.image_size = image_size
        self.views = views
        # Use an odd FFT grid so the positive and negative radial bins pair at
        # every angle, giving real-valued projections of a real image.
        grid_size = math.ceil(math.sqrt(2.0) * image_size * expansion)
        self.detectors = grid_size | 1
        before = self.detectors // 2 - image_size // 2
        after = self.detectors - image_size - before
        self.padding = (before, after, before, after)

        angles = np.arange(views)[:, None] * (np.pi / views)
        radius = np.arange(self.detectors) - self.detectors // 2
        cols = np.rint(np.cos(angles) * radius).astype(np.int64) % self.detectors
        rows = np.rint(-np.sin(angles) * radius).astype(np.int64) % self.detectors
        indices = rows * self.detectors + cols
        bins, first_hit = np.unique(indices, return_index=True)
        self._slice_indices = torch.as_tensor(indices, device=device)
        self._grid_indices = torch.as_tensor(bins, device=device)
        self._view_indices = torch.as_tensor(first_hit, device=device)

    def _spectrum(self, image: Tensor) -> Tensor:
        padded = F.pad(image, self.padding)
        return torch.fft.fft2(torch.fft.ifftshift(padded, dim=SPATIAL_DIMS))

    def prepare_measurement(self, measurement: Tensor) -> Tensor:
        return _prepare_measurement(measurement, (self.views, self.detectors)).to(
            device=self._slice_indices.device, dtype=torch.complex64
        )

    def forward(self, image: Tensor) -> Tensor:
        lines = self._spectrum(image).flatten(-2)[..., self._slice_indices]
        projections = torch.fft.ifft(torch.fft.ifftshift(lines, dim=-1))
        return torch.fft.fftshift(projections, dim=-1)

    def preview(self, measurement: Tensor) -> Tensor:
        """Preview the sampled Fourier slices with all other coefficients set to zero."""
        shape = (measurement.shape[0], 1, self.image_size, self.image_size)
        empty_image = measurement.real.new_zeros(shape)
        return self.project(empty_image, measurement)

    def residual(self, image: Tensor, measurement: Tensor) -> Tensor:
        return self.forward(image) - measurement

    def project(self, image: Tensor, measurement: Tensor, *, damping: float = 0.0) -> Tensor:
        spectrum = self._spectrum(image).flatten(-2)
        lines = torch.fft.fft(torch.fft.ifftshift(measurement, dim=-1))
        lines = torch.fft.fftshift(lines, dim=-1).flatten(-2)
        # Multiple rays can land at one Cartesian frequency. Resolve these
        # collisions with a fixed ray order instead of a duplicate-index write.
        measured_bins = lines.index_select(-1, self._view_indices)
        predicted_bins = spectrum.index_select(-1, self._grid_indices)
        updated_bins = (measured_bins + damping * predicted_bins) / (1.0 + damping)
        spectrum = spectrum.index_copy(-1, self._grid_indices, updated_bins)
        spectrum = spectrum.unflatten(-1, (self.detectors, self.detectors))
        padded = torch.fft.fftshift(torch.fft.ifft2(spectrum), dim=SPATIAL_DIMS)
        start = self.padding[0]
        stop = start + self.image_size
        return padded.real[..., start:stop, start:stop]


Degradation = MRI | ParallelBeamCT


def _mri_mask(name: str, size: int, acceleration: float, mask_type: str | None, seed: int) -> np.ndarray:
    if not math.isfinite(acceleration) or acceleration < 1:
        raise ValueError("MRI acceleration must be finite and at least one.")

    if name == "fastmri-brain":
        if mask_type is not None:
            raise ValueError("The super-resolution preset uses a central square mask.")
        retained = max(1, round(size / math.sqrt(acceleration)))
        start = size // 2 - retained // 2
        mask = np.zeros((size, size), dtype=bool)
        mask[start:start + retained, start:start + retained] = True
        return mask

    # Match the original experiment masks, including repeat draws in the
    # Gaussian scheme. A local generator makes masks independent of inference RNG.
    rng = np.random.RandomState(seed % 2**32)
    columns = np.zeros(size, dtype=bool)
    if name == "brats" and mask_type in {None, "uniform1d"}:
        center = max(1, round(size * 0.04))
        start = (size - center) // 2
        total = round(size / acceleration)
        mask_type = "uniform1d"
    else:
        center = max(1, int(size * 0.04))
        start = size // 2 - center // 2
        total = int(size / acceleration)
        mask_type = mask_type or "gaussian1d"

    if mask_type not in {"uniform1d", "gaussian1d"}:
        raise ValueError(f"Unsupported MRI mask type: {mask_type}.")

    columns[start:start + center] = True
    if acceleration == 1:
        columns[:] = True
    elif mask_type == "uniform1d":
        available = np.flatnonzero(~columns)
        count = max(0, total - center)
        columns[rng.choice(available, size=count, replace=False)] = True
    else:
        draws = rng.normal(size // 2, size * (15.0 / 128), size=int(total * 1.2))
        columns[draws.astype(np.int64).clip(0, size - 1)] = True
    return np.tile(columns, (size, 1))


def build_degradation(
    spec: Task,
    *,
    device: torch.device,
    seed: int,
    mask_type: str | None = None,
    mask: np.ndarray | Tensor | None = None,
) -> Degradation:
    """Build one operator for a preset and optionally use a caller's MRI mask."""
    size = spec.image_size
    if size < 1:
        raise ValueError("Image size must be positive.")
    if spec.name == "lidc":
        if mask is not None or mask_type is not None:
            raise ValueError("The CT preset uses projection views instead of an MRI mask.")
        if spec.views is None:
            raise ValueError("CT needs a projection view count.")
        return ParallelBeamCT(size, spec.views, spec.expansion, device)
    if spec.name not in {"brats", "fastmri-knee", "fastmri-brain"}:
        raise ValueError(f"Unknown reconstruction preset: {spec.name}.")

    if mask is None:
        if spec.acceleration is None:
            raise ValueError("MRI needs an acceleration factor or an explicit mask.")
        mask = _mri_mask(spec.name, size, spec.acceleration, mask_type or spec.mask_type, seed)
    elif mask_type is not None:
        raise ValueError("Choose either an explicit MRI mask or a mask type.")
    mask_tensor = mask.to(device) if isinstance(mask, Tensor) else torch.tensor(mask, device=device)
    if mask_tensor.shape not in {(size, size), (1, size, size), (1, 1, size, size)}:
        raise ValueError(f"MRI mask must have spatial shape {size} × {size} and only singleton leading axes.")
    if torch.is_complex(mask_tensor) or not ((mask_tensor == 0) | (mask_tensor == 1)).all():
        raise ValueError("MRI mask must be binary, with entries 0 and 1.")
    if not mask_tensor.any():
        raise ValueError("MRI mask must retain at least one k-space sample.")
    return MRI(mask_tensor.bool().reshape(1, 1, size, size), spec.is_complex)
