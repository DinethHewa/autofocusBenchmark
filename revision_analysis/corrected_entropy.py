"""Dtype-safe fixed-range histogram entropy for microscopy images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

DEFAULT_BINS = 256


@dataclass(frozen=True)
class EntropyResult:
    entropy_bits: float
    histogram_mass: int
    pixel_count: int


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3].astype(np.float64, copy=False)
        array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got {array.shape}")
    if array.size == 0:
        raise ValueError("empty images are not defined")
    return array


def fixed_range_histogram(
    image: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    float_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Return a mass-preserving histogram under a fixed radiometric convention.

    Integer arrays are mapped affinely from the full representable dtype range
    to ``bins`` equal bins. Floating-point arrays are interpreted in the
    caller-declared fixed range (unit intensity by default). No per-image
    normalization is performed.
    """
    if bins <= 1:
        raise ValueError("bins must be greater than one")
    array = _gray(image)
    if not np.all(np.isfinite(array)):
        raise ValueError("image contains NaN or infinite values")

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        # Float arithmetic avoids overflow for wide signed integer types. The
        # upper endpoint is explicitly assigned to the last bin.
        scaled = (array.astype(np.float64) - float(info.min)) / float(info.max - info.min)
        indices = np.floor(scaled * bins).astype(np.int64)
        np.clip(indices, 0, bins - 1, out=indices)
        histogram = np.bincount(indices.ravel(), minlength=bins)
    elif np.issubdtype(array.dtype, np.floating):
        low, high = map(float, float_range)
        if not np.isfinite(low) or not np.isfinite(high) or not high > low:
            raise ValueError("float_range must contain two finite increasing bounds")
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(low), abs(high)) * 8
        observed_low = float(np.min(array))
        observed_high = float(np.max(array))
        if observed_low < low - tolerance or observed_high > high + tolerance:
            raise ValueError(
                f"floating image range [{observed_low}, {observed_high}] lies outside "
                f"declared fixed range [{low}, {high}]"
            )
        scaled = (array.astype(np.float64, copy=False) - low) / (high - low)
        indices = np.floor(scaled * bins).astype(np.int64)
        np.clip(indices, 0, bins - 1, out=indices)
        histogram = np.bincount(indices.ravel(), minlength=bins)
    else:
        raise TypeError(f"unsupported image dtype: {array.dtype}")

    if int(histogram.sum()) != int(array.size):
        raise RuntimeError("histogram mass does not equal pixel count")
    return histogram.astype(np.int64, copy=False)


def histogram_entropy(
    image: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    float_range: Tuple[float, float] = (0.0, 1.0),
    return_details: bool = False,
) -> float | EntropyResult:
    histogram = fixed_range_histogram(image, bins=bins, float_range=float_range)
    mass = int(histogram.sum())
    probabilities = histogram[histogram > 0].astype(np.float64) / mass
    value = float(-np.sum(probabilities * np.log2(probabilities)))
    if not np.isfinite(value):
        raise RuntimeError("entropy calculation returned a non-finite value")
    result = EntropyResult(value, mass, int(np.asarray(image).shape[0] * np.asarray(image).shape[1]))
    return result if return_details else value


__all__ = ["DEFAULT_BINS", "EntropyResult", "fixed_range_histogram", "histogram_entropy"]
