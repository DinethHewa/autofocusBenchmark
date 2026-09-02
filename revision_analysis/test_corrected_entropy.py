from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_revision_code"))
from corrected_entropy import fixed_range_histogram, histogram_entropy


def test_constant_image_is_zero_and_finite():
    value = histogram_entropy(np.full((32, 32), 17, dtype=np.uint8))
    assert value == pytest.approx(0.0)
    assert np.isfinite(value)


def test_bimodal_image_is_one_bit():
    image = np.vstack([np.zeros((16, 32)), np.ones((16, 32)) * 255]).astype(np.uint8)
    assert histogram_entropy(image) == pytest.approx(1.0)


def test_textured_image_has_more_entropy_than_bimodal():
    textured = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
    bimodal = (textured >= 128).astype(np.uint8) * 255
    assert histogram_entropy(textured) > histogram_entropy(bimodal)


def test_uint8_uint16_equivalence():
    image8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image16 = image8.astype(np.uint16) * np.uint16(257)
    assert histogram_entropy(image8) == pytest.approx(histogram_entropy(image16), abs=1e-12)


def test_unit_float_equivalence():
    image8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image_float = image8.astype(np.float64) / 255.0
    assert histogram_entropy(image8) == pytest.approx(histogram_entropy(image_float), abs=1e-12)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_histogram_mass_equals_pixel_count(dtype):
    rng = np.random.default_rng(42)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        image = rng.integers(info.min, info.max, size=(25, 31), dtype=dtype)
    else:
        image = rng.random((25, 31)).astype(dtype)
    histogram = fixed_range_histogram(image)
    assert int(histogram.sum()) == image.size
    assert np.isfinite(histogram_entropy(image))


def test_nonfinite_float_rejected():
    image = np.zeros((8, 8), dtype=float)
    image[0, 0] = np.nan
    with pytest.raises(ValueError):
        histogram_entropy(image)


def test_out_of_range_float_rejected_without_hidden_normalization():
    with pytest.raises(ValueError):
        histogram_entropy(np.array([[0.0, 2.0]], dtype=float))
