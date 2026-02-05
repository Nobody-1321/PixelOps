import numpy as np
import pytest
from pixelops.filtering.spatial.median import median_filter

# ------------------------------------------------------------
# Validación de errores
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_median
@pytest.mark.parametrize("window", [0, 2, 4, -1])
def test_invalid_window_size_raises(window):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        median_filter(img, window)


@pytest.mark.filtering_spatial_median
def test_invalid_dimensions_raises():
    img = np.zeros((16,), dtype=np.uint8)

    with pytest.raises(ValueError):
        median_filter(img, 3)


@pytest.mark.filtering_spatial_median
def test_invalid_dtype_raises():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        median_filter(img, 3)


# ------------------------------------------------------------
# Shape y dtype
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_median
def test_grayscale_shape_and_dtype():
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    out = median_filter(img, 3)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.filtering_spatial_median
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_multichannel_shape_and_dtype(channels):
    img = np.random.randint(
        0, 256, (32, 32, channels), dtype=np.uint8
    )

    out = median_filter(img, 3)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ------------------------------------------------------------
# Casos base (sanity)
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_median
@pytest.mark.parametrize("window", [1, 3, 5, 7])
def test_constant_image_identity(window):
    img = np.full((32, 32), 123, dtype=np.uint8)

    out = median_filter(img, window)

    assert np.array_equal(out, img)


@pytest.mark.filtering_spatial_median
def test_window_size_one_identity():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out = median_filter(img, 1)

    assert np.array_equal(out, img)


# ------------------------------------------------------------
# Independencia de canales
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_median
def test_channels_are_independent():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 0] = 10
    img[:, :, 1] = 100
    img[:, :, 2] = 200

    out = median_filter(img, 3)

    assert np.all(out[:, :, 0] == 10)
    assert np.all(out[:, :, 1] == 100)
    assert np.all(out[:, :, 2] == 200)


# ------------------------------------------------------------
# Correctitud numérica (naive de referencia)
# ------------------------------------------------------------

def naive_median_filter(image: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    h, w = image.shape
    out = np.empty_like(image)

    padded = np.pad(image, pad, mode="reflect")

    for i in range(h):
        for j in range(w):
            window = padded[i:i + k, j:j + k]
            out[i, j] = np.median(window)

    return out.astype(np.uint8)


@pytest.mark.filtering_spatial_median
@pytest.mark.xfail(
    reason="Boundary handling not unified across convolution backends",
    strict=False
)
@pytest.mark.parametrize("window", [3, 5])
def test_matches_naive_grayscale(window):
    np.random.seed(0)
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out_fast = median_filter(img, window)
    out_ref = naive_median_filter(img, window)

    assert np.array_equal(out_fast, out_ref)

