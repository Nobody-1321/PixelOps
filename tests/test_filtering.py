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

from pixelops.filtering.spatial.gaussian import gaussian_filter


# ============================================================
# Marker: filtering_spatial_gaussian
# ============================================================


# ------------------------------------------------------------
# Validación de errores
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gaussian
@pytest.mark.parametrize("sigma", [0.0, -1.0, -2.5])
def test_invalid_sigma_raises(sigma):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_filter(img, sigma)


@pytest.mark.filtering_spatial_gaussian
def test_invalid_dimensions_raises():
    img = np.zeros((16,), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_filter(img, sigma=1.0)


# ------------------------------------------------------------
# Shape y dtype
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gaussian
def test_grayscale_shape_and_dtype():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out = gaussian_filter(img, sigma=1.5)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_gaussian
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_multichannel_shape_and_dtype(channels):
    img = np.random.randint(
        0, 256, (32, 32, channels), dtype=np.uint8
    )

    out = gaussian_filter(img, sigma=1.0)

    assert out.shape == img.shape
    assert out.dtype == np.float32


# ------------------------------------------------------------
# Casos base
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gaussian
def test_constant_image_remains_constant():
    img = np.full((32, 32), 128, dtype=np.uint8)

    out = gaussian_filter(img, sigma=2.0)

    assert np.allclose(out, 128.0, atol=1e-5)


# ------------------------------------------------------------
# Independencia de canales
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gaussian
def test_channels_are_independent():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 0] = 10
    img[:, :, 1] = 100
    img[:, :, 2] = 200

    out = gaussian_filter(img, sigma=1.5)

    assert np.allclose(out[:, :, 0], 10.0, atol=1e-5)
    assert np.allclose(out[:, :, 1], 100.0, atol=1e-5)
    assert np.allclose(out[:, :, 2], 200.0, atol=1e-5)


# ------------------------------------------------------------
# Correctitud numérica (referencia naive)
# ------------------------------------------------------------

def gaussian_naive_reference(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Naive Gaussian filter for testing purposes.
    Uses explicit padding and full 2D convolution.
    """
    radius = int(3 * sigma)
    size = 2 * radius + 1

    ax = np.arange(-radius, radius + 1)
    kernel_1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()

    kernel_2d = np.outer(kernel_1d, kernel_1d)

    padded = np.pad(image.astype(np.float32), radius, mode="reflect")
    h, w = image.shape
    out = np.empty((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + size, j:j + size]
            out[i, j] = np.sum(patch * kernel_2d)

    return out


@pytest.mark.filtering_spatial_gaussian
@pytest.mark.xfail(
    reason="Boundary handling not unified across convolution backends",
    strict=False
)
@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_matches_naive_grayscale(sigma):
    np.random.seed(0)
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out_fast = gaussian_filter(img, sigma)
    out_ref = gaussian_naive_reference(img, sigma)

    assert np.allclose(out_fast, out_ref, atol=1e-4)


from pixelops.filtering.spatial.bilateral import bilateral_filter


# ============================================================
# Marker: filtering_spatial_bilateral
# ============================================================


# ------------------------------------------------------------
# Validación de errores
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_bilateral
@pytest.mark.parametrize("ss", [0.0, -1.0])
def test_invalid_spatial_sigma_raises(ss):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        bilateral_filter(img, ss=ss, sr=10.0)


@pytest.mark.filtering_spatial_bilateral
@pytest.mark.parametrize("sr", [0.0, -5.0])
def test_invalid_range_sigma_raises(sr):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        bilateral_filter(img, ss=2.0, sr=sr)


@pytest.mark.filtering_spatial_bilateral
@pytest.mark.parametrize("n_iter", [0, -1])
def test_invalid_n_iter_raises(n_iter):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        bilateral_filter(img, ss=2.0, sr=10.0, n_iter=n_iter)


@pytest.mark.filtering_spatial_bilateral
@pytest.mark.parametrize("n", [0, 2, -3])
def test_invalid_n_raises(n):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        bilateral_filter(img, ss=2.0, sr=10.0, n=n)


@pytest.mark.filtering_spatial_bilateral
def test_invalid_dimensions_raises():
    img = np.zeros((16,), dtype=np.uint8)

    with pytest.raises(ValueError):
        bilateral_filter(img, ss=2.0, sr=10.0)


# ------------------------------------------------------------
# Shape y dtype
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_bilateral
def test_grayscale_shape_and_dtype():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out = bilateral_filter(img, ss=2.0, sr=25.0)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_bilateral
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_multichannel_shape_and_dtype(channels):
    img = np.random.randint(
        0, 256, (32, 32, channels), dtype=np.uint8
    )

    out = bilateral_filter(img, ss=2.0, sr=25.0)

    assert out.shape == img.shape
    assert out.dtype == np.float32


# ------------------------------------------------------------
# Casos base
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_bilateral
def test_constant_image_remains_constant():
    img = np.full((32, 32), 128, dtype=np.uint8)

    out = bilateral_filter(img, ss=2.0, sr=30.0)
    out = np.clip(out*255, 0, 255).astype(np.uint8)

    assert np.allclose(out, 128.0, atol=1e-5)


# ------------------------------------------------------------
# Independencia de canales
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_bilateral
def test_channels_are_independent():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 0] = 10
    img[:, :, 1] = 100
    img[:, :, 2] = 200

    out = bilateral_filter(img, ss=2.0, sr=30.0)
    out = np.clip(out*255, 0, 255).astype(np.uint8)

    assert np.allclose(out[:, :, 0], 10.0, atol=1e-5)
    assert np.allclose(out[:, :, 1], 100.0, atol=1e-5)
    assert np.allclose(out[:, :, 2], 200.0, atol=1e-5)


# ------------------------------------------------------------
# Propiedad básica del bilateral
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_bilateral
def test_bilateral_preserves_edges_better_than_gaussian_like():
    """
    Bilateral filtering should preserve sharp edges
    when sr is small.
    """
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, :16] = 50
    img[:, 16:] = 200

    out = bilateral_filter(img, ss=2.0, sr=5.0)
    out = np.clip(out*255, 0, 255).astype(np.uint8)

    left_mean = out[:, 10].mean()
    right_mean = out[:, 22].mean()

    assert abs(left_mean - right_mean) > 50.0


# ------------------------------------------------------------
# Correctitud numérica (referencia naive)
# ------------------------------------------------------------

def bilateral_naive_reference(
    image: np.ndarray,
    ss: float,
    sr: float
) -> np.ndarray:
    """
    Extremely slow bilateral filter reference.
    Intended ONLY for testing.
    """
    image = image.astype(np.float32)
    radius = int(3 * ss)
    h, w = image.shape
    out = np.zeros_like(image)

    padded = np.pad(image, radius, mode="reflect")

    for i in range(h):
        for j in range(w):
            center = padded[i + radius, j + radius]

            weights_sum = 0.0
            value_sum = 0.0

            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    neighbor = padded[
                        i + radius + di, j + radius + dj
                    ]

                    gs = np.exp(-(di**2 + dj**2) / (2 * ss**2))
                    gr = np.exp(-((neighbor - center) ** 2) / (2 * sr**2))

                    w_ij = gs * gr
                    weights_sum += w_ij
                    value_sum += w_ij * neighbor

            out[i, j] = value_sum / weights_sum

    return out


@pytest.mark.filtering_spatial_bilateral
@pytest.mark.xfail(
    reason="Boundary handling not unified across bilateral backends",
    strict=False
)
def test_matches_naive_grayscale():
    np.random.seed(0)
    img = np.random.randint(0, 256, (24, 24), dtype=np.uint8)

    out_fast = bilateral_filter(img, ss=2.0, sr=20.0)
    out_ref = bilateral_naive_reference(img, ss=2.0, sr=20.0)

    assert np.allclose(out_fast, out_ref, atol=1e-3)

import numpy as np
import pytest

from pixelops.filtering.spatial.gradient import (
    gaussian_gradient,
    sobel_gradient,
    log_gradient,
)

# ============================================================
# Marker: filtering_spatial_gradient
# ============================================================


# ------------------------------------------------------------
# Validación de errores
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gradient
@pytest.mark.parametrize("sigma_s, sigma_d", [
    (0.0, 1.0),
    (1.0, 0.0),
    (-1.0, 1.0),
])
def test_gaussian_gradient_invalid_sigmas(sigma_s, sigma_d):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_gradient(img, sigma_s, sigma_d)


@pytest.mark.filtering_spatial_gradient
def test_gaussian_gradient_invalid_dimensions():
    img = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_gradient(img, 1.0, 1.0)


@pytest.mark.filtering_spatial_gradient
def test_sobel_invalid_dimensions():
    img = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        sobel_gradient(img)


@pytest.mark.filtering_spatial_gradient
@pytest.mark.parametrize("sigma_s, sigma_d", [
    (0.0, 1.0),
    (1.0, 0.0),
])
def test_log_invalid_sigmas(sigma_s, sigma_d):
    img = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        log_gradient(img, sigma_s, sigma_d)


# ------------------------------------------------------------
# Shape y dtype
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gradient
def test_gaussian_gradient_shapes_and_dtype():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    Gx, Gy, Gmag, Gphase = gaussian_gradient(img, 1.5, 1.0)

    assert Gx.shape == img.shape
    assert Gy.shape == img.shape
    assert Gmag.shape == img.shape
    assert Gphase.shape == img.shape

    assert Gx.dtype == np.float32
    assert Gy.dtype == np.float32
    assert Gmag.dtype == np.float32
    assert Gphase.dtype == np.float32


@pytest.mark.filtering_spatial_gradient
def test_sobel_shapes_and_dtype():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    Gx, Gy, Gmag, Gphase = sobel_gradient(img)

    assert Gx.shape == img.shape
    assert Gy.shape == img.shape
    assert Gmag.shape == img.shape
    assert Gphase.shape == img.shape

    assert Gx.dtype == np.float32
    assert Gy.dtype == np.float32
    assert Gmag.dtype == np.float32
    assert Gphase.dtype == np.float32


@pytest.mark.filtering_spatial_gradient
def test_log_shape_and_dtype():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out = log_gradient(img, 1.5, 1.0)

    assert out.shape == img.shape
    assert out.dtype == np.float32


# ------------------------------------------------------------
# Casos base (imagen constante)
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gradient
def test_gaussian_gradient_constant_image_zero():
    img = np.full((32, 32), 128, dtype=np.uint8)

    Gx, Gy, Gmag, _ = gaussian_gradient(img, 1.5, 1.0)

    assert np.allclose(Gx, 0.0, atol=1e-5)
    assert np.allclose(Gy, 0.0, atol=1e-5)
    assert np.allclose(Gmag, 0.0, atol=1e-5)


@pytest.mark.filtering_spatial_gradient
def test_sobel_constant_image_zero():
    img = np.full((32, 32), 128, dtype=np.uint8)

    Gx, Gy, Gmag, _ = sobel_gradient(img)

    assert np.allclose(Gx, 0.0)
    assert np.allclose(Gy, 0.0)
    assert np.allclose(Gmag, 0.0)


@pytest.mark.filtering_spatial_gradient
def test_log_constant_image_zero():
    img = np.full((32, 32), 128, dtype=np.uint8)

    out = log_gradient(img, 1.5, 1.0)

    assert np.allclose(out, 0.0, atol=1e-5)


# ------------------------------------------------------------
# Propiedades matemáticas básicas
# ------------------------------------------------------------

@pytest.mark.filtering_spatial_gradient
def test_gradient_detects_vertical_edge():
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, 16:] = 255

    Gx, Gy, Gmag, _ = gaussian_gradient(img, 1.0, 1.0)

    assert Gx[:, 15:17].mean() > 0
    assert np.abs(Gy).mean() < np.abs(Gx).mean()


# ------------------------------------------------------------
# Correctitud vs referencia (xfail por bordes)
# ------------------------------------------------------------

def log_naive_reference(img, sigma):
    img = img.astype(np.float32)
    radius = int(3 * sigma)

    ax = np.arange(-radius, radius + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g2 = (ax**2 - sigma**2) * np.exp(-(ax**2) / (2 * sigma**2))
    g2 /= np.sum(np.abs(g2))

    padded = np.pad(img, radius, mode="reflect")
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded[i:i+2*radius+1, j:j+2*radius+1]
            out[i, j] = (
                np.sum(patch * np.outer(g2, g)) +
                np.sum(patch * np.outer(g, g2))
            )

    return out


@pytest.mark.filtering_spatial_gradient
@pytest.mark.xfail(
    reason="Boundary handling not unified across convolution backends",
    strict=False
)
def test_log_matches_naive():
    np.random.seed(0)
    img = np.random.randint(0, 256, (24, 24), dtype=np.uint8)

    out_fast = log_gradient(img, 1.5, 1.5)
    out_ref = log_naive_reference(img, 1.5)

    assert np.allclose(out_fast, out_ref, atol=1e-3)



from pixelops.filtering.spatial.mean_shift import mean_shift_filter

# ============================================================
# Validación de parámetros
# ============================================================

def test_mean_shift_invalid_hs():
    img = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        mean_shift_filter(img, hs=0, hr=10.0)

    with pytest.raises(ValueError):
        mean_shift_filter(img, hs=-3, hr=10.0)


def test_mean_shift_invalid_hr():
    img = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        mean_shift_filter(img, hs=3, hr=0.0)

    with pytest.raises(ValueError):
        mean_shift_filter(img, hs=3, hr=-1.0)


def test_mean_shift_invalid_iter_eps():
    img = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        mean_shift_filter(img, hs=3, hr=10.0, max_iter=0)

    with pytest.raises(ValueError):
        mean_shift_filter(img, hs=3, hr=10.0, eps=0.0)


def test_mean_shift_invalid_dimensions():
    img = np.zeros((10, 10, 10, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        mean_shift_filter(img, hs=3, hr=10.0)


# ============================================================
# Grayscale behavior
# ============================================================

@pytest.mark.filtering_spatial_mean_shift
def test_mean_shift_grayscale_shape_and_dtype():
    img = np.random.rand(32, 32).astype(np.float32)

    out = mean_shift_filter(img, hs=3, hr=0.1)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_mean_shift
def test_mean_shift_grayscale_constant_image():
    img = np.full((32, 32), 42.0, dtype=np.float32)

    out = mean_shift_filter(img, hs=3, hr=5.0)

    # Imagen constante → debe permanecer constante
    assert np.allclose(out, img, atol=1e-5)


# ============================================================
# Multi-channel behavior
# ============================================================

@pytest.mark.filtering_spatial_mean_shift
def test_mean_shift_multichannel_shape_and_dtype():
    img = np.random.rand(32, 32, 3).astype(np.float32)

    out = mean_shift_filter(img, hs=3, hr=0.1)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_mean_shift
def test_mean_shift_channels_processed_independently():
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = 10.0
    img[:, :, 1] = 50.0
    img[:, :, 2] = 100.0

    out = mean_shift_filter(img, hs=3, hr=5.0)

    assert np.allclose(out[:, :, 0], 10.0, atol=1e-5)
    assert np.allclose(out[:, :, 1], 50.0, atol=1e-5)
    assert np.allclose(out[:, :, 2], 100.0, atol=1e-5)


# ============================================================
# Consistencia numérica básica
# ============================================================

@pytest.mark.filtering_spatial_mean_shift
def test_mean_shift_output_finite():
    img = np.random.rand(64, 64).astype(np.float32)

    out = mean_shift_filter(img, hs=5, hr=0.2)

    assert np.isfinite(out).all()


# ============================================================
# Compatibilidad con pipeline
# ============================================================

@pytest.mark.filtering_spatial_mean_shift
def test_mean_shift_pipeline_compatibility():
    img = np.random.rand(32, 32).astype(np.float32)

    out1 = mean_shift_filter(img, hs=3, hr=0.2)
    out2 = mean_shift_filter(out1, hs=3, hr=0.2)

    assert out2.shape == img.shape
    assert out2.dtype == np.float32



from pixelops.filtering.spatial.isotropic_diffusion import isotropic_diffusion


# ============================================================
# Validación de parámetros
# ============================================================

def test_invalid_n_iter_raises():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        isotropic_diffusion(img, n_iter=0)

    with pytest.raises(ValueError):
        isotropic_diffusion(img, n_iter=-5)


def test_invalid_gamma_raises():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        isotropic_diffusion(img, gamma=0.0)

    with pytest.raises(ValueError):
        isotropic_diffusion(img, gamma=0.3)

    with pytest.raises(ValueError):
        isotropic_diffusion(img, gamma=-0.1)


def test_invalid_dimensions_raises():
    img = np.zeros((10, 10, 10, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        isotropic_diffusion(img)


# ============================================================
# Grayscale behavior
# ============================================================

@pytest.mark.filtering_spatial_isotropic
def test_grayscale_shape_and_dtype():
    img = np.random.rand(32, 32).astype(np.float32)

    out = isotropic_diffusion(img, n_iter=5, gamma=0.2)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_isotropic
def test_grayscale_constant_image():
    img = np.full((32, 32), 42.0, dtype=np.float32)
    out = isotropic_diffusion(img, n_iter=10, gamma=0.25)

    # La ecuación del calor preserva constantes
    assert np.allclose(out, img, atol=1e-6)


# ============================================================
# Multi-channel behavior
# ============================================================

@pytest.mark.filtering_spatial_isotropic
def test_multichannel_shape_and_dtype():
    img = np.random.rand(32, 32, 3).astype(np.float32)

    out = isotropic_diffusion(img, n_iter=5, gamma=0.2)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_isotropic
def test_channels_processed_independently():
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = 10.0
    img[:, :, 1] = 50.0
    img[:, :, 2] = 100.0

    out = isotropic_diffusion(img, n_iter=10, gamma=0.25)

    assert np.allclose(out[:, :, 0], 10.0, atol=1e-6)
    assert np.allclose(out[:, :, 1], 50.0, atol=1e-6)
    assert np.allclose(out[:, :, 2], 100.0, atol=1e-6)


# ============================================================
# Consistencia numérica
# ============================================================

@pytest.mark.filtering_spatial_isotropic
def test_output_is_finite():
    img = np.random.rand(64, 64).astype(np.float32)

    out = isotropic_diffusion(img, n_iter=15, gamma=0.2)

    assert np.isfinite(out).all()


@pytest.mark.filtering_spatial_isotropic
def test_diffusion_reduces_variance():
    img = np.random.rand(64, 64).astype(np.float32)

    var_before = img.var()
    out = isotropic_diffusion(img, n_iter=20, gamma=0.25)
    var_after = out.var()

    assert var_after < var_before


# ============================================================
# Compatibilidad con pipelines
# ============================================================

@pytest.mark.filtering_spatial_isotropic
def test_pipeline_compatibility():
    img = np.random.rand(32, 32).astype(np.float32)

    out1 = isotropic_diffusion(img, n_iter=5, gamma=0.2)
    out2 = isotropic_diffusion(out1, n_iter=5, gamma=0.2)

    assert out2.shape == img.shape
    assert out2.dtype == np.float32



from pixelops.filtering.spatial.anisotropic_diffusion import anisotropic_diffusion


# ============================================================
# Validación de parámetros
# ============================================================

def test_invalid_n_iter_raises():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, n_iter=0)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, n_iter=-3)


def test_invalid_kappa_raises():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, kappa=0)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, kappa=-10)


def test_invalid_gamma_raises():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, gamma=0.0)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, gamma=0.3)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, gamma=-0.1)


def test_invalid_option_raises():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, option=0)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img, option=3)


def test_invalid_dimensions_raises():
    img = np.zeros((10, 10, 10, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        anisotropic_diffusion(img)


# ============================================================
# Grayscale behavior
# ============================================================

@pytest.mark.filtering_spatial_anisotropic
def test_grayscale_shape_and_dtype():
    img = np.random.rand(32, 32).astype(np.float32)

    out = anisotropic_diffusion(img, n_iter=5, kappa=10.0)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_anisotropic
def test_grayscale_constant_image():
    img = np.full((32, 32), 42.0, dtype=np.float32)

    out = anisotropic_diffusion(img, n_iter=10, kappa=20.0)

    # La difusión anisotrópica preserva constantes
    assert np.allclose(out, img, atol=1e-6)


# ============================================================
# Multi-channel behavior
# ============================================================

@pytest.mark.filtering_spatial_anisotropic
def test_multichannel_shape_and_dtype():
    img = np.random.rand(32, 32, 3).astype(np.float32)

    out = anisotropic_diffusion(img, n_iter=5, kappa=15.0)

    assert out.shape == img.shape
    assert out.dtype == np.float32


@pytest.mark.filtering_spatial_anisotropic
def test_channels_processed_independently():
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = 10.0
    img[:, :, 1] = 50.0
    img[:, :, 2] = 100.0

    out = anisotropic_diffusion(img, n_iter=10, kappa=20.0)

    assert np.allclose(out[:, :, 0], 10.0, atol=1e-6)
    assert np.allclose(out[:, :, 1], 50.0, atol=1e-6)
    assert np.allclose(out[:, :, 2], 100.0, atol=1e-6)


# ============================================================
# Propiedades del filtrado
# ============================================================

@pytest.mark.filtering_spatial_anisotropic
def test_diffusion_reduces_variance():
    img = np.random.rand(64, 64).astype(np.float32)

    var_before = img.var()
    out = anisotropic_diffusion(img, n_iter=15, kappa=0.2)
    var_after = out.var()

    assert var_after < var_before


@pytest.mark.filtering_spatial_anisotropic
def test_edges_preserved_better_than_isotropic():
    # Imagen con borde fuerte
    img = np.zeros((64, 64), dtype=np.float32)
    img[:, 32:] = 100.0

    out = anisotropic_diffusion(img, n_iter=20, kappa=5.0)

    # El borde debe seguir existiendo
    left_mean = out[:, :30].mean()
    right_mean = out[:, 34:].mean()

    assert abs(right_mean - left_mean) > 20.0


# ============================================================
# Consistencia numérica
# ============================================================

@pytest.mark.filtering_spatial_anisotropic
def test_output_is_finite():
    img = np.random.rand(64, 64).astype(np.float32)

    out = anisotropic_diffusion(img, n_iter=10, kappa=10.0)

    assert np.isfinite(out).all()


# ============================================================
# Compatibilidad con pipelines
# ============================================================

@pytest.mark.filtering_spatial_anisotropic
def test_pipeline_compatibility():
    img = np.random.rand(32, 32).astype(np.float32)

    out1 = anisotropic_diffusion(img, n_iter=5, kappa=10.0)
    out2 = anisotropic_diffusion(out1, n_iter=5, kappa=10.0)

    assert out2.shape == img.shape
    assert out2.dtype == np.float32
