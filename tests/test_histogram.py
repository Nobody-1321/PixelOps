import pytest
import numpy as np
from pixelops.histogram.equalization import (
    histogram_equalization_channel,
    histogram_equalization,
    clahe,
)


# ============================================================
# HISTOGRAM EQUALIZATION CHANNEL
# ============================================================


# ------------------------------------------------------------
# Validación de errores
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_channel_invalid_type():
    with pytest.raises(TypeError):
        histogram_equalization_channel([1, 2, 3])


@pytest.mark.histogram_equalization
def test_equalization_channel_invalid_dimensions():
    img = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        histogram_equalization_channel(img)


@pytest.mark.histogram_equalization
def test_equalization_channel_invalid_dtype():
    img = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        histogram_equalization_channel(img)


# ------------------------------------------------------------
# Shape y dtype
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_channel_shape_and_dtype():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out = histogram_equalization_channel(img)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ------------------------------------------------------------
# Casos base
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_channel_constant_image():
    img = np.full((32, 32), 128, dtype=np.uint8)

    out = histogram_equalization_channel(img)

    # Imagen constante debe permanecer constante (o cercana)
    assert np.unique(out).size == 1


@pytest.mark.histogram_equalization
def test_equalization_channel_zero_image():
    img = np.zeros((32, 32), dtype=np.uint8)

    out = histogram_equalization_channel(img)

    # Imagen con todos ceros
    assert np.all(out == 0)


# ------------------------------------------------------------
# Propiedades matemáticas
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_channel_output_range():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    out = histogram_equalization_channel(img)

    assert out.min() >= 0
    assert out.max() <= 255


@pytest.mark.histogram_equalization
def test_equalization_channel_spreads_histogram():
    # Imagen con histograma concentrado
    img = np.random.randint(100, 150, (64, 64), dtype=np.uint8)

    out = histogram_equalization_channel(img)

    # Después de ecualizar, el rango debe ser mayor
    assert (out.max() - out.min()) >= (img.max() - img.min())


# ============================================================
# HISTOGRAM EQUALIZATION (GRAYSCALE Y BGR)
# ============================================================


# ------------------------------------------------------------
# Validación de errores
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_invalid_type():
    with pytest.raises(TypeError):
        histogram_equalization([1, 2, 3])


@pytest.mark.histogram_equalization
def test_equalization_invalid_dimensions():
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        histogram_equalization(img)


@pytest.mark.histogram_equalization
def test_equalization_1d_raises():
    img = np.zeros((16,), dtype=np.uint8)

    with pytest.raises(ValueError):
        histogram_equalization(img)


# ------------------------------------------------------------
# Shape y dtype
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_grayscale_shape_and_dtype():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    out = histogram_equalization(img)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.histogram_equalization
def test_equalization_bgr_shape_and_dtype():
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    out = histogram_equalization(img)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ------------------------------------------------------------
# Casos base
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_grayscale_constant():
    img = np.full((32, 32), 100, dtype=np.uint8)

    out = histogram_equalization(img)

    assert np.unique(out).size == 1


@pytest.mark.histogram_equalization
def test_equalization_bgr_constant():
    img = np.full((32, 32, 3), 100, dtype=np.uint8)

    out = histogram_equalization(img)

    # Imagen constante debería permanecer constante
    assert out.shape == img.shape


# ------------------------------------------------------------
# Propiedades
# ------------------------------------------------------------

@pytest.mark.histogram_equalization
def test_equalization_output_range():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    out = histogram_equalization(img)

    assert out.min() >= 0
    assert out.max() <= 255


@pytest.mark.histogram_equalization
def test_equalization_bgr_output_range():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    out = histogram_equalization(img)

    assert out.min() >= 0
    assert out.max() <= 255


# ============================================================
# CLAHE
# ============================================================


# ------------------------------------------------------------
# Validación de errores
# ------------------------------------------------------------

@pytest.mark.histogram_clahe
def test_clahe_invalid_dtype():
    img = np.zeros((32, 32), dtype=np.float32)

    with pytest.raises(TypeError):
        clahe(img)


@pytest.mark.histogram_clahe
def test_clahe_invalid_dimensions():
    img = np.zeros((32, 32, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        clahe(img)


@pytest.mark.histogram_clahe
def test_clahe_grid_larger_than_image():
    img = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        clahe(img, grid_size=(8, 8))


# ------------------------------------------------------------
# Shape y dtype
# ------------------------------------------------------------

@pytest.mark.histogram_clahe
def test_clahe_grayscale_shape_and_dtype():
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=(8, 8))

    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.histogram_clahe
def test_clahe_bgr_shape_and_dtype():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=(8, 8))

    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ------------------------------------------------------------
# Casos base
# ------------------------------------------------------------

@pytest.mark.histogram_clahe
def test_clahe_constant_image():
    img = np.full((64, 64), 128, dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=(8, 8))

    # Imagen constante debería permanecer constante
    assert np.unique(out).size == 1


@pytest.mark.histogram_clahe
def test_clahe_zero_image():
    img = np.zeros((64, 64), dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=(8, 8))

    assert np.all(out == 0)


@pytest.mark.histogram_clahe
def test_clahe_max_image():
    img = np.full((64, 64), 255, dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=(8, 8))

    assert np.all(out == 255)


# ------------------------------------------------------------
# Propiedades
# ------------------------------------------------------------

@pytest.mark.histogram_clahe
def test_clahe_output_range():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=(8, 8))

    assert out.min() >= 0
    assert out.max() <= 255


@pytest.mark.histogram_clahe
def test_clahe_improves_local_contrast():
    # Imagen con regiones de bajo contraste
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:32, :] = 50
    img[32:, :] = 55

    out = clahe(img, clip_limit=2, grid_size=(4, 4))

    # CLAHE debería mejorar el contraste local
    assert (out.max() - out.min()) >= (img.max() - img.min())


@pytest.mark.histogram_clahe
def test_clahe_bgr_output_range():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=(8, 8))

    assert out.min() >= 0
    assert out.max() <= 255


# ------------------------------------------------------------
# Parámetros
# ------------------------------------------------------------

@pytest.mark.histogram_clahe
@pytest.mark.parametrize("clip_limit", [2, 5, 10, 20])
def test_clahe_different_clip_limits(clip_limit):
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    out = clahe(img, clip_limit=clip_limit, grid_size=(8, 8))

    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.histogram_clahe
@pytest.mark.parametrize("grid_size", [(4, 4), (8, 8), (16, 16)])
def test_clahe_different_grid_sizes(grid_size):
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    out = clahe(img, clip_limit=10, grid_size=grid_size)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ------------------------------------------------------------
# Consistencia
# ------------------------------------------------------------

@pytest.mark.histogram_clahe
def test_clahe_deterministic():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    out1 = clahe(img, clip_limit=10, grid_size=(8, 8))
    out2 = clahe(img, clip_limit=10, grid_size=(8, 8))

    assert np.array_equal(out1, out2)
