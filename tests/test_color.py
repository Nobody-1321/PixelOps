import numpy as np
import pytest

from pixelops.color.reinhard import (
    bgr_to_lab_real,
    lab_real_to_bgr,
    reinhard_color_transfer,
    reinhard_color_transfer_controlled,
)

# ============================================================
# Helpers
# ============================================================

def random_bgr(h=64, w=64):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ============================================================
# bgr_to_lab_real
# ============================================================

@pytest.mark.color_reinhard
def test_bgr_to_lab_real_invalid_type():
    with pytest.raises(TypeError):
        bgr_to_lab_real([1, 2, 3])


@pytest.mark.color_reinhard
def test_bgr_to_lab_real_invalid_shape():
    img = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        bgr_to_lab_real(img)


@pytest.mark.color_reinhard
def test_bgr_to_lab_real_invalid_dtype():
    img = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        bgr_to_lab_real(img)


@pytest.mark.color_reinhard
def test_bgr_to_lab_real_output_shape_and_dtype():
    img = random_bgr()
    lab = bgr_to_lab_real(img)

    assert lab.shape == img.shape
    assert lab.dtype == np.float32


@pytest.mark.color_reinhard
def test_bgr_to_lab_real_ranges():
    img = random_bgr(128, 128)
    lab = bgr_to_lab_real(img)

    assert np.all(lab[..., 0] >= 0.0)
    assert np.all(lab[..., 0] <= 100.0)
    assert np.all(lab[..., 1] >= -128.0)
    assert np.all(lab[..., 1] <= 127.0)
    assert np.all(lab[..., 2] >= -128.0)
    assert np.all(lab[..., 2] <= 127.0)


# ============================================================
# lab_real_to_bgr
# ============================================================

@pytest.mark.color_reinhard
def test_lab_real_to_bgr_invalid_type():
    with pytest.raises(TypeError):
        lab_real_to_bgr([1, 2, 3])


@pytest.mark.color_reinhard
def test_lab_real_to_bgr_invalid_shape():
    lab = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        lab_real_to_bgr(lab)


@pytest.mark.color_reinhard
def test_lab_real_to_bgr_invalid_dtype():
    lab = np.zeros((32, 32, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        lab_real_to_bgr(lab)


@pytest.mark.color_reinhard
def test_lab_real_to_bgr_output_shape_and_dtype():
    lab = np.zeros((32, 32, 3), dtype=np.float32)
    bgr = lab_real_to_bgr(lab)

    assert bgr.shape == lab.shape
    assert bgr.dtype == np.uint8


# ============================================================
# Round-trip consistency
# ============================================================

@pytest.mark.color_reinhard
def test_bgr_lab_roundtrip_lab_consistency():
    img = random_bgr(128, 128)

    lab1 = bgr_to_lab_real(img)
    lab2 = bgr_to_lab_real(lab_real_to_bgr(lab1))

    diff = np.abs(lab1 - lab2)

    assert np.max(diff[..., 0]) <= 1.0      # L*
    assert np.max(diff[..., 1]) <= 2.0      # a*
    assert np.max(diff[..., 2]) <= 2.0      # b*

@pytest.mark.color_reinhard
def test_reinhard_controlled_identity_in_lab_when_alpha_zero():
    src = random_bgr()
    tgt = random_bgr()

    out = reinhard_color_transfer_controlled(
        src,
        tgt,
        alpha_L=0.0,
        alpha_ab=0.0
    )

    src_lab = bgr_to_lab_real(src)
    out_lab = bgr_to_lab_real(out)

    diff = np.abs(src_lab - out_lab)

    assert np.max(diff[..., 0]) <= 1.0
    assert np.max(diff[..., 1]) <= 2.0
    assert np.max(diff[..., 2]) <= 2.0


# ============================================================
# Reinhard color transfer (classic)
# ============================================================

@pytest.mark.color_reinhard
def test_reinhard_invalid_type():
    with pytest.raises(TypeError):
        reinhard_color_transfer([1, 2, 3], [1, 2, 3])


@pytest.mark.color_reinhard
def test_reinhard_invalid_shape():
    src = np.zeros((32, 32), dtype=np.uint8)
    tgt = random_bgr()

    with pytest.raises(ValueError):
        reinhard_color_transfer(src, tgt)


@pytest.mark.color_reinhard
def test_reinhard_output_shape_and_dtype():
    src = random_bgr()
    tgt = random_bgr()

    out = reinhard_color_transfer(src, tgt)

    assert out.shape == src.shape
    assert out.dtype == np.uint8


@pytest.mark.color_reinhard
def test_reinhard_matches_target_statistics():
    src = random_bgr(128, 128)
    tgt = random_bgr(128, 128)

    out = reinhard_color_transfer(src, tgt)

    out_lab = bgr_to_lab_real(out)
    tgt_lab = bgr_to_lab_real(tgt)

    for c in range(3):
        assert np.isclose(
            np.mean(out_lab[..., c]),
            np.mean(tgt_lab[..., c]),
            atol=1.0
        )

        assert np.isclose(
            np.std(out_lab[..., c]),
            np.std(tgt_lab[..., c]),
            atol=1.0
        )


# ============================================================
# Reinhard color transfer (controlled)
# ============================================================


@pytest.mark.color_reinhard
def test_reinhard_controlled_output_shape_and_dtype():
    src = random_bgr()
    tgt = random_bgr()

    out = reinhard_color_transfer_controlled(src, tgt)

    assert out.shape == src.shape
    assert out.dtype == np.uint8


@pytest.mark.color_reinhard
def test_reinhard_controlled_value_range():
    src = random_bgr(128, 128)
    tgt = random_bgr(128, 128)

    out = reinhard_color_transfer_controlled(src, tgt)

    assert np.min(out) >= 0
    assert np.max(out) <= 255


# ============================================================
# Edge cases
# ============================================================

@pytest.mark.color_reinhard
def test_constant_image_stability():
    src = np.full((64, 64, 3), 128, dtype=np.uint8)
    tgt = random_bgr(64, 64)

    out = reinhard_color_transfer(src, tgt)

    assert out.dtype == np.uint8
    assert out.shape == src.shape
