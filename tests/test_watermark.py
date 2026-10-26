"""Tests for the watermark detector. Run: python -m pytest -q"""
import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

import detectors.watermark as wm


@pytest.fixture(autouse=True)
def test_key(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", "test-key-for-pytest-only")


def photo_like(seed=0, size=(512, 384)):
    """A textured test image (smooth gradients + detail), similar to a photo."""
    rng = np.random.default_rng(seed)
    w, h = size
    x, y = np.meshgrid(np.linspace(0, 6, w), np.linspace(0, 4, h))
    base = np.stack([np.sin(x + i) * np.cos(y * (i + 1)) for i in range(3)], -1)
    arr = ((base + 1) * 110 + rng.normal(0, 12, (h, w, 3))).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1))


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue()))


def test_embed_then_verify_is_authentic():
    marked, info = wm.embed(photo_like(), "RVT")
    r = wm.verify(marked)
    assert r["status"] == "authentic" and r["owner"] == "RVT" and r["date"] == info["date"]
    assert info["quality_psnr_db"] > 30  # invisible to the eye


def test_survives_jpeg_compression():
    marked, _ = wm.embed(photo_like(1), "ABC")
    assert wm.verify(jpeg(marked, 80))["status"] == "authentic"


def test_survives_whatsapp_style_resize():
    marked, _ = wm.embed(photo_like(4, size=(1600, 1200)), "RVT")
    shrunk = marked.resize((800, 600))  # e.g. a messaging app halving the size
    assert wm.verify(jpeg(shrunk, 75))["status"] == "authentic"


def test_wrong_key_reports_forged(monkeypatch):
    marked, _ = wm.embed(photo_like(2), "RVT")
    monkeypatch.setenv("WATERMARK_KEY", "a-different-secret")
    assert wm.verify(marked)["status"] == "forged"


def test_plain_image_has_no_watermark():
    assert wm.verify(photo_like(3))["status"] == "none"


def test_small_image_rejected():
    with pytest.raises(ValueError):
        wm.embed(photo_like(size=(200, 200)), "RVT")
