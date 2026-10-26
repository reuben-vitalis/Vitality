"""Tests for the AI-image detector (model replaced by a stub so tests run offline and fast)."""
import io

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

import detectors.ai_image as ai


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(ai, "model_probability", lambda img: 0.5)


def photo_like(seed=0, size=(512, 384)):
    rng = np.random.default_rng(seed)
    w, h = size
    x, y = np.meshgrid(np.linspace(0, 6, w), np.linspace(0, 4, h))
    base = np.stack([np.sin(x + i) * np.cos(y * (i + 1)) for i in range(3)], -1)
    return Image.fromarray(((base + 1) * 110 + rng.normal(0, 10, (h, w, 3))).clip(0, 255).astype(np.uint8))


def test_stable_diffusion_prompt_in_png_is_evidence(tmp_path):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a cat on a sofa, Steps: 30, Sampler: Euler a, Model: sdxl")
    path = tmp_path / "sd.png"
    photo_like().save(path, pnginfo=meta)
    r = ai.analyse(str(path))
    assert r["verdict"] == "Likely AI-generated"
    assert any("Generation settings" in e for e in r["evidence"])


def test_plain_image_has_no_evidence_and_uses_model():
    r = ai.analyse(photo_like())
    assert r["evidence"] == [] and r["model_probability"] == 0.5
    assert 0 <= r["ai_probability"] <= 1 and "model" in r["basis"]


def test_fft_returns_score_and_spectrum():
    score, spec = ai.fft_analysis(photo_like())
    assert 0 <= score <= 1 and spec.size == (512, 512)


def test_grid_artifacts_raise_fft_score():
    clean = photo_like(1, size=(768, 640))
    arr = np.asarray(clean).astype(float)
    arr[::4, ::4] += 60                       # simulate a regular upsampling grid
    grid = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    assert ai.fft_analysis(grid)[0] > ai.fft_analysis(clean)[0]
