"""Vitality — AI-generated image detector.

Combines three independent signals into one verdict:
  1. Evidence in the file  - AI-tool tags, generation prompts, C2PA content credentials,
                             and the public Stable Diffusion watermark (from detectors.watermark).
  2. A pretrained model    - Organika/sdxl-detector (Swin transformer, 98.1% validation accuracy
                             on SDXL images; licence CC-BY-NC-3.0, educational use).
  3. A frequency check     - FFT spectrum; AI upsampling often leaves regular grid-like peaks.

Lesson from the Week 9 lab: missing camera EXIF is NOT treated as evidence of AI,
because messaging apps (e.g. WhatsApp) strip EXIF from real photos.
"""
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

from detectors.watermark import _check_stable_diffusion, _to_bgr, MIN_PIXELS

MODEL_ID = "Organika/sdxl-detector"
AI_WORDS = ["stable diffusion", "midjourney", "dall-e", "dall·e", "dalle", "firefly", "comfyui",
            "automatic1111", "novelai", "leonardo", "ideogram", "flux", "imagen", "sdxl", "openai",
            "generative fill", "ai generated", "ai-generated"]
PROMPT_KEYS = ["parameters", "prompt", "workflow", "negative_prompt", "sd-metadata", "dream"]


def _open(source):
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
        return Image.open(io.BytesIO(raw)), raw
    return source, None


# ---------- Signal 1: evidence in the file ----------
def file_evidence(img: Image.Image, raw: bytes | None) -> dict:
    found, camera = [], None
    exif = img.getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()} if exif else {}
    if tags.get("Make") or tags.get("Model"):
        camera = f"{tags.get('Make', '')} {tags.get('Model', '')}".strip()
    texts = [str(v) for k, v in tags.items() if k in ("Software", "ImageDescription", "Artist", "UserComment")]
    for k, v in (img.info or {}).items():          # PNG text chunks, e.g. Stable Diffusion "parameters"
        if isinstance(v, (str, bytes)):
            if str(k).lower() in PROMPT_KEYS:
                found.append(f"Generation settings stored in file ('{k}')")
            texts.append(v.decode("latin-1", "ignore") if isinstance(v, bytes) else v)
    blob = " ".join(texts).lower()
    found += [f"AI tool named in metadata: '{w}'" for w in AI_WORDS if w in blob]
    if raw is not None:
        head = raw[:2_000_000].lower()
        if b"c2pa" in head or b"jumb" in head:
            found.append("C2PA content credentials present (check them at contentcredentials.org/verify)")
            if b"trainedalgorithmicmedia" in head:
                found.append("C2PA declares the image as AI-generated (trainedAlgorithmicMedia)")
    bgr = _to_bgr(img)
    if bgr.shape[0] * bgr.shape[1] >= MIN_PIXELS:
        sd = _check_stable_diffusion(bgr)
        if sd:
            found.append(f"Hidden {sd} watermark detected")
    return {"evidence": sorted(set(found)), "camera": camera}


# ---------- Signal 2: pretrained model ----------
try:                                    # Hugging Face ZeroGPU: borrow a GPU only while the model runs
    import spaces
    _gpu = spaces.GPU(duration=30)
except ImportError:                     # running locally: no-op decorator
    def _gpu(fn):
        return fn


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline   # imported lazily: slow to load
    return pipeline("image-classification", model=MODEL_ID)


@_gpu
def _run_model(img: Image.Image):
    return _classifier()(img.convert("RGB"))


def model_probability(img: Image.Image) -> float | None:
    """Probability (0-1) that the image is AI-generated, or None if the model is unavailable."""
    try:
        preds = _run_model(img)
    except Exception as exc:  # e.g. no internet on first run
        print(f"[ai_image] model unavailable: {exc}")
        return None
    ai_like = ("artificial", "ai", "fake", "generated", "synthetic")
    for p in preds:
        if any(w in p["label"].lower() for w in ai_like):
            return float(p["score"])
    real = [p for p in preds if any(w in p["label"].lower() for w in ("human", "real"))]
    return 1 - float(real[0]["score"]) if real else None


# ---------- Signal 3: frequency (FFT) check ----------
def fft_analysis(img: Image.Image, size: int = 512):
    """Return (artifact score 0-1, spectrum image). Higher score = sharper grid-like peaks
    in the high frequencies, a common trace of AI upsampling layers."""
    g = img.convert("L")
    w, h = g.size
    if min(w, h) >= size:                                     # crop, don't resize: resizing hides the grid
        left, top = (w - size) // 2, (h - size) // 2
        g = g.crop((left, top, left + size, top + size))
    else:
        g = g.resize((size, size))
    g = np.asarray(g, dtype=np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(size), np.hanning(size))
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    c = size // 2
    yy, xx = np.ogrid[:size, :size]
    r = np.hypot(yy - c, xx - c)
    rb = r.astype(int)
    # Remove the normal smooth fall-off of a photo's spectrum, leaving only isolated peaks.
    profile = np.array([np.median(mag[rb == k]) if (rb == k).any() else 0.0 for k in range(rb.max() + 1)])
    residual = mag - profile[rb]
    band = residual[(r > size * 0.2) & (r < size * 0.49)]
    peakiness = float(band.max() / (band.std() + 1e-6))      # real photos measured ~2.5-4.5
    score = float(np.clip((peakiness - 4.5) / 4.0, 0, 1))
    spec = ((mag - mag.min()) / (np.ptp(mag) + 1e-6) * 255).astype(np.uint8)
    return round(score, 3), Image.fromarray(spec)


# ---------- Combined verdict ----------
def analyse(source) -> dict:
    img, raw = _open(source)
    ev = file_evidence(img, raw)
    p_model = model_probability(img)
    fft_score, spectrum = fft_analysis(img)

    if ev["evidence"]:
        prob, basis = 0.99, "direct evidence found in the file"
    elif p_model is not None:
        prob, basis = 0.85 * p_model + 0.15 * fft_score, "model (85%) + frequency check (15%)"
    else:
        prob, basis = fft_score, "frequency check only (model unavailable)"
    if ev["camera"] and not ev["evidence"]:
        prob, basis = max(0.0, prob - 0.10), basis + "; camera metadata present (-10%)"

    verdict = ("Likely AI-generated" if prob >= 0.70 else
               "Uncertain - treat with caution" if prob >= 0.40 else "Likely a real photo")
    return {
        "verdict": verdict,
        "ai_probability": round(prob, 3),
        "basis": basis,
        "evidence": ev["evidence"],
        "camera": ev["camera"],
        "model_probability": None if p_model is None else round(p_model, 3),
        "fft_score": fft_score,
        "spectrum": spectrum,
    }


if __name__ == "__main__":
    import sys
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        paths += sorted(f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if p.is_dir() else [p]
    if not paths:
        print("Usage: python -m detectors.ai_image <image or folder> [...]")
    for path in paths:
        r = analyse(str(path))
        print(f"{path.name:<28} {r['verdict']:<32} AI={r['ai_probability']:.2f}  "
              f"model={r['model_probability']}  fft={r['fft_score']}  evidence={r['evidence']}")
