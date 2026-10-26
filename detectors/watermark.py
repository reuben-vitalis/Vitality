"""Vitality — Watermark detector.

Embeds and verifies an invisible, cryptographically signed watermark.

Watermark payload (20 bytes, hidden in the image's frequency domain):
    b"V"          1 byte   magic marker ("this is a Vitality watermark")
    day           2 bytes  date stamped, as days since 2020-01-01
    owner         3 bytes  owner tag, up to 3 letters/digits (e.g. "RVT")
    tag           6 bytes  HMAC-SHA256(secret key, first 6 bytes), truncated to 48 bits
    + 8 bytes Reed-Solomon error correction (repairs bits damaged by JPEG/resizing)

Verification outcomes:
    authentic  - watermark found and its HMAC signature is valid for our key
    forged     - a Vitality-style watermark is present but the signature is wrong
    none       - no Vitality watermark found
It also checks for the public watermark that Stable Diffusion embeds in AI images.
"""
import datetime as dt
import hashlib
import hmac
import io
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image
from reedsolo import ReedSolomonError, RSCodec

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

METHOD = "dwtDctSvd"
SCALES = [0, 50, 0]          # embedding strength (tested: survives JPEG 70 and 50% resize)
MAGIC = b"V"
EPOCH = dt.date(2020, 1, 1)
DATA_LEN, ECC_LEN = 12, 8
RS = RSCodec(ECC_LEN)
MIN_PIXELS = 256 * 256
WORK_SIDE = 512              # watermark lives at a fixed working size, so resized copies still verify

# Public Stable Diffusion watermarks (dwtDct method)
SD1_MESSAGE = b"StableDiffusionV1"
SDXL_BITS = [int(b) for b in "101100111110110010010000011110111011000110011110"]


def _key() -> bytes:
    key = os.getenv("WATERMARK_KEY", "")
    if not key or key.startswith("replace-with"):
        raise RuntimeError("WATERMARK_KEY is not set. Add it to your .env file (or as a Hugging Face secret).")
    return key.encode()


def _to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _work_size(bgr: np.ndarray) -> tuple:
    """(width, height) scaled so the longer side is WORK_SIDE pixels, keeping the aspect ratio."""
    h, w = bgr.shape[:2]
    s = WORK_SIDE / max(h, w)
    return max(1, round(w * s)), max(1, round(h * s))


def _to_work(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(bgr, _work_size(bgr), interpolation=cv2.INTER_AREA)


def _open(source):
    """Accept a file path or a PIL image. Returns (PIL image, raw file bytes or None)."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
        return Image.open(io.BytesIO(raw)), raw
    return source, None


def sha256_fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sign(body: bytes) -> bytes:
    return hmac.new(_key(), body, hashlib.sha256).digest()[:6]


def _clean_owner(owner: str) -> bytes:
    tag = "".join(ch for ch in (owner or "").upper() if ch.isalnum())[:3]
    if not tag:
        raise ValueError("Owner tag must contain 1-3 letters or digits, e.g. RVT.")
    return tag.ljust(3).encode("ascii")


def embed(source, owner: str, date: dt.date | None = None):
    """Return (watermarked PIL image, info dict). Save the result as PNG to keep it intact."""
    img, _ = _open(source)
    bgr = _to_bgr(img)
    if bgr.shape[0] * bgr.shape[1] < MIN_PIXELS:
        raise ValueError("Image is too small to watermark (needs at least 256 x 256 pixels).")
    date = date or dt.date.today()
    body = MAGIC + (date - EPOCH).days.to_bytes(2, "big") + _clean_owner(owner)
    payload = bytes(RS.encode(body + _sign(body)))
    enc = WatermarkEncoder()
    enc.set_watermark("bytes", payload)
    # Embed at the working size, then add the (invisible) difference back onto the full-size image.
    small = _to_work(bgr)
    if small.shape[0] * small.shape[1] < MIN_PIXELS:
        raise ValueError("Image shape is too narrow to watermark (needs roughly 256 x 256 pixels or more).")
    residual = enc.encode(small, METHOD, scales=SCALES).astype(np.float32) - small.astype(np.float32)
    residual_full = cv2.resize(residual, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    out_bgr = np.clip(bgr.astype(np.float32) + residual_full, 0, 255).round().astype(np.uint8)
    out = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return out, {
        "owner": body[3:6].decode().strip(),
        "date": date.isoformat(),
        "quality_psnr_db": round(float(cv2.PSNR(bgr, out_bgr)), 1),
        "sha256": sha256_fingerprint(buf.getvalue()),
    }


def _check_vitality(bgr: np.ndarray) -> dict:
    small = _to_work(bgr)
    if small.shape[0] * small.shape[1] < MIN_PIXELS:
        return {"status": "none"}
    raw = WatermarkDecoder("bytes", (DATA_LEN + ECC_LEN) * 8).decode(small, METHOD, scales=SCALES)
    try:
        data = bytes(RS.decode(raw)[0])
    except ReedSolomonError:
        return {"status": "none"}
    body, tag = data[:6], data[6:]
    if body[:1] != MAGIC:
        return {"status": "none"}
    info = {
        "owner": body[3:6].decode("ascii", "replace").strip(),
        "date": (EPOCH + dt.timedelta(days=int.from_bytes(body[1:3], "big"))).isoformat(),
    }
    ok = hmac.compare_digest(tag, _sign(body))
    return {"status": "authentic" if ok else "forged", **info}


def _check_stable_diffusion(bgr: np.ndarray) -> str | None:
    """Detect the public watermark Stable Diffusion adds to its outputs (fragile: lost after editing)."""
    bits = np.array(WatermarkDecoder("bits", 48).decode(bgr, "dwtDct"))
    if (bits == np.array(SDXL_BITS)).sum() >= 44:
        return "Stable Diffusion XL"
    sd1 = WatermarkDecoder("bytes", len(SD1_MESSAGE) * 8).decode(bgr, "dwtDct")
    matches = sum(8 - bin(a ^ b).count("1") for a, b in zip(sd1, SD1_MESSAGE))
    if matches >= len(SD1_MESSAGE) * 8 * 0.9:
        return "Stable Diffusion 1.x/2.x"
    return None


def verify(source) -> dict:
    img, raw = _open(source)
    bgr = _to_bgr(img)
    result = _check_vitality(bgr)
    big_enough = bgr.shape[0] * bgr.shape[1] >= MIN_PIXELS
    result["ai_watermark"] = _check_stable_diffusion(bgr) if big_enough else None
    result["sha256"] = sha256_fingerprint(raw) if raw else None
    result["message"] = {
        "authentic": f"Authentic Vitality watermark: owner {result.get('owner')}, stamped {result.get('date')}. Signature valid.",
        "forged": "A watermark is present but its signature is INVALID: forged, altered or made with a different key.",
        "none": "No Vitality watermark found.",
    }[result["status"]]
    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python -m detectors.watermark path\\to\\image.jpg")
    else:
        marked, info = embed(path, "RVT")
        marked.save("samples/watermarked.png")
        print("Embedded:", info)
        print("Verify:  ", verify("samples/watermarked.png"))
        print("Original:", verify(path))
