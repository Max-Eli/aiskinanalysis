"""
CV Service — FastAPI

POST /analyze
  Body: { "image": "<base64 data URL>" }
  Returns: { image_quality, quality_issues, cropped_face, skin_coverage }
  The cropped_face is a base64 JPEG of just the face region for Gemini Vision.
"""

from __future__ import annotations
import base64
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from image_quality import check_quality
from face_segment import get_skin_mask
from skin_measure import measure_skin
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading MediaPipe Face Landmarker…")
    try:
        from face_segment import _load_model
        _load_model()
        log.info("MediaPipe Face Landmarker loaded.")
    except Exception as e:
        log.warning(f"Model warm-up failed (will retry on first request): {e}")
    yield


app = FastAPI(title="Skin CV Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    image: str  # data URL: "data:image/jpeg;base64,..."


def _decode_image(data_url: str) -> Image.Image:
    try:
        header, b64 = data_url.split(",", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image data URL format.")
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _crop_face(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Crop to face bounding box with 10% padding."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return img
    h, w = mask.shape
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    pad_y = int((y1 - y0) * 0.10)
    pad_x = int((x1 - x0) * 0.10)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    return img.crop((x0, y0, x1, y1))


def _to_base64(img: Image.Image) -> str:
    """Encode PIL image as base64 JPEG data URL."""
    max_side = 1024
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # 1. Decode
    try:
        img = _decode_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Image quality gate
    quality = check_quality(img)
    if quality["quality"] == "poor":
        return {
            "image_quality": "poor",
            "quality_issues": quality["issues"],
            "cropped_face": None,
            "skin_coverage": 0.0,
        }

    # 3. Face segmentation
    try:
        skin_mask = get_skin_mask(img)
    except Exception as e:
        log.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail="Segmentation failed.")

    skin_coverage = float(skin_mask.mean())
    log.info(f"Skin coverage: {skin_coverage:.3f}")

    if skin_coverage < 0.03:
        return {
            "image_quality": "poor",
            "quality_issues": ["No face detected. Ensure your face is clearly visible and well-lit."],
            "cropped_face": None,
            "skin_coverage": skin_coverage,
        }

    # 4. Objective pixel measurements (anchor for Gemini Vision)
    try:
        measurements = measure_skin(img, skin_mask)
    except Exception as e:
        log.warning(f"Measurements failed (non-fatal): {e}")
        measurements = {}

    log.info(f"CV measurements: spots={measurements.get('dark_spot_count')}, "
             f"redness={measurements.get('redness_index')}, "
             f"oiliness={measurements.get('oiliness_pct')}%")

    # 5. Crop face region for Gemini Vision
    face_crop = _crop_face(img, skin_mask)
    cropped_b64 = _to_base64(face_crop)

    return {
        "image_quality": quality["quality"],
        "quality_issues": quality["issues"],
        "cropped_face": cropped_b64,
        "skin_coverage": round(skin_coverage, 4),
        "cv_measurements": measurements,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
