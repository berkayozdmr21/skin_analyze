"""
GOZENEK.2 test uygulamasi - bagimsiz FastAPI sunucusu.

Gozenek.1 ile ayni iskelet, ama black-hat z-score'u artik TEK global
referans yerine YUZUN HER ALT-BOLGESI (alin/burun/cene/yanak) kendi
icinde ayri ayri hesaplaniyor (bkz. gozenek.py). Ana skin-scan
projesinden BAGIMSIZ calisir.

Calistirma:
    <skin-scan venv39 python> app.py
"""
import base64
import io
import logging

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from face_mesh import FaceMeshDetector, make_face_scan_mask, make_region_scan_masks
from preprocess import preprocess
from gozenek import pore_map, pore_score, find_marked_regions
from visualize import create_heatmap_overlay, draw_detection_markers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gozenek.2")

app = FastAPI(title="Gozenek.2 - Bolgesel Referans")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_image_bgr(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def encode_png_base64(img_bgr_or_bgra: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", img_bgr_or_bgra)
    if not success:
        raise ValueError("PNG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.get("/health")
async def health():
    return {"ok": True, "variant": "gozenek.2 - bolgesel referans"}


@app.post("/scan")
async def scan(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        logger.info(f"Received image: {image.filename}, size: {len(image_data)} bytes")

        img_bgr = read_image_bgr(image_data)
        img_processed = preprocess(img_bgr, max_size=1024)

        detector = FaceMeshDetector()
        landmarks = detector.detect(img_processed)
        if landmarks is None:
            raise HTTPException(status_code=400, detail="Fotografta yuz bulunamadi")

        face_scan_mask = make_face_scan_mask(landmarks, img_processed.shape)
        region_scan_masks = make_region_scan_masks(landmarks, img_processed.shape)

        if face_scan_mask is None:
            raise HTTPException(status_code=400, detail="Yuz cildi bolgesi tespit edilemedi")

        pmap = pore_map(img_processed, face_scan_mask, region_scan_masks)
        score, mean_pore_z, affected_area_pct, region_count, breakdown = pore_score(
            img_processed, face_scan_mask, region_scan_masks
        )
        contours = find_marked_regions(img_processed, face_scan_mask, region_scan_masks)

        overlay_rgba = create_heatmap_overlay(pmap, colormap=cv2.COLORMAP_HOT, alpha=0.6)
        marked_bgr = draw_detection_markers(img_processed, contours, color=(0, 255, 255))

        overlay_b64 = encode_png_base64(overlay_rgba)
        marked_b64 = encode_png_base64(marked_bgr)

        return {
            "variant": "gozenek.2",
            "method": "Black-hat + bolgesel (alin/burun/cene/yanak ayri) z-score, skor tespit edilen bolgelerin yukunden hesaplanir",
            "score": score,
            "mean_pore_z": mean_pore_z,
            "affected_area_pct": affected_area_pct,
            "detected_region_count": region_count,
            "region_breakdown": breakdown,
            "overlay": overlay_b64,
            "marked_image": marked_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Scan error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return FileResponse("client.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8402)
