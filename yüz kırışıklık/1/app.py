"""
YUZ KIRISIKLIK.1 test uygulamasi - bagimsiz FastAPI sunucusu.

Goz Alti Morluk / Akne / Gozenek / Yuz Lekesi / Yaglilik aileleriyle
ayni calisma prensibi: kendi kendine yeten, ana skin-scan projesinden
BAGIMSIZ bir klasor.

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

from face_mesh import FaceMeshDetector, make_face_scan_mask
from preprocess import preprocess
from kirisiklik import wrinkle_map, wrinkle_score, find_marked_regions
from visualize import create_heatmap_overlay, draw_severity_lines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yuz_kirisiklik.1")

app = FastAPI(title="Yuz Kirisiklik.1 - Cok Yonelimli Gabor Filtre Bankasi")

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
    return {"ok": True, "variant": "yuz_kirisiklik.1 - cok yonelimli gabor filtre bankasi"}


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

        if face_scan_mask is None:
            raise HTTPException(status_code=400, detail="Yuz cildi bolgesi tespit edilemedi")

        wmap = wrinkle_map(img_processed, face_scan_mask)
        score, weighted_length, total_length, mean_z, region_count = wrinkle_score(
            img_processed, face_scan_mask
        )
        contours, severities = find_marked_regions(img_processed, face_scan_mask)

        overlay_rgba = create_heatmap_overlay(wmap, colormap=cv2.COLORMAP_HOT, alpha=0.7)
        marked_bgr = draw_severity_lines(img_processed, contours, severities)

        overlay_b64 = encode_png_base64(overlay_rgba)
        marked_b64 = encode_png_base64(marked_bgr)

        return {
            "variant": "yuz_kirisiklik.1",
            "method": "Cok yonelimli Gabor filtre bankasi + robust koyuluk dogrulamasi + uzama filtresi",
            "score": score,
            "weighted_length_px": weighted_length,
            "total_length_px": total_length,
            "mean_severity_z": mean_z,
            "detected_region_count": region_count,
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
    uvicorn.run(app, host="127.0.0.1", port=8701)
