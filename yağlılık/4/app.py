"""
YAGLILIK.4 test uygulamasi - bagimsiz FastAPI sunucusu.

Yaglilik.3'un tum tespit/gosterim ozellikleri korunuyor, ama HSV
specular highlight sinyaline EK olarak doku puruzsuzlugu (Laplacian
varyansi) sinyali de ekleniyor - arastirmanin onerdigi tamamlayici
teknik (bkz. yaglilik.py). Ana skin-scan projesinden BAGIMSIZ calisir.

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
from yaglilik import oiliness_map, oiliness_score, find_marked_regions
from visualize import create_heatmap_overlay, draw_filled_regions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yaglilik.4")

app = FastAPI(title="Yaglilik.4 - Specular + Doku Puruzsuzlugu Birlesimi")

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
    return {"ok": True, "variant": "yaglilik.4 - specular + doku puruzsuzlugu birlesimi"}


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

        omap = oiliness_map(img_processed, face_scan_mask, region_scan_masks)
        (score, coverage_pct, region_count, t_zone_pct, cheek_pct,
         skin_type, specular_pct, texture_pct) = oiliness_score(
            img_processed, face_scan_mask, region_scan_masks
        )
        contours = find_marked_regions(img_processed, face_scan_mask, region_scan_masks)

        overlay_rgba = create_heatmap_overlay(omap, colormap=cv2.COLORMAP_JET, alpha=0.6)
        marked_bgr = draw_filled_regions(img_processed, contours)

        overlay_b64 = encode_png_base64(overlay_rgba)
        marked_b64 = encode_png_base64(marked_bgr)

        return {
            "variant": "yaglilik.4",
            "method": "HSV specular highlight VEYA doku puruzsuzlugu (Laplacian varyansi) - iki bagimsiz sinyal",
            "score": score,
            "coverage_pct": coverage_pct,
            "detected_region_count": region_count,
            "t_zone_pct": t_zone_pct,
            "cheek_pct": cheek_pct,
            "skin_type": skin_type,
            "specular_pct": specular_pct,
            "texture_pct": texture_pct,
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
    uvicorn.run(app, host="127.0.0.1", port=8604)
