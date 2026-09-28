"""
GOZ ALTI MORLUK.5 test uygulamasi - bagimsiz FastAPI sunucusu.

goz alti morluk.4 ile ayni iskelet, ama darkness_map/score artik TEK
global ten ortalamasi yerine YEREL (spatially-adaptive) bir aydinlatma
referansi ve bu referansa gore hesaplanmis rezidüel kovaryans
kullaniyor (bkz. morluk.py). Ana skin-scan projesinden BAGIMSIZ calisir.

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

from face_mesh import FaceMeshDetector, make_region_masks, make_skin_tone_mask, make_under_eye_mask
from preprocess import preprocess
from morluk import darkness_map, darkness_score, find_marked_regions
from visualize import create_heatmap_overlay, draw_detection_markers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("goz_alti_morluk.5")

app = FastAPI(title="Goz Alti Morluk.5 - Yerel Aydinlatma Duzeltmeli Mahalanobis Mesafesi")

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
    return {"ok": True, "variant": "goz_alti_morluk.5 - yerel aydinlatma duzeltmeli mahalanobis mesafesi"}


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

        region_masks = make_region_masks(landmarks, img_processed.shape)
        skin_tone_mask = make_skin_tone_mask(region_masks, img_processed.shape)
        under_eye_mask = make_under_eye_mask(landmarks, img_processed.shape)

        if skin_tone_mask is None or under_eye_mask is None:
            raise HTTPException(status_code=400, detail="Ten veya goz alti bolgesi tespit edilemedi")

        dmap = darkness_map(img_processed, under_eye_mask, skin_tone_mask)
        score, mean_mahalanobis, mean_z_l, mean_z_a, mean_z_b = darkness_score(
            img_processed, under_eye_mask, skin_tone_mask
        )
        contours = find_marked_regions(img_processed, under_eye_mask, skin_tone_mask)

        overlay_rgba = create_heatmap_overlay(dmap, colormap=cv2.COLORMAP_VIRIDIS, alpha=0.6)
        marked_bgr = draw_detection_markers(img_processed, contours, color=(255, 0, 255))

        overlay_b64 = encode_png_base64(overlay_rgba)
        marked_b64 = encode_png_base64(marked_bgr)

        return {
            "variant": "goz_alti_morluk.5",
            "method": "Yerel (spatially-adaptive) aydinlatma duzeltmeli, rezidüel kovaryansli Mahalanobis mesafesi",
            "score": score,
            "mean_mahalanobis_distance": mean_mahalanobis,
            "mean_z_darkness": mean_z_l,
            "mean_z_red_magenta": mean_z_a,
            "mean_z_blue_tone": mean_z_b,
            "detected_region_count": len(contours),
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
    uvicorn.run(app, host="127.0.0.1", port=8205)
