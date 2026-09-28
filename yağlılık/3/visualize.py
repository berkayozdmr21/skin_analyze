"""Heatmap overlay + tespit edilen yagli/parlak bolgeleri isaretleme.

YAGLILIK.1: bu proje ailesindeki (goz alti morluk/akne/gozenek/yuz
lekesi) digerlerinden farkli bir isaretleme kullanir. Onceki projeler
AYRIK, kucuk, yuvarlak "leke"leri buluyordu - numarali daire ile
isaretlemek dogruydu. Yagliliktaki "parlama" (specular highlight) ise
DOGASI GEREGI YAYGIN/DUZENSIZ SEKILLI bolgelerdir (T-bolgesi boyunca
uzanan, birbirine baglanan parlak alanlar) - bu yuzden burada
konturlarin GERCEK SEKLI (daire degil) yari saydam olarak dolduruluyor.
"""
import cv2
import numpy as np


def create_heatmap_overlay(
    map_data: np.ndarray,
    colormap: int = cv2.COLORMAP_OCEAN,
    alpha: float = 0.6
) -> np.ndarray:
    map_uint8 = (map_data * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(map_uint8, colormap)
    heatmap_rgba = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
    alpha_channel = (map_data * alpha * 255).astype(np.uint8)
    heatmap_rgba[..., 3] = alpha_channel
    return heatmap_rgba


def find_detection_contours(binary_mask: np.ndarray, min_area: int = 15) -> list:
    """
    Ikili (0/255) maskeden, kucuk gurultuyu eleyerek tespit edilen
    bolgelerin konturlarini dondurur.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def draw_filled_regions(
    base_img_bgr: np.ndarray,
    contours: list,
    color: tuple = (255, 220, 60),
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """
    Tespit edilen yagli/parlak bolgelerin GERCEK SEKLINI (daire veya
    dikdortgene sadeleştirmeden) yari saydam doldurarak isaretler -
    bu bolgeler dogasi geregi duzensiz/yayilmis oldugu icin (bir
    "sivilce" veya "gozenek" gibi kompakt bir noktaya indirgemek
    yaniltici olurdu).
    """
    marked = base_img_bgr.copy()
    if not contours:
        return marked

    overlay = base_img_bgr.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
    marked = cv2.addWeighted(overlay, alpha, marked, 1 - alpha, 0)

    if outline:
        cv2.drawContours(marked, contours, -1, color, 2, lineType=cv2.LINE_AA)

    return marked


def draw_detection_markers(base_img_bgr: np.ndarray, contours: list, color=(0, 255, 255)) -> np.ndarray:
    """Onceki ailedeki sabit renkli kontur+cerceve yontemi - geriye donuk uyumluluk icin tutuldu."""
    marked = base_img_bgr.copy()
    for c in contours:
        cv2.drawContours(marked, [c], -1, color, 2)
        x, y, w, h = cv2.boundingRect(c)
        pad = 4
        cv2.rectangle(marked, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 1)
    return marked
