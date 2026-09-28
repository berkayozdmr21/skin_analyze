"""Heatmap overlay + tespit edilen kirisiklik cizgilerini isaretleme.

YUZ KIRISIKLIK.1: kirisiklikler CIZGI/EGRI seklinde (Yuz Lekesi ailesindeki
gibi kompakt lekeler veya Yaglilik'teki gibi yayilmis alanlar degil) -
bu yuzden isaretleme de kirisiklik ailesine ozgu: her tespit edilen
cizginin GERCEK SEKLI, siddetine gore renklendirilerek (Yuz Lekesi.2'deki
sari->kirmizi renk mantigi, ama daire yerine cizginin kendisi) ciziliyor.
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


def _severity_color(severity: float) -> tuple:
    """Siddet (0-1) -> BGR renk: 0 = sari (hafif), 1 = kirmizi (belirgin)."""
    severity = max(0.0, min(1.0, severity))
    b = 0
    g = int(round(255 * (1.0 - severity)))
    r = 255
    return (b, g, r)


def draw_severity_lines(
    base_img_bgr: np.ndarray,
    contours: list,
    severities: list,
    thickness: int = 2,
) -> np.ndarray:
    """
    Her tespit edilen kirisiklik CIZGISINI, kendi siddetine gore
    renklendirilmis (sari=hafif -> kirmizi=belirgin) bir egri olarak
    cizer. Daire veya dikdortgene indirgemez - kirisiklik dogasi geregi
    ince/uzun bir sekildir, gercek konturu takip etmek daha dogru.
    """
    marked = base_img_bgr.copy()
    for contour, severity in zip(contours, severities):
        color = _severity_color(severity)
        cv2.drawContours(marked, [contour], -1, color, thickness, lineType=cv2.LINE_AA)
    return marked


def draw_filled_regions(
    base_img_bgr: np.ndarray,
    contours: list,
    color: tuple = (255, 220, 60),
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """Genel amacli: konturlarin gercek seklini yari saydam doldurur."""
    marked = base_img_bgr.copy()
    if not contours:
        return marked

    overlay = base_img_bgr.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
    marked = cv2.addWeighted(overlay, alpha, marked, 1 - alpha, 0)

    if outline:
        cv2.drawContours(marked, contours, -1, color, 2, lineType=cv2.LINE_AA)

    return marked
