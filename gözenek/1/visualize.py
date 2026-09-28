"""Heatmap overlay + tespit edilen bolgeleri kontur ile isaretleme."""
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
    bolgelerin konturlarini dondurur. "Buralarda buldum" isaretlemesi
    icin kullanilir.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def draw_detection_markers(base_img_bgr: np.ndarray, contours: list, color=(0, 255, 255)) -> np.ndarray:
    """
    Tespit edilen her bolgenin etrafina dikkat cekici bir cerceve (dikdortgen)
    ve kontur cizgisi cizer - "buralarda tespit ettim" gostergesi.
    """
    marked = base_img_bgr.copy()
    for c in contours:
        cv2.drawContours(marked, [c], -1, color, 2)
        x, y, w, h = cv2.boundingRect(c)
        pad = 4
        cv2.rectangle(marked, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 1)
    return marked
