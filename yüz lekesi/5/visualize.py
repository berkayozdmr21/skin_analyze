"""Heatmap overlay + tespit edilen bolgeleri isaretleme.

YUZ LEKESI.2: isaretleme SEKLI degisti. Onceki surumlerde (goz alti
morluk / akne / gozenek ailesi) tum tespitler AYNI sabit sari renkte,
kontur+dikdortgen cerceve ile isaretleniyordu - bu, "kac tane bulundu"
sorusuna cevap veriyordu ama "hangisi DAHA CIDDI" sorusuna cevap
vermiyordu.

Bu surumde draw_severity_circles() eklendi: her tespit, KENDI SIDDETINE
gore renklendirilmis (sari=hafif -> kirmizi=belirgin) yari saydam bir
daire ile isaretleniyor, ustune sira numarasi yaziliyor - hem "nerede"
hem "ne kadar ciddi" hem de "kacinci sirada" bilgisini tek bir gorsel
katmanda veriyor.
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
    bolgelerin konturlarini dondurur. "Buralarda buldum" isaretlemesi
    icin kullanilir.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _severity_color(severity: float) -> tuple:
    """
    Siddet (0-1) -> BGR renk: 0 = sari (hafif), 1 = kirmizi (belirgin).
    Aradaki degerler turuncu tonlarinda yumusak gecis yapar.
    """
    severity = max(0.0, min(1.0, severity))
    b = 0
    g = int(round(255 * (1.0 - severity)))
    r = 255
    return (b, g, r)


def draw_severity_circles(
    base_img_bgr: np.ndarray,
    contours: list,
    severities: list,
    alpha: float = 0.45,
    min_radius: int = 6,
) -> np.ndarray:
    """
    Her tespit edilen bolgeyi, KENDI SIDDETINE gore renklendirilmis
    (sari=hafif -> kirmizi=belirgin) yari saydam bir daire ile isaretler,
    ustune sira numarasini yazar.

    Onceki "kontur + sabit sari cerceve" yontemine gore fark: (1) renk
    artik bilgi tasiyor (siddet), (2) daire seklinde, orijinal lekenin
    duzensiz konturunu degil, onu KAPSAYAN kompakt bir alani gosteriyor
    - "burada YAKLASIK burada" seklinde daha okunakli bir isaret,
    (3) numaralandirma ile rapor/liste ile gorsel eslestirme kolaylasiyor.

    Args:
        base_img_bgr: isaretlenecek goruntu
        contours: tespit edilen konturlar listesi
        severities: her kontur icin 0-1 arasi siddet degeri (contours ile
            ayni sirada, ayni uzunlukta)
    """
    marked = base_img_bgr.copy()
    overlay = base_img_bgr.copy()

    circles = []
    for contour, severity in zip(contours, severities):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = max(min_radius, int(round(radius * 1.25)))
        center = (int(round(x)), int(round(y)))
        color = _severity_color(severity)
        cv2.circle(overlay, center, radius, color, -1, lineType=cv2.LINE_AA)
        circles.append((center, radius, color))

    marked = cv2.addWeighted(overlay, alpha, marked, 1 - alpha, 0)

    for i, (center, radius, color) in enumerate(circles, start=1):
        cv2.circle(marked, center, radius, color, 2, lineType=cv2.LINE_AA)

        label = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_org = (center[0] - tw // 2, center[1] + th // 2)

        cv2.putText(marked, label, text_org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(marked, label, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return marked


def draw_detection_markers(base_img_bgr: np.ndarray, contours: list, color=(0, 255, 255)) -> np.ndarray:
    """Onceki surumlerdeki sabit renkli kontur+cerceve yontemi - geriye donuk uyumluluk icin tutuldu."""
    marked = base_img_bgr.copy()
    for c in contours:
        cv2.drawContours(marked, [c], -1, color, 2)
        x, y, w, h = cv2.boundingRect(c)
        pad = 4
        cv2.rectangle(marked, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 1)
    return marked
