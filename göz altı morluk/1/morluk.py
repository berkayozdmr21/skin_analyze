"""
GOZ ALTI MORLUK.1 - Bolgesel L* farki yontemi (yanak referansli)

Mantik (daha once tartistigimiz orijinal yontemle ayni):
Goz altindaki L* (Lab parlaklik kanali) ortalamasi, yanak ROI'sindeki
L* ortalamasiyla kiyaslanir:

    delta = cheek_L_mean - eye_pixel_L

Pozitif ve buyuk fark = goz alti, yanaktan belirgin sekilde daha koyu =
morluk/golge adayi.

Bu surumde EKSTRA olarak: sadece bir "ortalama skor" degil, delta'nin
belli bir esigi astigi NOKTALAR (blob'lar) bulunup gorsel uzerinde
KONTUR + CERCEVE ile isaretleniyor - "tam olarak nerede buldum" gostergesi.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours, draw_detection_markers

# Delta'nin (a* biriminde degil, L* biriminde) "tam kirmizi/morluk" (skor=1.0)
# sayildigi ust sinir - gorsellestirme ve normalize icin.
DELTA_VISUAL_CAP = 30.0

# Bir pikselin "tespit edilen morluk noktasi" (kontur/cerceve ile isaretlenecek)
# sayilmasi icin delta'nin asmasi gereken esik (L* birimi)
DETECTION_THRESHOLD = 12.0

# Kontur bulurken elenecek minimum piksel alani (gurultu temizligi)
MIN_CONTOUR_AREA = 20


def _compute_delta(img_bgr: np.ndarray, cheeks_mask: np.ndarray) -> np.ndarray:
    """cheek_L_mean - L(piksel) farkini tum goruntu icin hesaplar."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    cheek_region = cheeks_mask > 0
    if not cheek_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    cheek_l_mean = float(np.mean(l_channel[cheek_region]))
    delta = cheek_l_mean - l_channel
    return delta


def darkness_map(img_bgr: np.ndarray, under_eye_mask: np.ndarray, cheeks_mask: np.ndarray) -> np.ndarray:
    """
    Goz alti morluk haritasi.

    Returns:
        [0,1] araliginda normalize edilmis morluk haritasi (sadece goz alti
        bandi icinde, geri kalani 0)
    """
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    delta = _compute_delta(img_bgr, cheeks_mask)
    delta_positive = np.clip(delta, 0, None)

    normalized = delta_positive / DELTA_VISUAL_CAP
    normalized = np.clip(normalized, 0, 1)
    normalized[~eye_region] = 0

    return normalized.astype(np.float32)


def darkness_score(img_bgr: np.ndarray, under_eye_mask: np.ndarray, cheeks_mask: np.ndarray) -> float:
    """Goz alti bandindaki ORTALAMA (pozitif) delta, [0,1]'e olceklenmis."""
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return 0.0

    delta = _compute_delta(img_bgr, cheeks_mask)
    delta_positive = np.clip(delta, 0, None)

    eye_mean_delta = float(np.mean(delta_positive[eye_region]))
    score = eye_mean_delta / DELTA_VISUAL_CAP
    return float(np.clip(score, 0, 1))


def find_marked_regions(img_bgr: np.ndarray, under_eye_mask: np.ndarray, cheeks_mask: np.ndarray) -> list:
    """
    DETECTION_THRESHOLD'u asan piksellerden, goz alti bandi icinde kalan
    ve yeterince buyuk olan (gurultu degil) kontur listesini dondurur.
    Bu konturlar, fotografta "buralarda morluk tespit ettim" diye
    isaretlenecek yerler.
    """
    eye_region_mask = (under_eye_mask > 0).astype(np.uint8) * 255
    if not (eye_region_mask > 0).any():
        return []

    delta = _compute_delta(img_bgr, cheeks_mask)
    binary = (delta > DETECTION_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, eye_region_mask)

    # Kucuk gurultu noktalarini temizlemek icin hafif morfolojik acma
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)
