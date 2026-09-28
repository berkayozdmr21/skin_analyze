"""
GOZ ALTI MORLUK.3 - Renk imzasi (koyuluk + mavi-mor ton) yontemi

Onceki iki versiyon SADECE koyuluga (L*) bakiyordu - ama bu, gercek bir
morlugu (damarlarin ince ciltten gorunmesiyle olusan MAVI-MOR renkli
bolge) sadece bir GOLGE ile (ornegin ust goz kapaginin dustugu, ayni
tonda ama daha koyu bir alan) ayirt edemiyordu. Golge notr/gri renktedir,
gercek morluk ise renk olarak da kayar.

Bu surumde IKI ayri z-score birlikte kullaniliyor:

    1. Koyuluk z-score (z_L): goz alti morluk.2 ile ayni - ten L*
       ortalamasina gore ne kadar koyu.
    2. Mavi-mor ton z-score (z_b): Lab b* kanali (mavi<->sari ekseni).
       Ten b* ortalamasindan DUSUK olan (yani tenin normal sarimsi
       tonundan daha maviye kaymis) bolgeler pozitif z_b alir.

Ikisi birlikte, agirlikli olarak birlestiriliyor:

    combined_z = 0.6 * z_L + 0.4 * z_b

Boylece hem koyu HEM DE mavimsi olan bolgeler en yuksek skoru alir -
sadece golgeli (koyu ama renk kaymamis) bolgeler daha dusuk skor alir.
Bu, "golge mi yoksa gercek morluk mu" ayrimini biraz daha netlestirmeye
calisan bir yaklasim.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Agirliklar: koyuluk mu, renk mi daha baskin sayilsin
WEIGHT_DARKNESS = 0.6
WEIGHT_BLUE_TONE = 0.4

Z_SCORE_CAP = 2.5
DETECTION_Z_THRESHOLD = 1.2
MIN_CONTOUR_AREA = 20


def _compute_combined_z_map(img_bgr: np.ndarray, skin_tone_mask: np.ndarray) -> np.ndarray:
    """
    combined_z = WEIGHT_DARKNESS * z_L + WEIGHT_BLUE_TONE * z_b

    z_L: (ten_L_ort - L) / ten_L_std        -> pozitif = daha koyu
    z_b: (ten_b_ort - b) / ten_b_std        -> pozitif = tenin normal
                                                sarimsi tonundan daha
                                                MAVIYE kaymis
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]
    b_channel = lab[..., 2]

    skin_region = skin_tone_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    skin_l_mean = float(np.mean(l_channel[skin_region]))
    skin_l_std = float(np.std(l_channel[skin_region])) + 1e-6

    skin_b_mean = float(np.mean(b_channel[skin_region]))
    skin_b_std = float(np.std(b_channel[skin_region])) + 1e-6

    z_l = (skin_l_mean - l_channel) / skin_l_std
    z_b = (skin_b_mean - b_channel) / skin_b_std

    # Sadece pozitif yonleri anlamli (daha koyu / daha mavi)
    z_l = np.clip(z_l, 0, None)
    z_b = np.clip(z_b, 0, None)

    combined = WEIGHT_DARKNESS * z_l + WEIGHT_BLUE_TONE * z_b
    return combined


def darkness_map(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray) -> np.ndarray:
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    combined = _compute_combined_z_map(img_bgr, skin_tone_mask)

    normalized = combined / Z_SCORE_CAP
    normalized = np.clip(normalized, 0, 1)
    normalized[~eye_region] = 0

    return normalized.astype(np.float32)


def darkness_score(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray):
    """
    Returns:
        (score, mean_combined_z, mean_z_l, mean_z_b) - raporlama icin
        koyuluk ve mavi-ton bilesenleri ayri ayri da donuluyor, boylece
        "bu bolge neden yuksek skor aldi - koyu oldugu icin mi, mavi
        oldugu icin mi, ikisi icin mi" gorulebilir.
    """
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return 0.0, 0.0, 0.0, 0.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]
    b_channel = lab[..., 2]

    skin_region = skin_tone_mask > 0
    skin_l_mean = float(np.mean(l_channel[skin_region]))
    skin_l_std = float(np.std(l_channel[skin_region])) + 1e-6
    skin_b_mean = float(np.mean(b_channel[skin_region]))
    skin_b_std = float(np.std(b_channel[skin_region])) + 1e-6

    z_l_map = np.clip((skin_l_mean - l_channel) / skin_l_std, 0, None)
    z_b_map = np.clip((skin_b_mean - b_channel) / skin_b_std, 0, None)

    mean_z_l = float(np.mean(z_l_map[eye_region]))
    mean_z_b = float(np.mean(z_b_map[eye_region]))
    mean_combined = WEIGHT_DARKNESS * mean_z_l + WEIGHT_BLUE_TONE * mean_z_b

    score = float(np.clip(mean_combined / Z_SCORE_CAP, 0, 1))
    return score, mean_combined, mean_z_l, mean_z_b


def find_marked_regions(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray) -> list:
    eye_region_mask = (under_eye_mask > 0).astype(np.uint8) * 255
    if not (eye_region_mask > 0).any():
        return []

    combined = _compute_combined_z_map(img_bgr, skin_tone_mask)
    binary = (combined > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, eye_region_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)
