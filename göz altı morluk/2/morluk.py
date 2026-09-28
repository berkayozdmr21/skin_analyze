"""
GOZ ALTI MORLUK.2 - Ten rengi referansli, Z-SCORE (bagil) yontem

Goz alti morluk.1'deki sorun: skor sabit bir olcekle (0-30 L* birimi)
hesaplaniyordu ve sadece YANAK referans aliniyordu. Sonuc: "hafif morluk
var" denecek bir fotografta skor 1.0'a (tavan) vuruyordu - yanlis
kalibrasyon.

Bu surumde iki degisiklik var:

    1. REFERANS GENISLETILDI: artik sadece yanak degil, kisinin TUM
       teni (alin+burun+cene+yanak birlesimi) referans aliniyor. Bu,
       "ten rengi" dedigimizde kastedilen daha genel/stabil bir olcum -
       yanaktaki lokal bir golge/kizariklik tek basina yanlis
       yonlendirmesin diye.

    2. OLCEK SABIT DEGIL, Z-SCORE: skor artik sabit "30 L* birimi"
       gibi keyfi bir sayiya degil, KISININ KENDI TENININ DOGAL
       VARYASYONUNA (standart sapmasina) gore olceklenir:

           z = (ten_L_ortalama - L_piksel) / ten_L_std

       Bu, "bu kisinin teninde normalde ne kadar dalgalanma var, goz
       alti bu dalgalanmanin KAC KATI kadar koyu" sorusunu soruyor.
       Duz/pursuz tenli birinde kucuk bir fark bile anlamli cikar;
       zaten dokulu/gölgeli tenli birinde ayni mutlak fark daha az
       carpici sayilir - bu da daha ADIL/DOGRU bir kiyaslama saglar.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Z-score'un "tam morluk" (skor=1.0) sayildigi ust sinir - kac standart
# sapmalik bir fark "maksimum" kabul edilsin.
Z_SCORE_CAP = 2.5

# Bir pikselin "tespit edilen morluk noktasi" sayilmasi icin gereken
# minimum z-score (standart sapma cinsinden)
DETECTION_Z_THRESHOLD = 1.2

MIN_CONTOUR_AREA = 20


def _compute_z_map(img_bgr: np.ndarray, skin_tone_mask: np.ndarray) -> np.ndarray:
    """
    Tum goruntu icin z = (ten_L_ortalama - L) / ten_L_std haritasini
    hesaplar (sadece pozitif kisim anlamli - negatif = tenden daha
    parlak, morluk degil).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    skin_region = skin_tone_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    skin_l_mean = float(np.mean(l_channel[skin_region]))
    skin_l_std = float(np.std(l_channel[skin_region])) + 1e-6

    z = (skin_l_mean - l_channel) / skin_l_std
    return z


def darkness_map(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray) -> np.ndarray:
    """
    Goz alti morluk haritasi - TEN REFERANSLI Z-SCORE yontemi.

    Returns:
        [0,1] araliginda normalize edilmis morluk haritasi
    """
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    z = _compute_z_map(img_bgr, skin_tone_mask)
    z_positive = np.clip(z, 0, None)

    normalized = z_positive / Z_SCORE_CAP
    normalized = np.clip(normalized, 0, 1)
    normalized[~eye_region] = 0

    return normalized.astype(np.float32)


def darkness_score(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray):
    """
    Returns:
        (score, mean_z) - mean_z raporlama/README testleri icin (kac
        standart sapmalik bir ortalama farkla karsi karsiya oldugumuzu
        gormek onemli, sadece 0-1 skor bunu gizler).
    """
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return 0.0, 0.0

    z = _compute_z_map(img_bgr, skin_tone_mask)
    z_positive = np.clip(z, 0, None)

    mean_z = float(np.mean(z_positive[eye_region]))
    score = float(np.clip(mean_z / Z_SCORE_CAP, 0, 1))
    return score, mean_z


def find_marked_regions(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray) -> list:
    """DETECTION_Z_THRESHOLD'u asan bolgelerin konturlarini bulur."""
    eye_region_mask = (under_eye_mask > 0).astype(np.uint8) * 255
    if not (eye_region_mask > 0).any():
        return []

    z = _compute_z_map(img_bgr, skin_tone_mask)
    binary = (z > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, eye_region_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)
