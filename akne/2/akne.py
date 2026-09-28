"""
AKNE.2 - Tum Ten Referansli, Z-Score (bagil) Yontem

Akne.1'deki iki sorun (Goz Alti Morluk.1 -> .2 gecisiyle BIREBIR ayni
motivasyon):

  1. Referans (yanak) convex hull'u DUDAK bolgesini de icine aliyordu -
     dudagin dogal kirmizi rengi "tipik ten kirmizilik" ortalamasini
     YUKARI cekiyordu. Bu, esigin yanlislikla yuksek bir referansa gore
     olculmesine, yani akne tespitinin GEREKTIGINDEN AZ HASSAS olmasina
     yol aciyordu.
  2. Skor sabit bir olcekle (18 a* birimi) hesaplaniyordu - herkesin
     teninde ayni "dogal kirmizilik dalgalanmasi" yok, bu yuzden sabit
     olcek adil bir kiyaslama degildi.

Bu surumde iki degisiklik (Goz Alti Morluk.2 ile birebir ayni mantik):

  1. REFERANS TEMIZLENDI VE GENISLETILDI: artik sadece yanak degil,
     DISLAMA UYGULANMIS tum ten (alin+burun+cene+yanak, goz/kas/dudak
     haric - `face_scan_mask`'in kendisi) referans aliniyor. Boylece
     dudak kontaminasyonu ortadan kalkiyor VE referans daha genis/stabil
     hale geliyor. Hedef (taranan) bolge zaten Akne.1'den beri tum yuzdu
     - bu surumde referans ve hedef AYNI (dislama uygulanmis) maskeye
     oturuyor, yani her piksel KENDI teninin genel dagilimina gore
     degerlendiriliyor.

  2. OLCEK SABIT DEGIL, Z-SCORE: skor artik sabit "18 a* birimi" gibi
     keyfi bir sayiya degil, KISININ KENDI TENININ DOGAL KIRMIZILIK
     DALGALANMASINA (standart sapmasina) gore olceklenir:

         z = (piksel_a - ten_a_ortalama) / ten_a_std

     Puruzsuz/az dokulu tenli birinde kucuk bir kirmizilik farki bile
     anlamli z-score verir; dogal olarak daha kirmizimsi/dokulu tenli
     birinde ayni mutlak fark daha az carpici sayilir - bu da kisiden
     kisiye daha ADIL bir kiyaslama saglar.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Z-score'un "tam akne/kizariklik" (skor=1.0) sayildigi ust sinir - kac
# standart sapmalik bir fark "maksimum" kabul edilsin. Goz alti morluk
# ailesiyle ayni sabit (karsilastirilabilirlik icin).
Z_SCORE_CAP = 2.5

# Bir pikselin "tespit edilen akne noktasi" sayilmasi icin gereken
# minimum z-score (standart sapma cinsinden)
DETECTION_Z_THRESHOLD = 1.2

# Kontur bulurken elenecek minimum piksel alani - Akne.1'deki gibi
# kucuk (sivilce buyuklugunde), goz alti morluk'un 20'si degil.
MIN_CONTOUR_AREA = 6


def _compute_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """
    Tum goruntu icin z = (piksel_a - ten_a_ortalama) / ten_a_std
    haritasini hesaplar (sadece pozitif kisim anlamli - negatif = tenden
    daha az kirmizi, akne degil).

    Referans istatistigi (ortalama/std) SADECE face_scan_mask (dislama
    uygulanmis tum ten) icindeki piksellerden hesaplaniyor - dudak/goz/kas
    katkida bulunmuyor.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_channel = lab[..., 1]

    skin_region = face_scan_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    skin_a_mean = float(np.mean(a_channel[skin_region]))
    skin_a_std = float(np.std(a_channel[skin_region])) + 1e-6

    z = (a_channel - skin_a_mean) / skin_a_std
    return z


def redness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """
    Akne/kizariklik haritasi - TUM TEN REFERANSLI Z-SCORE yontemi.

    Returns:
        [0,1] araliginda normalize edilmis kizariklik haritasi
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    z = _compute_z_map(img_bgr, face_scan_mask)
    z_positive = np.clip(z, 0, None)

    normalized = z_positive / Z_SCORE_CAP
    normalized = np.clip(normalized, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def redness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (score, mean_z) - mean_z raporlama icin (kac standart sapmalik
        bir ortalama farkla karsi karsiya oldugumuzu gormek onemli,
        sadece 0-1 skor bunu gizler).
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return 0.0, 0.0

    z = _compute_z_map(img_bgr, face_scan_mask)
    z_positive = np.clip(z, 0, None)

    mean_z = float(np.mean(z_positive[scan_region]))
    score = float(np.clip(mean_z / Z_SCORE_CAP, 0, 1))
    return score, mean_z


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> list:
    """DETECTION_Z_THRESHOLD'u asan bolgelerin konturlarini bulur."""
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    if not (scan_mask_u8 > 0).any():
        return []

    z = _compute_z_map(img_bgr, face_scan_mask)
    binary = (z > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)
