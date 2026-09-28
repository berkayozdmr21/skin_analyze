"""
YUZ LEKESI.2 - Tum Yuz Referansli Z-Score

## Yuz Lekesi.1'in sorunu (Goz Alti Morluk.1 -> .2 / Akne.1 -> .2 ile
## BIREBIR AYNI motivasyon)

Yuz Lekesi.1, referans olarak SADECE yanagi (`cheeks`) kullaniyordu ve
sabit bir olcekle (`DELTA_VISUAL_CAP = 30` L* birimi) degerlendiriyordu.
Bunun iki bilinen sorunu var:

  1. Yanak convex hull'u DUDAK bolgesini de icine alabiliyordu - bu,
     "tipik ten" ortalamasini yanlis yonlendirebiliyordu (Akne.1 -> .2
     gecisinde aynen tespit edilen sorun).
  2. Sabit bir olcek (30 L* birimi) HERKES icin ayni isliyordu - ama
     insanlarin ten dokusu/dogal koyuluk dalgalanmasi farkli olabilir.

## Bu surumde ne degisti (kucuk ama anlamli bir istatistik degisikligi)

1. **Referans genisletildi**: artik sadece yanak degil, DISLAMA
   UYGULANMIS tum ten (`face_scan_mask`'in kendisi - alin+burun+cene+
   yanak, goz/kas/dudak haric) referans aliniyor.
2. **Olcek sabit degil, Z-SCORE**: skor artik sabit "30 L* birimi" yerine
   kisinin KENDI TENININ DOGAL KOYULUK DALGALANMASINA (standart
   sapmasina) gore olceklenir:

       z = (ten_L_ortalama - L_piksel) / ten_L_std

   Puruzsuz/duz tenli biri icin kucuk bir fark bile anlamli z-score
   verir; dogal olarak dokulu/golgeli tenli biri icin ayni mutlak fark
   daha az carpici sayilir.

## Degismeyenler (Yuz Lekesi.1 ile ayni)

- Hedef bolge: tum taranabilir yuz cildi.
- Sekil filtresi (dairesellik + maksimum alan) - lekelerin akne/
  gozenekten daha buyuk/duzensiz olabilecegi kabulu korunuyor.
- Cozunurlukten bagimsiz (goreli) kernel/alan boyutlandirma.
- Skor, tespit edilen lekelerin SAYISI/SIDDETINDEN (alan degil)
  hesaplaniyor - Akne.5/Gozenek.3'un cozunurluk-kararsizligi hatasindan
  kacinmak icin.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

Z_SCORE_CAP = 2.5
DETECTION_Z_THRESHOLD = 1.2

CLEAN_KERNEL_RATIO = 0.008
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 15

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

MIN_CIRCULARITY = 0.22
MAX_SPOT_AREA_RATIO = 0.02

# Yuz Lekesi.1'deki COUNT_CAP, L* delta biriminde kalibre edilmisti;
# z-score'a gecince siddet hesabi degisti, bu yuzden gercek bir test
# fotografiyla yeniden kalibre edildi (bkz. README.md).
COUNT_CAP = 22.0


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _compute_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """z = (ten_L_ortalama - L) / ten_L_std haritasini hesaplar."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    skin_region = face_scan_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    skin_l_mean = float(np.mean(l_channel[skin_region]))
    skin_l_std = float(np.std(l_channel[skin_region])) + 1e-6

    z = (skin_l_mean - l_channel) / skin_l_std
    return z


def _is_leke_like(contour: np.ndarray, max_area: float) -> bool:
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """Tek bir tespit sureci - z haritasi ve kontur listesi birlikte doner."""
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    z = _compute_z_map(img_bgr, face_scan_mask)

    if not (scan_mask_u8 > 0).any():
        return z, []

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    binary = (z > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = find_detection_contours(binary, min_area=min_area)

    max_spot_area = max(min_area * 4, scan_px * MAX_SPOT_AREA_RATIO)
    leke_contours = [c for c in contours if _is_leke_like(c, max_spot_area)]

    return z, leke_contours


def leke_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    z, _ = _detect(img_bgr, face_scan_mask)
    z_positive = np.clip(z, 0, None)

    normalized = np.clip(z_positive / Z_SCORE_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (contours, severities) - her kontur icin 0-1 arasi siddet degeri
        de birlikte donuyor (Yuz Lekesi.2'nin yeni siddet-renkli
        isaretleme yontemi icin gerekli).
    """
    z, contours = _detect(img_bgr, face_scan_mask)

    severities = []
    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_z = float(z[region_pixels].mean()) if region_pixels.any() else 0.0
        severities.append(float(np.clip(mean_z / Z_SCORE_CAP, 0, 1)))

    return contours, severities


def leke_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, tespit edilen lekelerin SAYISI/SIDDETINDEN hesaplaniyor -
    ALANA DEGIL (Yuz Lekesi.1 ile ayni felsefe). 0 leke tespit
    edildiyse skor MATEMATIKSEL OLARAK 0.0'dir.

    Returns:
        (score, mean_leke_z, affected_area_pct, region_count)
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0.0, 0

    z, contours = _detect(img_bgr, face_scan_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0

    weighted_count = 0.0
    total_area = 0
    z_values = []

    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        area = int(region_pixels.sum())
        if area == 0:
            continue

        mean_z = float(z[region_pixels].mean())
        severity = float(np.clip(mean_z / Z_SCORE_CAP, 0, 1))

        weighted_count += severity
        total_area += area
        z_values.append(mean_z)

    score = float(np.clip(weighted_count / COUNT_CAP, 0, 1))
    mean_leke_z = float(np.mean(z_values)) if z_values else 0.0
    affected_area_pct = float(total_area / scan_area * 100)

    return score, mean_leke_z, affected_area_pct, len(contours)
