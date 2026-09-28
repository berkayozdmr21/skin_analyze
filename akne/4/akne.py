"""
AKNE.4 - Cozunurlukten Bagimsiz (Resolution-Invariant) Tespit

## Bulunan sorun (kullanicinin gercek testleriyle ortaya cikti)

Akne.3, ayni fotografi FARKLI cozunurluklerde test ettigimizde COK
FARKLI sonuclar veriyordu - somut olcum (ayni yuz, sadece kaynak
fotograf genisligi degistirildi):

    kaynak genislik 170px  -> 15 bolge
    kaynak genislik 700px  -> 13 bolge
    kaynak genislik 1200px -> 32 bolge
    kaynak genislik 2000px -> 28 bolge
    kaynak genislik 3000px -> 48 bolge

Sebep: `MIN_CONTOUR_AREA` (4 piksel) ve morfolojik gurultu temizleme
cekirdegi (3x3, SABIT boyut) her ikisi de MUTLAK piksel sayilariydi.
Yuksek cozunurluklu bir fotografta (ornegin bir telefon kamerasi
ciktisi), tenin dogal DOKUSU (gozenekler, ince kil golgeleri, hafif
isik degisimleri) binlerce KUCUK ama az-cok yuvarlak piksel grubu
olusturuyor - dusuk cozunurlukte bunlar zaten tek bir piksele
sikisip kayboluyordu ama yuksek cozunurlukte HER BIRI kendi basina
"4 pikselden buyuk, dairesel bir bolge" sayilip sivilce olarak
isaretleniyordu. Kullanicinin bildirdigi "85 tespit" / "22 tespit"
farkli fotograflarin farkli cozunurluklerde olmasindan kaynaklaniyordu.

## Cozum: goreli (scan bolgesine oranli) esikler

Artik butun piksel-tabanli sabitler, o FOTOGRAFA OZGU taranan yuz
alaninin (`face_scan_mask`) piksel sayisindan TURETILIYOR - mutlak
sayi degil, ORAN:

    face_scale = sqrt(scan_piksel_sayisi)   # karakteristik "uzunluk"

    kernel_boyutu = max(3, round(face_scale * KERNEL_SIZE_RATIO)) | tek sayi
    min_kontur_alani = max(6, round(scan_piksel_sayisi * MIN_AREA_RATIO))

Boylece yuksek cozunurluklu bir fotografta hem gurultu temizleme
cekirdegi ORANTILI olarak buyuyor (kucuk dokusal gurultuyu daha
agresif siliyor) hem de "sivilce sayilmak icin gereken minimum alan"
ORANTILI olarak buyuyor - dusuk cozunurluktekiyle AYNI GORECELI
buyuklukte bir leke gerekiyor, mutlak piksel sayisi degil.

`MAX_SPOT_AREA_RATIO` zaten Akne.3'te bir ORANDI (degismedi) - sadece
KUCUK taraftaki esikler mutlak sayidan orana cevrildi.

## Degismeyenler (Akne.3 ile ayni)

- Kirmizilik (a*) + doygunluk (HSV S) z-score birlesimi:
  `combined_z = 0.6*z_a + 0.4*z_s`
- Dairesellik filtresi (`MIN_CIRCULARITY`) - genis anatomik konturlari
  (burun koprusu, cene hatti) eleme mantigi degismedi.
- `DETECTION_Z_THRESHOLD` biraz yukseltildi (1.0 -> 1.15) - alan/kernel
  duzeltmesine EK bir guvenlik payi olarak, cok yuksek cozunurluklu
  fotograflarda bile dokusal gurultunun esigi asma ihtimalini daha da
  azaltmak icin.

Detaylar ve cozunurlukler arasi dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

WEIGHT_REDNESS = 0.6
WEIGHT_SATURATION = 0.4

Z_SCORE_CAP = 2.5

# Akne.3'teki 1.0'dan 1.15'e yukseltildi - alan/kernel duzeltmesine ek
# bir guvenlik payi.
DETECTION_Z_THRESHOLD = 1.15

MIN_CIRCULARITY = 0.35
MAX_SPOT_AREA_RATIO = 0.012

# --- YENI: cozunurlukten bagimsiz (goreli) boyutlandirma ---

# Gurultu temizleme cekirdeginin boyutu, taranan alanin karakteristik
# uzunlugunun (sqrt(scan_piksel_sayisi)) bu ORANI olarak hesaplanir.
KERNEL_SIZE_RATIO = 0.012
MIN_KERNEL_SIZE = 3
MAX_KERNEL_SIZE = 15

# Minimum kontur alani, taranan yuz alaninin (piksel sayisi) bu ORANI
# olarak hesaplanir - Akne.3'teki SABIT "4"un yerini aliyor.
MIN_AREA_RATIO = 0.00012
MIN_AREA_FLOOR = 6


def _relative_sizing(face_scan_mask: np.ndarray) -> tuple[int, int]:
    """
    Taranan yuz alaninin piksel sayisindan, o FOTOGRAFA OZGU gurultu
    temizleme cekirdek boyutunu ve minimum kontur alanini turetir.

    Returns:
        (kernel_size, min_contour_area)
    """
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))

    kernel_size = int(round(face_scale * KERNEL_SIZE_RATIO))
    kernel_size = max(MIN_KERNEL_SIZE, min(MAX_KERNEL_SIZE, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    return kernel_size, min_area


def _compute_combined_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """
    combined_z = WEIGHT_REDNESS * z_a + WEIGHT_SATURATION * z_s

    z_a: (a - ten_a_ort) / ten_a_std          -> pozitif = daha kirmizi
    z_s: (S - ten_S_ort) / ten_S_std          -> pozitif = daha doygun/canli
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    a_channel = lab[..., 1]
    s_channel = hsv[..., 1]

    skin_region = face_scan_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    skin_a_mean = float(np.mean(a_channel[skin_region]))
    skin_a_std = float(np.std(a_channel[skin_region])) + 1e-6
    skin_s_mean = float(np.mean(s_channel[skin_region]))
    skin_s_std = float(np.std(s_channel[skin_region])) + 1e-6

    z_a = (a_channel - skin_a_mean) / skin_a_std
    z_s = (s_channel - skin_s_mean) / skin_s_std

    z_a = np.clip(z_a, 0, None)
    z_s = np.clip(z_s, 0, None)

    combined = WEIGHT_REDNESS * z_a + WEIGHT_SATURATION * z_s
    return combined


def redness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    combined = _compute_combined_z_map(img_bgr, face_scan_mask)
    normalized = np.clip(combined / Z_SCORE_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def redness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (score, mean_combined, mean_z_redness, mean_z_saturation)
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return 0.0, 0.0, 0.0, 0.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    a_channel = lab[..., 1]
    s_channel = hsv[..., 1]

    skin_a_mean = float(np.mean(a_channel[scan_region]))
    skin_a_std = float(np.std(a_channel[scan_region])) + 1e-6
    skin_s_mean = float(np.mean(s_channel[scan_region]))
    skin_s_std = float(np.std(s_channel[scan_region])) + 1e-6

    z_a_map = np.clip((a_channel - skin_a_mean) / skin_a_std, 0, None)
    z_s_map = np.clip((s_channel - skin_s_mean) / skin_s_std, 0, None)

    mean_z_a = float(np.mean(z_a_map[scan_region]))
    mean_z_s = float(np.mean(z_s_map[scan_region]))
    mean_combined = WEIGHT_REDNESS * mean_z_a + WEIGHT_SATURATION * mean_z_s

    score = float(np.clip(mean_combined / Z_SCORE_CAP, 0, 1))
    return score, mean_combined, mean_z_a, mean_z_s


def _is_spot_like(contour: np.ndarray, max_area: float) -> bool:
    """Kucuk, yuvarlak/kompakt bolgeleri tutar; uzun/genis konturlari eler."""
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> list:
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    if not (scan_mask_u8 > 0).any():
        return []

    kernel_size, min_contour_area = _relative_sizing(face_scan_mask)

    combined = _compute_combined_z_map(img_bgr, face_scan_mask)
    binary = (combined > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = find_detection_contours(binary, min_area=min_contour_area)

    scan_area = int((face_scan_mask > 0).sum())
    max_spot_area = max(min_contour_area * 4, scan_area * MAX_SPOT_AREA_RATIO)

    return [c for c in contours if _is_spot_like(c, max_spot_area)]
