"""
AKNE.5 - Tespit Edilen Bolgelere Dayali Skor

## Bulunan sorun (kullanicinin ekran goruntusuyle bildirdigi)

Akne.4'te (ve tum onceki surumlerde) SKOR ile TESPIT SAYISI birbirinden
TAMAMEN BAGIMSIZ iki ayri hesaplamaydi:

  - Skor: taranan YUZUN TAMAMINDAKI ortalama combined_z (binlerce piksel
    - neredeyse hepsi normal ten - uzerinden ortalama).
  - Tespit sayisi: sadece DETECTION_Z_THRESHOLD'u asan, sivilce-seklinde
    bolgelerin SAYISI.

Bu ikisi FARKLI esiklerle/FARKLI kapsamlarla hesaplandigi icin
celiskili sonuclar cikabiliyordu - kullanicinin bildirdigi ornekte:
"0 adet sivilce/akne tespit edildi" ama skor "%6" gosteriyordu. Bu
kullanici acisindan anlamsiz: hic sivilce yoksa skor neden sifir degil?

## Cozum: skor artik DOGRUDAN tespit edilen bolgelerden hesaplaniyor

Artik iki AYRI hesap yok - TEK bir tespit surecinin (`_detect()`) iki
farkli ciktisi var: hem kontur LISTESI (isaretleme icin) hem de bu
konturlardan turetilen SKOR. Formul:

    her tespit edilen bolge i icin:
        siddet_i = clip(bolge_i_ortalama_combined_z / Z_SCORE_CAP, 0, 1)
        katki_i  = siddet_i * bolge_i_alani (piksel)

    toplam_yuk = sum(katki_i)
    skor = clip(toplam_yuk / (taranan_yuz_alani * BURDEN_AREA_FRACTION), 0, 1)

Yani skor artik "yuzun KAPLANAN (tespit edilmis) kismi, ne kadar
SIDDETLI (ortalama z) ve ne kadar BUYUK (piksel alani)" sorusuna
dogrudan cevap veriyor - dermatolojideki "lezyon yuku" (lesion burden)
mantigiyla ayni fikir: cok sayida/genis/siddetli bolge -> yuksek skor;
az sayida/kucuk/hafif bolge -> dusuk skor; SIFIR bolge -> SKOR KESINLIKLE
SIFIR (artik matematiksel olarak garanti, ayri bir hesap degil).

`BURDEN_AREA_FRACTION = 0.05`: "yuzun %5'i, MAKSIMUM siddette
(Z_SCORE_CAP'e esit veya uzeri) sivilcelerle kapli olsa, skor %100'e
ulasir" anlamina gelen ampirik bir referans - dermatolojik siddet
derecelendirmelerinde "yuzeyin %5'i lezyonlu" zaten belirgin/ciddi bir
akne durumu sayilir.

## Degismeyenler (Akne.4 ile ayni)

- Kirmizilik (a*) + doygunluk (HSV S) z-score birlesimi.
- Cozunurlukten bagimsiz (goreli) kernel/min alan boyutlandirma.
- Dairesellik + maksimum alan filtresi (sivilce SEKLI kontrolu).

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

WEIGHT_REDNESS = 0.6
WEIGHT_SATURATION = 0.4

Z_SCORE_CAP = 2.5
DETECTION_Z_THRESHOLD = 1.15

MIN_CIRCULARITY = 0.35
MAX_SPOT_AREA_RATIO = 0.012

KERNEL_SIZE_RATIO = 0.012
MIN_KERNEL_SIZE = 3
MAX_KERNEL_SIZE = 15

MIN_AREA_RATIO = 0.00012
MIN_AREA_FLOOR = 6

# YENI: skorun "tespit edilen bolgelerin yuku" uzerinden hesaplanmasi
# icin normalizasyon sabiti - bkz. yukaridaki aciklama.
BURDEN_AREA_FRACTION = 0.05


def _relative_sizing(face_scan_mask: np.ndarray) -> tuple[int, int]:
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


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Tek bir tespit sureci - hem kontur listesini (isaretleme icin) hem
    de bunlarin turetildigi combined_z haritasini (skor hesaplamak icin)
    birlikte dondurur. find_marked_regions() ve redness_score() ARTIK
    AYNI bu fonksiyona dayaniyor - iki ayri/tutarsiz hesap riski ortadan
    kalkiyor.
    """
    h, w = img_bgr.shape[:2]
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255

    combined = _compute_combined_z_map(img_bgr, face_scan_mask)

    if not (scan_mask_u8 > 0).any():
        return combined, []

    kernel_size, min_area = _relative_sizing(face_scan_mask)

    binary = (combined > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = find_detection_contours(binary, min_area=min_area)

    scan_area = int((face_scan_mask > 0).sum())
    max_spot_area = max(min_area * 4, scan_area * MAX_SPOT_AREA_RATIO)

    spot_contours = [c for c in contours if _is_spot_like(c, max_spot_area)]
    return combined, spot_contours


def redness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    combined, _ = _detect(img_bgr, face_scan_mask)
    normalized = np.clip(combined / Z_SCORE_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> list:
    _, contours = _detect(img_bgr, face_scan_mask)
    return contours


def redness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, TESPIT EDILEN BOLGELERIN kendisinden hesaplaniyor (bkz. dosya
    basindaki aciklama) - artik yuzun tamamindaki ayri bir ortalamadan
    DEGIL. Bu yuzden: 0 bolge tespit edildiyse skor MATEMATIKSEL OLARAK
    0.0'dir, ayri bir hesap tutarsizligi olamaz.

    Returns:
        (score, mean_lesion_z, affected_area_pct, region_count)

        mean_lesion_z: tespit edilen bolgelerin ORTALAMA siddeti (kac
            standart sapma) - raporlama icin.
        affected_area_pct: taranan yuz alaninin tespit edilen
            bolgelerce kaplanan yuzdesi.
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0.0, 0

    combined, contours = _detect(img_bgr, face_scan_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0

    total_burden = 0.0
    total_area = 0
    severities = []

    lesion_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        lesion_mask[:] = 0
        cv2.drawContours(lesion_mask, [c], -1, 255, -1)
        region_pixels = lesion_mask > 0
        area = int(region_pixels.sum())
        if area == 0:
            continue

        mean_z = float(combined[region_pixels].mean())
        severity = float(np.clip(mean_z / Z_SCORE_CAP, 0, 1))

        total_burden += severity * area
        total_area += area
        severities.append(mean_z)

    score = float(np.clip(total_burden / (scan_area * BURDEN_AREA_FRACTION), 0, 1))
    mean_lesion_z = float(np.mean(severities)) if severities else 0.0
    affected_area_pct = float(total_area / scan_area * 100)

    return score, mean_lesion_z, affected_area_pct, len(contours)
