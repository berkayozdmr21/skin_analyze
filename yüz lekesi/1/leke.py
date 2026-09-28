"""
YUZ LEKESI.1 - Yanak Referansli Koyuluk (L*) Farki

Goz Alti Morluk.1 / Akne.1 ile AYNI mimari, yuz lekesi (gunes lekesi,
melazma, hiperpigmentasyon) tespiti icin uyarlandi.

## Yontem

1. Lab renk uzayina cevrilir, **L\*** (parlaklik) kanali alinir - lekeler
   oncelikle bir KOYULUK/pigmentasyon belirtisi (akne'nin RENK/kirmizilik
   belirtisinden farkli olarak, goz alti morluk'un KOYULUK mantigina
   daha yakin).
2. **Referans bolge**: yanaklar (`cheeks`) - goz alti morluk.1/akne.1 ile
   birebir ayni mantik, "tipik/lekesiz ten" ornegi.
3. **Hedef (taranan) bolge**: akne.1'deki gibi TUM taranabilir yuz cildi
   (alin+burun+cene+yanak, goz/kas/dudak haric) - lekeler goz alti
   morluk'taki gibi sabit bir bantta degil, yuzun HERHANGI bir yerinde
   olabilir.
4. Her piksel icin:
   ```
   delta = cheek_L_mean - L(piksel)
   ```
   Pozitif ve buyuk deger = piksel, yanaktan belirgin sekilde daha KOYU
   -> leke adayi.

## Baştan dogru kurulan tasarim kararlari (Akne/Gozenek ailesinin
## SONRADAN ogrendigi dersler)

Akne ve Gozenek ailelerinde ONEMLI iki sorun SONRADAN cikip duzeltildi:

  1. Sabit piksel-tabanli esikler (gurultu temizleme cekirdegi, minimum
     kontur alani) cozunurlukle ORANTISIZ olcekleniyordu -> burada
     BASTAN goreli (taranan alana oranli) kuruldu.
  2. Skoru ALANA bolmek (Akne.5, Gozenek.3) COZUNURLUK KARARSIZLIGINA
     yol aciyordu (ayni foto farkli cozunurlukte COK farkli skor
     veriyordu - Gozenek.3'te %58 -> %5 gibi). Gozenek.4'te bu, skoru
     ALANA DEGIL SABIT BIR SAYI REFERANSINA (COUNT_CAP) bolerek
     duzeltildi.

Bu proje, ic BASTAN (2) numarali cozumle kuruluyor: skor, tespit edilen
lekelerin SAYISI/SIDDETINDEN (alan degil) hesaplaniyor, sabit bir
COUNT_CAP referansina gore olcekleniyor.

## Sekil filtresi

Lekeler (gunes lekesi, melazma) akne benekleri veya gozeneklerden daha
BUYUK ve daha DUZENSIZ/BLOTCHY sinirlarina sahip olabilir - bu yuzden
`MIN_CIRCULARITY` akne'dekinden (0.35) daha DUSUK (0.22) ve
`MAX_SPOT_AREA_RATIO` daha YUKSEK (0.02) tutuldu.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Delta'nin (L* biriminde) "tam leke" (siddet=1.0) sayildigi ust sinir -
# goz alti morluk.1 ile ayni sabit (ayni kanal, ayni olcek).
DELTA_VISUAL_CAP = 30.0

# Bir pikselin "tespit edilen leke noktasi" sayilmasi icin gereken
# minimum delta (L* birimi) - goz alti morluk.1 ile ayni.
DETECTION_THRESHOLD = 12.0

# Gurultu temizleme cekirdegi ve minimum kontur alani - cozunurlukten
# bagimsiz (goreli), Gozenek ailesindeki duzeltilmis mantik.
CLEAN_KERNEL_RATIO = 0.008
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 15

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

# Sekil filtresi: lekeler akne/gozenekten daha buyuk ve duzensiz sinirli
# olabilir - daha gevsek dairesellik, daha genis ust alan siniri.
MIN_CIRCULARITY = 0.22
MAX_SPOT_AREA_RATIO = 0.02

# Skor, tespit edilen lekelerin SAYISI/SIDDETINDEN hesaplaniyor (Akne.5
# ve Gozenek.3'un ALAN tabanli hatalarindan kacinmak icin, Gozenek.4'te
# dogrulanan yontem BASTAN uygulaniyor) - COUNT_CAP, gercek bir test
# fotografiyla kalibre edildi (bkz. README.md).
COUNT_CAP = 25.0


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


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


def _is_leke_like(contour: np.ndarray, max_area: float) -> bool:
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray, cheeks_mask: np.ndarray):
    """
    Tek bir tespit sureci - hem delta haritasini hem de kontur listesini
    birlikte dondurur (Akne.5/Gozenek ailesindeki tutarlilik prensibi:
    skor ve gorsellestirme AYNI kaynaga dayanmali).
    """
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    delta = _compute_delta(img_bgr, cheeks_mask)

    if not (scan_mask_u8 > 0).any():
        return delta, []

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    binary = (delta > DETECTION_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = find_detection_contours(binary, min_area=min_area)

    max_spot_area = max(min_area * 4, scan_px * MAX_SPOT_AREA_RATIO)
    leke_contours = [c for c in contours if _is_leke_like(c, max_spot_area)]

    return delta, leke_contours


def leke_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, cheeks_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    delta, _ = _detect(img_bgr, face_scan_mask, cheeks_mask)
    delta_positive = np.clip(delta, 0, None)

    normalized = np.clip(delta_positive / DELTA_VISUAL_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray, cheeks_mask: np.ndarray) -> list:
    _, contours = _detect(img_bgr, face_scan_mask, cheeks_mask)
    return contours


def leke_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray, cheeks_mask: np.ndarray):
    """
    Skor, tespit edilen lekelerin SAYISI/SIDDETINDEN hesaplaniyor -
    ALANA DEGIL (Akne.5/Gozenek.3'un cozunurluk-kararsizligi hatasindan
    kacinmak icin). 0 leke tespit edildiyse skor MATEMATIKSEL OLARAK
    0.0'dir.

    Returns:
        (score, mean_leke_delta, affected_area_pct, region_count)
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0.0, 0

    delta, contours = _detect(img_bgr, face_scan_mask, cheeks_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0

    weighted_count = 0.0
    total_area = 0
    deltas = []

    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        area = int(region_pixels.sum())
        if area == 0:
            continue

        mean_delta = float(delta[region_pixels].mean())
        severity = float(np.clip(mean_delta / DELTA_VISUAL_CAP, 0, 1))

        weighted_count += severity
        total_area += area
        deltas.append(mean_delta)

    score = float(np.clip(weighted_count / COUNT_CAP, 0, 1))
    mean_leke_delta = float(np.mean(deltas)) if deltas else 0.0
    affected_area_pct = float(total_area / scan_area * 100)

    return score, mean_leke_delta, affected_area_pct, len(contours)
