"""
YUZ LEKESI.4 - Yerel Uyarlamali Esikleme (Adaptive Thresholding)

## Farkli teknik, ayni islev

Yuz Lekesi.1-3, hep AYNI temel teknige dayaniyordu: bir REFERANS
BOLGE'nin (yanak, sonra tum yuz) istatistigini (ortalama/std veya
medyan/MAD) hesaplayip, HER pikseli bu TEK, KURESEL (global/bolgesel)
referansla kiyaslamak. Bu, klasik bir "global esikleme" yaklasimidir.

Bu surumde TAMAMEN FARKLI bir teknik kullaniliyor: **yerel uyarlamali
esikleme** (adaptive thresholding) - goruntu isleme literaturunde
`cv2.adaptiveThreshold` ile bilinen, klasik ama BAMBASKA bir yontem.
Fikir: her pikseli TUM YUZUN ortalamasiyla degil, SADECE KENDI YAKIN
CEVRESININ (kucuk bir pencere/blok) ortalamasiyla kiyasla:

    yerel_ortalama(x,y) = pikselin cevresindeki KUCUK bir pencerenin
                           (Gauss agirlikli) ortalamasi
    delta(x,y) = yerel_ortalama(x,y) - piksel(x,y)

Bu, Yuz Lekesi.1-3'teki "referans bolgesi" kavramini TAMAMEN ORTADAN
KALDIRIYOR - artik ne yanak ne de tum yuz referans aliniyor, HER
PIKSELIN KENDI YEREL KOMSULUGU kendi referansi.

## Bu teknigin dogal bir avantaji: Yuz Lekesi.3'un sorununu COZUYOR

Yuz Lekesi.3'te ugrastigimiz sorun (buyuk bir vitiligo/melazma alaninin
GLOBAL/bolgesel referansi bozmasi) burada DOGASI GEREGI ortadan
kalkiyor: yerel pencere KUCUK oldugu icin (yuz olceginin kucuk bir
orani), UZAKTAKI buyuk bir atipik bolge, bir pikselin YEREL komsulugunu
ETKILEMEZ - o pencere zaten o bolgeye ulasmiyor. Medyan/MAD gibi
"robust ama yine de TUM veriye bakan" bir cozume bile gerek kalmiyor,
cunku sorun kaynagindan (yerel olmayan referans) ortadan kaldiriliyor.

## Maske-duyarli yerel ortalama (normalize edilmis konvolusyon)

Yerel ortalama hesaplanirken, saf bir Gauss bulaniklastirma
KULLANILMIYOR - cunku bu, maske sinirina yakin piksellerde SAC/ARKA
PLAN degerlerini de ortalamaya karistirirdi (Yuz Lekesi.3'teki sinir
artefakti sorununun BASKA bir kaynagi). Bunun yerine, Goz Alti
Morluk.5'teki ile ayni "normalize edilmis konvolusyon" teknigi
kullaniliyor - yerel ortalama SADECE ten bolgesi piksellerinden
hesaplaniyor:

    yerel_ortalama = GaussianBlur(ten_maskesi x L) / GaussianBlur(ten_maskesi)

## Degismeyenler (Yuz Lekesi.1-3 ile ayni islev)

- Hedef bolge: tum taranabilir yuz cildi.
- Sekil filtresi (dairesellik + maksimum alan).
- Cozunurlukten bagimsiz (goreli) boyutlandirma.
- Skor: sayi/siddet agirlikli (alan degil).
- Isaretleme: siddet-renkli numarali daire.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Yerel pencere (blok) boyutu, yuz olcegine oranli - bir lekenin
# cevresindeki "normal ten" ornegini almaya yetecek kadar genis, ama
# baska bir anatomik bolgeye tasmayacak kadar dar.
#
# NOT (iki basarisiz deneme sonrasi bulunan gercek sorun): kucuk bir
# pencere (0.05/taban 15) ince ten dokusuna asiri hassas kaliyordu;
# buyuk bir pencere (0.09/taban 25) ise GOZ/KAS/DUDAK "delikleri"nin
# KENARLARINDAKI normalize-konvolusyon artefaktlarini DAHA GENIS bir
# alana yayiyordu (delik kenarina yakin piksellerde yerel ortalama
# guvenilmez oluyor, bu bolge buyudukce daha fazla sahte tespit
# olusuyordu). Gercek cozum pencere boyutunu "dogru" ayarlamak degil,
# tespit alanini bu kenar bolgelerinden ARINDIRMAKTI - bkz.
# DETECTION_MARGIN_RATIO ve _erode_for_detection().
BLOCK_SIZE_RATIO = 0.045
MIN_BLOCK_SIZE = 21
MAX_BLOCK_SIZE = 61

# Esikleme sabiti (L* birimi) - bir pikselin, KENDI yerel ortalamasindan
# bu kadar koyu olmasi gerekiyor "leke adayi" sayilmasi icin.
C_OFFSET = 13.0

# YENI: tespit alani, delik (goz/kas/dudak) ve maske kenarlarindan bu
# oranda icebukuluyor - normalize konvolusyonun guvenilmez oldugu kenar
# bolgesini DETECTION icin devre disi birakiyor (yerel ortalama HESABI
# icin hala tam maske kullaniliyor, sadece SONUC bu daha dar bolgede
# degerlendiriliyor).
DETECTION_MARGIN_RATIO = 0.05
MIN_DETECTION_MARGIN = 9
MAX_DETECTION_MARGIN = 41

# Siddet/isi haritasi normalizasyonu icin ust sinir (L* birimi) - Yuz
# Lekesi.1 ile ayni sabit (ayni kanal, ayni olcek mantigi).
DELTA_VISUAL_CAP = 30.0

CLEAN_KERNEL_RATIO = 0.008
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 15

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

MIN_CIRCULARITY = 0.22
MAX_SPOT_AREA_RATIO = 0.02

# Teknik degistigi icin (yerel delta, farkli bir dagilima sahip) gercek
# bir test fotografiyla yeniden kalibre edildi (bkz. README.md).
COUNT_CAP = 18.0


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _local_mean(l_channel: np.ndarray, face_scan_mask: np.ndarray, block_size: int) -> np.ndarray:
    """
    Maske-duyarli yerel ortalama - normalize edilmis konvolusyon
    (Goz Alti Morluk.5 ile ayni teknik). Sadece ten bolgesi pikselleri
    katkida bulunur, sac/arka plan degil.
    """
    mask_f = (face_scan_mask > 0).astype(np.float32)
    weighted = l_channel * mask_f

    blurred_weighted = cv2.GaussianBlur(weighted, (block_size, block_size), 0)
    blurred_mask = cv2.GaussianBlur(mask_f, (block_size, block_size), 0)

    return blurred_weighted / np.clip(blurred_mask, 1e-6, None)


def _compute_delta_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    delta(x,y) = yerel_ortalama(x,y) - L(x,y) - pozitif deger, pikselin
    KENDI yakin cevresinden daha koyu oldugu anlamina gelir.

    Returns:
        (delta_map, block_size)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32), MIN_BLOCK_SIZE

    block_size = _relative_kernel(face_scan_mask, BLOCK_SIZE_RATIO, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
    local_mean = _local_mean(l_channel, face_scan_mask, block_size)

    delta = local_mean - l_channel
    return delta, block_size


def _is_leke_like(contour: np.ndarray, max_area: float) -> bool:
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def _erode_for_detection(face_scan_mask: np.ndarray) -> np.ndarray:
    """
    Tespit alanini delik (goz/kas/dudak) ve maske kenarlarindan icebuker
    - normalize konvolusyonun guvenilmez oldugu kenar bolgesini DISI
    birakir. Yerel ortalama HESABI icin bu fonksiyon KULLANILMAZ (tam
    maske kullanilir) - sadece SONUCUN degerlendirildigi bolgeyi daraltir.
    """
    margin = _relative_kernel(face_scan_mask, DETECTION_MARGIN_RATIO, MIN_DETECTION_MARGIN, MAX_DETECTION_MARGIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    eroded = cv2.erode(face_scan_mask, kernel, iterations=1)

    if not (eroded > 0).any():
        return face_scan_mask
    return eroded


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """Tek bir tespit sureci - delta haritasi ve kontur listesi birlikte doner."""
    delta, _ = _compute_delta_map(img_bgr, face_scan_mask)

    detection_mask = _erode_for_detection(face_scan_mask)
    scan_mask_u8 = (detection_mask > 0).astype(np.uint8) * 255

    if not (scan_mask_u8 > 0).any():
        return delta, []

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    scan_px = int((detection_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    binary = (delta > C_OFFSET).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = find_detection_contours(binary, min_area=min_area)

    max_spot_area = max(min_area * 4, scan_px * MAX_SPOT_AREA_RATIO)
    leke_contours = [c for c in contours if _is_leke_like(c, max_spot_area)]

    return delta, leke_contours


def leke_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    delta, _ = _detect(img_bgr, face_scan_mask)
    delta_positive = np.clip(delta, 0, None)

    normalized = np.clip(delta_positive / DELTA_VISUAL_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (contours, severities)
    """
    delta, contours = _detect(img_bgr, face_scan_mask)

    severities = []
    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_delta = float(delta[region_pixels].mean()) if region_pixels.any() else 0.0
        severities.append(float(np.clip(mean_delta / DELTA_VISUAL_CAP, 0, 1)))

    return contours, severities


def leke_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, tespit edilen lekelerin SAYISI/SIDDETINDEN hesaplaniyor.
    0 leke tespit edildiyse skor MATEMATIKSEL OLARAK 0.0'dir.

    Returns:
        (score, mean_leke_delta, affected_area_pct, region_count)
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0.0, 0

    delta, contours = _detect(img_bgr, face_scan_mask)

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
