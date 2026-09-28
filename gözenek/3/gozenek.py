"""
GOZENEK.3 - Sayi (Yogunluk) Tabanli Skor

## Kullanicinin bildirdigi iki sorun

1. "Tutarsiz - gozenek arttikca skor artsin." Gozenek.2'de skor,
   ALAN-agirlikli bir "yuk" idi: `skor = sum(siddet_i * alan_i) / ...`.
   Bu, PIKSEL ALANINA gore agirlikliydi - COUNT'a degil. Sonuc: az
   sayida ama BUYUK gozenek/leke, cok sayida ama KUCUK gozenekten daha
   yuksek skor verebiliyordu. Kullanicinin sezgisel beklentisi ("daha
   COK gozenek = daha YUKSEK skor") ile bu davranis UYUSMUYORDU -
   cunku skor, SAYIYA degil ALANA bakiyordu.

2. "Yuzdeler az geliyor, arttir." `BURDEN_AREA_FRACTION = 0.15`
   ("yuzun %15'i kaplansa skor %100") gercekci hicbir fotografta
   yaklasilamayacak kadar yuksek bir esikti - gercek bir yuzde gorunur
   gozeneklerin kapladigi alan yuzdesi (`affected_area_pct`) genelde
   %1-3 civarinda kaliyor, bu da skoru sistematik olarak DUSUK
   gosteriyordu.

## Cozum: skor artik SAYI (yogunluk) tabanli

Alan-agirlikli "yuk" yerine, her tespit edilen gozenek KENDI SIDDETIYLE
(0-1 arasi, z-score'dan) sayilıyor - alani DEGIL:

    agirlikli_sayi = sum(siddet_i)   [i = her tespit edilen gozenek]

Bu, ONEMLI bir garanti getiriyor: her siddeti sifirdan buyuk (yani
DETECTION_Z_THRESHOLD'u gecmis, dolayisiyla HER ZAMAN pozitif) bir
gozenek tespiti, toplami KESINLIKLE ARTIRIR - hicbir zaman AZALTMAZ.
Yani "daha fazla gozenek tespit edildi -> skor hicbir zaman dusmez,
genellikle artar" garantisi matematiksel olarak saglaniyor - kullanicinin
istedigi tutarlilik bu.

Skor, bu agirlikli sayiyi yuzun buyuklugune (karsilastirilabilirlik
icin) oranlayip normalize ediyor:

    yogunluk = agirlikli_sayi / (taranan_yuz_alani / 100)   [yuzde-1'lik alana dusen agirlikli gozenek sayisi]
    skor = clip(yogunluk / DENSITY_CAP, 0, 1)

`DENSITY_CAP = 0.20` gercek bir test fotografiyla kalibre edildi:
gozle GORULUR olcude gozenekli bir yuzde (119 tespit, ortalama siddet
~0.91) yogunluk ~0.116 cikiyordu - bu deger `%58` gibi, artik daha
"hissedilir" bir skora karsilik geliyor (Gozenek.2'deki `%14.5`
yerine).

## Degismeyenler (Gozenek.2 ile ayni)

- Black-hat morfolojik donusum, cozunurlukten bagimsiz cekirdek.
- BOLGESEL (alin/burun/cene/yanak ayri) z-score referansi.
- Sekil filtresi (dairesellik + maksimum alan) - bu SADECE tespit
  asamasinda hangi bolgelerin "gozenek" sayilacagini belirliyor, skor
  hesaplamasini degil.
- Bolge dagilimi raporlamasi.

`affected_area_pct` hala raporlaniyor (seffaflik icin) ama artik
SKORUN kendisini BELIRLEMIYOR - sadece ek bilgi.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

BLACKHAT_KERNEL_RATIO = 0.018
MIN_BLACKHAT_KERNEL = 3
MAX_BLACKHAT_KERNEL = 21

Z_SCORE_CAP = 2.5
DETECTION_Z_THRESHOLD = 1.3

CLEAN_KERNEL_RATIO = 0.006
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 9

MIN_AREA_RATIO = 0.00004
MIN_AREA_FLOOR = 3

MIN_CIRCULARITY = 0.3
MAX_SPOT_AREA_RATIO = 0.0025

MIN_REGION_PIXELS = 300

# YENI: skor artik agirlikli SAYI (yogunluk) tabanli - bkz. yukaridaki
# aciklama. DENSITY_CAP, "yuzun %1'lik alanina dusen agirlikli gozenek
# sayisi bu degere ulasirsa skor %100" anlamina gelir - gercek bir test
# fotografiyla kalibre edildi (bkz. README.md).
DENSITY_CAP = 0.20


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _compute_pore_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    """Black-hat donusumu + bolgesel z-score (Gozenek.2 ile ayni)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[..., 0]

    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32), MIN_BLACKHAT_KERNEL

    kernel_size = _relative_kernel(face_scan_mask, BLACKHAT_KERNEL_RATIO, MIN_BLACKHAT_KERNEL, MAX_BLACKHAT_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    global_mean = float(np.mean(blackhat[scan_region]))
    global_std = float(np.std(blackhat[scan_region])) + 1e-6

    z = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    covered = np.zeros(img_bgr.shape[:2], dtype=bool)

    for mask in region_scan_masks.values():
        region_px = mask > 0
        count = int(region_px.sum())
        if count < MIN_REGION_PIXELS:
            continue

        r_mean = float(np.mean(blackhat[region_px]))
        r_std = float(np.std(blackhat[region_px])) + 1e-6

        z[region_px] = (blackhat[region_px] - r_mean) / r_std
        covered |= region_px

    uncovered = scan_region & ~covered
    if uncovered.any():
        z[uncovered] = (blackhat[uncovered] - global_mean) / global_std

    return z, kernel_size


def _is_pore_like(contour: np.ndarray, max_area: float) -> bool:
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    z, _ = _compute_pore_z_map(img_bgr, face_scan_mask, region_scan_masks)

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
    pore_contours = [c for c in contours if _is_pore_like(c, max_spot_area)]

    return z, pore_contours


def pore_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    z, _ = _detect(img_bgr, face_scan_mask, region_scan_masks)
    normalized = np.clip(z / Z_SCORE_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict) -> list:
    _, contours = _detect(img_bgr, face_scan_mask, region_scan_masks)
    return contours


def _region_of_contour(contour: np.ndarray, region_scan_masks: dict) -> str:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        x, y = contour[0][0]
    else:
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])

    for name, mask in region_scan_masks.items():
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            return name
    return "diger"


def pore_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    """
    Skor artik AGIRLIKLI SAYI (yogunluk) tabanli - bkz. dosya basindaki
    aciklama. Her tespit edilen gozenek, alani NE OLURSA OLSUN, kendi
    siddetiyle (>0) toplama katkida bulunur - bu yuzden "daha fazla
    gozenek -> skor hicbir zaman dusmez" garantisi var.

    Returns:
        (score, mean_pore_z, affected_area_pct, region_count, region_breakdown)

        affected_area_pct hala raporlaniyor (seffaflik) ama artik
        SKORU belirlemiyor.
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0.0, 0, {}

    z, contours = _detect(img_bgr, face_scan_mask, region_scan_masks)

    if not contours:
        return 0.0, 0.0, 0.0, 0, {}

    weighted_count = 0.0
    total_area = 0
    severities = []
    breakdown: dict = {}

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
        severities.append(mean_z)

        region_name = _region_of_contour(c, region_scan_masks)
        breakdown[region_name] = breakdown.get(region_name, 0) + 1

    density = weighted_count / (scan_area / 100)  # yuzde-1'lik alana dusen agirlikli gozenek sayisi
    score = float(np.clip(density / DENSITY_CAP, 0, 1))

    mean_pore_z = float(np.mean(severities)) if severities else 0.0
    affected_area_pct = float(total_area / scan_area * 100)

    return score, mean_pore_z, affected_area_pct, len(contours), breakdown
