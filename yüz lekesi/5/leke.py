"""
YUZ LEKESI.5 - Morfolojik Kapama (Closing) Tabanli Arka Plan Tahmini

## Kullanicinin bildirdigi sorun (ekran goruntusuyle)

Yuz Lekesi.4, yogun cilli/benli bir yuzde GOZLE GORULUR pek cok lekeyi
KACIRIYORDU - "40 tespit" derken fotografta bariz sekilde COK DAHA
FAZLA leke vardi. Bu bir yanlis-pozitif (false positive) degil, bir
RECALL (kacirma) sorunuydu.

## Kok neden

Yuz Lekesi.4, "yerel ortalama"yi Gauss BULANIKLASTIRMA ile
hesapliyordu (komsu piksellerin AGIRLIKLI ORTALAMASI). Ama yogun cilli
bir bolgede, bir pikselin yakin cevresi zaten BASKA cillerle dolu -
yani "yerel ortalama"nin KENDISI de bu cillerden dolayi asagi
cekiliyor. Sonuc: bir cil, cevresindeki DIGER cillerle kiyaslandiginda
artik o kadar "farkli/koyu" gorunmuyor - benekler BIRBIRINI
MASKELIYOR. Bu, ortalama-tabanli her yontemin (Yuz Lekesi.1-4 dahil)
paylastigi bir zayiflik: YOGUN kumeler halinde bir "seyin", kendi
ortalamasi da o "seyden" etkilenir.

## Cozum: ortalama yerine morfolojik KAPAMA (closing) ile arka plan
## tahmini

`ortalama` yerine, Gozenek ailesindeki black-hat teknigiyle AYNI
matematiksel araci - ama TERSTEN, bir arka plan TAHMIN EDICI olarak -
kullaniyoruz. Morfolojik KAPAMA (once yerel MAKSIMUM alma - dilate,
sonra yerel MINIMUM alma - erode) islemi, KUCUK KOYU noktalari
(cilleri) "gormezden gelerek" onlarin ALTINDA/ARASINDA olmasi beklenen
DUZ TEN yuzeyini yeniden insa eder:

    arka_plan(x,y) = closing(L)(x,y)   [cilleri "doldurarak" DUZ ten tahmini]
    delta(x,y)      = arka_plan(x,y) - L(x,y)

**Neden bu isliyor:** Kapamanin ilk adimi (dilate = yerel maksimum),
KUCUK koyu benekleri (kapama cekirdeginden kucukse) TAMAMEN yok sayar -
o bolgede en YAKIN parlak/normal ten degerini "yayar". Ikinci adim
(erode) bu yayilmayi geri daraltir ama koyu benekler bir daha
GORUNMEZ - cunku zaten ilk adimda "silinmislerdi". Bu, YOGUN cil
kumelerinde bile, HER BIR cilin kendi ETRAFINDAKI GERCEK (cilsiz)
ten seviyesine gore dogru degerlendirilmesini sagliyor - ortalama
gibi komsu benekler tarafindan "asagi cekilmiyor".

## Delik (goz/kas/dudak) kontaminasyonunu onlemek: inpainting

Kapama islemi mask-duyarli (normalize konvolusyon gibi) YAPILAMAZ -
bu yuzden once goz/kas/dudak/arka plan gibi disi birakilan alanlar
`cv2.inpaint` ile civarindaki ten degerleriyle "dolduruluyor" (gecici,
sadece arka plan hesabi icin), SONRA kapama uygulaniyor. Boylece
kapama, gercekten koyu olan goz/dudak bolgelerinden yanlislikla
etkilenmiyor.

## Degismeyenler (Yuz Lekesi.4 ile ayni islev/felsefe)

- Yerel (global degil) yaklasim korunuyor - sadece "yerel ortalama"
  yerine "yerel arka plan tahmini" kullaniliyor.
- Hedef bolge, sekil filtresi, cozunurlukten bagimsiz boyutlandirma,
  sayi/siddet agirlikli skor, siddet-renkli numarali daire isaretleme.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Kapama cekirdeginin boyutu, yuz olcegine oranli - tek bir cilden
# BUYUK olmali (aksi halde cili "gormezden gelemez") ama anatomik
# ozellikleri (burun, cene hatti) bozmayacak kadar kucuk kalmali.
CLOSING_KERNEL_RATIO = 0.035
MIN_CLOSING_KERNEL = 9
MAX_CLOSING_KERNEL = 45

# Esikleme sabiti (L* birimi). NOT: kapama-tabanli delta, bulanikligi
# temelli delta'dan cok daha GENIS bir dagilima sahip (uzun bir kuyruk -
# p95=21, p99=49, bkz. README.md) - bu yuzden ilk denemedeki 9.0 COK
# dusuk kaldi (pikselerin >%10'unu esik-ustu sayiyordu). Gercek dagilim
# olculerek p95-p97 civarina kalibre edildi.
C_OFFSET = 25.0

DELTA_VISUAL_CAP = 30.0

CLEAN_KERNEL_RATIO = 0.006
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 11

MIN_AREA_RATIO = 0.00010
MIN_AREA_FLOOR = 4

# Sekil filtresi biraz gevsetildi - Yuz Lekesi.4'te bazi gercek (kucuk,
# hafif duzensiz) cilller dairesellik/alan filtresinden eleniyordu.
MIN_CIRCULARITY = 0.18
MAX_SPOT_AREA_RATIO = 0.02

# Tespit alani, delik/kenar bolgelerinden hala biraz icebukuluyor -
# ama artik arka plan hesabi inpainting ile kontamine olmadigi icin
# Yuz Lekesi.4'teki paydan (0.05) daha kucuk bir marj yeterli.
DETECTION_MARGIN_RATIO = 0.02
MIN_DETECTION_MARGIN = 5
MAX_DETECTION_MARGIN = 21

# Gercek tespit sayisi arttigi icin (yogun cilli yuzlerde) yeniden
# kalibre edildi - bkz. README.md.
COUNT_CAP = 55.0


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _background_estimate(l_channel: np.ndarray, face_scan_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Once disi birakilan alanlari (goz/kas/dudak/arka plan) inpainting
    ile doldurur, sonra morfolojik KAPAMA ile "cilsiz/lekesiz" ten
    yuzeyini yeniden insa eder - bkz. dosya basindaki aciklama.
    """
    l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
    inpaint_mask = (face_scan_mask == 0).astype(np.uint8) * 255

    if (inpaint_mask > 0).any() and (inpaint_mask > 0).sum() < inpaint_mask.size:
        inpainted = cv2.inpaint(l_uint8, inpaint_mask, 5, cv2.INPAINT_TELEA)
    else:
        inpainted = l_uint8

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(inpainted, cv2.MORPH_CLOSE, kernel)

    return background.astype(np.float32)


def _compute_delta_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    delta(x,y) = arka_plan(x,y) - L(x,y) - pozitif deger, pikselin
    KENDI civarindaki (cilsiz) ten seviyesinden daha koyu oldugu
    anlamina gelir.

    Returns:
        (delta_map, kernel_size)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32), MIN_CLOSING_KERNEL

    kernel_size = _relative_kernel(face_scan_mask, CLOSING_KERNEL_RATIO, MIN_CLOSING_KERNEL, MAX_CLOSING_KERNEL)
    background = _background_estimate(l_channel, face_scan_mask, kernel_size)

    delta = background - l_channel
    return delta, kernel_size


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
