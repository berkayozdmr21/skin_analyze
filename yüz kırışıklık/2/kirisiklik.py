"""
YUZ KIRISIKLIK.2 - Yerel Arka Plan (Morfolojik Kapama) Referansi

## Kullanicinin bildirdigi sorun (ekran goruntusuyle)

Yasli bir kadinin YOGUN kirisikliklerle kapli yuzunde, Yuz Kirisiklik.1
gozle GORULUR pek cok kirisikligi KACIRIYORDU - sadece 21 kucuk tespit,
oysa fotografta alin, goz cevresi, yanaklar, cene DAHIL neredeyse yuzun
TAMAMI derin kirisikliklarla kapliydi.

## Kok neden (Yuz Lekesi.5'teki AYNI sorun sinifi)

Yuz Kirisiklik.1, koyuluk referansini TUM YUZUN medyani/MAD'i ile
hesapliyordu. Ama eger yuzun BUYUK bir kismi ZATEN kirisiklarla kapliysa,
bu medyan/MAD REFERANSININ KENDISI kirisikliklardan etkilenip
KARARIYOR/BOZULUYOR - yani "normal/kirisiksiz ten" tahmini artik dogru
degil, kismen KIRISIKLI tenin ortalamasi oluyor. Sonuc: gercek bir
kirisiklik pikseli, artik "zaten kirisikli" olan referansa gore o kadar
"anormal koyu" gorunmuyor - KIRISIKLIKLAR BIRBIRINI MASKELIYOR.

Bu, Yuz Lekesi.5'te "yogun cil kumelerinin ortalamayi asagi cekip
birbirini maskelemesi" ile BIREBIR AYNI sorun sinifi - ve AYNI COZUM
burada da uygulaniyor.

## Cozum: morfolojik KAPAMA ile yerel arka plan tahmini

Tum yuzun TEK bir medyan/MAD degeri yerine, HER piksel icin KENDI yerel
cevresinde "kirisiksiz olsaydi nasil gorunurdu" tahmini yapiliyor -
Yuz Lekesi.5'teki ayni teknik: morfolojik KAPAMA (once yerel maksimum -
dilate, sonra yerel minimum - erode), KUCUK KOYU cizgileri (kirisikliklari)
"gormezden gelerek" onlarin ALTINDA olmasi beklenen DUZ ten yuzeyini
yeniden insa eder.

    arka_plan(x,y) = closing(L)(x,y)   [kirisikliklari "doldurarak" duz ten tahmini]
    delta(x,y)      = arka_plan(x,y) - L(x,y)

Kapama cekirdegi, TEK BIR kirisiklik cizgisinin GENISLIGINDEN (kalinligindan)
BUYUK olmali (ki onu "gormezden gelebilsin") - bu, cizginin UZUNLUGUNDAN
bagimsiz calisir (kapama YEREL bir islemdir), yani cok UZUN ama INCE
kirisikliklar bile dogru sekilde "arka plana gomulup" cikarilir.

Goz/kas/dudak gibi DISLANAN alanlarin kapamayi yanlis yonlendirmemesi
icin, once bu alanlar `cv2.inpaint` ile civar ten degerleriyle
dolduruluyor (Yuz Lekesi.5 ile ayni teknik).

## Degismeyenler (Yuz Kirisiklik.1 ile ayni)

- Cok yonelimli (6 acida) Gabor filtre bankasi.
- Uzama (elongation) filtresi.
- Tespit alani icin kenar/delik marji.
- Skor: agirlikli UZUNLUK, yuzun karakteristik uzunluguna oranli
  (alana DEGIL) - cozunurluk kararliligi icin.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

GABOR_ORIENTATIONS_DEG = (0, 30, 60, 90, 120, 150)

GABOR_KSIZE_RATIO = 0.030
MIN_GABOR_KSIZE = 7
MAX_GABOR_KSIZE = 41
GABOR_LAMBDA_RATIO = 0.022
MIN_GABOR_LAMBDA = 4.0
GABOR_SIGMA_FACTOR = 0.30
GABOR_GAMMA = 0.5

# Yuz Lekesi.5'teki ayni ders: yerel referansa gecince, gercek
# kirisikliklarin cogu daha DOGRU/GUCLU sekilde ayirt edilebiliyor -
# bu yuzden Gabor esigi Yuz Kirisiklik.1'deki 88'den dusuruldu (daha
# once "gurultu" sayilip elenen zayif ama gercek kirisiklik sinyalleri
# artik yakalanabiliyor, cunku esas secicilik artik YEREL KOYULUK
# kontrolunden geliyor).
GABOR_STRENGTH_PERCENTILE = 75.0

# YENI: yerel koyuluk referansi icin morfolojik kapama cekirdegi -
# tek bir kirisiklik cizgisinin GENISLIGINDEN buyuk olmali.
CLOSING_KERNEL_RATIO = 0.020
MIN_CLOSING_KERNEL = 5
MAX_CLOSING_KERNEL = 25

# Yerel koyuluk delta esigi (L* birimi) - Yuz Kirisiklik.1'deki
# medyan/MAD z-score yerine, artik DOGRUDAN yerel arka plan farki
# kullaniliyor (Yuz Lekesi.1'in DELTA_VISUAL_CAP mantigiyla ayni ruhta).
DARK_DELTA_THRESHOLD = 4.0
DELTA_VISUAL_CAP = 14.0

CLEAN_KERNEL_RATIO = 0.006
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 11

MIN_AREA_RATIO = 0.00012
MIN_AREA_FLOOR = 5

MIN_ELONGATION_RATIO = 2.0
MAX_LENGTH_RATIO_OF_FACE = 0.55

LENGTH_DENSITY_CAP = 4.0

DETECTION_MARGIN_RATIO = 0.07
MIN_DETECTION_MARGIN = 9
MAX_DETECTION_MARGIN = 65


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: float, max_k: float) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(int(min_k), min(int(max_k), k))
    if k % 2 == 0:
        k += 1
    return k


def _face_scale(face_scan_mask: np.ndarray) -> float:
    scan_px = int((face_scan_mask > 0).sum())
    return float(np.sqrt(max(scan_px, 1)))


def _gabor_line_strength(l_channel: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    ksize = _relative_kernel(face_scan_mask, GABOR_KSIZE_RATIO, MIN_GABOR_KSIZE, MAX_GABOR_KSIZE)
    face_scale = _face_scale(face_scan_mask)
    lambd = max(MIN_GABOR_LAMBDA, face_scale * GABOR_LAMBDA_RATIO)
    sigma = ksize * GABOR_SIGMA_FACTOR

    responses = []
    for deg in GABOR_ORIENTATIONS_DEG:
        theta = np.deg2rad(deg)
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, GABOR_GAMMA, 0, ktype=cv2.CV_32F
        )
        kernel -= kernel.mean()
        response = cv2.filter2D(l_channel, cv2.CV_32F, kernel)
        responses.append(np.abs(response))

    return np.maximum.reduce(responses)


def _local_background_delta(l_channel: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """
    Once disi birakilan alanlari (goz/kas/dudak/arka plan) inpainting
    ile doldurur, sonra morfolojik KAPAMA ile "kirisiksiz" ten
    yuzeyini yeniden insa eder - bkz. dosya basindaki aciklama.

    Returns:
        delta = arka_plan - L  (pozitif = yerel arka plandan daha koyu)
    """
    l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
    inpaint_mask = (face_scan_mask == 0).astype(np.uint8) * 255

    if (inpaint_mask > 0).any() and (inpaint_mask > 0).sum() < inpaint_mask.size:
        inpainted = cv2.inpaint(l_uint8, inpaint_mask, 5, cv2.INPAINT_TELEA)
    else:
        inpainted = l_uint8

    kernel_size = _relative_kernel(face_scan_mask, CLOSING_KERNEL_RATIO, MIN_CLOSING_KERNEL, MAX_CLOSING_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(inpainted, cv2.MORPH_CLOSE, kernel).astype(np.float32)

    return background - l_channel


def _compute_candidate_mask(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        zero_u8 = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        zero_f = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        return zero_u8, zero_f

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    gabor_strength = _gabor_line_strength(l_channel, face_scan_mask)
    gabor_cutoff = float(np.percentile(gabor_strength[scan_region], GABOR_STRENGTH_PERCENTILE))
    strong_line = gabor_strength > gabor_cutoff

    dark_delta = _local_background_delta(l_channel, face_scan_mask)
    dark_enough = dark_delta > DARK_DELTA_THRESHOLD

    candidate = (strong_line & dark_enough).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, (face_scan_mask > 0).astype(np.uint8) * 255)

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, clean_kernel_size - 2), max(3, clean_kernel_size - 2)))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    return cleaned, dark_delta


def _elongation_ratio(contour: np.ndarray) -> float:
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    short = min(w, h)
    long_ = max(w, h)
    if short <= 1e-3:
        return float("inf") if long_ > 1 else 0.0
    return long_ / short


def _contour_length(contour: np.ndarray) -> float:
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return float(max(w, h))


def _is_wrinkle_like(contour: np.ndarray, face_scale: float) -> bool:
    elongation = _elongation_ratio(contour)
    if elongation < MIN_ELONGATION_RATIO:
        return False

    length = _contour_length(contour)
    if length > face_scale * MAX_LENGTH_RATIO_OF_FACE:
        return False

    return True


def _erode_for_detection(face_scan_mask: np.ndarray) -> np.ndarray:
    margin = _relative_kernel(face_scan_mask, DETECTION_MARGIN_RATIO, MIN_DETECTION_MARGIN, MAX_DETECTION_MARGIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    eroded = cv2.erode(face_scan_mask, kernel, iterations=1)

    if not (eroded > 0).any():
        return face_scan_mask
    return eroded


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """Tek bir tespit sureci - yerel koyuluk deltasi ve kontur listesi birlikte doner."""
    candidate_mask, dark_delta = _compute_candidate_mask(img_bgr, face_scan_mask)

    detection_mask = _erode_for_detection(face_scan_mask)
    candidate_mask = cv2.bitwise_and(candidate_mask, (detection_mask > 0).astype(np.uint8) * 255)

    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    raw_contours = find_detection_contours(candidate_mask, min_area=min_area)

    face_scale = _face_scale(face_scan_mask)
    wrinkle_contours = [c for c in raw_contours if _is_wrinkle_like(c, face_scale)]

    return dark_delta, wrinkle_contours


def wrinkle_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    dark_delta, contours = _detect(img_bgr, face_scan_mask)

    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    delta_positive = np.clip(dark_delta, 0, None)
    normalized = np.clip(delta_positive / DELTA_VISUAL_CAP, 0, 1)
    normalized[mask == 0] = 0
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (contours, severities) - her kontur icin 0-1 arasi siddet degeri.
    """
    dark_delta, contours = _detect(img_bgr, face_scan_mask)

    severities = []
    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_delta = float(dark_delta[region_pixels].mean()) if region_pixels.any() else 0.0
        severities.append(float(np.clip(mean_delta / DELTA_VISUAL_CAP, 0, 1)))

    return contours, severities


def wrinkle_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, tespit edilen kirisikliklerin AGIRLIKLI TOPLAM UZUNLUGUNDAN
    hesaplaniyor - yuzun karakteristik UZUNLUGUNA oranli (Yuz
    Kirisiklik.1 ile ayni tasarim, bkz. o surumun README'si).

    Returns:
        (score, weighted_length_px, total_length_px, mean_delta, region_count)
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return 0.0, 0.0, 0.0, 0.0, 0

    dark_delta, contours = _detect(img_bgr, face_scan_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0.0, 0

    weighted_length = 0.0
    total_length = 0.0
    delta_values = []

    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        length = _contour_length(c)

        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_delta = float(dark_delta[region_pixels].mean()) if region_pixels.any() else 0.0
        severity = float(np.clip(mean_delta / DELTA_VISUAL_CAP, 0, 1))

        weighted_length += length * severity
        total_length += length
        delta_values.append(mean_delta)

    face_scale = _face_scale(face_scan_mask)
    score = float(np.clip(weighted_length / (face_scale * LENGTH_DENSITY_CAP), 0, 1))
    mean_delta_overall = float(np.mean(delta_values)) if delta_values else 0.0

    return score, weighted_length, total_length, mean_delta_overall, len(contours)
