"""
YAGLILIK.4 - Doku Puruzsuzlugu (Laplacian Varyansi) Ikinci Sinyali

## Farklilastirma: HSV'ye ek bir BAGIMSIZ sinyal

Arastirma sirasinda (bkz. Yaglilik.1 README) bulunan bir ipucu tam
kullanilmamisti: "Beyond specular highlight detection, Local Binary
Patterns or Laplacian variance over patches can be used to COMPLEMENT
the oiliness detection." Yaglilik.1-3 SADECE HSV specular highlight
(parlaklik+doygunluk) sinyaline dayaniyordu. Bu surumde, arastirmanin
onerdigi TAMAMLAYICI sinyal eklendi: **doku puruzsuzlugu** (Laplacian
varyansi).

## Neden ayri bir sinyal gerekiyor

HSV specular testi, "cok parlak VE cok az doygun" olan (aynasal
yansima noktalari) pikselleri buluyor - bu KESIN parlama noktalarini
yakalar ama YAGLI CILDIN GENEL GORUNUMUNU (parlak bir "nokta" olmasa
bile PURUZSUZ/gozeneksiz gorunen genis alanlar) KACIRABILIR. Gercek
yagli cilt, sadece birkac parlak nokta degil, genellikle DAHA GENIS bir
alanda "donuk parlaklik" ve DUSUK DOKU (az gorunur gozenek/kil) sergiler
- bu, HSV parlaklik esigini GECMEYEBILIR ama YINE DE tipik bir
"yaglilik" gorunumudur.

**Doku puruzsuzlugu** (Laplacian varyansi - klasik bir "bulaniklik/
netlik" olcusu, goruntu isleme literaturunde yaygin) bunu yakalar:
puruzsuz/yagli bir yuzeyde yerel Laplacian enerjisi DUSUKTUR (netlik
detayi az), gozenekli/kuru/dokulu bir yuzeyde YUKSEKTIR.

## Iki sinyalin birlesimi

    ozel_aday = (HSV specular: yuksek V + dusuk S + cok olcekli duzluk)
    doku_aday = (orta-yuksek parlaklik) VE (COK DUSUK yerel Laplacian enerjisi - bolgesel yuzdelik dilim)

    nihai_aday = ozel_aday VEYA doku_aday

Bu bir "VE" (AND) degil, "VEYA" (OR) birlesimi - Goz Alti Morluk.3/
Akne.3'teki agirlikli-toplam yerine burada IKI BAGIMSIZ KRITER
kullaniliyor: herhangi biri yeterliyse piksel "yagli adayi" sayilir.
Bu, HASSASIYETI ARTIRIR (Yaglilik.1-3'un SADECE parlak nokta arayip
kacirdigi "donuk ama puruzsuz" bolgeleri de yakalar).

## Raporlama

Artik hangi PIKSELLERIN hangi sinyalden geldigi de ayri ayri
izlenebiliyor (`specular_pct` / `texture_pct`) - "bu yaglilik
tespiti belirgin parlama noktalarindan mi, yoksa genel puruzsuz
gorunumden mi geliyor" sorusuna cevap.

## Degismeyenler (Yaglilik.3 ile ayni)

- Bolgesel V/S yuzdelik dilimleri, cok olcekli gradyan dogrulamasi.
- T-bolgesi/yanak ikili gosterimi + cilt tipi siniflandirmasi.
- Kaplama-alani tabanli skor.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

V_PERCENTILE = 92.0
S_PERCENTILE = 35.0

GRADIENT_KERNEL_SMALL = 3
GRADIENT_KERNEL_LARGE = 9
GRADIENT_SMOOTH_PERCENTILE = 70.0

# YENI: doku puruzsuzlugu (Laplacian varyansi) esigi - bolgesel
# yuzdelik dilim, ayni felsefe. Sadece en puruzsuz (dusuk enerjili)
# %TEXTURE_SMOOTH_PERCENTILE'lik kisim "yagli/puruzsuz doku adayi"
# sayilir - bu, HSV testinden cok daha SIKI (kucuk bir yuzdelik dilim)
# cunku "genel puruzsuzluk" tek basina cok genel bir kriter, yanlis-
# pozitifi sinirlamak icin dar tutuldu.
TEXTURE_WINDOW_RATIO = 0.02
MIN_TEXTURE_WINDOW = 5
MAX_TEXTURE_WINDOW = 25
# NOT: ilk denemede %15/medyan-parlaklik COK GEVSEK cikti - normal
# yanak dokusunun buyuk bir kismi bile bu kritere uyuyordu (kaplama
# %5'ten %14'e sicradi, yanaklar yanlislikla "yagli" sayildi). Gercek
# test fotografiyla gozlemlenip SIKILASTIRILDI: sadece EN puruzsuz
# %8'lik dilim VE EN AZ %75'lik dilimde parlaklik - HSV testine daha
# yakin bir sicilik, ama hala ayri/tamamlayici bir sinyal.
TEXTURE_SMOOTH_PERCENTILE = 8.0
# Doku adayinin ayrica "belirgin sekilde parlak" olmasi sarti -
# tamamen karanlik ama duz bolgeleri (golge, sac golgesi) elemek icin.
TEXTURE_MIN_V_PERCENTILE = 75.0

CLEAN_KERNEL_RATIO = 0.010
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 21

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

OILY_COVERAGE_CAP = 0.10

MIN_REGION_PIXELS = 300

OILY_REGION_THRESHOLD_PCT = 8.0

T_ZONE_REGIONS = ("forehead", "nose", "chin")
CHEEK_REGIONS = ("cheeks",)


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _regional_cutoffs(channel: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict, percentile: float):
    scan_region = face_scan_mask > 0
    global_cutoff = float(np.percentile(channel[scan_region], percentile))

    cutoff_map = np.full(channel.shape, global_cutoff, dtype=np.float32)

    for mask in region_scan_masks.values():
        region_px = mask > 0
        count = int(region_px.sum())
        if count < MIN_REGION_PIXELS:
            continue

        region_cutoff = float(np.percentile(channel[region_px], percentile))
        cutoff_map[region_px] = region_cutoff

    return cutoff_map


def _multiscale_smoothness(gray: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0

    gx_s = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=GRADIENT_KERNEL_SMALL)
    gy_s = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=GRADIENT_KERNEL_SMALL)
    grad_small = np.sqrt(gx_s ** 2 + gy_s ** 2)

    gx_l = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=GRADIENT_KERNEL_LARGE)
    gy_l = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=GRADIENT_KERNEL_LARGE)
    grad_large = np.sqrt(gx_l ** 2 + gy_l ** 2)

    if not scan_region.any():
        return np.zeros(gray.shape, dtype=bool)

    cutoff_small = float(np.percentile(grad_small[scan_region], GRADIENT_SMOOTH_PERCENTILE))
    cutoff_large = float(np.percentile(grad_large[scan_region], GRADIENT_SMOOTH_PERCENTILE))

    smooth_small = grad_small < cutoff_small
    smooth_large = grad_large < cutoff_large

    return smooth_small & smooth_large


def _texture_smoothness_candidate(gray: np.ndarray, v_channel: np.ndarray, face_scan_mask: np.ndarray,
                                   region_scan_masks: dict) -> np.ndarray:
    """
    Yerel Laplacian ENERJISI (bulaniklik/netlik olcusu) hesaplar; her
    bolgede en dusuk enerjili (en puruzsuz) %TEXTURE_SMOOTH_PERCENTILE
    VE en az orta parlaklikta olan pikselleri "doku-tabanli yaglilik
    adayi" sayar.
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(gray.shape, dtype=bool)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_energy = laplacian ** 2

    win = _relative_kernel(face_scan_mask, TEXTURE_WINDOW_RATIO, MIN_TEXTURE_WINDOW, MAX_TEXTURE_WINDOW)
    local_energy = cv2.boxFilter(lap_energy, -1, (win, win))

    energy_cutoff_map = _regional_cutoffs(local_energy, face_scan_mask, region_scan_masks, TEXTURE_SMOOTH_PERCENTILE)
    v_cutoff_map = _regional_cutoffs(v_channel, face_scan_mask, region_scan_masks, TEXTURE_MIN_V_PERCENTILE)

    smooth_texture = local_energy < energy_cutoff_map
    bright_enough = v_channel > v_cutoff_map

    return smooth_texture & bright_enough


def _compute_oily_mask(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        zero = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        return zero, zero, zero

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]
    s_channel = hsv[..., 1]

    v_cutoff_map = _regional_cutoffs(v_channel, face_scan_mask, region_scan_masks, V_PERCENTILE)
    s_cutoff_map = _regional_cutoffs(s_channel, face_scan_mask, region_scan_masks, S_PERCENTILE)

    high_v = v_channel > v_cutoff_map
    low_s = s_channel < s_cutoff_map
    specular = high_v & low_s

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = _multiscale_smoothness(gray, face_scan_mask)

    specular_candidate = (specular & smooth).astype(np.uint8) * 255
    specular_candidate = cv2.bitwise_and(specular_candidate, (face_scan_mask > 0).astype(np.uint8) * 255)

    texture_bool = _texture_smoothness_candidate(gray, v_channel, face_scan_mask, region_scan_masks)
    texture_candidate = texture_bool.astype(np.uint8) * 255
    texture_candidate = cv2.bitwise_and(texture_candidate, (face_scan_mask > 0).astype(np.uint8) * 255)

    combined = cv2.bitwise_or(specular_candidate, texture_candidate)

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    return cleaned, specular_candidate, texture_candidate


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    binary, specular_candidate, texture_candidate = _compute_oily_mask(img_bgr, face_scan_mask, region_scan_masks)

    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    contours = find_detection_contours(binary, min_area=min_area)
    return binary, contours, specular_candidate, texture_candidate


def oiliness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    binary, _, _, _ = _detect(img_bgr, face_scan_mask, region_scan_masks)
    normalized = (binary > 0).astype(np.float32)
    normalized[~scan_region] = 0

    return normalized


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict) -> list:
    _, contours, _, _ = _detect(img_bgr, face_scan_mask, region_scan_masks)
    return contours


def _region_coverage_pct(oily: np.ndarray, mask: np.ndarray) -> float:
    region_px = mask > 0
    total = int(region_px.sum())
    if total == 0:
        return 0.0
    return float(int((oily & region_px).sum()) / total * 100)


def _group_coverage_pct(oily: np.ndarray, region_scan_masks: dict, region_names: tuple) -> float:
    values = []
    for name in region_names:
        if name in region_scan_masks:
            values.append(_region_coverage_pct(oily, region_scan_masks[name]))
    return float(max(values)) if values else 0.0


def classify_skin_type(t_zone_pct: float, cheek_pct: float) -> str:
    t_oily = t_zone_pct >= OILY_REGION_THRESHOLD_PCT
    cheek_oily = cheek_pct >= OILY_REGION_THRESHOLD_PCT

    if t_oily and cheek_oily:
        return "Yagli Cilt"
    if t_oily and not cheek_oily:
        return "Kombine Cilt"
    if not t_oily and not cheek_oily:
        return "Normal Cilt"
    return "Yanak Agirlikli Yaglilik"


def oiliness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    """
    Returns:
        (score, coverage_pct, region_count, t_zone_pct, cheek_pct,
         skin_type, specular_pct, texture_pct)

        specular_pct/texture_pct: taranan yuzun ne kadarinin HANGI
        sinyalden (parlama noktasi mi, genel puruzsuz doku mu) yagli
        sayildigi - seffaflik icin (bkz. dosya basindaki aciklama).
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0, 0.0, 0.0, "Belirsiz", 0.0, 0.0

    binary, contours, specular_candidate, texture_candidate = _detect(img_bgr, face_scan_mask, region_scan_masks)
    oily = binary > 0

    oily_px = int(oily[scan_region].sum())
    coverage_ratio = oily_px / scan_area
    score = float(np.clip(coverage_ratio / OILY_COVERAGE_CAP, 0, 1))
    coverage_pct = float(coverage_ratio * 100)

    t_zone_pct = _group_coverage_pct(oily, region_scan_masks, T_ZONE_REGIONS)
    cheek_pct = _group_coverage_pct(oily, region_scan_masks, CHEEK_REGIONS)
    skin_type = classify_skin_type(t_zone_pct, cheek_pct)

    specular_pct = float(int((specular_candidate > 0)[scan_region].sum()) / scan_area * 100)
    texture_pct = float(int((texture_candidate > 0)[scan_region].sum()) / scan_area * 100)

    return score, coverage_pct, len(contours), t_zone_pct, cheek_pct, skin_type, specular_pct, texture_pct
