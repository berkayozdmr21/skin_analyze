"""
YAGLILIK.2 - Bolgesel (Alin/Burun/Cene/Yanak Ayri) Yuzdelik Dilim Referansi

## Yaglilik.1'in sorunu (Gozenek.1 -> .2 ile BIREBIR AYNI motivasyon)

Yaglilik.1, "parlama" esigini (V/S yuzdelik dilimlerini) TEK BIR GLOBAL
referansla hesapliyordu: taranan yuzun TAMAMI (alin+burun+cene+yanak
birlikte) uzerinden tek bir yuzdelik dilim kesme degeri. Ama yaglilik/
parlama gorunurlugu yuzun farkli bolgelerinde ANATOMIK OLARAK farkli -
T-bolgesi (alin+burun+cene) dogal olarak daha fazla yag bezi icerir,
SAGLIKLI bir tende bile yanaklardan daha parlak/yagli gorunur. Tek bir
global yuzdelik dilim kullanmak, bu dogal farki yanlis degerlendirebilir:

  - T-bolgesi genel dagilimi "parlaklik" yonunde cektigi icin,
    yanaklardaki GERCEK bir yaglilik, global esigin YUKSEK kalmasi
    yuzunden KACIRILABILIYORDU.
  - Tam tersi, T-bolgesindeki TAMAMEN NORMAL/SAGLIKLI parlaklik bile
    (yanaklara kiyasla) "olagan disi" sayilip GEREKSIZ YERE
    isaretlenebiliyordu.

## Cozum: her bolge kendi icinde degerlendiriliyor

Bu, Gozenek.1 -> .2 gecisiyle ayni ruhta - TEK bir "tum yuz" yuzdelik
dilimi yerine, YUZUN HER ALT-BOLGESI (alin, burun, cene, yanaklar)
KENDI ICINDE degerlendiriliyor:

    her bolge r icin ayri ayri:
        V_esik_r = persentil(V[bolge_r], V_PERCENTILE)
        S_esik_r = persentil(S[bolge_r], S_PERCENTILE)

Yani "bu alin pikseli, DIGER ALIN piksellerine gore ne kadar parlak/
yagli" ve "bu yanak pikseli, DIGER YANAK piksellerine gore ne kadar
parlak/yagli" ayri ayri sorulariyor - T-bolgesinin dogal olarak daha
yuksek parlakligi, artik yanaklarin degerlendirmesini BOZMUYOR.

## YENI: bolge dagilimi raporlamasi

Artik yagli/parlak alanin hangi yuz bolgesinde (alin/burun/cene/yanak)
ne kadar kapladigi da raporlaniyor - "T-bolgesi varsayiminin" gercekten
dogru oldugunu GORMEK icin faydali bir seffaflik (Gozenek.2'deki ayni
fikir).

## Degismeyenler (Yaglilik.1 ile ayni)

- HSV specular highlight + gradyan dogrulama teknigi.
- Skor, KAPLANAN ALAN YUZDESINE dayanir (ayrik leke sayisina degil).
- Cozunurlukten bagimsiz (goreli) gurultu temizleme boyutlandirma.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

V_PERCENTILE = 92.0
S_PERCENTILE = 35.0

GRADIENT_THRESHOLD = 50.0

CLEAN_KERNEL_RATIO = 0.010
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 21

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

OILY_COVERAGE_CAP = 0.08

# Bir alt-bolgenin KENDI yuzdelik dilim esiginin guvenilir sayilmasi
# icin gereken minimum piksel sayisi - Gozenek.2 ile ayni guvenlik agi.
MIN_REGION_PIXELS = 300


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _regional_cutoffs(channel: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict, percentile: float):
    """
    Her bolge icin ayri bir persentil-kesme DEGERI hesaplar, sonra bu
    degeri o bolgenin piksellerine uygulayarak TUM GORUNTU icin bir
    "esik haritasi" (her pikselin kendi bolgesine gore esigi) dondurur.

    Bir bolgenin piksel sayisi MIN_REGION_PIXELS'in altindaysa, o bolge
    GLOBAL (tum taranan yuz) esigine duser.
    """
    scan_region = face_scan_mask > 0
    global_cutoff = float(np.percentile(channel[scan_region], percentile))

    cutoff_map = np.full(channel.shape, global_cutoff, dtype=np.float32)
    covered = np.zeros(channel.shape, dtype=bool)

    for mask in region_scan_masks.values():
        region_px = mask > 0
        count = int(region_px.sum())
        if count < MIN_REGION_PIXELS:
            continue

        region_cutoff = float(np.percentile(channel[region_px], percentile))
        cutoff_map[region_px] = region_cutoff
        covered |= region_px

    return cutoff_map


def _compute_oily_mask(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    """
    Bolgesel yuzdelik dilim esiklemesi (yuksek V + dusuk S) + gradyan
    dogrulamasi ile specular highlight (parlama) adaylarini bulur.

    Returns:
        binary_mask (uint8, 0/255)
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]
    s_channel = hsv[..., 1]

    v_cutoff_map = _regional_cutoffs(v_channel, face_scan_mask, region_scan_masks, V_PERCENTILE)
    s_cutoff_map = _regional_cutoffs(s_channel, face_scan_mask, region_scan_masks, S_PERCENTILE)

    high_v = v_channel > v_cutoff_map
    low_s = s_channel < s_cutoff_map
    specular = high_v & low_s

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    smooth = grad_mag < GRADIENT_THRESHOLD

    candidate = (specular & smooth).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, (face_scan_mask > 0).astype(np.uint8) * 255)

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    cleaned = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    return cleaned


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    binary = _compute_oily_mask(img_bgr, face_scan_mask, region_scan_masks)

    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    contours = find_detection_contours(binary, min_area=min_area)
    return binary, contours


def oiliness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    binary, _ = _detect(img_bgr, face_scan_mask, region_scan_masks)
    normalized = (binary > 0).astype(np.float32)
    normalized[~scan_region] = 0

    return normalized


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


def oiliness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    """
    Skor, KAPLANAN ALAN YUZDESINE dayanir (Yaglilik.1 ile ayni felsefe).

    Returns:
        (score, coverage_pct, region_count, region_coverage_breakdown)

        region_coverage_breakdown: {"forehead": pct, "nose": pct, ...}
        - her bolgenin KENDI ICINDE ne kadarinin yagli/parlak
          isaretlendigi (o bolgenin kendi piksel sayisina oranla).
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0, {}

    binary, contours = _detect(img_bgr, face_scan_mask, region_scan_masks)
    oily = binary > 0

    oily_px = int(oily[scan_region].sum())
    coverage_ratio = oily_px / scan_area
    score = float(np.clip(coverage_ratio / OILY_COVERAGE_CAP, 0, 1))
    coverage_pct = float(coverage_ratio * 100)

    region_breakdown = {}
    for name, mask in region_scan_masks.items():
        region_px = mask > 0
        region_total = int(region_px.sum())
        if region_total == 0:
            continue
        region_oily = int((oily & region_px).sum())
        region_breakdown[name] = float(region_oily / region_total * 100)

    return score, coverage_pct, len(contours), region_breakdown
