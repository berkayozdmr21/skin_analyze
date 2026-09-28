"""
GOZENEK.2 - Bolgesel (Alin/Burun/Cene/Yanak Ayri Ayri) Referans

## Gozenek.1'in sorunu

Gozenek.1, black-hat tepkisinin z-score'unu TEK BIR GLOBAL referansla
hesapliyordu: taranan yuzun TAMAMI (alin+burun+cene+yanak birlikte)
uzerinden tek bir ortalama/std. Ama gozenek gorunurlugu yuzun farkli
bolgelerinde ANATOMIK OLARAK farkli olan bir sey - burun ve alin
(T-bolgesi) dogal olarak daha fazla yag bezi icerir, gozenekleri
SAGLIKLI bir tende bile yanaklardan daha buyuk/belirgin gorunur. Tek
bir global referans kullanmak, bu dogal farki "anormallik" gibi
degerlendirebiliyordu:

  - Eger T-bolgesi genel ortalamayi yukari cekiyorsa, yanaklardaki
    GERCEK bir gozenek sorunu, global std'nin sisirilmesi yuzunden
    OLDUGUNDAN DUSUK z-score alip KACIRILABILIYORDU.
  - Tam tersi, T-bolgesindeki TAMAMEN NORMAL/SAGLIKLI gozenekler bile
    global ortalamadan sapma gosterip GEREKSIZ YERE tespit
    edilebiliyordu.

## Cozum: her bolge kendi icinde degerlendiriliyor

Bu, Goz Alti Morluk ailesindeki "tek referans -> daha stabil/dogru
referans" ilerlemesiyle ayni ruhta (M.1'de yanak, M.2'de tum ten oldugu
gibi) - burada ise TEK bir "tum yuz" referansi yerine, YUZUN HER
ALT-BOLGESI (alin, burun, cene, yanaklar) KENDI ICINDE degerlendiriliyor:

    her bolge r icin ayri ayri:
        z_r = (blackhat_piksel - bolge_r_ortalama) / bolge_r_std

Yani "bu burun pikseli, DIGER BURUN piksellerine gore ne kadar
gozenekli" ve "bu yanak pikseli, DIGER YANAK piksellerine gore ne kadar
gozenekli" ayri ayri sorulariyor - T-bolgesinin dogal olarak daha
belirgin gozenekleri, yanaklarin degerlendirmesini artik BOZMUYOR.

## Degismeyenler (Gozenek.1 ile ayni)

- Black-hat morfolojik donusum, cozunurlukten bagimsiz (goreli)
  cekirdek boyutu.
- Sekil filtresi (dairesellik + maksimum alan).
- Skor, tespit edilen bolgelerin yukunden (alan x siddet) hesaplaniyor -
  0 bolge = kesinlikle %0 skor.

## YENI: bolge dagilimi raporlamasi

Artik kac gozenegin hangi yuz bolgesinde (alin/burun/cene/yanak)
tespit edildigi de raporlaniyor - bu, T-bolgesi varsayiminin gercekten
dogru oldugunu GORMEK icin de faydali bir seffaflik.

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

BURDEN_AREA_FRACTION = 0.15

# Bir alt-bolgenin KENDI istatistiginin guvenilir sayilmasi icin
# gereken minimum piksel sayisi - cok kucuk/parcali bir bolge (ornegin
# yuz kismen kadraj disindaysa) kendi ortalamasini guvenilmez sekilde
# hesaplayabilir; bu durumda o bolgenin pikselleri GLOBAL referansa
# (tum taranan yuz) duser.
MIN_REGION_PIXELS = 300


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _compute_pore_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    """
    Black-hat donusumunu hesaplayip, HER BOLGEYI KENDI ICINDE
    z-score'a cevirir (bkz. dosya basindaki aciklama). Bir bolgenin
    piksel sayisi MIN_REGION_PIXELS'in altindaysa, o bolge GLOBAL
    (tum taranan yuz) istatistige duser.

    Returns:
        (z_map, blackhat_kernel_size)
    """
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
    """
    Tek bir tespit sureci - z-score haritasi ve kontur listesi birlikte
    donuyor (Gozenek.1/Akne.5 ile ayni tutarlilik prensibi).
    """
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
    """Bir konturun agirlik merkezinin hangi yuz bolgesine dustugunu bulur."""
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
    Returns:
        (score, mean_pore_z, affected_area_pct, region_count, region_breakdown)

        region_breakdown: {"forehead": n, "nose": n, "chin": n, "cheeks": n, "diger": n}
        - hangi yuz bolgesinde kac gozenek tespit edildigi (seffaflik icin).
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0.0, 0, {}

    z, contours = _detect(img_bgr, face_scan_mask, region_scan_masks)

    if not contours:
        return 0.0, 0.0, 0.0, 0, {}

    total_burden = 0.0
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

        total_burden += severity * area
        total_area += area
        severities.append(mean_z)

        region_name = _region_of_contour(c, region_scan_masks)
        breakdown[region_name] = breakdown.get(region_name, 0) + 1

    score = float(np.clip(total_burden / (scan_area * BURDEN_AREA_FRACTION), 0, 1))
    mean_pore_z = float(np.mean(severities)) if severities else 0.0
    affected_area_pct = float(total_area / scan_area * 100)

    return score, mean_pore_z, affected_area_pct, len(contours), breakdown
