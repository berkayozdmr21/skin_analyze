"""
YAGLILIK.3 - Cok Olcekli Gradyan Dogrulama + Cilt Tipi Siniflandirmasi

## 1. Teknik farklilastirma: cok olcekli gradyan/duzluk dogrulamasi

Yaglilik.1-2, specular highlight adaylarini TEK bir Sobel cekirdegi
(ksize=3) ile dogruluyordu: "bu piksel yumusak/duz mu?" sorusu SADECE
en ince/lokal olcekte soruluyordu. Ama gercek parlama bolgeleri farkli
BUYUKLUKLERDE olabilir - burun ucundaki kucuk, sivri bir parlama ile
alindaki genis, yayilmis bir parlama AYNI olcekte degerlendirilirse,
biri (kucuk ama gercek) kacirilabilir, digeri (genis ama komsu dokudan
etkilenen) yanlis degerlendirilebilir.

Gozenek.4'teki "cok olcekli black-hat" ile AYNI mantik burada da
uygulaniyor: IKI farkli Sobel cekirdek boyutuyla (kucuk=3, buyuk=9)
gradyan buyuklugu hesaplanip, bir pikselin "yumusak" sayilmasi icin
HER IKI olcekte de dusuk gradyan gostermesi gerekiyor - bu, tek
olcekli kontrolden daha SIKI (daha az yanlis-pozitif) bir dogrulama.

## 2. Yuzde gosterimi farklilastirma: T-bolgesi / Yanak ikili gosterimi
## + cilt tipi siniflandirmasi

Yaglilik.1-2, TEK bir toplam "kaplanan alan yuzdesi" gosteriyordu.
Dermatolojide/cilt bakiminda cilt yagliligi genelde BOYLE
raporlanmaz - "kombine cilt" (T-bolgesi yagli, yanaklar normal/kuru)
ve "yagli cilt" (her yerde yagli) gibi standart cilt tipi
siniflandirmalari, T-BOLGESI ve YANAK kaplamasinin AYRI AYRI
karsilastirilmasina dayanir.

Bu surumde skor/yuzde gosterimi bu mantiga gore YENIDEN TASARLANDI:

    T_bolgesi_kaplama_% = (alin+burun+cene birlesiminde yagli piksel) / (o birlesimin toplam pikseli)
    Yanak_kaplama_%     = (yanaklarda yagli piksel) / (yanaklarin toplam pikseli)

    Cilt tipi:
        T-bolgesi YUKSEK VE Yanak YUKSEK  -> "Yagli Cilt"
        T-bolgesi YUKSEK VE Yanak DUSUK   -> "Kombine Cilt"
        T-bolgesi DUSUK VE Yanak DUSUK    -> "Normal Cilt"
        T-bolgesi DUSUK VE Yanak YUKSEK   -> "Yanak Agirlikli Yaglilik" (nadir/beklenmedik)

`OILY_REGION_THRESHOLD_PCT`, bir bolgenin "yagli" sayilmasi icin
gereken minimum kaplama yuzdesi - gercek test fotografiyla kalibre
edildi (bkz. README.md).

Genel (tek) skor da geriye donuk uyumluluk/karsilastirma icin hala
hesaplaniyor ve donuluyor, ama arayuzde artik T-bolgesi/yanak ikilisi
ve cilt tipi ETIKETI on planda.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

V_PERCENTILE = 92.0
S_PERCENTILE = 35.0

# Cok olcekli gradyan dogrulamasi - iki farkli Sobel cekirdegi.
#
# NOT: ilk denemede GRADIENT_THRESHOLD_LARGE=90 SABIT bir deger olarak
# secilmisti - ama bu TAMAMEN YANLIS cikti (her sey sifira dustu).
# Sebep: Sobel cekirdekleri NORMALIZE EDILMEMIS - ksize buyudukce
# gradyan buyuklugu OLCEGI patlar (olculdu: ksize=3'te medyan~25,
# ksize=9'da medyan~88.000!). Sabit bir esik boylece TAMAMEN farkli
# olceklerdeki iki gradyan haritasina uygulanamazdi.
#
# Cozum, bu dosyadaki V/S esikleriyle AYNI felsefe: sabit sayi yerine
# YUZDELIK DILIM - her olcekte, o olcegin KENDI gradyan dagiliminin
# en "yumusak" (dusuk) %GRADIENT_SMOOTH_PERCENTILE'i "duz/parlama
# adayi" sayilir. Olcek ne olursa olsun otomatik uyum saglar.
GRADIENT_KERNEL_SMALL = 3
GRADIENT_KERNEL_LARGE = 9
GRADIENT_SMOOTH_PERCENTILE = 70.0

CLEAN_KERNEL_RATIO = 0.010
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 21

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

OILY_COVERAGE_CAP = 0.08

MIN_REGION_PIXELS = 300

# Bir bolgenin (T-bolgesi veya yanak) "yagli" sayilmasi icin gereken
# minimum kaplama yuzdesi - cilt tipi siniflandirmasi icin.
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
    """
    Iki farkli Sobel cekirdek boyutuyla gradyan buyuklugu hesaplar; bir
    piksel SADECE HER IKI olcekte de (o olcegin KENDI dagilimina gore)
    dusuk gradyan gosteriyorsa "yumusak/duz" (gercek parlama adayi)
    sayilir.
    """
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


def _compute_oily_mask(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
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
    smooth = _multiscale_smoothness(gray, face_scan_mask)

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


def _region_coverage_pct(oily: np.ndarray, mask: np.ndarray) -> float:
    region_px = mask > 0
    total = int(region_px.sum())
    if total == 0:
        return 0.0
    return float(int((oily & region_px).sum()) / total * 100)


def _group_coverage_pct(oily: np.ndarray, region_scan_masks: dict, region_names: tuple) -> float:
    """
    Bir grubun (T-bolgesi = alin+burun+cene) kaplama yuzdesi, alt
    bolgelerin PIKSEL SAYISINA GORE AGIRLIKLI ORTALAMASI (havuzlanmis/
    pooled birlesim) DEGIL, alt bolgelerin KENDI yuzdelerinin
    MAKSIMUMU olarak hesaplanir.

    Neden: alin, burun/cenenin kat kat FAZLA pikseline sahip (test
    fotografinda 50.000 vs 764/2.836) - havuzlanmis bir ortalama, alnin
    (genelde daha az yagli) buyuklugu yuzunden burun/cenedeki GERCEK
    ve BELIRGIN yaglilik sinyalini "SEYRELTIR". Bir dermatoloğun
    sordugu soru "T-bolgesinde HERHANGI BIR yerde belirgin yaglilik
    var mi" - bu yuzden en YUKSEK alt-bolge degeri kullanmak, kombine
    cilt tespiti icin havuzlanmis ortalamadan daha DOGRU/HASSAS.
    """
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
        (score, coverage_pct, region_count, t_zone_pct, cheek_pct, skin_type)

        score/coverage_pct: genel (tum yuz) kaplama - onceki surumlerle
            karsilastirma icin hala hesaplaniyor.
        t_zone_pct/cheek_pct/skin_type: YENI, on plandaki gosterim -
            bkz. dosya basindaki aciklama.
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0, 0.0, 0.0, "Belirsiz"

    binary, contours = _detect(img_bgr, face_scan_mask, region_scan_masks)
    oily = binary > 0

    oily_px = int(oily[scan_region].sum())
    coverage_ratio = oily_px / scan_area
    score = float(np.clip(coverage_ratio / OILY_COVERAGE_CAP, 0, 1))
    coverage_pct = float(coverage_ratio * 100)

    t_zone_pct = _group_coverage_pct(oily, region_scan_masks, T_ZONE_REGIONS)
    cheek_pct = _group_coverage_pct(oily, region_scan_masks, CHEEK_REGIONS)
    skin_type = classify_skin_type(t_zone_pct, cheek_pct)

    return score, coverage_pct, len(contours), t_zone_pct, cheek_pct, skin_type
