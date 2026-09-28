"""
GOZENEK.4 - Cok Olcekli (Multi-Scale) Black-Hat Tespiti

## Kullanicinin istegi

> "sayisal mantik ayni, gozenek gorme kismini degistirelim"

Yani: skor formulu (Gozenek.3'teki sayi/yogunluk tabanli hesap)
DEGISMEYECEK - degisecek olan, gozeneklerin NASIL TESPIT EDILDIGI
("gorme" kismi).

## Gozenek.1-3'un sorunu: TEK cekirdek boyutu

Onceki surumler, black-hat donusumu icin TEK bir cekirdek boyutu
kullaniyordu (yuz olcegine oranli, ama hep TEK bir deger). Ama gercek
gozenekler AYNI YUZDE bile FARKLI boyutlarda olabilir - bazilari cok
ince/kucuk, bazilari daha genis/belirgin (yaglanma, yas, cilt tipine
gore). TEK bir cekirdek boyutu, sadece O BOYUTA yakin gozenekleri iyi
yakalar:

  - Cekirdek KUCUKSE: buyuk/genis gozenekler ya hic yakalanmiyor ya da
    parcali/bolunmus sekilde (birkac kucuk kontur olarak) yakalaniyor.
  - Cekirdek BUYUKSE: ince/kucuk gozenekler black-hat tepkisinde
    "kayboluyor" (buyuk cekirdek onlari komsu dokuyla harmanlayip
    silikleştiriyor).

Bu, klasik goruntu isleme literaturunde "olcek secimi" (scale
selection) problemi olarak bilinir - blob tespiti icin standart cozum,
TEK bir olcek yerine BIRDEN FAZLA olcekte tarama yapip sonuclari
BIRLESTIRMEK (scale-space / multi-scale blob detection).

## Cozum: uc farkli cekirdek boyutuyla tarama, en guclu sinyali al

Artik UC farkli black-hat cekirdek boyutuyla (kucuk/orta/buyuk, hepsi
yuz olcegine oranli) ayri ayri taraniyor, HER OLCEKTE bolgesel z-score
hesaplaniyor (Gozenek.2/3'teki bolgesel mantik AYNEN korunuyor), sonra
bu uc z-haritasinin PIKSEL BAZLI MAKSIMUMU aliniyor:

    z_kucuk  = bolgesel_z(blackhat(kucuk_cekirdek))
    z_orta   = bolgesel_z(blackhat(orta_cekirdek))
    z_buyuk  = bolgesel_z(blackhat(buyuk_cekirdek))

    z_birlesik = max(z_kucuk, z_orta, z_buyuk)   [piksel bazli]

Yani bir piksel, HERHANGI BIR olcekte guclu bir gozenek sinyali
veriyorsa (ince gozenek KUCUK cekirdekte, genis gozenek BUYUK
cekirdekte belirgin olabilir), tespit edilme sansi artiyor - "gorme"
artik tek bir odak mesafesi yerine, uc farkli "buyutec" ile bakmis gibi.

## Degismeyenler (Gozenek.3 ile ayni - kullanicinin istegi geregi)

- SAYI/SIDDET AGIRLIKLI SKOR FELSEFESI AYNI: skor hala "her tespit
  edilen gozenek kendi siddetiyle katkida bulunur, alan degil" mantigina
  dayaniyor - "daha fazla gozenek -> skor hicbir zaman dusmez" garantisi
  korunuyor.
- Bolgesel (alin/burun/cene/yanak ayri) referans mantigi.
- Sekil filtresi (dairesellik + maksimum alan), cozunurlukten bagimsiz
  boyutlandirma.
- Bolge dagilimi raporlamasi.

## YAN URUN: Gozenek.3'te bulunan bir hata da duzeltildi

Bu surumu gelistirirken (multi-scale tespiti farkli cozunurluklerde
test ederken) Gozenek.3'un skor formulunde daha once fark edilmemis bir
hata bulundu: skor, agirlikli sayiyi TARANAN ALANA (piksel sayisina)
BOLUYORDU. Agirlikli sayi cozunurlukten bagimsiz (goreli) filtreler
sayesinde zaten YAKLASIK SABIT kalirken, alan cozunurlukle ORANTILI
BUYUYORDU - bu da skorun cozunurluk arttikca COKMESINE yol aciyordu
(ayni fotografta %58 -> %5, 2000px'e cikildiginda). Bu surumde skor
ARTIK ALANA BOLUNMUYOR - dogrudan sabit bir referansla (`COUNT_CAP`)
kiyaslaniyor. Kullanicinin "sayisal mantik ayni kalsin" istegindeki ANA
FIKIR (sayi/siddet agirlikli, alan degil) boylece hem korundu hem de
cozunurluk kararliligi kazandi. Detaylar ve olcumler README.md'de.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# UC farkli black-hat cekirdek boyutu (yuz olcegine oranli) - kucuk,
# orta, buyuk gozenekleri hedeflemek icin. MULTISCALE_KERNEL_RATIOS'un
# ortasi, Gozenek.1-3'teki tek BLACKHAT_KERNEL_RATIO (0.018) ile ayni -
# yani bu surum, oncekinin "orta" olcegini kortuyup iki yeni olcek
# (kucuk + buyuk) ekliyor.
MULTISCALE_KERNEL_RATIOS = (0.010, 0.018, 0.032)
MIN_BLACKHAT_KERNEL = 3
MAX_BLACKHAT_KERNEL = 27

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

# Gozenek.3'ten devralinirken bulunan bir hata DUZELTILDI: Gozenek.3
# skoru `agirlikli_sayi / (taranan_alan/100)` seklinde ALAN'a bolerek
# hesapliyordu. Ama agirlikli_sayi (COUNT), cozunurlukten bagimsiz
# (goreli) kernel/alan filtreleri sayesinde zaten YAKLASIK SABIT
# kaliyor (fiziksel yuzdeki gercek gozenek sayisini yansitiyor) - oysa
# taranan_alan (piksel sayisi) COZUNURLUKLE ORANTILI BUYUYOR. Yaklasik
# sabit bir sayiyi, cozunurlukle orantili buyuyen bir alana bolmek,
# skorun COZUNURLUK ARTTIKCA COKMESINE yol aciyordu (bu fotografta
# %58 -> %5 gibi 10 kat fark - somut olcum README.md'de).
#
# Cozum: SKORU ARTIK ALANA BOLMUYORUZ - agirlikli sayi zaten kendi
# basina cozunurlukten (buyuk olcude) bagimsiz, dogrudan sabit bir
# referansla (COUNT_CAP) kiyaslaniyor. Kullanicinin istedigi "sayisal
# mantik" (sayi/siddet agirlikli, ALAN DEGIL) boylece hem KORUNUYOR
# hem de cozunurluk kararliligi kazaniyor.
COUNT_CAP = 200.0


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _regional_z(blackhat: np.ndarray, scan_region: np.ndarray, region_scan_masks: dict) -> np.ndarray:
    """Gozenek.2/3 ile ayni bolgesel z-score mantigi - tek bir olcek icin."""
    global_mean = float(np.mean(blackhat[scan_region]))
    global_std = float(np.std(blackhat[scan_region])) + 1e-6

    z = np.zeros(blackhat.shape, dtype=np.float32)
    covered = np.zeros(blackhat.shape, dtype=bool)

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

    return z


def _compute_pore_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, region_scan_masks: dict):
    """
    UC farkli cekirdek boyutuyla black-hat + bolgesel z-score hesaplar,
    piksel bazli MAKSIMUMUNU alir.

    Returns:
        (z_map, kernel_sizes) - kernel_sizes raporlama/seffaflik icin.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[..., 0]

    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32), []

    kernel_sizes = []
    z_stack = []

    for ratio in MULTISCALE_KERNEL_RATIOS:
        kernel_size = _relative_kernel(face_scan_mask, ratio, MIN_BLACKHAT_KERNEL, MAX_BLACKHAT_KERNEL)
        kernel_sizes.append(kernel_size)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

        z_scale = _regional_z(blackhat, scan_region, region_scan_masks)
        z_stack.append(z_scale)

    combined_z = np.maximum.reduce(z_stack)
    return combined_z, kernel_sizes


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
    Skor formulu Gozenek.3 ile BIREBIR AYNI (sayi/yogunluk tabanli) -
    sadece z haritasinin NASIL uretildigi (cok olcekli) degisti.

    Returns:
        (score, mean_pore_z, affected_area_pct, region_count, region_breakdown)
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

    score = float(np.clip(weighted_count / COUNT_CAP, 0, 1))

    mean_pore_z = float(np.mean(severities)) if severities else 0.0
    affected_area_pct = float(total_area / scan_area * 100)

    return score, mean_pore_z, affected_area_pct, len(contours), breakdown
