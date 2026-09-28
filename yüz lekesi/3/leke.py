"""
YUZ LEKESI.3 - Robust (Medyan/MAD) Referans

## Kullanicinin bildirdigi sorun (ekran goruntusuyle)

Buyuk, acik renkli bir pigmentasyon alanina (vitiligo/melazma benzeri)
sahip bir yuzde, Yuz Lekesi.2 "bazi yerleri karistiriyordu" - isi
haritasinda genis, dagitik parlak bolgeler ve sinir/kenar cizgileri
gorunuyordu, sadece kucuk/nokta seklindeki gercek lekeler degil.

## Kok neden

Yuz Lekesi.2, referans istatistigini (ortalama + std) HAM (ROBUST
OLMAYAN) sekilde hesapliyordu:

    skin_l_mean = mean(L[ten_bolgesi])
    skin_l_std  = std(L[ten_bolgesi])

Eger ten bolgesinin BUYUK bir kismi (ornegin vitiligo gibi genis bir
depigmentasyon alani) NORMAL tenden COK farkli bir parlaklikta ise, bu
BUYUK "aykiri" grup, ORTALAMAYI kendi yonune ciddi sekilde CEKER.
Sonuc: geri kalan NORMAL ten bile, bu kaydirilmis ortalamaya gore
"anormal derecede farkli" gorunur - bu da yuzun GENIS bolgelerinin
(sadece gercek kucuk lekelerin degil) yanlislikla "leke" olarak
isaretlenmesine yol aciyor. Ortalama/std, ISTATISTIKTE "robust
olmayan" (aykiri degerlere/buyuk azinlik gruplarina karsi hassas)
olcumler olarak bilinir.

## Cozum: medyan + MAD (Median Absolute Deviation)

Goz Alti Morluk ailesinde (M.4/M.5) benzer bir sorun icin kullanilan
cozumle ayni ruhta: ortalama/std yerine MEDYAN ve MAD kullaniliyor.

    skin_l_medyan = medyan(L[ten_bolgesi])
    skin_l_mad    = medyan(|L[ten_bolgesi] - skin_l_medyan|)
    skin_l_std_robust = skin_l_mad * 1.4826   [normal dagilimla tutarli olcek]

    z = (skin_l_medyan - L) / skin_l_std_robust

**Neden bu daha dayanikli:** Medyan, verinin %50'sinden AZ bir kismi
aykiri/farkli olduğu surece, o "tipik/cogunluk" degerine sabit kalir -
ortalamanin aksine, birkac (hatta yuzeyin onemli bir kismini kaplayan)
aykiri deger onu KAYDIRAMAZ. Yani yuzun coguu normal tonda oldugu
surece (vitiligo/melazma alani yuzeyin YARISINDAN AZ oldugu surece),
medyan hala "gercek tipik ten rengini" doğru temsil eder.

## YENI: maske sinirinda hafif icebukey (erosion)

Isi haritasinda gozlemlenen "sinir cizgisi" artefaktlarini azaltmak
icin, `face_scan_mask` artik hafifce icebukulmus (eroded) - sac
cizgisi/cene hatti gibi GECIS bolgelerindeki piksel artik taramaya
dahil edilmiyor (bu bolgeler zaten "gercek ten" degil, gecis/golge
bolgesi).

## Degismeyenler (Yuz Lekesi.2 ile ayni)

- Hedef bolge: tum taranabilir yuz cildi (artik hafifce icebukey).
- Sekil filtresi, cozunurlukten bagimsiz boyutlandirma.
- Skor: sayi/siddet agirlikli (alan degil).
- Isaretleme: siddet-renkli numarali daire.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

Z_SCORE_CAP = 2.5
DETECTION_Z_THRESHOLD = 1.2

CLEAN_KERNEL_RATIO = 0.008
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 15

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

MIN_CIRCULARITY = 0.22
MAX_SPOT_AREA_RATIO = 0.02

# Yuz Lekesi.2 ile ayni kalibrasyon mantigi (z-score birimi degismedi,
# medyan/MAD normal dagilimda mean/std'ye yaklasik esdeger olacak
# sekilde olceklendigi icin ayni COUNT_CAP kullanilabilir).
COUNT_CAP = 22.0

# YENI: face_scan_mask'in sinirini icebukmek (erode) icin cekirdek
# orani - sac cizgisi/cene hatti gecis bolgelerini disi birakmak icin.
EDGE_ERODE_RATIO = 0.01
MIN_EDGE_ERODE_KERNEL = 3
MAX_EDGE_ERODE_KERNEL = 15


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _erode_scan_mask(face_scan_mask: np.ndarray) -> np.ndarray:
    """Tarama bolgesinin sinirini hafifce icebukerek gecis/kenar piksellerini disi birakir."""
    kernel_size = _relative_kernel(face_scan_mask, EDGE_ERODE_RATIO, MIN_EDGE_ERODE_KERNEL, MAX_EDGE_ERODE_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(face_scan_mask, kernel, iterations=1)

    # Guvenlik: erosion maskeyi tamamen yok ederse (cok kucuk bolge),
    # orijinal maskeyi kullan.
    if not (eroded > 0).any():
        return face_scan_mask
    return eroded


def _compute_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """
    z = (ten_L_medyan - L) / (ten_L_MAD * 1.4826) haritasini hesaplar.

    Medyan/MAD, ortalama/std'ye gore ROBUST (aykiri deger/buyuk azinlik
    gruplarina karsi dayanikli) - bkz. dosya basindaki aciklama.

    NOT: k-means tabanli iki-kumeli (bimodal) ayirma da denendi, ama
    testlerde BEKLENMEDIK sekilde kotu sonuc verdi - yuzdeki DOGAL
    parlaklik varyasyonu (burun/alin parlamasi, cene golgesi) ile
    ATIPIK bir pigmentasyon bolgesini (vitiligo/melazma) guvenilir
    sekilde ayirt edemiyordu, bazen YANLIS grubu "cogunluk/normal ten"
    sayabiliyordu. Bu yuzden daha basit VE daha ongorulebilir olan
    medyan/MAD tercih edildi - tek bir istatistiksel islem, ekstra
    hata riski yok.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    skin_region = face_scan_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    skin_values = l_channel[skin_region]
    skin_l_median = float(np.median(skin_values))
    skin_l_mad = float(np.median(np.abs(skin_values - skin_l_median)))
    skin_l_std_robust = skin_l_mad * 1.4826 + 1e-6

    z = (skin_l_median - l_channel) / skin_l_std_robust
    return z


def _is_leke_like(contour: np.ndarray, max_area: float) -> bool:
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """Tek bir tespit sureci - z haritasi ve kontur listesi birlikte doner."""
    scan_mask = _erode_scan_mask(face_scan_mask)
    scan_mask_u8 = (scan_mask > 0).astype(np.uint8) * 255
    z = _compute_z_map(img_bgr, scan_mask)

    if not (scan_mask_u8 > 0).any():
        return z, []

    clean_kernel_size = _relative_kernel(scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    scan_px = int((scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    binary = (z > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = find_detection_contours(binary, min_area=min_area)

    max_spot_area = max(min_area * 4, scan_px * MAX_SPOT_AREA_RATIO)
    leke_contours = [c for c in contours if _is_leke_like(c, max_spot_area)]

    return z, leke_contours


def leke_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    z, _ = _detect(img_bgr, face_scan_mask)
    z_positive = np.clip(z, 0, None)

    normalized = np.clip(z_positive / Z_SCORE_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (contours, severities)
    """
    z, contours = _detect(img_bgr, face_scan_mask)

    severities = []
    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_z = float(z[region_pixels].mean()) if region_pixels.any() else 0.0
        severities.append(float(np.clip(mean_z / Z_SCORE_CAP, 0, 1)))

    return contours, severities


def leke_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, tespit edilen lekelerin SAYISI/SIDDETINDEN hesaplaniyor.
    0 leke tespit edildiyse skor MATEMATIKSEL OLARAK 0.0'dir.

    Returns:
        (score, mean_leke_z, affected_area_pct, region_count)
    """
    scan_region_full = face_scan_mask > 0
    scan_area_full = int(scan_region_full.sum())
    if scan_area_full == 0:
        return 0.0, 0.0, 0.0, 0

    z, contours = _detect(img_bgr, face_scan_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0

    weighted_count = 0.0
    total_area = 0
    z_values = []

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
        z_values.append(mean_z)

    score = float(np.clip(weighted_count / COUNT_CAP, 0, 1))
    mean_leke_z = float(np.mean(z_values)) if z_values else 0.0
    affected_area_pct = float(total_area / scan_area_full * 100)

    return score, mean_leke_z, affected_area_pct, len(contours)
