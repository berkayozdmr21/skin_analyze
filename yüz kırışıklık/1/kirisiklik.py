"""
YUZ KIRISIKLIK.1 - Cok Yonelimli Gabor Filtre Bankasi + Cizgi Tespiti

## Yontem arastirmasi

Kirisiklik tespiti, goruntu isleme literaturunde "cizgi/sirt (ridge/line)
tespiti" problemi olarak ele alinir. Arastirmada tutarli sekilde ortaya
cikan iki standart teknik:

    1. Gabor filtre bankasi - farkli YONELIMLERDE (orientation) ince
       cizgisel yapilari vurgulayan bir filtre ailesi. Kirisiklik
       "derinligi" (wrinkle depth), Gabor filtre yanitiyla dogrudan
       iliskilendiriliyor (bkz. README.md kaynaklari).
    2. Hessian/Frangi tabanli sirt (ridge) filtreleri - damar
       segmentasyonundan uyarlanan, cizgisel yapilari bulan yontemler.

Bu projede Gabor filtre bankasi kullanildi - cunku (a) dogrudan OpenCV
icinde hazir (`cv2.getGaborKernel`), ek bagimlilik gerektirmiyor, (b)
YONELIM SECICI: kirisiklikler HER ACIDA olabilir (alinda yatay, kaslar
arasinda dikey, goz kenarinda capraz "kaz ayagi") - tek bir filtre
degil, birden fazla YONELIMDE filtre bankasi kullanip EN GUCLU yaniti
almak gerekiyor.

## Yontem

1. **Alti farkli yonelimde** (0, 30, 60, 90, 120, 150 derece - bir
   cizgi kendisiyle 180 derece sonra ayni gorundugu icin 180 derecelik
   yelpaze yeterli) Gabor cekirdekleri olusturulur.
2. Her yonelimde goruntu filtrelenir, MUTLAK deger alinir (cizgi
   varligini, yonunu degil siddetini olcmek icin).
3. Tum yonelimlerdeki en YUKSEK yanit piksel-bazli alinir - "bu piksel,
   HANGI yonelimde olursa olsun, bir cizgi yapisinin parcasi mi"
   sorusuna cevap.
4. **Koyuluk dogrulamasi**: Gabor filtresi TEK BASINA yonlu herhangi
   bir cizgisel yapiya (parlak veya koyu) tepki verir - ama kirisiklik
   ozellikle KOYU bir cizgidir (deri kivrimindaki golge). Bu yuzden
   Gabor yaniti, ayrica "bu piksel ten referansindan daha koyu mu"
   kontroluyle (medyan/MAD - Yuz Lekesi.3'te dogrulanan robust yontem)
   BIRLESTIRILIYOR (VE/AND) - sadece koyu VE cizgisel olan pikseller
   kirisiklik adayi sayilir.
5. **Uzama (elongation) filtresi**: tespit edilen her bolge icin
   `cv2.minAreaRect` ile en/boy orani hesaplanir - SADECE yeterince
   UZUN/INCE (dairesel/yuvarlak DEGIL) bolgeler kirisiklik sayilir. Bu,
   Akne/Gozenek/Yuz Lekesi ailesindeki "sadece yuvarlak" filtresinin
   TAM TERSI - kirisiklik bir CIZGIDIR, bir NOKTA degil.

## Skor mantigi - kirisiklige ozgu bir tasarim karari

Kirisiklik, Yuz Lekesi/Akne/Gozenek gibi SAYILABILIR noktalar degil,
Yaglilik gibi yayilmis bir alan da degil - bir CIZGI TOPLULUGU. Dogru
olculen sey "kac tane" veya "ne kadar alan" degil, TOPLAM CIZGI
UZUNLUGU'dur (dermatolojide "kirisiklik yogunlugu" tam olarak boyle
tanimlanir).

    her tespit edilen cizgi i icin:
        uzunluk_i = cv2.minAreaRect ile bulunan UZUN kenar (piksel)
        siddet_i  = clip(cizgi_i_ortalama_koyuluk_z / Z_SCORE_CAP, 0, 1)

    agirlikli_uzunluk = Σ (uzunluk_i * siddet_i)

**Cozunurluk kararliligi icin ONEMLI bir tasarim karari**: agirlikli
uzunluk bir "UZUNLUK" (1 boyutlu) miktar - cozunurlukle DOGRUSAL olarak
olceklenir (Akne/Gozenek ailesindeki hatalardan cikarilan ders: bir
miktarin cozunurlukle nasil olceklendigini ANLAMADAN gelisi guzel bir
paydaya bolmek, skorun cozunurlukte cokmesine/patlamasina yol acar).
Bu yuzden agirlikli uzunluk, ALANA (cozunurlukle KARESEL olceklenir)
DEGIL, yuzun KARAKTERISTIK UZUNLUGUNA (face_scale = sqrt(taranan_alan),
cozunurlukle DOGRUSAL olceklenir) bolunuyor - boylece oran
cozunurlukten (buyuk olcude) bagimsiz kaliyor:

    skor = clip( agirlikli_uzunluk / (face_scale * LENGTH_DENSITY_CAP), 0, 1)

Detaylar ve dogrulama olcumleri icin README.md.

Sources:
- https://github.com/elliotnaylor/Wrinkle-detection
- https://www.sciencedirect.com/science/article/abs/pii/S0031320314002945
- https://e-space.mmu.ac.uk/597156/2/accv2014finalpaper604.pdf
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Gabor filtre bankasi yonelimleri (derece) - bir cizgi 180 derece
# sonra kendisiyle ayni gorundugu icin 0-150 araligi yeterli.
GABOR_ORIENTATIONS_DEG = (0, 30, 60, 90, 120, 150)

# Gabor cekirdek boyutu ve dalga boyu, yuz olcegine ORANLI - kirisiklik
# genisligini/araligini "hedeflemek" icin.
GABOR_KSIZE_RATIO = 0.030
MIN_GABOR_KSIZE = 7
MAX_GABOR_KSIZE = 41
GABOR_LAMBDA_RATIO = 0.022
MIN_GABOR_LAMBDA = 4.0
GABOR_SIGMA_FACTOR = 0.30   # sigma = ksize * bu oran
GABOR_GAMMA = 0.5            # spatial aspect ratio - cizgiye dik yonde daraltir

# Gabor yanitinin "guclu cizgi" sayilmasi icin gereken yuzdelik dilim.
GABOR_STRENGTH_PERCENTILE = 88.0

# Koyuluk dogrulamasi - medyan/MAD (robust, Yuz Lekesi.3'te dogrulandi).
DARK_Z_THRESHOLD = 0.6
Z_SCORE_CAP = 2.0

# Gurultu temizleme + minimum alan, cozunurlukten bagimsiz (goreli).
CLEAN_KERNEL_RATIO = 0.006
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 11

MIN_AREA_RATIO = 0.00012
MIN_AREA_FLOOR = 5

# Uzama (elongation) filtresi - minAreaRect'in uzun/kisa kenar orani
# bu DEGERDEN BUYUK olmali (kirisiklik bir CIZGI, bir NOKTA degil).
MIN_ELONGATION_RATIO = 2.2
# Kirisiklik cok da UZUN olmamali (yuzun tamamini kaplayan bir "cizgi"
# muhtemelen gercek bir yapi degil, artefakttir).
MAX_LENGTH_RATIO_OF_FACE = 0.55

# Skorun %100'e ulastigi, yuzun karakteristik uzunluguna ORANLI agirlikli
# kirisiklik uzunlugu esigi - gercek test fotografiyla kalibre edildi.
# (Dogrulama: agirlikli_uzunluk/face_scale orani 170px-3000px arasinda
# 1.72-1.98 araliginda olculdu - cok stabil, bkz. README.md.)
LENGTH_DENSITY_CAP = 4.0

# YENI (gorsel dogrulamada bulundu): goz soketi kenari ve yuz dis
# sinirinin (sac/arka plan gecisi) KENDISI de guclu, koyu, uzun bir
# "cizgi" gibi davranip yanlislikla kirisiklik sayilabiliyordu. Tespit
# alani, delik (goz/kas/dudak) ve maske kenarlarindan bu oranda
# icebukuluyor - Yuz Lekesi.4/5'teki ayni cozum.
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
    """
    Alti yonelimde Gabor filtre bankasi uygular, piksel-bazli en yuksek
    (mutlak deger) yaniti dondurur - "bu piksel HANGI yonde olursa
    olsun bir cizgi yapisinin parcasi mi" olcusu.
    """
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
        kernel -= kernel.mean()  # DC bilesenini cikar - duz alanlara tepki vermesin
        response = cv2.filter2D(l_channel, cv2.CV_32F, kernel)
        responses.append(np.abs(response))

    return np.maximum.reduce(responses)


def _darkness_z_map(l_channel: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """(ten_medyan - L) / (ten_MAD * 1.4826) - Yuz Lekesi.3 ile ayni robust yontem."""
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(l_channel.shape, dtype=np.float32)

    skin_values = l_channel[scan_region]
    median = float(np.median(skin_values))
    mad = float(np.median(np.abs(skin_values - median)))
    std_robust = mad * 1.4826 + 1e-6

    return (median - l_channel) / std_robust


def _compute_candidate_mask(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8), np.zeros(img_bgr.shape[:2], dtype=np.float32)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    gabor_strength = _gabor_line_strength(l_channel, face_scan_mask)
    gabor_cutoff = float(np.percentile(gabor_strength[scan_region], GABOR_STRENGTH_PERCENTILE))
    strong_line = gabor_strength > gabor_cutoff

    dark_z = _darkness_z_map(l_channel, face_scan_mask)
    dark_enough = dark_z > DARK_Z_THRESHOLD

    candidate = (strong_line & dark_enough).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, (face_scan_mask > 0).astype(np.uint8) * 255)

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, clean_kernel_size - 2), max(3, clean_kernel_size - 2)))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    return cleaned, dark_z


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
    """Tespit alanini delik/kenar bolgelerinden icebuker - bkz. DETECTION_MARGIN_RATIO aciklamasi."""
    margin = _relative_kernel(face_scan_mask, DETECTION_MARGIN_RATIO, MIN_DETECTION_MARGIN, MAX_DETECTION_MARGIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    eroded = cv2.erode(face_scan_mask, kernel, iterations=1)

    if not (eroded > 0).any():
        return face_scan_mask
    return eroded


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """Tek bir tespit sureci - koyuluk-z haritasi ve kontur listesi birlikte doner."""
    candidate_mask, dark_z = _compute_candidate_mask(img_bgr, face_scan_mask)

    detection_mask = _erode_for_detection(face_scan_mask)
    candidate_mask = cv2.bitwise_and(candidate_mask, (detection_mask > 0).astype(np.uint8) * 255)

    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    raw_contours = find_detection_contours(candidate_mask, min_area=min_area)

    face_scale = _face_scale(face_scan_mask)
    wrinkle_contours = [c for c in raw_contours if _is_wrinkle_like(c, face_scale)]

    return dark_z, wrinkle_contours


def wrinkle_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    dark_z, contours = _detect(img_bgr, face_scan_mask)

    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    normalized = np.clip(dark_z / Z_SCORE_CAP, 0, 1)
    normalized[mask == 0] = 0
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (contours, severities) - her kontur icin 0-1 arasi siddet degeri.
    """
    dark_z, contours = _detect(img_bgr, face_scan_mask)

    severities = []
    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_z = float(dark_z[region_pixels].mean()) if region_pixels.any() else 0.0
        severities.append(float(np.clip(mean_z / Z_SCORE_CAP, 0, 1)))

    return contours, severities


def wrinkle_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, tespit edilen kirisikliklerin AGIRLIKLI TOPLAM UZUNLUGUNDAN
    hesaplaniyor - yuzun karakteristik UZUNLUGUNA (alanina DEGIL)
    oranli, cozunurluk kararliligi icin (bkz. dosya basindaki aciklama).
    0 kirisiklik tespit edildiyse skor MATEMATIKSEL OLARAK 0.0'dir.

    Returns:
        (score, weighted_length_px, total_length_px, mean_z, region_count)
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return 0.0, 0.0, 0.0, 0.0, 0

    dark_z, contours = _detect(img_bgr, face_scan_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0.0, 0

    weighted_length = 0.0
    total_length = 0.0
    z_values = []

    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        length = _contour_length(c)

        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_z = float(dark_z[region_pixels].mean()) if region_pixels.any() else 0.0
        severity = float(np.clip(mean_z / Z_SCORE_CAP, 0, 1))

        weighted_length += length * severity
        total_length += length
        z_values.append(mean_z)

    face_scale = _face_scale(face_scan_mask)
    score = float(np.clip(weighted_length / (face_scale * LENGTH_DENSITY_CAP), 0, 1))
    mean_z_overall = float(np.mean(z_values)) if z_values else 0.0

    return score, weighted_length, total_length, mean_z_overall, len(contours)
