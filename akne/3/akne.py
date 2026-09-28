"""
AKNE.3 - Kirmizilik (a*) + Doygunluk (HSV S) Birlesimi

Akne.2'nin sorunu: SADECE a* (Lab kirmizi-yesil ekseni) kanaline bakiyordu.
Ama gercek iltihaplı/aktif bir sivilce, cevre tenden sadece renk EKSENI
acisindan degil, ayni zamanda DOYGUNLUK (ne kadar "canli/parlak" renk)
acisindan da farklidir. Kullanici testinde ("sivilcelere benziyor,
onlari da saysin") gozlemlenen sorun: bazi gercek sivilceler a*'da cok
yuksek cikmiyor ama HSV doygunlugunda (S) belirgin sekilde daha yuksek -
Akne.2 bunlari kacırıyordu.

Bu surumde iki z-score birlikte kullaniliyor (Goz Alti Morluk.3 ile
BIREBIR AYNI YAPIDA - iki kanal, agirlikli toplam):

    z_a: Akne.2 ile ayni - ten a* ortalamasina gore ne kadar kirmizi.
    z_s: HSV S (doygunluk) kanali - ten S ortalamasindan ne kadar daha
         doygun/canli.

    combined_z = 0.6 * z_a + 0.4 * z_s

Bu bir "VE" (AND) sarti DEGIL, agirlikli TOPLAM: sadece kirmizilikte
VEYA sadece doygunlukta guclu cikan bir piksel de hala yuksek combined_z
alabiliyor. Boylece Akne.2'nin kacirdigi "doygun ama a*'da orta" turu
sivilceler de yakalanabiliyor - ekleme, hassasiyeti DUSURMUYOR, tam
tersine iki farkli belirti turunu de kapsayarak ARTIRIYOR.

## YENI: goruntu cozunurlugu duzeltmesi (preprocess.py)

Kullanicinin gercek test fotografi kucuk cozunurlukteydi (170x226 piksel).
Bu cozunurlukte, sabit boyutlu morfolojik gurultu temizligi (3x3 kernel)
ORANTISIZ agresif kaliyor ve gercek sivilce adaylarinin neredeyse tamamini
siliyordu - somut olcum: 914 esik-ustu pikselden sadece 1 bolge
kaliyordu. `preprocess.py`'ye kucuk goruntuleri en az `MIN_WORKING_SIZE`
piksele BUYUTEN bir adim eklendi (bkz. o dosyadaki aciklama).

## Esik ayari

Iki kanalin birlestirilmesiyle gurultu riski dustugu icin
`DETECTION_Z_THRESHOLD`, Akne.2'nin 1.2'sinden 1.0'a indirildi -
gercek fotografta test edilip dogrulandi (bkz. README.md).

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Agirliklar: kirmizilik mi, doygunluk mu daha baskin sayilsin
WEIGHT_REDNESS = 0.6
WEIGHT_SATURATION = 0.4

Z_SCORE_CAP = 2.5

# Akne.2'deki 1.2'den 1.0'a indirildi - gercek (dusuk cozunurluklu) test
# fotografinda gozle gorulur sivilcelerin kacirilmasini onlemek icin,
# iki kanalli birlesim sayesinde gurultu riski daha dusuk oldugundan.
DETECTION_Z_THRESHOLD = 1.0

# Akne.2'deki 6'dan 4'e indirildi - dusuk cozunurluklu fotograflarda bile
# kucuk sivilceleri kacirmamak icin.
MIN_CONTOUR_AREA = 4

# Bir bolgenin "sivilce SEKLINDE" (kucuk, yuvarlak/kompakt) sayilmasi
# icin gereken minimum dairesellik (4*pi*alan/cevre^2, 1.0 = mukemmel
# daire). Bu ESIGIN ALTINDA kalan bolgeler (uzun/ince seritler, yuzun
# dogal hatlarini takip eden konturlar - burun koprusu, cene golgesi,
# nazolabial kivrim gibi) sivilce DEGIL, genis anatomik/dokusal fark
# sayilip ELENIR.
MIN_CIRCULARITY = 0.35

# Bir bolgenin taranan yuz alaninin bu ORANINDAN BUYUK olmasi durumunda
# (tek bir sivilcenin kaplayabileceginden fazla), sivilce degil genis
# bir bolgesel fark sayilip elenir.
MAX_SPOT_AREA_RATIO = 0.012


def _compute_combined_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """
    combined_z = WEIGHT_REDNESS * z_a + WEIGHT_SATURATION * z_s

    z_a: (a - ten_a_ort) / ten_a_std          -> pozitif = daha kirmizi
    z_s: (S - ten_S_ort) / ten_S_std          -> pozitif = daha doygun/canli
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    a_channel = lab[..., 1]
    s_channel = hsv[..., 1]

    skin_region = face_scan_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    skin_a_mean = float(np.mean(a_channel[skin_region]))
    skin_a_std = float(np.std(a_channel[skin_region])) + 1e-6
    skin_s_mean = float(np.mean(s_channel[skin_region]))
    skin_s_std = float(np.std(s_channel[skin_region])) + 1e-6

    z_a = (a_channel - skin_a_mean) / skin_a_std
    z_s = (s_channel - skin_s_mean) / skin_s_std

    z_a = np.clip(z_a, 0, None)
    z_s = np.clip(z_s, 0, None)

    combined = WEIGHT_REDNESS * z_a + WEIGHT_SATURATION * z_s
    return combined


def redness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    combined = _compute_combined_z_map(img_bgr, face_scan_mask)
    normalized = np.clip(combined / Z_SCORE_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def redness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (score, mean_combined, mean_z_redness, mean_z_saturation) -
        bilesenler ayri ayri da donuluyor, "bu bolge neden yuksek skor
        aldi - kirmizi oldugu icin mi, doygun oldugu icin mi, ikisi icin
        mi" gorulebilsin diye (Goz Alti Morluk.3'teki seffaflik fikri).
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return 0.0, 0.0, 0.0, 0.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    a_channel = lab[..., 1]
    s_channel = hsv[..., 1]

    skin_a_mean = float(np.mean(a_channel[scan_region]))
    skin_a_std = float(np.std(a_channel[scan_region])) + 1e-6
    skin_s_mean = float(np.mean(s_channel[scan_region]))
    skin_s_std = float(np.std(s_channel[scan_region])) + 1e-6

    z_a_map = np.clip((a_channel - skin_a_mean) / skin_a_std, 0, None)
    z_s_map = np.clip((s_channel - skin_s_mean) / skin_s_std, 0, None)

    mean_z_a = float(np.mean(z_a_map[scan_region]))
    mean_z_s = float(np.mean(z_s_map[scan_region]))
    mean_combined = WEIGHT_REDNESS * mean_z_a + WEIGHT_SATURATION * mean_z_s

    score = float(np.clip(mean_combined / Z_SCORE_CAP, 0, 1))
    return score, mean_combined, mean_z_a, mean_z_s


def _is_spot_like(contour: np.ndarray, max_area: float) -> bool:
    """
    Bir konturun "sivilce SEKLINDE" (kucuk, yuvarlak/kompakt) olup
    olmadigini kontrol eder. Uzun/ince seritler (burun koprusu, cene
    golge siniri, nazolabial kivrim gibi genis anatomik hatlar) DUSUK
    dairesellige sahiptir ve burada elenir - gercek bir sivilce, alani ne
    olursa olsun, yaklasik yuvarlak bir leke seklindedir.
    """
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> list:
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    if not (scan_mask_u8 > 0).any():
        return []

    combined = _compute_combined_z_map(img_bgr, face_scan_mask)
    binary = (combined > DETECTION_Z_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours = find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)

    scan_area = int((face_scan_mask > 0).sum())
    max_spot_area = max(MIN_CONTOUR_AREA * 4, scan_area * MAX_SPOT_AREA_RATIO)

    return [c for c in contours if _is_spot_like(c, max_spot_area)]
