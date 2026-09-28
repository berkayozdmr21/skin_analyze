"""
GOZ ALTI MORLUK.5 - Yerel (spatially-adaptive) aydinlatma duzeltmeli,
reziduel kovaryansli Mahalanobis mesafesi.

M.4, ten bolgesi icin TEK BIR global ortalama vektor (L,a,b) ve TEK BIR
global kovaryans matrisi kullaniyordu. Bunun istatistiksel bir kor
noktasi var: yuzun aydinlatmasi HER ZAMAN duzgun/homojen degildir - ornegin
yandan gelen isik, bir yanagin diger yanaktan daha parlak/koyu gorunmesine
sebep olabilir. Tek bir global ortalama bu durumda "yanlis" bir referans
verir: dogal olarak daha golgeli tarafta kalan goz altinin skoru,
sirf o taraf global ortalamadan uzak oldugu icin YAPAY OLARAK yukselir/
duser - morlukla ilgisi olmayan bir sapma.

## Bu surumde ne degisti

### 1. Global ortalama -> yerel (piksel-bazli) ortalama alani

Tek bir skaler `ten_ortalama_vektoru` yerine, her piksel konumu icin
AYRI bir beklenen ten degeri hesapliyoruz - "normalize edilmis
konvolusyon" (normalized convolution) tekniğiyle:

    yerel_ortalama(x,y) = GaussianBlur(ten_maskesi * L/a/b)(x,y)
                           / GaussianBlur(ten_maskesi)(x,y)

Bu, ten bolgesindeki komsu piksellerin agirlikli (Gauss cekirdekli)
ortalamasini, goz alti dahil TUM goruntuye "yayarak" (extrapole ederek)
her noktada "buradaki cilt normalde nasil olmali" sorusuna piksele ozel
bir cevap verir. Cekirdek genisligi (`sigma`), yuz genisliginin bir
oranidir (`LOCAL_BASELINE_SIGMA_RATIO`) - yeterince genis ki gurultu
degil TREND yakalansin, yeterince dar ki gercekten "yerel" kalsin.

### 2. Ham kovaryans -> rezidüel (detrend edilmis) kovaryans

M.4'te kovaryans matrisi, ten piksellerinin HAM (L,a,b) degerlerinden
hesaplaniyordu - bu da aydinlatma gradyaninin kendisini de varyansa
dahil ediyordu (yuzun bir ucundan digerine dogal parlaklik degisimi,
gercek "doku" gurultusunden cok daha buyuk bir varyans kaynagi olabilir).
Sonuc: kovaryans olmasi gerekenden BUYUK, mesafe olcusu olmasi
gerekenden AZ HASSAS oluyordu.

Bu surumde, her ten pikselinin kendi KONUMUNDAKI yerel ortalamadan
sapmasi (`rezidüel = piksel - o_noktadaki_yerel_ortalama`) hesaplanip,
kovaryans BU rezidüellerden cikariliyor. Aydinlatma trendi cikarildigi
icin, kalan varyans gercekten "doku/gurultu" varyansidir - daha kucuk,
daha SIKI bir kovaryans, dolayisiyla ayni büyüklükteki bir renk/koyuluk
sapmasi artik daha YUKSEK bir Mahalanobis mesafesi veriyor (cunku
"normal" dalgalanma daha dar tanimlanmis oluyor).

### 3. Yon kisitlamasi ve raporlama

M.4 ile ayni mantik korunuyor: sadece yerel referansa gore "daha koyu
(L) VE daha mavi (b)" yonundeki sapmalar sayiliyor. Kanal bazli z-score
raporlamasi da (z_L, z_a, z_b) artik yerel rezidüel std'ye gore.

## Istatistik - skor NEYE gore veriliyor?

    skor = clip( mean(yon_kisitli_D_yerel[goz_altı_bandı]) / MAHALANOBIS_CAP, 0, 1)

Cerceve M.4 ile ayni kaliyor (χ²(3) dagilimi, %99 persentili ≈ 3.37,
tespit esigi %80 persentili ≈ 2.15) - degisen sey mesafenin NEYE GORE
hesaplandigi (yerel trend + rezidüel kovaryans), olcegin kendisi degil.
Bu, iki surumun skorlarinin kabaca karsilastirilabilir kalmasini saglar.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

MAHALANOBIS_CAP = 3.37              # sqrt(chi2.ppf(0.99, df=3))
DETECTION_THRESHOLD = 2.15          # sqrt(chi2.ppf(0.80, df=3))
MIN_CONTOUR_AREA = 20
WINSOR_PERCENTILES = (5, 95)        # rezidüel kovaryans icin aykiri deger kirpma araligi
COV_REG_EPS = 1e-3                  # kovaryans matrisini tersine cevirebilmek icin regularizasyon
LOCAL_BASELINE_SIGMA_RATIO = 0.18   # Gauss cekirdek sigma'si = yuz genisligi * bu oran
MIN_SIGMA = 15.0                    # cekirdek cok kuculmesin diye alt sinir


def _winsorize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo_v, hi_v = np.percentile(values, [lo, hi])
    return np.clip(values, lo_v, hi_v)


def _face_width_from_mask(skin_tone_mask: np.ndarray, img_shape: tuple) -> float:
    ys, xs = np.where(skin_tone_mask > 0)
    if xs.size == 0:
        return float(max(img_shape[:2]))
    return float(xs.max() - xs.min())


def _local_skin_baseline(lab: np.ndarray, skin_tone_mask: np.ndarray, sigma: float):
    """
    Normalize edilmis konvolusyon ile her piksel icin yerel "beklenen ten
    rengi" alanini (H,W,3) hesaplar - ten bolgesindeki komsu piksellerin
    Gauss-agirlikli ortalamasi, tum goruntuye extrapole edilir.
    """
    h, w = skin_tone_mask.shape[:2]
    ksize = int(sigma * 3) | 1  # tek sayi

    mask_f = (skin_tone_mask > 0).astype(np.float64)
    blurred_mask = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
    blurred_mask_safe = np.clip(blurred_mask, 1e-6, None)

    local_mean = np.empty((h, w, 3), dtype=np.float64)
    for c in range(3):
        weighted_channel = lab[..., c] * mask_f
        blurred_channel = cv2.GaussianBlur(weighted_channel, (ksize, ksize), sigma)
        local_mean[..., c] = blurred_channel / blurred_mask_safe

    return local_mean, blurred_mask


def _residual_gaussian(lab: np.ndarray, skin_tone_mask: np.ndarray, local_mean: np.ndarray):
    """
    Ten pikselleri icin rezidüel (piksel - o noktadaki yerel ortalama)
    vektorlerinden robust (winsorize edilmis) ortalama/kovaryans cikarir.
    Rezidüel ortalamasi ~0 olmali (yerel trend zaten cikarildigi icin) -
    yine de kucuk bir kayma varsa raporlamada hesaba katilsin diye donuluyor.
    """
    skin_region = skin_tone_mask > 0
    residual = lab - local_mean  # (H,W,3): piksel - yerel_beklenen

    r_l = residual[..., 0][skin_region]
    r_a = residual[..., 1][skin_region]
    r_b = residual[..., 2][skin_region]

    lo, hi = WINSOR_PERCENTILES
    r_l_w = _winsorize(r_l, lo, hi)
    r_a_w = _winsorize(r_a, lo, hi)
    r_b_w = _winsorize(r_b, lo, hi)

    data = np.stack([r_l_w, r_a_w, r_b_w], axis=1).astype(np.float64)
    residual_mean = data.mean(axis=0)

    cov = np.cov(data, rowvar=False)
    cov = cov + np.eye(3) * COV_REG_EPS
    cov_inv = np.linalg.inv(cov)
    std_vec = np.sqrt(np.clip(np.diag(cov), 1e-9, None))

    return residual_mean, cov_inv, std_vec


def _compute_local_mahalanobis_map(img_bgr: np.ndarray, skin_tone_mask: np.ndarray):
    """
    Her piksel icin, YEREL ten referansina gore yon-kisitli Mahalanobis
    mesafesini hesaplar.

    Returns:
        (D_directional, diff_local, cov_inv, std_vec)
        diff_local: yerel_ortalama - piksel, (H,W,3) - raporlama icin de kullanilir.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    h, w = img_bgr.shape[:2]

    skin_region = skin_tone_mask > 0
    if not skin_region.any():
        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, np.zeros((h, w, 3)), np.eye(3), np.ones(3)

    face_width = _face_width_from_mask(skin_tone_mask, img_bgr.shape)
    sigma = max(face_width * LOCAL_BASELINE_SIGMA_RATIO, MIN_SIGMA)

    local_mean, _blurred_mask = _local_skin_baseline(lab, skin_tone_mask, sigma)
    residual_mean, cov_inv, std_vec = _residual_gaussian(lab, skin_tone_mask, local_mean)

    # diff_local: bu noktada "yerel ten normali" ile pikselin farki,
    # rezidüel ortalamasindaki kucuk kaymaya gore merkezlenmis.
    diff_local = (local_mean - residual_mean.reshape(1, 1, 3)) - lab

    mahalanobis_sq = np.einsum('hwi,ij,hwj->hw', diff_local, cov_inv, diff_local)
    mahalanobis_sq = np.clip(mahalanobis_sq, 0, None)
    distance = np.sqrt(mahalanobis_sq).astype(np.float32)

    # Yon kisitlamasi: sadece daha koyu (diff_L > 0) VE daha mavi (diff_b > 0)
    darker = diff_local[..., 0] > 0
    bluer = diff_local[..., 2] > 0
    directional_gate = darker & bluer

    distance[~directional_gate] = 0.0
    return distance, diff_local, cov_inv, std_vec


def darkness_map(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray) -> np.ndarray:
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    distance, *_ = _compute_local_mahalanobis_map(img_bgr, skin_tone_mask)

    normalized = distance / MAHALANOBIS_CAP
    normalized = np.clip(normalized, 0, 1)
    normalized[~eye_region] = 0

    return normalized.astype(np.float32)


def darkness_score(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray):
    """
    Returns:
        (score, mean_mahalanobis, mean_z_l, mean_z_a, mean_z_b)

    z_L/a/b: goz alti bandindaki her kanalin, YEREL referansa gore
    (rezidüel std'ye bolunmus) ortalama tek-kanalli sapmasi - raporlama
    icindir, skoru etkilemez (skor tamamen D uzerinden).
    """
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0

    distance, diff_local, cov_inv, std_vec = _compute_local_mahalanobis_map(img_bgr, skin_tone_mask)

    z_per_channel = np.clip(diff_local, 0, None) / std_vec.reshape(1, 1, 3)

    mean_distance = float(np.mean(distance[eye_region]))
    mean_z_l = float(np.mean(z_per_channel[..., 0][eye_region]))
    mean_z_a = float(np.mean(z_per_channel[..., 1][eye_region]))
    mean_z_b = float(np.mean(z_per_channel[..., 2][eye_region]))

    score = float(np.clip(mean_distance / MAHALANOBIS_CAP, 0, 1))
    return score, mean_distance, mean_z_l, mean_z_a, mean_z_b


def find_marked_regions(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray) -> list:
    eye_region_mask = (under_eye_mask > 0).astype(np.uint8) * 255
    if not (eye_region_mask > 0).any():
        return []

    distance, *_ = _compute_local_mahalanobis_map(img_bgr, skin_tone_mask)
    binary = (distance > DETECTION_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, eye_region_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)
