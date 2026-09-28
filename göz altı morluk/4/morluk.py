"""
GOZ ALTI MORLUK.4 - Cok degiskenli (multivariate) Mahalanobis mesafesi,
robust (winsorized) istatistik yontemi.

Onceki surumler renk kanallarini BIRBIRINDEN BAGIMSIZ ele aliyordu:
  - M.2: sadece L* (koyuluk) z-score.
  - M.3: L* ve b* (mavi-mor ton) z-score'lari ayri ayri hesaplanip,
    SABIT/ELLE SECILMIS agirliklarla (0.6 / 0.4) toplaniyordu.

Bu yaklasimin iki istatistiksel zayifligi var:
  1. Kanallar arasindaki KORELASYON goz ardi ediliyor. Ornegin bir kisinin
     teninde L* dusukken b*'nin de dogal olarak biraz dusuk olma egilimi
     varsa, bu iki kanali bagimsizmis gibi toplamak sapmayi OLDUGUNDAN
     FAZLA buyutur (cift sayim).
  2. Agirliklar (0.6/0.4) veriden degil, elle secildi - "dogru" oldugunu
     gosteren istatistiksel bir dayanak yok.

## Bu surumde ne degisti

1. **Mahalanobis mesafesi**: L*, a*, b* uc kanali birlikte, ten
   bolgesinin GERCEK kovaryans yapisini (Sigma) kullanarak tek bir
   mesafeye indiriyoruz:

       diff  = ten_ortalama_vektoru - piksel_vektoru   (3 boyutlu: L,a,b)
       D     = sqrt(diff^T . Sigma^-1 . diff)

   Bu, iki ayri z-score'u elle agirliklandirip toplamak yerine, kanallar
   arasi korelasyonu otomatik olarak hesaba katan, istatistikte standart
   olan "bu nokta dagilimin merkezinden kac birim uzakta" olcusudur.

2. **a* kanali da dahil edildi**: M.3 sadece L* ve b* kullaniyordu.
   Gercek bir morlukta (damarlarin ince ciltten gorunmesi) sadece maviye
   degil, hafifce KIRMIZI/MAGENTA yonune de kayma olabilir (mor = mavi +
   kirmizi). a* kanalini da vektore ekleyip Sigma'ya dahil ederek bu
   bilgiyi de (varsa) mesafeye yansitiyoruz.

3. **Robust (winsorized) istatistik**: Ten bolgesinin ortalama/kovaryans
   hesaplamasi ARTIK ham (mean/std) degil, once her kanal %5-%95
   persentillerine kirpiliyor (winsorize). Boylece ten maskesine yanlislikla
   giren birkac asiri piksel (parlama/highlight, kirpik golgesi, sac teli)
   referans istatistigini bozamiyor - M.2/M.3'te bu bir risk olarak
   README'de belirtilmisti, burada dogrudan ele alindi.

4. **Yon kisitlamasi korundu**: Mahalanobis mesafesi yonsuzdur (parlak
   bir piksel de yuksek mesafe verebilir). M.3'teki gibi, sadece
   "tenden daha KOYU (L dusuk) VE daha MAVI (b dusuk)" yonundeki
   sapmalar sayiliyor - aksi yondeki sapmalar (ör. parlama) skora
   katkida bulunmuyor.

## Istatistik - skor NEYE gore veriliyor?

    skor = clip( mean(yon_kisitli_D[goz_altı_bandı]) / MAHALANOBIS_CAP, 0, 1)

`MAHALANOBIS_CAP`, z-score'daki "2.5 sigma" mantiginin cok degiskenli
karsiligi: 3 serbestlik dereceli bir ki-kare (chi-squared) dagiliminda,
D^2 degerinin %99 persentili ~11.3'tur, yani D ~ 3.37. Bu, "3 kanalda
birlikte, tesadufen olusma ihtimali %1'den kucuk bir sapma" anlamina
gelir - M.2/M.3'teki 2.5 sigma'nin (tek kanalda %99'luk dilim disi)
doğal, cok-degiskenli genellemesi.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

MAHALANOBIS_CAP = 3.37          # sqrt(chi2.ppf(0.99, df=3)) - bkz. yukaridaki not
DETECTION_THRESHOLD = 2.15      # sqrt(chi2.ppf(0.80, df=3)) - tespit isaretleme esigi
MIN_CONTOUR_AREA = 20
WINSOR_PERCENTILES = (5, 95)    # ten referans istatistigi icin aykiri deger kirpma araligi
COV_REG_EPS = 1e-3              # kovaryans matrisini tersine cevirebilmek icin kucuk regularizasyon


def _winsorize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo_v, hi_v = np.percentile(values, [lo, hi])
    return np.clip(values, lo_v, hi_v)


def _skin_gaussian(img_lab: np.ndarray, skin_region: np.ndarray):
    """
    Ten bolgesinden robust bir 3-boyutlu (L,a,b) Gauss modeli cikarir:
    ortalama vektor + kovaryans matrisinin tersi. Winsorize edilmis
    verilerden hesaplanir (aykiri piksellere karsi dayaniklilik icin).

    Returns:
        mean_vec (3,), cov_inv (3,3), std_vec (3,) - std_vec sadece
        raporlama/per-kanal z-score gosterimi icin (kovaryansin
        kosegeninin karekoku).
    """
    l_vals = img_lab[..., 0][skin_region]
    a_vals = img_lab[..., 1][skin_region]
    b_vals = img_lab[..., 2][skin_region]

    lo, hi = WINSOR_PERCENTILES
    l_w = _winsorize(l_vals, lo, hi)
    a_w = _winsorize(a_vals, lo, hi)
    b_w = _winsorize(b_vals, lo, hi)

    data = np.stack([l_w, a_w, b_w], axis=1).astype(np.float64)
    mean_vec = data.mean(axis=0)

    cov = np.cov(data, rowvar=False)
    cov = cov + np.eye(3) * COV_REG_EPS
    cov_inv = np.linalg.inv(cov)
    std_vec = np.sqrt(np.clip(np.diag(cov), 1e-9, None))

    return mean_vec, cov_inv, std_vec


def _compute_mahalanobis_map(img_bgr: np.ndarray, skin_tone_mask: np.ndarray):
    """
    Her piksel icin yon-kisitli Mahalanobis mesafesini hesaplar.

    Returns:
        (D_directional, mean_vec, cov_inv, std_vec)
        D_directional: sadece "tenden daha koyu VE daha mavi" yonunde
        sapan pikseller icin > 0, digerleri icin 0.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    h, w = img_bgr.shape[:2]

    skin_region = skin_tone_mask > 0
    if not skin_region.any():
        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, np.zeros(3), np.eye(3), np.ones(3)

    mean_vec, cov_inv, std_vec = _skin_gaussian(lab, skin_region)

    diff = mean_vec.reshape(1, 1, 3) - lab  # ten_ortalama - piksel, (H,W,3)

    mahalanobis_sq = np.einsum('hwi,ij,hwj->hw', diff, cov_inv, diff)
    mahalanobis_sq = np.clip(mahalanobis_sq, 0, None)
    distance = np.sqrt(mahalanobis_sq).astype(np.float32)

    # Yon kisitlamasi: sadece daha koyu (diff_L > 0) VE daha mavi (diff_b > 0)
    darker = diff[..., 0] > 0
    bluer = diff[..., 2] > 0
    directional_gate = darker & bluer

    distance[~directional_gate] = 0.0
    return distance, mean_vec, cov_inv, std_vec


def darkness_map(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray) -> np.ndarray:
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    distance, *_ = _compute_mahalanobis_map(img_bgr, skin_tone_mask)

    normalized = distance / MAHALANOBIS_CAP
    normalized = np.clip(normalized, 0, 1)
    normalized[~eye_region] = 0

    return normalized.astype(np.float32)


def darkness_score(img_bgr: np.ndarray, under_eye_mask: np.ndarray, skin_tone_mask: np.ndarray):
    """
    Returns:
        (score, mean_mahalanobis, mean_z_l, mean_z_a, mean_z_b)

    mean_z_l/a/b: goz alti bandindaki her kanalin, ten referansina gore
    (robust std'ye bolunmus) ortalama tek-kanalli sapmasi - "bu bolge
    neden yuksek cikti" sorusuna kanal bazinda cevap icin (M.3'teki
    seffaflik fikrinin 3 kanala genellenmis hali). Bu degerler skoru
    ETKILEMIYOR, sadece raporlama icindir - skor tamamen D (Mahalanobis)
    uzerinden hesaplanir.
    """
    eye_region = under_eye_mask > 0
    if not eye_region.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0

    distance, mean_vec, cov_inv, std_vec = _compute_mahalanobis_map(img_bgr, skin_tone_mask)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    diff = mean_vec.reshape(1, 1, 3) - lab
    z_per_channel = np.clip(diff, 0, None) / std_vec.reshape(1, 1, 3)

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

    distance, *_ = _compute_mahalanobis_map(img_bgr, skin_tone_mask)
    binary = (distance > DETECTION_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, eye_region_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)
