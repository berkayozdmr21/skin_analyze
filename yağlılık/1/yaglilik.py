"""
YAGLILIK.1 - HSV Specular Highlight (Parlama) Tespiti

## Yontem arastirmasi

Cilt yagliligi, OpenCV/goruntu isleme literaturunde tipik olarak
"specular highlight" (aynasal/parlak yansima) tespitiyle olculur - ana
skin-scan projesindeki `src/pipeline/maps/oiliness.py` de dahil olmak
uzere, bu alandaki standart/kanitlanmis yontem:

    1. HSV renk uzayina cevir.
    2. YUKSEK V (Value - parlaklik) VE DUSUK S (Saturation - doygunluk)
       olan pikselleri bul - yagli/parlak ten, isigi neredeyse dogrudan
       yansittigi icin BEYAZA YAKIN (doygunlugu dusuk) ama PARLAK
       gorunur - matbaacilikta/fotografcilikta "specular highlight"
       olarak bilinen klasik bir optik ozellik.
    3. Bu tek basina yeterli degil - beyaz bir nesne (ör. dis, goz aki)
       veya cok parlak/duz bir arka plan da ayni kritere uyabilir. Bunu
       elemek icin GRADYAN BUYUKLUGU (Sobel) ile dogrulama yapilir:
       gercek bir "parlama" ten uzerinde YUMUSAK/DUZ bir gecis yapar
       (dusuk gradyan) - keskin kenarli (yuksek gradyan) parlak
       bolgeler byte/nesne siniri olabilir, parlama degil.

Bu proje ailesindeki (goz alti morluk/akne/gozenek/yuz lekesi)
digerlerinden ONEMLI BIR FARKI var: onceki projeler hep KOYU
lekeler ariyordu (V dusuk = koyu). Yaglilik ise TAM TERSINE PARLAK
bolgeler ariyor (V yuksek) - bu yuzden black-hat degil, kavramsal
olarak "top-hat" mantiginda (parlak/kucuk yapilari bulma) ama burada
dogrudan HSV esikleme + gradyan dogrulamasi ile yapiliyor (literatürdeki
standart yontem).

## Sabit esikten yuzdelik dilime gecis (gercek test fotografiyla bulunan sorun)

Ana skin-scan projesindeki `oiliness.py`, SABIT esikler kullaniyordu
(V>200, S<80). Bu proje de baslangicta ayni sabitlerle kuruldu - ama
gercek test fotografinda (`test edilen goruntuler/File.jpg`) MEDYAN
V degerinin zaten 202 oldugu goruldu! Yani sabit "V>200" esigi, bu
fotografta YUZUN NEREDEYSE YARISINI "yagli" sayiyordu - acikca yanlis
kalibrasyon (fotograflar arasi genel pozlama/parlaklik farki cok
degisebiliyor).

Bu proje ailesinde defalarca ise yarayan cozum burada da uygulandi:
SABIT deger yerine, o FOTOGRAFIN KENDI parlaklik dagilimina gore
YUZDELIK DILIM esigi (`V_PERCENTILE=92`, `S_PERCENTILE=35`) - "bu
yuzun en parlak %8'i VE en az doygun %35'i" - hangi fotografta olursa
olsun, o fotografin KENDI ozel parlama noktalarini buluyor, sabit bir
sayiya guvenmek yerine.

## Skor mantigi - bu projeye ozgu bir tasarim karari

Akne/Gozenek/Yuz Lekesi ailesinde skor, tespit edilen AYRIK
lekelerin SAYISI/SIDDETINDEN hesaplaniyordu - cunku o lekeler dogasi
geregi kucuk, ayri, sayilabilir noktalardi. Yaglilik/parlama ise
FARKLI bir fiziksel dogaya sahip: T-bolgesi boyunca birbirine baglanan,
DUZENSIZ/YAYGIN bir ORTUSME (coverage) olgusu - "kac tane parlama var"
sorusu anlamli degil (genelde TEK BUYUK bir baglanti kumesi olusur).

Bu yuzden skor burada KAPLANAN ALAN YUZDESINE dayaniyor:

    skor = clip( (yagli_piksel_sayisi / taranan_yuz_alani) / OILY_COVERAGE_CAP, 0, 1)

`OILY_COVERAGE_CAP`, "yuzun bu yuzdesi parlak/yagli olsa skor %100'e
ulasir" anlamina gelen ampirik bir referans - gercek bir test
fotografiyla kalibre edildi (bkz. README.md).

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# HSV esikleri - ana skin-scan projesindeki oiliness.py'nin SABIT
# degerleriyle (V>200, S<80) baslandi, ama gercek test fotografinda
# (medyan V=202!) bu esigin YUZUN YARISINI "yagli" saydigi gorulunce
# vazgecildi. Sebep: fotograflar arasinda genel pozlama/parlaklik COK
# degisebiliyor (flas, on-isleme gamma duzeltmesi vb.) - sabit bir V
# esigi, parlak cekilmis bir fotografta anlamsizlasiyor.
#
# Cozum, bu proje ailesinde defalarca ise yarayan ayni fikir: SABIT
# deger yerine, o FOTOGRAFIN KENDI parlaklik dagilimina gore YUZDELIK
# DILIM esigi. "Bu yuzun en parlak %V_PERCENTILE'i VE en az doygun
# %S_PERCENTILE'i" - hangi fotografta olursa olsun, ozel/gercek
# parlama noktalarini (fotografin geri kalanina GORE) yakalar.
V_PERCENTILE = 92.0
S_PERCENTILE = 35.0

# Gradyan buyuklugu esigi - bunun ALTINDAKI (yumusak/duz) bolgeler
# "parlama" sayilir, USTUNDEKI (keskin kenar) bolgeler nesne/nokta
# siniri olarak elenir.
GRADIENT_THRESHOLD = 50.0

# Gurultu temizleme cekirdegi, cozunurlukten bagimsiz (goreli).
CLEAN_KERNEL_RATIO = 0.010
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 21

MIN_AREA_RATIO = 0.00015
MIN_AREA_FLOOR = 6

# Skorun %100'e ulastigi kaplanan alan orani - gercek test fotografiyla
# kalibre edildi (bkz. README.md).
OILY_COVERAGE_CAP = 0.08


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _compute_oily_mask(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    HSV esikleme (yuksek V + dusuk S) + gradyan dogrulamasi ile
    specular highlight (parlama) adaylarini bulur.

    Returns:
        (binary_mask uint8 0/255, clean_kernel_size)
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8), MIN_CLEAN_KERNEL

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]
    s_channel = hsv[..., 1]

    v_values = v_channel[scan_region]
    s_values = s_channel[scan_region]
    v_cutoff = float(np.percentile(v_values, V_PERCENTILE))
    s_cutoff = float(np.percentile(s_values, S_PERCENTILE))

    high_v = v_channel > v_cutoff
    low_s = s_channel < s_cutoff
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

    return cleaned, clean_kernel_size


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """Tek bir tespit sureci - ikili maske ve kontur listesi birlikte doner."""
    binary, _ = _compute_oily_mask(img_bgr, face_scan_mask)

    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    contours = find_detection_contours(binary, min_area=min_area)
    return binary, contours


def oiliness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """[0,1] araliginda ikili (yagli=1, degil=0) bir harita - heatmap gorsellestirmesi icin."""
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    binary, _ = _detect(img_bgr, face_scan_mask)
    normalized = (binary > 0).astype(np.float32)
    normalized[~scan_region] = 0

    return normalized


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> list:
    _, contours = _detect(img_bgr, face_scan_mask)
    return contours


def oiliness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, KAPLANAN ALAN YUZDESINE dayanir (bkz. dosya basindaki
    aciklama) - Akne/Gozenek/Yuz Lekesi ailesindeki "sayi/siddet
    agirlikli" yaklasimdan BILINCLI BIR SAPMA, cunku yaglilik/parlama
    ayrik lekeler degil yaygin bir ortusme olgusu.

    Returns:
        (score, coverage_pct, region_count)
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0

    binary, contours = _detect(img_bgr, face_scan_mask)

    oily_px = int((binary > 0).sum())
    coverage_ratio = oily_px / scan_area

    score = float(np.clip(coverage_ratio / OILY_COVERAGE_CAP, 0, 1))
    coverage_pct = float(coverage_ratio * 100)

    return score, coverage_pct, len(contours)
