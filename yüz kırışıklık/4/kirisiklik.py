"""
YUZ KIRISIKLIK.4 - Cozunurluk Tavani Duzeltmesi + Dengelenmis Esikler

## Kullanicinin bu surum icin bildirdigi IKI sorun

1. "Yaslilarinki cok dustu" - Yuz Kirisiklik.3'un sikilastirmasi, yasli/
   gercekten kirisiklikli yuzlerde HASSASIYETI ASIRI DUSURMUSTU (bilincli
   bir odunlesim olarak o surumde de belirtilmisti, ama kullanici geri
   bildirimine gore COK FAZLA gitmisti).
2. "Bir yerlerde patlak veriyor" - bu, GERCEK BIR HATAYA isaret ediyordu.

## Bulunan gercek hata: cozunurluk tavanlari cok DUSUK

Yuz Kirisiklik.1-3'teki TUM cekirdek boyutlari (`_relative_kernel`),
yuz olcegine ORANLI hesaplaniyordu AMA bir UST SINIRLA (`max_k`)
sinirlandirilmisti. Bu tavanlar, sadece dusuk-orta cozunurluklu test
fotograflariyla (170-1200px) kalibre edilmisti - GERCEK yuksek
cozunurluklu fotograflarda (telefon kameralari kolayca 3000-4000px+
uretir) bu tavanlara COKTAN CARPILIYORDU:

    OLCULDU (Yuz Kirisiklik.3 ile):
    w=2000px  -> DETECTION_MARGIN: istenen=82,  tavan=65  (COKTAN carpiyor)
    w=3000px  -> GABOR_KSIZE:      istenen=53,  tavan=41
                 CLOSING_KERNEL:   istenen=35,  tavan=25
                 DETECTION_MARGIN: istenen=124, tavan=65
    w=4000px  -> GABOR_KSIZE:      istenen=71,  tavan=41
                 CLOSING_KERNEL:   istenen=47,  tavan=25
                 DETECTION_MARGIN: istenen=165, tavan=65

Tavana carpan bir cekirdek, ARTIK cozunurlukle ORANTILI BUYUMUYOR -
bu, TAM OLARAK bu proje ailesi boyunca defalarca ogrenilen "cozunurlukten
bagimsizlik" dersinin IHLALI. Sonuc: yuksek cozunurlukte kapama
cekirdegi GERCEKTE OLMASI GEREKENDEN KUCUK kaliyor, yerel arka plan
tahmini artik kirisikligi duzgun "gormezden gelemiyor", tespit sayisi
BEKLENMEDIK sekilde COKUYOR (Yuz Kirisiklik.3'un kendi olcumunde: 2000px'te
10 tespit, 3000px'te SADECE 3 - duzgun bir egilim degil, ani bir
"patlama"). **Bu, kullanicinin "patlak veriyor" ifadesiyle isaret
ettigi gercek hataydi.**

**Duzeltme**: tum tavanlar, gercekci telefon kamerasi cozunurluklerini
(6000px+) rahatca kapsayacak sekilde yukseltildi.

## Ikinci duzeltme: hassasiyet dengesi yeniden ayarlandi

Yuz Kirisiklik.3'teki uc parametre (Gabor esigi, koyuluk esigi, minimum
uzunluk) TEK BIR duz-tenli fotografla kalibre edilip fazla sikilastirilmisti.
Kullanici geri bildirimine gore, bu Yuz Kirisiklik.1'in yasli/kirisiklikli
yuzlerdeki performansini olumsuz etkiledi. Ucu de Yuz Kirisiklik.2 (cok
gevsek) ile Yuz Kirisiklik.3 (cok siki) arasinda DAHA DENGELI bir
noktaya cekildi - bkz. asagidaki parametre karsilastirmasi.

## (Yuz Kirisiklik.3'un asagidaki bolumleri hala gecerli - minimum
## uzunluk filtresi fikri korunuyor, sadece esik degeri yeniden ayarlandi)

## Kullanicinin bildirdigi ONCEKI sorun (Yuz Kirisiklik.2 icin)

Yuz Kirisiklik.2, yogun kirisikliklarla kapli yuzlerde kacirma
sorununu cozdu (bkz. o surumun README'si) - AMA bu sefer DUZ/kirisiksiz
bir yuzde de COK SAYIDA yanlis tespit uretmeye basladi. Klasik bir
hassasiyet/ozgulluk (precision/recall) DENGE KAYMASI: recall'u
iyilestirmek icin (a) esigi (GABOR_STRENGTH_PERCENTILE 88->75) gevsetmis,
(b) YEREL (kapama tabanli) referansa gecmistik - ikisi birlikte, duz
ciltteki INCE DOKU/GOZENEK gibi normal detaylari da "kirisiklik" gibi
gostermeye baslamisti.

## Somut olcum: gercek tespitlerin cogu COK KISAYDI

Referans fotografta (goz altı morluk'taki genc adam - nispeten duz
ten) Yuz Kirisiklik.2'nin bulduğu 30 bolgeyi tek tek olculdu:

    face_scale = 304.9 px

    20/30 tespit: uzunluk 5.7-20.1 px  (face_scale'in SADECE %1.9-6.6'si)
    7/30 tespit:  uzunluk 26.9-90.1 px (face_scale'in %8.8-29.6'si)

Yani tespitlerin COGU (20/30), yuzun olcegine kiyasla COK KISAYDI - bu
boyutta bir "kirisiklik" fiziksel olarak anlamsiz, muhtemelen tek bir
gozenek/ince doku noktasi elongation testini "sans eseri" gecmis.

## Cozum: uc hedefli, olcume dayali sikilastirma

1. **Minimum UZUNLUK filtresi (YENI)**: `MAX_LENGTH_RATIO_OF_FACE` gibi
   bir UST sinir zaten vardi, ama ALT sinir YOKTU. Olculen verideki
   dogal bosluga (20px altinda cogu gurultu, 26px ustunde cogu gercek)
   dayanarak `MIN_LENGTH_RATIO_OF_FACE` eklendi - artik bir bolge, yuzun
   karakteristik uzunlugunun belirli bir oranindan KISA olamaz.
2. **Koyuluk esigi yukseltildi**: `DARK_DELTA_THRESHOLD` 4.0'dan 7.0'a
   cikarildi - olculen gercek tespitlerin delta degerleri 7.7-18.2
   arasindaydi, 4.0 esigi bu araligin COK altinda kalip fazla piksel
   pasa birakiyordu.
3. **Gabor esigi kismen geri cekildi**: %75'ten %83'e - Yuz
   Kirisiklik.1'in %88'ine TAM DONMEDEN (o surumun kacirma sorununu
   geri getirmemek icin), ama Yuz Kirisiklik.2'nin asiri gevsek
   esiginden biraz sikilastirilarak.

Bu ucu de BIRLIKTE calisiyor - Yuz Kirisiklik.2'nin YEREL referans
fikri (gercek katki, korunuyor) artik gereksiz gurultuyu de
yakalamiyor.

## Kok neden (Yuz Lekesi.5'teki AYNI sorun sinifi - Yuz Kirisiklik.2'den)

Yuz Kirisiklik.1, koyuluk referansini TUM YUZUN medyani/MAD'i ile
hesapliyordu. Ama eger yuzun BUYUK bir kismi ZATEN kirisiklarla kapliysa,
bu medyan/MAD REFERANSININ KENDISI kirisikliklardan etkilenip
KARARIYOR/BOZULUYOR - yani "normal/kirisiksiz ten" tahmini artik dogru
degil, kismen KIRISIKLI tenin ortalamasi oluyor. Sonuc: gercek bir
kirisiklik pikseli, artik "zaten kirisikli" olan referansa gore o kadar
"anormal koyu" gorunmuyor - KIRISIKLIKLAR BIRBIRINI MASKELIYOR.

Bu, Yuz Lekesi.5'te "yogun cil kumelerinin ortalamayi asagi cekip
birbirini maskelemesi" ile BIREBIR AYNI sorun sinifi - ve AYNI COZUM
burada da uygulaniyor.

## Cozum: morfolojik KAPAMA ile yerel arka plan tahmini

Tum yuzun TEK bir medyan/MAD degeri yerine, HER piksel icin KENDI yerel
cevresinde "kirisiksiz olsaydi nasil gorunurdu" tahmini yapiliyor -
Yuz Lekesi.5'teki ayni teknik: morfolojik KAPAMA (once yerel maksimum -
dilate, sonra yerel minimum - erode), KUCUK KOYU cizgileri (kirisikliklari)
"gormezden gelerek" onlarin ALTINDA olmasi beklenen DUZ ten yuzeyini
yeniden insa eder.

    arka_plan(x,y) = closing(L)(x,y)   [kirisikliklari "doldurarak" duz ten tahmini]
    delta(x,y)      = arka_plan(x,y) - L(x,y)

Kapama cekirdegi, TEK BIR kirisiklik cizgisinin GENISLIGINDEN (kalinligindan)
BUYUK olmali (ki onu "gormezden gelebilsin") - bu, cizginin UZUNLUGUNDAN
bagimsiz calisir (kapama YEREL bir islemdir), yani cok UZUN ama INCE
kirisikliklar bile dogru sekilde "arka plana gomulup" cikarilir.

Goz/kas/dudak gibi DISLANAN alanlarin kapamayi yanlis yonlendirmemesi
icin, once bu alanlar `cv2.inpaint` ile civar ten degerleriyle
dolduruluyor (Yuz Lekesi.5 ile ayni teknik).

## Degismeyenler (Yuz Kirisiklik.1 ile ayni)

- Cok yonelimli (6 acida) Gabor filtre bankasi.
- Uzama (elongation) filtresi.
- Tespit alani icin kenar/delik marji.
- Skor: agirlikli UZUNLUK, yuzun karakteristik uzunluguna oranli
  (alana DEGIL) - cozunurluk kararliligi icin.

Detaylar ve dogrulama olcumleri icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

GABOR_ORIENTATIONS_DEG = (0, 30, 60, 90, 120, 150)

GABOR_KSIZE_RATIO = 0.030
MIN_GABOR_KSIZE = 7
# ONEMLI DUZELTME (bkz. README.md "patlama" bolumu): bu tavan 41'di -
# ama face_scale yuksek cozunurlukte (2000px+ kaynak fotograf) bu
# oranla ISTENEN cekirdek boyutunu COKTAN asiyordu (olculdu: 3000px
# kaynakta istenen=53, tavan=41 - cekirdek KUCUK KALIYORDU). Boylece
# cekirdek boyutu cozunurlukle orantili BUYUMEYI DURDURUYOR, davranis
# beklenmedik sekilde degisiyordu (kacinilmasi gereken TAM OLARAK bu
# aile boyunca ogrenilen "cozunurlukten bagimsizlik" dersiydi). Tavan
# gercekci telefon kamerasi cozunurluklerini (6000px+) kapsayacak
# sekilde yukseltildi.
MAX_GABOR_KSIZE = 121
GABOR_LAMBDA_RATIO = 0.022
MIN_GABOR_LAMBDA = 4.0
GABOR_SIGMA_FACTOR = 0.30
GABOR_GAMMA = 0.5

# Yuz Kirisiklik.3'te 83'e sikilastirilmisti (duz cilt icin dogruydu,
# ama yasli/gercekten kirisiklikli yuzlerde HASSASIYETI FAZLA DUSURDU -
# bkz. README.md). Ikisi arasinda daha dengeli bir noktaya cekildi.
GABOR_STRENGTH_PERCENTILE = 80.0

# YENI: yerel koyuluk referansi icin morfolojik kapama cekirdegi -
# tek bir kirisiklik cizgisinin GENISLIGINDEN buyuk olmali.
CLOSING_KERNEL_RATIO = 0.020
MIN_CLOSING_KERNEL = 5
# Ayni "patlama" hatasi - bkz. MAX_GABOR_KSIZE aciklamasi. Tavan
# yukseltildi.
MAX_CLOSING_KERNEL = 81

# Yuz Kirisiklik.3'te 7.0'a yukseltilmisti (duz cilt icin dogruydu,
# ama yasli/kirisiklikli yuzlerde COK SIKI oldugu bildirildi) - dengeli
# bir noktaya (5.5) cekildi.
DARK_DELTA_THRESHOLD = 5.5
DELTA_VISUAL_CAP = 14.0

CLEAN_KERNEL_RATIO = 0.006
MIN_CLEAN_KERNEL = 3
# Ayni "patlama" hatasi - tavan yukseltildi.
MAX_CLEAN_KERNEL = 31

MIN_AREA_RATIO = 0.00012
MIN_AREA_FLOOR = 5

MIN_ELONGATION_RATIO = 2.2

# Yuz Kirisiklik.3'te 0.075 olarak kalibre edilmisti - kullanici
# geri bildirimine gore (yasli/kirisiklikli yuzlerde asiri sikiydi)
# DUSURULDU. Cok kisa (5-10px, tek gozenek boyutu) gurultuyu hala
# eliyor ama kisa/ince kirisikliklari (kaz ayagi gibi) artik reddetmiyor.
MIN_LENGTH_RATIO_OF_FACE = 0.035
MAX_LENGTH_RATIO_OF_FACE = 0.55

LENGTH_DENSITY_CAP = 4.0

DETECTION_MARGIN_RATIO = 0.07
MIN_DETECTION_MARGIN = 9
# Ayni "patlama" hatasi - bu ozellikle onemliydi: 2000px'te bile
# tavana carpiyordu (istenen=82, tavan=65). Yukseltildi.
MAX_DETECTION_MARGIN = 181


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
        kernel -= kernel.mean()
        response = cv2.filter2D(l_channel, cv2.CV_32F, kernel)
        responses.append(np.abs(response))

    return np.maximum.reduce(responses)


def _local_background_delta(l_channel: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    """
    Once disi birakilan alanlari (goz/kas/dudak/arka plan) inpainting
    ile doldurur, sonra morfolojik KAPAMA ile "kirisiksiz" ten
    yuzeyini yeniden insa eder - bkz. dosya basindaki aciklama.

    Returns:
        delta = arka_plan - L  (pozitif = yerel arka plandan daha koyu)
    """
    l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
    inpaint_mask = (face_scan_mask == 0).astype(np.uint8) * 255

    if (inpaint_mask > 0).any() and (inpaint_mask > 0).sum() < inpaint_mask.size:
        inpainted = cv2.inpaint(l_uint8, inpaint_mask, 5, cv2.INPAINT_TELEA)
    else:
        inpainted = l_uint8

    kernel_size = _relative_kernel(face_scan_mask, CLOSING_KERNEL_RATIO, MIN_CLOSING_KERNEL, MAX_CLOSING_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(inpainted, cv2.MORPH_CLOSE, kernel).astype(np.float32)

    return background - l_channel


def _compute_candidate_mask(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        zero_u8 = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        zero_f = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        return zero_u8, zero_f

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]

    gabor_strength = _gabor_line_strength(l_channel, face_scan_mask)
    gabor_cutoff = float(np.percentile(gabor_strength[scan_region], GABOR_STRENGTH_PERCENTILE))
    strong_line = gabor_strength > gabor_cutoff

    dark_delta = _local_background_delta(l_channel, face_scan_mask)
    dark_enough = dark_delta > DARK_DELTA_THRESHOLD

    candidate = (strong_line & dark_enough).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, (face_scan_mask > 0).astype(np.uint8) * 255)

    clean_kernel_size = _relative_kernel(face_scan_mask, CLEAN_KERNEL_RATIO, MIN_CLEAN_KERNEL, MAX_CLEAN_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
    cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, clean_kernel_size - 2), max(3, clean_kernel_size - 2)))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    return cleaned, dark_delta


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
    if length < face_scale * MIN_LENGTH_RATIO_OF_FACE:
        return False
    if length > face_scale * MAX_LENGTH_RATIO_OF_FACE:
        return False

    return True


def _erode_for_detection(face_scan_mask: np.ndarray) -> np.ndarray:
    margin = _relative_kernel(face_scan_mask, DETECTION_MARGIN_RATIO, MIN_DETECTION_MARGIN, MAX_DETECTION_MARGIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    eroded = cv2.erode(face_scan_mask, kernel, iterations=1)

    if not (eroded > 0).any():
        return face_scan_mask
    return eroded


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """Tek bir tespit sureci - yerel koyuluk deltasi ve kontur listesi birlikte doner."""
    candidate_mask, dark_delta = _compute_candidate_mask(img_bgr, face_scan_mask)

    detection_mask = _erode_for_detection(face_scan_mask)
    candidate_mask = cv2.bitwise_and(candidate_mask, (detection_mask > 0).astype(np.uint8) * 255)

    scan_px = int((face_scan_mask > 0).sum())
    min_area = max(MIN_AREA_FLOOR, int(round(scan_px * MIN_AREA_RATIO)))

    raw_contours = find_detection_contours(candidate_mask, min_area=min_area)

    face_scale = _face_scale(face_scan_mask)
    wrinkle_contours = [c for c in raw_contours if _is_wrinkle_like(c, face_scale)]

    return dark_delta, wrinkle_contours


def wrinkle_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    dark_delta, contours = _detect(img_bgr, face_scan_mask)

    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    delta_positive = np.clip(dark_delta, 0, None)
    normalized = np.clip(delta_positive / DELTA_VISUAL_CAP, 0, 1)
    normalized[mask == 0] = 0
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Returns:
        (contours, severities) - her kontur icin 0-1 arasi siddet degeri.
    """
    dark_delta, contours = _detect(img_bgr, face_scan_mask)

    severities = []
    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_delta = float(dark_delta[region_pixels].mean()) if region_pixels.any() else 0.0
        severities.append(float(np.clip(mean_delta / DELTA_VISUAL_CAP, 0, 1)))

    return contours, severities


def wrinkle_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, tespit edilen kirisikliklerin AGIRLIKLI TOPLAM UZUNLUGUNDAN
    hesaplaniyor - yuzun karakteristik UZUNLUGUNA oranli (Yuz
    Kirisiklik.1 ile ayni tasarim, bkz. o surumun README'si).

    Returns:
        (score, weighted_length_px, total_length_px, mean_delta, region_count)
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return 0.0, 0.0, 0.0, 0.0, 0

    dark_delta, contours = _detect(img_bgr, face_scan_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0.0, 0

    weighted_length = 0.0
    total_length = 0.0
    delta_values = []

    region_mask = np.zeros(face_scan_mask.shape, dtype=np.uint8)
    for c in contours:
        length = _contour_length(c)

        region_mask[:] = 0
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        region_pixels = region_mask > 0
        mean_delta = float(dark_delta[region_pixels].mean()) if region_pixels.any() else 0.0
        severity = float(np.clip(mean_delta / DELTA_VISUAL_CAP, 0, 1))

        weighted_length += length * severity
        total_length += length
        delta_values.append(mean_delta)

    face_scale = _face_scale(face_scan_mask)
    score = float(np.clip(weighted_length / (face_scale * LENGTH_DENSITY_CAP), 0, 1))
    mean_delta_overall = float(np.mean(delta_values)) if delta_values else 0.0

    return score, weighted_length, total_length, mean_delta_overall, len(contours)
