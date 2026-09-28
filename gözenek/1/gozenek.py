"""
GOZENEK.1 - Black-Hat (Top-Hat) Tabanli Gozenek Tespiti

## Fikir

Gozenekler (pores), ten yuzeyinde cevresinden biraz DAHA KOYU, KUCUK ve
YUVARLAK noktalardir - akne/kizariklik gibi RENK farkindan degil, temel
olarak DOKU/GOLGE farkindan kaynaklanir. Bu yuzden akne ailesindeki gibi
a*/S (renk) kanallarina degil, klasik bir goruntu isleme teknigine
dayaniyoruz: **black-hat morfolojik donusum**.

Black-hat = closing(goruntu) - goruntu

"Closing" (genisletme sonra asindirma) islemi, kucuk KOYU noktalari
DOLDURUR (yok eder), buyuk yapilari (kas, burun golgesi gibi) korur.
Bu doldurulmus versiyondan orijinali cikarinca, geriye SADECE "closing
tarafindan doldurulmus" - yani cevresinden belirgin sekilde koyu VE
KUCUK olan - noktalar kalir. Bu, aynen gozenek/nokta tespiti icin
tasarlanmis standart bir teknik (skin-scan ana projesindeki
`blemishes.py`'de de top-hat/black-hat varyanti kullaniliyordu).

## Istatistik (Akne ailesiyle ayni ruhta - z-score)

Black-hat cikisi ham (mutlak) degerlerle degil, o FOTOGRAFA OZGU
z-score ile degerlendiriliyor - Akne.2'den beri ailede kullanilan
"kisinin kendi tenine gore" mantigi:

    z = (blackhat_piksel - ten_blackhat_ortalama) / ten_blackhat_std

Boylece farkli fotograf/isik kosullarinda "ne kadar black-hat tepkisi
NORMAL sayilir" otomatik olarak kalibre oluyor - sabit bir esik yerine.

## Cozunurlukten bagimsizlik VE skor mantigi (Akne.4/.5'ten ogrenilen
## dersler, BASTAN dogru kurulsun diye)

Akne ailesinde iki hata sonradan duzeltildi:
  1. Sabit piksel-tabanli esikler (kernel/min alan) cozunurlukle
     orantisiz olcekleniyordu -> burada BASTAN goreli (scan alanina
     oranli) kuruldu.
  2. Skor, tespit sayisindan BAGIMSIZ ayri bir ortalamaydi -> burada
     BASTAN skor DOGRUDAN tespit edilen gozenek bolgelerinden (sayi +
     alan + siddet) hesaplaniyor, 0 bolge = kesinlikle %0 skor.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Black-hat cekirdeginin boyutu, taranan yuz alaninin karakteristik
# uzunlugunun (sqrt(scan_piksel_sayisi)) bu orani - gozenek boyutunu
# "hedeflemek" icin akne'nin gurultu-temizleme cekirdeginden daha kucuk
# bir oran kullaniliyor (gozenekler sivilcelerden kucuktur).
BLACKHAT_KERNEL_RATIO = 0.018
MIN_BLACKHAT_KERNEL = 3
MAX_BLACKHAT_KERNEL = 21

# z-score esikleri (Akne ailesiyle ayni cerceve - bkz. o projelerin
# README'leri): Z_SCORE_CAP skorun "tam siddet" sayildigi ust sinir,
# DETECTION_Z_THRESHOLD bir pikselin "gozenek adayi" sayilmasi icin
# gereken minimum z.
Z_SCORE_CAP = 2.5
DETECTION_Z_THRESHOLD = 1.3

# Gurultu temizleme cekirdegi ve minimum kontur alani - Akne.4'teki
# gibi COZUNURLUGE ORANLI (mutlak piksel sayisi degil).
CLEAN_KERNEL_RATIO = 0.006
MIN_CLEAN_KERNEL = 3
MAX_CLEAN_KERNEL = 9

MIN_AREA_RATIO = 0.00004
MIN_AREA_FLOOR = 3

# Sekil filtresi: gozenekler KUCUK ve YUVARLAK olmali - akne'dekiyle
# ayni mantik ama daha kucuk bir ust alan sinirina sahip (bir gozenek,
# bir sivilceden cok daha kucuk olmali).
MIN_CIRCULARITY = 0.3
MAX_SPOT_AREA_RATIO = 0.0025

# Skor, tespit edilen gozeneklerin toplam "yukunden" (alan x siddet)
# hesaplaniyor - Akne.5'teki lezyon-yuku mantigi. Gozenekler akneden
# COK daha fazla sayida ve KUCUK oldugu icin, bu oran akne'nin
# BURDEN_AREA_FRACTION'indan (0.05) daha yuksek: normal bir tende bile
# gorunur onlarca gozenek olmasi beklenir, bu yuzden "yuzun %X'i
# gozenekli = %100 skor" esigi daha yuksek tutuluyor.
BURDEN_AREA_FRACTION = 0.15


def _relative_kernel(face_scan_mask: np.ndarray, ratio: float, min_k: int, max_k: int) -> int:
    scan_px = int((face_scan_mask > 0).sum())
    face_scale = float(np.sqrt(max(scan_px, 1)))
    k = int(round(face_scale * ratio))
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _compute_pore_z_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Black-hat donusumunun z-score haritasini hesaplar.

    Returns:
        (z_map, blackhat_kernel_size)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[..., 0]

    skin_region = face_scan_mask > 0
    if not skin_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32), MIN_BLACKHAT_KERNEL

    kernel_size = _relative_kernel(face_scan_mask, BLACKHAT_KERNEL_RATIO, MIN_BLACKHAT_KERNEL, MAX_BLACKHAT_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    bh_mean = float(np.mean(blackhat[skin_region]))
    bh_std = float(np.std(blackhat[skin_region])) + 1e-6

    z = (blackhat - bh_mean) / bh_std
    return z, kernel_size


def _is_pore_like(contour: np.ndarray, max_area: float) -> bool:
    """Kucuk, yuvarlak/kompakt bolgeleri tutar; uzun/genis konturlari eler."""
    area = cv2.contourArea(contour)
    if area <= 0 or area > max_area:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= MIN_CIRCULARITY


def _detect(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Tek bir tespit sureci - hem z-score haritasini hem de kontur
    listesini birlikte dondurur. Hem gorsellestirme hem skor BU
    fonksiyona dayaniyor, boylece ikisi arasinda tutarsizlik olamaz
    (Akne.5'te duzeltilen sorunun ayni cozumu, bastan uygulandi).
    """
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    z, _ = _compute_pore_z_map(img_bgr, face_scan_mask)

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


def pore_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> np.ndarray:
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    z, _ = _detect(img_bgr, face_scan_mask)
    normalized = np.clip(z / Z_SCORE_CAP, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray) -> list:
    _, contours = _detect(img_bgr, face_scan_mask)
    return contours


def pore_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray):
    """
    Skor, TESPIT EDILEN GOZENEK BOLGELERINDEN (Akne.5 ile ayni "lezyon
    yuku" mantigi) hesaplanir - yuzun tamamindaki ayri bir ortalamadan
    DEGIL. 0 bolge tespit edildiyse skor MATEMATIKSEL OLARAK 0.0'dir.

    Returns:
        (score, mean_pore_z, affected_area_pct, region_count)
    """
    scan_region = face_scan_mask > 0
    scan_area = int(scan_region.sum())
    if scan_area == 0:
        return 0.0, 0.0, 0.0, 0

    z, contours = _detect(img_bgr, face_scan_mask)

    if not contours:
        return 0.0, 0.0, 0.0, 0

    total_burden = 0.0
    total_area = 0
    severities = []

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

    score = float(np.clip(total_burden / (scan_area * BURDEN_AREA_FRACTION), 0, 1))
    mean_pore_z = float(np.mean(severities)) if severities else 0.0
    affected_area_pct = float(total_area / scan_area * 100)

    return score, mean_pore_z, affected_area_pct, len(contours)
