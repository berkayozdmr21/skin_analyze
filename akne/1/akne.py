"""
AKNE.1 - Yanak Referansli Kirmizilik (a*) Farki

Goz Alti Morluk.1 ile BIREBIR AYNI mimari ve mantik (yanak referansli,
sabit-olcekli delta, esik-ustu bolgeleri kontur ile isaretleme), akne
(sivilce/kizariklik benegi) tespiti icin uyarlandi.

## Neyin degistigi

1. **Kanal: L* -> a\*.** Goz alti morluk KOYULUGA (L*, parlaklik) bakiyordu
   cunku morluk oncelikle bir golgelenme/koyulasma belirtisi. Akne ise
   oncelikle bir RENK (kizariklik/iltihap) belirtisi - bu yuzden Lab
   renk uzayinin a* (yesil<->kirmizi) kanali kullaniliyor.

2. **Hedef bolge: dar bant -> tum yuz.** Goz alti morluk'ta hedef sabit
   ve dar bir bolgeydi (goz alti bandi) - cunku morluk anatomik olarak
   hep ayni yerde olusur. Akne ise yuzun HERHANGI bir yerinde cikabilir,
   bu yuzden hedef bolge TUM taranabilir yuz cildi (alin+burun+cene+yanak).

3. **YENI: disleme (exclusion) maskesi.** Hedef bolge tum yuze
   genisleyince, dogal olarak akne OLMAYAN ama akneye benzer sekilde
   "kirmizi" ya da farkli renkte olan bolgeler (dudaklar basta olmak
   uzere, gozler ve kaslar da) devreye giriyor. Bunlar disi birakilmazsa
   ornegin dudaklar dogrudan "akne" olarak isaretlenir. face_mesh.py'deki
   make_exclusion_mask() bu bolgeleri (goz+kas+dudak, hafif genisletilmis)
   hedef bolgeden cikariyor.

4. **Referans bolge: DEGISMEDI.** Goz alti morluk.1 gibi, yanaklar
   (cheeks) "tipik/iltihapsiz ten" referansi olarak kullaniliyor.

## Formul (Goz Alti Morluk.1 ile birebir ayni yapida)

    delta = a(piksel) - cheek_a_mean

Pozitif ve buyuk deger = piksel, yanaktan belirgin sekilde daha KIRMIZI
-> akne/kizariklik adayi.

Detaylar icin README.md.
"""
import cv2
import numpy as np

from visualize import find_detection_contours

# Delta'nin (a* biriminde) "tam akne/kizariklik" (skor=1.0) sayildigi
# ust sinir - gorsellestirme ve normalize icin.
DELTA_VISUAL_CAP = 18.0

# Bir pikselin "tespit edilen akne noktasi" (kontur/cerceve ile
# isaretlenecek) sayilmasi icin delta'nin asmasi gereken esik (a* birimi)
DETECTION_THRESHOLD = 7.0

# Kontur bulurken elenecek minimum piksel alani (gurultu temizligi).
# Akne benekleri goz alti torbalarindan kucuk oldugu icin GAM.1'deki
# 20 yerine daha kucuk bir deger kullaniliyor.
MIN_CONTOUR_AREA = 6


def _compute_delta(img_bgr: np.ndarray, cheeks_mask: np.ndarray) -> np.ndarray:
    """a(piksel) - cheek_a_mean farkini tum goruntu icin hesaplar."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_channel = lab[..., 1]

    cheek_region = cheeks_mask > 0
    if not cheek_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    cheek_a_mean = float(np.mean(a_channel[cheek_region]))
    delta = a_channel - cheek_a_mean
    return delta


def redness_map(img_bgr: np.ndarray, face_scan_mask: np.ndarray, cheeks_mask: np.ndarray) -> np.ndarray:
    """
    Akne/kizariklik haritasi.

    Returns:
        [0,1] araliginda normalize edilmis kizariklik haritasi (sadece
        taranan yuz bolgesi icinde, geri kalani 0)
    """
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)

    delta = _compute_delta(img_bgr, cheeks_mask)
    delta_positive = np.clip(delta, 0, None)

    normalized = delta_positive / DELTA_VISUAL_CAP
    normalized = np.clip(normalized, 0, 1)
    normalized[~scan_region] = 0

    return normalized.astype(np.float32)


def redness_score(img_bgr: np.ndarray, face_scan_mask: np.ndarray, cheeks_mask: np.ndarray) -> float:
    """Taranan yuz bolgesindeki ORTALAMA (pozitif) delta, [0,1]'e olceklenmis."""
    scan_region = face_scan_mask > 0
    if not scan_region.any():
        return 0.0

    delta = _compute_delta(img_bgr, cheeks_mask)
    delta_positive = np.clip(delta, 0, None)

    mean_delta = float(np.mean(delta_positive[scan_region]))
    score = mean_delta / DELTA_VISUAL_CAP
    return float(np.clip(score, 0, 1))


def find_marked_regions(img_bgr: np.ndarray, face_scan_mask: np.ndarray, cheeks_mask: np.ndarray) -> list:
    """
    DETECTION_THRESHOLD'u asan piksellerden, taranan yuz bolgesi icinde
    kalan ve yeterince buyuk olan (gurultu degil) kontur listesini
    dondurur. Bu konturlar, fotografta "buralarda akne tespit ettim"
    diye isaretlenecek yerler.
    """
    scan_mask_u8 = (face_scan_mask > 0).astype(np.uint8) * 255
    if not (scan_mask_u8 > 0).any():
        return []

    delta = _compute_delta(img_bgr, cheeks_mask)
    binary = (delta > DETECTION_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, scan_mask_u8)

    # Kucuk gurultu noktalarini temizlemek icin hafif morfolojik acma
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return find_detection_contours(binary, min_area=MIN_CONTOUR_AREA)
