"""Image preprocessing - goz alti morluk/kizariklik ailesinden kopya,
AKNE.3'e ozgu bir ekleme ile: kucuk goruntuleri BUYUTME adimi.

Neden gerekti: gercek test fotograflarinin cogu (ornegin 170x226 piksel
gibi) ailenin varsayimladigi "buyuk fotograf, kucult" senaryosunun tam
TERSI - KUCUK. Orijinal preprocess() sadece max_size'dan BUYUK
goruntuleri kucultuyordu, kucukleri OLDUGU GIBI birakiyordu. Dusuk
cozunurlukte, akne.py'deki morfolojik gurultu temizligi (3x3 kernel) ve
minimum kontur alani esigi ORANTISIZ agresif kaliyor, gercek sivilce
bolgelerinin cogunu siliyordu (bkz. README.md - test edilen bir
fotografta 914 aday pikselden sadece 1 bolge kaliyordu).
"""
import cv2
import numpy as np

# Goruntu bu boyuttan KUCUKSE, morfolojik islemler ve alan esikleri
# tutarli calissin diye INTER_CUBIC ile buyutuluyor.
MIN_WORKING_SIZE = 700


def preprocess(img_bgr: np.ndarray, max_size: int = 1024) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif max(h, w) < MIN_WORKING_SIZE:
        scale = MIN_WORKING_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    img_bgr = gray_world_normalization(img_bgr)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img_bgr = gamma_correction(img_bgr, gamma=1.1)
    return img_bgr


def gray_world_normalization(img_bgr: np.ndarray) -> np.ndarray:
    img_float = img_bgr.astype(np.float32)
    b_mean = np.mean(img_float[..., 0])
    g_mean = np.mean(img_float[..., 1])
    r_mean = np.mean(img_float[..., 2])
    gray_mean = (b_mean + g_mean + r_mean) / 3
    if b_mean > 0:
        img_float[..., 0] *= gray_mean / b_mean
    if g_mean > 0:
        img_float[..., 1] *= gray_mean / g_mean
    if r_mean > 0:
        img_float[..., 2] *= gray_mean / r_mean
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)


def gamma_correction(img_bgr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_bgr, table)
