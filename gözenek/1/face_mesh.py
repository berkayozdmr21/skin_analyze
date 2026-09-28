"""Face mesh detection, ten bolgesi maskeleri, ve goz/kas/dudak disleme
(exclusion) maskesi.

Akne projesinden BIREBIR AYNI kopya - FaceMeshDetector ve bolge
maskeleri (forehead/nose/chin/cheeks) goz alti morluk ailesindeki
mantikla ayni. make_face_scan_mask() (tum ten, goz/kas/dudak haric)
gozenek tespiti icin de dogru hedef bolge: gozenekler de akne gibi
yuzun HERHANGI bir yerinde olabilir, dudak/goz/kas ise gozenek
SAYILMAMASI gereken bolgeler.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

# Goz (tam kontur, sadece alt kapak degil - disleme icin daha genis olmali)
RIGHT_EYE_FULL = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
LEFT_EYE_FULL = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

# Kaslar
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# Dudaklar (dis kontur)
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
              375, 321, 405, 314, 17, 84, 181, 91, 146]

EXCLUSION_GROUPS = (RIGHT_EYE_FULL, LEFT_EYE_FULL, RIGHT_EYEBROW, LEFT_EYEBROW, LIPS_OUTER)


class FaceMeshDetector:
    """MediaPipe face mesh detector."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = img_bgr.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])

        return np.array(landmarks, dtype=np.int32)

    def __del__(self):
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()


def make_region_masks(landmarks: np.ndarray, img_shape: tuple) -> dict[str, np.ndarray]:
    """forehead/nose/chin/cheeks bolge maskeleri - goz alti morluk ailesiyle ayni mantik."""
    h, w = img_shape[:2]
    masks = {}

    forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    nose_indices = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12,
                    13, 14, 15, 16, 17, 18, 200, 199, 175, 152]

    def create_mask_from_indices(indices, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if len(indices) == 0:
            return mask
        valid_indices = [i for i in indices if i < len(landmarks)]
        if len(valid_indices) < 3:
            return mask
        points = landmarks[valid_indices]
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    masks["forehead"] = create_mask_from_indices(forehead_indices[:20], (h, w))
    masks["nose"] = create_mask_from_indices(nose_indices, (h, w))
    masks["chin"] = create_mask_from_indices([152, 175, 200, 201, 18, 421, 406, 335, 273], (h, w))

    # Yanaklar: yuzu ortadan ikiye bolup her yarinin convex hull'u.
    # Goz alti morluk.1'in orijinal make_cheeks_mask()'i ile ayni
    # dayaniklilik: TEK bir taraf bile yeterli piksele sahipse (ornegin
    # yuz karede tam ortalanmamissa) "cheeks" yine de olusturuluyor -
    # her iki tarafin da bulunmasini SART kosmuyor.
    cheeks_mask = np.zeros((h, w), dtype=np.uint8)
    found_cheek = False

    left_side = landmarks[landmarks[:, 0] < w // 2]
    if len(left_side) > 3:
        hull = cv2.convexHull(left_side)
        cv2.fillConvexPoly(cheeks_mask, hull, 255)
        found_cheek = True

    right_side = landmarks[landmarks[:, 0] >= w // 2]
    if len(right_side) > 3:
        hull = cv2.convexHull(right_side)
        cv2.fillConvexPoly(cheeks_mask, hull, 255)
        found_cheek = True

    if found_cheek:
        masks["cheeks"] = cheeks_mask

    return masks


def make_skin_tone_mask(masks: dict[str, np.ndarray], img_shape: tuple) -> Optional[np.ndarray]:
    """Tum duz ten bolgelerinin (alin+burun+cene+yanak) birlesimi."""
    h, w = img_shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    found = False
    for region in ("forehead", "nose", "chin", "cheeks"):
        if region in masks:
            combined = cv2.bitwise_or(combined, masks[region])
            found = True
    return combined if found else None


def make_exclusion_mask(landmarks: np.ndarray, img_shape: tuple, dilate_px: int = 9) -> np.ndarray:
    """
    Goz + kas + dudak bolgelerinin birlesimi, ARANMAYACAK alanlar olarak
    isaretlenir. Kenarlarda pay birakmak icin hafifce genisletilir
    (dilate) - ornegin dudak kenarindaki dogal golge/renk gecisi yanlislikla
    "gozenek/akne" sayilmasin diye.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for indices in EXCLUSION_GROUPS:
        valid = [i for i in indices if i < len(landmarks)]
        if len(valid) < 3:
            continue
        points = landmarks[valid]
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def make_face_scan_mask(landmarks: np.ndarray, img_shape: tuple) -> Optional[np.ndarray]:
    """
    Akne taramasinin yapilacagi nihai bolge: TUM ten (alin+burun+cene+yanak)
    EKSI goz/kas/dudak dislama alani. Goz alti morluk projelerindeki dar
    "bant" yerine, akne anatomik olarak sabit bir yerde olmadigi icin
    TUM taranabilir yuz yuzeyi kullaniliyor.
    """
    region_masks = make_region_masks(landmarks, img_shape)
    skin_mask = make_skin_tone_mask(region_masks, img_shape)
    if skin_mask is None:
        return None

    exclusion = make_exclusion_mask(landmarks, img_shape)
    scan_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(exclusion))
    return scan_mask


def make_cheeks_mask(landmarks: np.ndarray, img_shape: tuple) -> Optional[np.ndarray]:
    """Referans bolgesi: yanaklar - goz alti morluk.1 ile birebir ayni mantik."""
    region_masks = make_region_masks(landmarks, img_shape)
    return region_masks.get("cheeks")
