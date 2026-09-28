"""Face mesh detection, TUM TEN (skin tone) bolge maskeleri, ve goz alti bandi.

goz alti morluk.1'den farkli: artik sadece yanak degil, yuzun TUM duz
ten bolgeleri (forehead+nose+chin+cheeks) referans olarak kullanilabiliyor
- make_skin_tone_mask() bu birlesimi doner.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LEFT_EYE_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263]


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
    """Ana skin-scan projesindeki ayni bolge maskeleri (forehead/nose/chin/cheeks)."""
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

    left_side = landmarks[landmarks[:, 0] < w // 2]
    if len(left_side) > 3:
        hull = cv2.convexHull(left_side)
        masks["left_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["left_cheek"], hull, 255)

    right_side = landmarks[landmarks[:, 0] >= w // 2]
    if len(right_side) > 3:
        hull = cv2.convexHull(right_side)
        masks["right_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["right_cheek"], hull, 255)

    if "left_cheek" in masks and "right_cheek" in masks:
        masks["cheeks"] = cv2.bitwise_or(masks["left_cheek"], masks["right_cheek"])
        del masks["left_cheek"]
        del masks["right_cheek"]

    return masks


def make_skin_tone_mask(masks: dict[str, np.ndarray], img_shape: tuple) -> Optional[np.ndarray]:
    """
    TUM duz ten bolgelerinin (alin+burun+cene+yanak) birlesimi - kisinin
    GENEL ten rengi referansi. goz alti morluk.1'deki gibi sadece yanak
    degil, yuzun butunu kullanilir - boylece yanaktaki lokal bir
    golge/kizariklik tek basina referansi yanlis yonlendiremez.
    """
    h, w = img_shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    found = False
    for region in ("forehead", "nose", "chin", "cheeks"):
        if region in masks:
            combined = cv2.bitwise_or(combined, masks[region])
            found = True
    return combined if found else None


def make_under_eye_mask(landmarks: np.ndarray, img_shape: tuple) -> Optional[np.ndarray]:
    """Goz alti bandi - goz alti morluk.1 ile ayni mantik."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    found_any = False

    for eye_indices in (RIGHT_EYE_LOWER, LEFT_EYE_LOWER):
        valid = [i for i in eye_indices if i < len(landmarks)]
        if len(valid) < 3:
            continue

        pts = landmarks[valid].astype(np.float64)
        eye_width = np.linalg.norm(pts[0] - pts[-1])
        if eye_width < 1:
            continue

        offset = eye_width * 0.5
        lower_edge = pts + np.array([0, offset])

        band_points = np.vstack([pts, lower_edge[::-1]]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(band_points), 255)
        found_any = True

    return mask if found_any else None
