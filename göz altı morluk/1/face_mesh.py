"""Face mesh detection, cheek region mask, and under-eye band mask.

FaceMeshDetector ve cheeks maskesi ana skin-scan projesinden kopyalandi.
make_under_eye_mask() de ana projedeki mantikla birebir ayni: alt goz
kapagi landmark zincirini alip, goz genisliginin yarisi kadar asagi
kaydirarak bir "bant" (band) olusturuyor.
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


def make_cheeks_mask(landmarks: np.ndarray, img_shape: tuple) -> Optional[np.ndarray]:
    """Referans bolgesi: yanaklar (ana projedeki ayni mantik - yuz yarilarindan hull)."""
    h, w = img_shape[:2]

    left_side = landmarks[landmarks[:, 0] < w // 2]
    right_side = landmarks[landmarks[:, 0] >= w // 2]

    mask = np.zeros((h, w), dtype=np.uint8)
    found = False

    if len(left_side) > 3:
        hull = cv2.convexHull(left_side)
        cv2.fillConvexPoly(mask, hull, 255)
        found = True

    if len(right_side) > 3:
        hull = cv2.convexHull(right_side)
        cv2.fillConvexPoly(mask, hull, 255)
        found = True

    return mask if found else None


def make_under_eye_mask(landmarks: np.ndarray, img_shape: tuple) -> Optional[np.ndarray]:
    """
    Goz alti bandi: alt goz kapagi landmark zincirinin (ust kenar) ve
    ayni noktalarin goz genisliginin yarisi kadar asagi kaydirilmis
    halinin (alt kenar) arasindaki serit.
    """
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
