"""
Classical Emotion Recognition Baseline (Criterion B & Rubric Comparison).

Uses OpenCV Haar Cascade face detector + Histogram of Oriented Gradients (HOG) descriptor
with a scikit-learn Linear Classifier (Logistic Regression / Linear SVM).

Emotion Mapping:
    1: Happy     (Quadrant 1)
    2: Angry     (Quadrant 2)
    3: Surprise  (Quadrant 3)
    4: Neutral   (Quadrant 4)
"""

import cv2
import numpy as np
import time
import os
from typing import Tuple, Optional, List

EMOTION_LABELS = {
    1: "Happy",
    2: "Angry",
    3: "Surprise",
    4: "Neutral"
}


class ClassicalEmotionClassifier:
    def __init__(self, model_path: Optional[str] = None):
        # Locate Haar Cascade XML (custom asset fallback or system cv2 data)
        local_cascade = os.path.join(os.path.dirname(__file__), "..", "assets", "haarcascade_frontalface_default.xml")
        cv2_cascade = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        cascade_file = local_cascade if os.path.exists(local_cascade) else cv2_cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

        # Standard HOG Descriptor for 64x64 cropped face
        # winSize=(64,64), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9
        self.hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

        # Lightweight linear weights (calibrated default or trained)
        self.model = None
        if model_path and os.path.exists(model_path):
            import joblib
            self.model = joblib.load(model_path)

    def extract_hog_features(self, face_gray: np.ndarray) -> np.ndarray:
        """Resizes face crop to 64x64 and computes 1764-dimensional HOG vector."""
        resized = cv2.resize(face_gray, (64, 64))
        features = self.hog.compute(resized)
        return features.flatten()

    def predict(self, frame_bgr: np.ndarray) -> Tuple[Optional[int], str, float, Optional[Tuple[int, int, int, int]]]:
        """
        Detects face, computes HOG, and predicts emotion class.

        Returns:
            Tuple: (emotion_id, label, inference_time_ms, face_bbox_xywh)
        """
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return None, "No Face Detected", elapsed_ms, None

        # Pick largest detected face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        features = self.extract_hog_features(face_roi)

        if self.model is not None:
            pred_id = int(self.model.predict([features])[0])
        else:
            # Deterministic geometric / energy heuristic fallback if no pre-trained weights file
            # Computes mouth-to-upper-face intensity and gradient ratio
            upper_roi = face_roi[0:int(h * 0.5), :]
            lower_roi = face_roi[int(h * 0.5):, :]
            mouth_energy = np.std(lower_roi)
            upper_energy = np.std(upper_roi)
            ratio = mouth_energy / (upper_energy + 1e-5)

            if ratio > 1.35:
                pred_id = 3  # Surprise (wide open mouth)
            elif ratio > 1.10:
                pred_id = 1  # Happy (smile curvature)
            elif upper_energy > mouth_energy * 1.15:
                pred_id = 2  # Angry (furrowed brow)
            else:
                pred_id = 4  # Neutral

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        label = EMOTION_LABELS.get(pred_id, "Unknown")
        return pred_id, label, elapsed_ms, (x, y, w, h)

