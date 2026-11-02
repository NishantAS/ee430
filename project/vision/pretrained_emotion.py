"""
Pretrained Emotion Recognition (no training required).

YuNet face detector (confidence-thresholded, OpenCV Zoo) + a pretrained expression CNN run through cv2.dnn.
Two interchangeable backends:
  - "hse"    : HSEmotion EfficientNet-B0 (trained on AffectNet, in-the-wild faces), 8 classes. Default.
               https://github.com/av-savchenko/face-emotion-recognition
  - "ferplus": ONNX Model Zoo FER+ CNN (64x64 grayscale), 8 classes.
               https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus

The 8 classes are folded into the project's 4 quadrants (best class in each group, not the sum, so a group with more
members isn't favoured; Neutral is down-weighted because it is the "default" answer on dim/odd-angle faces):
    Happy    <- happiness
    Angry    <- anger, disgust
    Surprise <- surprise, fear
    Neutral  <- neutral, sadness, contempt
"""

import os
import time
import cv2
import numpy as np
from typing import Tuple, Optional

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
YUNET_PATH = os.path.join(ASSETS, "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

BACKENDS = {
    "hse": {
        "path": os.path.join(ASSETS, "emotion-hsemotion-b0.onnx"),
        "url": "https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
        "classes": ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
    },
    "ferplus": {
        "path": os.path.join(ASSETS, "emotion-ferplus-8.onnx"),
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "classes": ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"],
    },
}

# (quadrant_id, label, member classes)
GROUPS = [
    (1, "Happy", ["happiness"]),
    (2, "Angry", ["anger", "disgust"]),
    (3, "Surprise", ["surprise", "fear"]),
    (4, "Neutral", ["neutral", "sadness", "contempt"]),
]
NEUTRAL_WEIGHT = 0.7  # <1 lets a clear expression beat a lukewarm "neutral"

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class PretrainedEmotionClassifier:
    def __init__(self, backend: str = "hse", face_conf: float = 0.6):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {list(BACKENDS)}")
        self.backend = backend
        cfg = BACKENDS[backend]
        self.classes = cfg["classes"]
        for path, url in ((cfg["path"], cfg["url"]), (YUNET_PATH, YUNET_URL)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}. Download it from:\n  {url}")
        self.net = cv2.dnn.readNetFromONNX(cfg["path"])
        self.detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (640, 480), face_conf, 0.3, 50)

        self.last_probs = None   # 8-class probabilities of the last face (diagnostics)
        self.last_scores = None  # 4 quadrant scores of the last face, order Happy, Angry, Surprise, Neutral
        self.last_crop = None    # the exact image patch that was fed to the network (diagnostics)

        # First forward pass is slow (graph setup); pay that here instead of in the first trial
        self._forward(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), np.uint8))

    def _forward(self, bgr_crop: np.ndarray, gray_crop: np.ndarray) -> np.ndarray:
        if self.backend == "hse":
            rgb = cv2.cvtColor(cv2.resize(bgr_crop, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.net.setInput(((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32))
        else:
            g = cv2.resize(gray_crop, (64, 64)).astype(np.float32)  # raw 0-255, no normalisation
            self.net.setInput(g.reshape(1, 1, 64, 64))
        logits = self.net.forward().flatten()
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def predict(self, frame_bgr: np.ndarray) -> Tuple[Optional[int], str, float, Optional[Tuple[int, int, int, int]]]:
        """Returns (quadrant_id, label, latency_ms, bbox_xywh); quadrant_id is None if no face is found."""
        t0 = time.perf_counter()
        h_img, w_img = frame_bgr.shape[:2]
        self.detector.setInputSize((w_img, h_img))
        _, dets = self.detector.detect(frame_bgr)
        if dets is None or len(dets) == 0:
            return None, "No Face Detected", (time.perf_counter() - t0) * 1000.0, None

        x, y, w, h = (int(v) for v in max(dets, key=lambda d: d[2] * d[3])[:4])
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, w_img - x), min(h, h_img - y)
        if w < 40 or h < 40:
            return None, "No Face Detected", (time.perf_counter() - t0) * 1000.0, None

        pad = int(0.1 * w)  # a little context around the box, as the AffectNet crops have
        crop = frame_bgr[max(y - pad, 0):min(y + h + pad, h_img), max(x - pad, 0):min(x + w + pad, w_img)]
        if self.backend == "ferplus":
            crop = frame_bgr[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self.last_crop = crop
        self.last_probs = self._forward(crop, gray)

        p = dict(zip(self.classes, self.last_probs))
        self.last_scores = np.array([max(p[c] for c in g[2]) for g in GROUPS], dtype=np.float64)
        self.last_scores[3] *= NEUTRAL_WEIGHT
        best = int(np.argmax(self.last_scores))
        return GROUPS[best][0], GROUPS[best][1], (time.perf_counter() - t0) * 1000.0, (x, y, w, h)
