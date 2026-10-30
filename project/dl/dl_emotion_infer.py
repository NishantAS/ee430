"""
Deep Learning Emotion Inference Wrapper (Criterion E Bonus).

Provides real-time GPU/CPU emotion inference using a lightweight CNN (MobileNetV2 / ResNet-18)
fine-tuned or pretrained for 4 classes:
    0: Happy    -> Target Quadrant 1
    1: Angry    -> Target Quadrant 2
    2: Surprise -> Target Quadrant 3
    3: Neutral  -> Target Quadrant 4
"""

import time
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Tuple, Optional

EMOTION_MAP = {
    0: (1, "Happy"),
    1: (2, "Angry"),
    2: (3, "Surprise"),
    3: (4, "Neutral")
}


class DLEmotionClassifier:
    def __init__(self, weights_path: Optional[str] = "project/dl/emotion_mobilenet.pth", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        local_cascade = os.path.join(os.path.dirname(__file__), "..", "assets", "haarcascade_frontalface_default.xml")
        cv2_cascade = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        cascade_file = local_cascade if os.path.exists(local_cascade) else cv2_cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

        # Build lightweight MobileNetV2 architecture with 4 output classes
        try:
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except Exception:
            # Fallback to unweighted architecture when offline
            self.model = models.mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 4)

        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded fine-tuned DL weights from: {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path} ({e}). Using initialized backbone.")

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing pipeline matching ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, frame_bgr: np.ndarray) -> Tuple[Optional[int], str, float, Optional[Tuple[int, int, int, int]]]:
        """
        Runs face detection and deep learning inference on frame.

        Returns:
            Tuple: (quadrant_id (1..4), label, latency_ms, bbox_xywh)
        """
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return None, "No Face Detected", elapsed_ms, None

        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]
        face_roi = frame_bgr[y:y + h, x:x + w]
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        tensor_img = self.transform(rgb_roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            class_idx = int(np.argmax(probs))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        quadrant_id, label = EMOTION_MAP[class_idx]
        return quadrant_id, label, elapsed_ms, (x, y, w, h)

