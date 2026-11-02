"""
Live emotion diagnostic: shows exactly what the robot camera feeds the emotion models.

    python project/test_emotion.py --cam 2

Shows the face box, the crop each model actually sees, and every raw class probability for BOTH backends
(HSEmotion and FER+) plus the folded 4-quadrant decision. Press s to save the frame + crop to
project/results/emotion_debug/, q to quit. Use it to check lighting / camera angle / which model reads you better.
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from vision.pretrained_emotion import PretrainedEmotionClassifier, GROUPS

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0)
args = ap.parse_args()

models = {name: PretrainedEmotionClassifier(name) for name in ("hse", "ferplus")}
cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
out_dir = os.path.join(os.path.dirname(__file__), "results", "emotion_debug")

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    vis = frame.copy()
    panel = np.full((frame.shape[0], 330, 3), 30, np.uint8)
    y0 = 22
    for name, m in models.items():
        qid, label, ms, bbox = m.predict(frame)
        cv2.putText(panel, f"{name.upper()}: {label} ({ms:.0f} ms)", (8, y0), 0, 0.55, (0, 255, 255), 1)
        y0 += 20
        if bbox is not None:
            if name == "hse":
                x, y, w, h = bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for cls, p in zip(m.classes, m.last_probs):
                cv2.rectangle(panel, (110, y0 - 9), (110 + int(p * 200), y0 + 2), (0, 200, 0), -1)
                cv2.putText(panel, f"{cls[:9]} {p:.2f}", (4, y0), 0, 0.4, (255, 255, 255), 1)
                y0 += 14
            thumb = cv2.resize(m.last_crop, (64, 64))
            panel[y0:y0 + 64, 8:72] = thumb
            cv2.putText(panel, "model input", (80, y0 + 36), 0, 0.4, (180, 180, 180), 1)
            y0 += 72
        y0 += 6
    cv2.imshow("emotion diagnostic (s=save, q=quit)", np.hstack([vis, panel]))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        os.makedirs(out_dir, exist_ok=True)
        stamp = int(time.time())
        cv2.imwrite(os.path.join(out_dir, f"{stamp}_frame.jpg"), frame)
        print("saved", stamp)
cap.release()
cv2.destroyAllWindows()
