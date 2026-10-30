"""
Lab Dataset Collector for Facial Expression Recognition (Rubric Component E).

Captures lab faces across group members to build:
- 120 training images (30 per emotion * 4 emotions)
- 80 held-out test images (20 per emotion * 4 emotions)

Saves images to project/dl/dataset/{train,test}/{happy,angry,surprise,neutral}/
"""

import os
import sys
import time
import argparse
import numpy as np

EMOTIONS = ["happy", "angry", "surprise", "neutral"]


def collect_dataset(video_source=0, target_count_per_class=50, output_dir="project/dl/dataset"):
    try:
        import cv2
    except ImportError:
        print("OpenCV not ready yet. Please run once virtualenv packages finish downloading.")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_source}")
        return

    local_cascade = os.path.join(os.path.dirname(__file__), "..", "assets", "haarcascade_frontalface_default.xml")
    cv2_cascade = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    cascade_file = local_cascade if os.path.exists(local_cascade) else cv2_cascade
    face_cascade = cv2.CascadeClassifier(cascade_file)

    print("==========================================================================")
    print(" EE430 Component E: Lab Face Dataset Collector")
    print("==========================================================================")
    print(f"Target count: {target_count_per_class} per class (Total: {target_count_per_class * 4})")
    print("Press SPACE to capture face frame for the active emotion.")
    print("Press 'n' to advance to the next emotion. Press 'q' to quit.")

    for emotion in EMOTIONS:
        train_path = os.path.join(output_dir, "train", emotion)
        test_path = os.path.join(output_dir, "test", emotion)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        count = 0
        train_quota = int(target_count_per_class * 0.6)  # 60% train, 40% test

        print(f"\n>>> ACTIVE EMOTION: [{emotion.upper()}] (Target: {target_count_per_class} frames)")

        while count < target_count_per_class:
            ret, frame = cap.read()
            if not ret:
                break

            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

            face_crop = None
            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                x, y, w, h = faces[0]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_crop = frame[y:y + h, x:x + w]

            cv2.putText(vis, f"Emotion: {emotion.upper()} | Captured: {count}/{target_count_per_class}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(vis, "SPACE: Capture | N: Next | Q: Quit", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("EE430 Dataset Collector", vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
            elif key == 32:  # SPACE
                if face_crop is not None:
                    target_dir = train_path if count < train_quota else test_path
                    img_name = f"{emotion}_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(os.path.join(target_dir, img_name), face_crop)
                    count += 1
                    print(f"Captured {emotion} [{count}/{target_count_per_class}] -> {target_dir}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDataset collection complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    collect_dataset(args.source, args.count)

