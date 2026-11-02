"""
Deep Learning vs Classical Pipeline Training, Evaluation, and Comparison (Criterion E).

Trains both on the lab dataset produced by dl/dataset_collector.py
(dataset/{train,test}/{happy,angry,surprise,neutral}/*.jpg face crops):
1. Classical Baseline: HOG (1764-d) + Linear SVM       -> dl/classical_hog_svm.joblib
2. Deep Learning:      MobileNetV2 (ImageNet backbone, trained 4-class head) -> dl/emotion_mobilenet.pth

Both are evaluated on the held-out test split; accuracy, latency and FPS are MEASURED here and written to
results/dl_vs_classical_metrics.csv. Nothing is hardcoded: with no dataset the script stops with an error.

Run from anywhere:  python project/dl/train_eval.py [--epochs 20]
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, "..")))

EMOTIONS = ["happy", "angry", "surprise", "neutral"]  # index 0..3 == DL class id, classical label = index + 1
DATASET_DIR = os.path.join(HERE, "dataset")
CLASSICAL_MODEL = os.path.join(HERE, "classical_hog_svm.joblib")
DL_WEIGHTS = os.path.join(HERE, "emotion_mobilenet.pth")
METRICS_CSV = os.path.abspath(os.path.join(HERE, "..", "results", "dl_vs_classical_metrics.csv"))


def load_split(split_dir):
    images, labels = [], []
    for idx, emo in enumerate(EMOTIONS):
        folder = os.path.join(split_dir, emo)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, name))
            if img is not None:
                images.append(img)
                labels.append(idx)
    return images, np.array(labels)


def train_classical(train_imgs, y_train, test_imgs, y_test):
    import joblib
    from sklearn.svm import LinearSVC
    from vision.classical_emotion import ClassicalEmotionClassifier

    feat = ClassicalEmotionClassifier(model_path=None, require_model=False)

    def feats(imgs):
        return np.array([feat.extract_hog_features(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)) for i in imgs])

    x_train = feats(train_imgs)
    # horizontal flips double the small training set
    x_train = np.vstack([x_train, feats([cv2.flip(i, 1) for i in train_imgs])])
    y_aug = np.concatenate([y_train, y_train])
    clf = LinearSVC(C=0.1, max_iter=5000).fit(x_train, y_aug)
    joblib.dump(clf, CLASSICAL_MODEL)

    acc = float((clf.predict(feats(test_imgs)) == y_test).mean())
    t0 = time.perf_counter()
    for i in test_imgs:
        clf.predict([feat.extract_hog_features(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))])
    lat = (time.perf_counter() - t0) * 1000.0 / len(test_imgs)
    return acc, lat, int(clf.coef_.size + clf.intercept_.size)


def train_dl(train_imgs, y_train, test_imgs, y_test, epochs):
    import torch
    import torch.nn as nn
    from dl.dl_emotion_infer import DLEmotionClassifier

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DLEmotionClassifier(weights_path=None, device=str(dev), require_weights=False)
    model, tf = net.model, net.transform

    for p in model.features.parameters():  # transfer learning: freeze backbone, train the head
        p.requires_grad = False
    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def tensor(img, flip=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return tf(cv2.flip(img, 1) if flip else img)

    rng = np.random.default_rng(0)
    for ep in range(epochs):
        model.train()
        model.features.eval()  # keep BN statistics of the frozen backbone
        order = rng.permutation(len(train_imgs))
        total = 0.0
        for s in range(0, len(order), 16):
            idx = order[s:s + 16]
            xb = torch.stack([tensor(train_imgs[i], flip=bool(rng.integers(2))) for i in idx]).to(dev)
            yb = torch.tensor(y_train[idx]).to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {ep + 1:02d}/{epochs}  train loss {total / len(order):.4f}")

    model.eval()
    torch.save(model.state_dict(), DL_WEIGHTS)

    preds, lats = [], []
    with torch.no_grad():
        for img in test_imgs:
            t0 = time.perf_counter()
            out = model(tensor(img).unsqueeze(0).to(dev))
            preds.append(int(out.argmax(1)))
            lats.append((time.perf_counter() - t0) * 1000.0)
    acc = float((np.array(preds) == y_test).mean())
    return acc, float(np.mean(lats)), sum(p.numel() for p in model.parameters()), str(dev)


def train_and_evaluate(epochs=20):
    train_imgs, y_train = load_split(os.path.join(DATASET_DIR, "train"))
    test_imgs, y_test = load_split(os.path.join(DATASET_DIR, "test"))
    if len(train_imgs) == 0 or len(test_imgs) == 0:
        sys.exit(f"No dataset found in {DATASET_DIR}. Collect one first:\n"
                 f"  python project/dl/dataset_collector.py --count 50")

    print(f"Train: {len(train_imgs)} images, Test: {len(test_imgs)} images")
    print("Training classical HOG + LinearSVC ...")
    c_acc, c_lat, c_params = train_classical(train_imgs, y_train, test_imgs, y_test)
    print(f"  test accuracy {c_acc * 100:.1f}%  latency {c_lat:.2f} ms")
    print("Training MobileNetV2 head ...")
    d_acc, d_lat, d_params, dev = train_dl(train_imgs, y_train, test_imgs, y_test, epochs)
    print(f"  test accuracy {d_acc * 100:.1f}%  latency {d_lat:.2f} ms ({dev})")

    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    rows = [
        {"Pipeline": "Classical (HOG + LinearSVC)", "Parameters": c_params, "Test_Images": len(test_imgs),
         "Test_Accuracy_Pct": round(c_acc * 100, 1), "Latency_ms_per_face": round(c_lat, 2),
         "FPS": round(1000.0 / c_lat, 1), "Device": "cpu"},
        {"Pipeline": "Deep Learning (MobileNetV2 head)", "Parameters": d_params, "Test_Images": len(test_imgs),
         "Test_Accuracy_Pct": round(d_acc * 100, 1), "Latency_ms_per_face": round(d_lat, 2),
         "FPS": round(1000.0 / d_lat, 1), "Device": dev},
    ]
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Measured metrics written to {METRICS_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    train_and_evaluate(ap.parse_args().epochs)
