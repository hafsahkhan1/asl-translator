"""
Evaluate every trained classifier on a folder of labeled images.

Works on our recorded test set (test_data/<person>/<label>/*.jpg) or any
folder where each image's parent folder is its label (e.g. a held-out split
of the Kaggle dataset). Running it on both gives the dataset vs. real-world
accuracy gap for the report.

Usage:
    python evaluate.py --data test_data                     # everyone's recordings
    python evaluate.py --data test_data/jack --tag jack     # one person
    python evaluate.py --data path/to/kaggle_test_split --tag dataset --limit 200

Writes to results/<tag>/:
    summary.csv                  one row per model: accuracy, detection rate, speed
    <model>_report.txt           per-letter precision / recall / F1
    <model>_confusion.png        confusion matrix
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from classifiers import load_classifiers
from hand_detector import HandDetector

IMG_EXTS = (".jpg", ".jpeg", ".png")
NO_HAND = "no hand"
NOTHING = "nothing"     # no hand detected on a "nothing" image is the right answer


def find_images(root, limit):
    """Return [(path, label)], label = name of the image's parent folder."""
    by_label = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.lower().endswith(IMG_EXTS):
                by_label[os.path.basename(dirpath)].append(os.path.join(dirpath, fn))
    items = []
    for label in sorted(by_label):
        paths = by_label[label][:limit] if limit else by_label[label]
        items += [(p, label) for p in paths]
    return items


def save_confusion(y_true, y_pred, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    true_labels = sorted(set(y_true))
    labels = true_labels + ([NO_HAND] if NO_HAND in y_pred else [])
    extra = sorted(set(y_pred) - set(labels))       # predictions of classes not in the test set
    cols = labels + extra
    cm = confusion_matrix(y_true, y_pred, labels=cols)[:len(true_labels)]
    # row-normalize so each row shows where that letter's frames went
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.4), max(7, len(true_labels) * 0.38)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(true_labels)), true_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of that letter's frames")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test_data")
    ap.add_argument("--tag", default="", help="results subfolder name (default: data folder name)")
    ap.add_argument("--limit", type=int, default=0, help="max images per label (0 = all)")
    ap.add_argument("--conf", type=float, default=0.5, help="min hand detection confidence")
    ap.add_argument("--flip", action="store_true", help="mirror images before detecting")
    args = ap.parse_args()

    from sklearn.metrics import classification_report

    classifiers = load_classifiers()
    if not classifiers:
        raise SystemExit("No models found in models/. Train one first (see README).")

    items = find_images(args.data, args.limit)
    if not items:
        raise SystemExit(f"No images found under {args.data}")
    print(f"{len(items)} images, {len({l for _, l in items})} labels")

    tag = args.tag or os.path.basename(os.path.normpath(args.data))
    out_dir = os.path.join("results", tag)
    os.makedirs(out_dir, exist_ok=True)

    y_true, hand_found = [], []
    y_pred = {name: [] for name in classifiers}
    clf_time = {name: [] for name in classifiers}
    det_time, detected = [], 0

    with HandDetector(mode="image", min_detection_confidence=args.conf) as detector:
        for n, (path, label) in enumerate(items, 1):
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            if args.flip:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            hand = detector.detect(rgb)
            det_time.append(time.perf_counter() - t0)
            y_true.append(label)
            hand_found.append(hand is not None)
            detected += hand is not None

            for name, clf in classifiers.items():
                if hand is None:
                    y_pred[name].append(NOTHING if label == NOTHING else NO_HAND)
                    continue
                t0 = time.perf_counter()
                pred, _ = clf.predict(hand, rgb)
                clf_time[name].append(time.perf_counter() - t0)
                y_pred[name].append(pred or NO_HAND)

            if n % 200 == 0:
                print(f"  {n}/{len(items)}")

    det_ms = 1000 * float(np.mean(det_time))
    rows = []
    for name in classifiers:
        preds = y_pred[name]
        correct = [t == p for t, p in zip(y_true, preds)]
        det_mask = hand_found
        acc_all = float(np.mean(correct))
        acc_det = float(np.mean([c for c, d in zip(correct, det_mask) if d])) if detected else 0.0
        clf_ms = 1000 * float(np.mean(clf_time[name])) if clf_time[name] else 0.0
        rows.append({
            "model": name,
            "images": len(y_true),
            "detection_rate": round(detected / len(y_true), 4),
            "accuracy_all_images": round(acc_all, 4),
            "accuracy_when_detected": round(acc_det, 4),
            "detect_ms": round(det_ms, 2),
            "classify_ms": round(clf_ms, 2),
            "est_fps": round(1000 / (det_ms + clf_ms), 1),
        })

        report = classification_report(y_true, preds, zero_division=0)
        with open(os.path.join(out_dir, f"{name}_report.txt"), "w") as f:
            f.write(f"{name} on {args.data}\n\n{report}")
        save_confusion(y_true, preds, f"{name} on {tag} (accuracy {acc_all:.1%})",
                       os.path.join(out_dir, f"{name}_confusion.png"))

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nHand detected in {detected}/{len(y_true)} images ({detected / len(y_true):.0%})")
    for r in rows:
        print(f"{r['model']:>10}: {r['accuracy_all_images']:.1%} overall, "
              f"{r['accuracy_when_detected']:.1%} when a hand was found, "
              f"~{r['est_fps']} FPS")
    print(f"Results -> {out_dir}/")


if __name__ == "__main__":
    main()
