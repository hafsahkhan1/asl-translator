"""
Run MediaPipe over the Kaggle ASL Alphabet dataset and save hand landmarks
to a CSV for training the landmark classifier. Optionally also saves hand
crops for training the CNN, cropped the same way the live app crops them.

Usage:
    python extract_landmarks.py --data path/to/asl_alphabet_train/asl_alphabet_train --limit 100
    python extract_landmarks.py --data path/to/asl_alphabet_train/asl_alphabet_train --save-crops crops

Output CSV columns:
    label, handedness, file,
    x0, y0, z0, ..., x20, y20, z20       raw MediaPipe coordinates
    nx0, ny0, nz0, ..., nx20, ny20, nz20  normalized (train on these)
Images where no hand is found are skipped and listed in <out>_missed.txt.
"""

import argparse
import csv
import os
import time

import cv2

from hand_detector import HandDetector, crop_hand

IMG_EXTS = (".jpg", ".jpeg", ".png")


def csv_header():
    header = ["label", "handedness", "file"]
    header += [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]
    header += [f"n{a}{i}" for i in range(21) for a in ("x", "y", "z")]
    return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing A/, B/, ... class folders")
    ap.add_argument("--out", default="asl_landmarks.csv")
    ap.add_argument("--limit", type=int, default=0, help="max images per class (0 = all)")
    ap.add_argument("--conf", type=float, default=0.3, help="min hand detection confidence")
    ap.add_argument("--save-crops", default="", help="folder to save hand crops for CNN training")
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(args.data)
                     if os.path.isdir(os.path.join(args.data, d)))
    print(f"Found {len(classes)} classes: {' '.join(classes)}")

    missed, total_hits, total = [], 0, 0
    start = time.time()

    with HandDetector(mode="image", min_detection_confidence=args.conf) as detector, \
            open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header())

        for label in classes:
            folder = os.path.join(args.data, label)
            files = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith(IMG_EXTS))
            if args.limit:
                files = files[:args.limit]
            if args.save_crops:
                os.makedirs(os.path.join(args.save_crops, label), exist_ok=True)
            hits = 0

            for fn in files:
                path = os.path.join(folder, fn)
                bgr = cv2.imread(path)
                if bgr is None:
                    missed.append(path)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                hand = detector.detect(rgb)
                if hand is None:
                    missed.append(path)
                    continue

                row = [label, hand.handedness, fn]
                row += [round(v, 6) for pt in hand.landmarks for v in pt]
                row += [round(v, 6) for v in hand.features()]
                writer.writerow(row)

                if args.save_crops:
                    crop = crop_hand(bgr, hand)
                    if crop.size:
                        cv2.imwrite(os.path.join(args.save_crops, label, fn), crop)
                hits += 1

            total_hits += hits
            total += len(files)
            print(f"{label:>8}: {hits}/{len(files)} detected "
                  f"({100 * hits / max(len(files), 1):.0f}%)  [{time.time() - start:.0f}s]")

    missed_path = os.path.splitext(args.out)[0] + "_missed.txt"
    with open(missed_path, "w") as f:
        f.write("\n".join(missed))

    print(f"\nDone: {total_hits}/{total} images with a hand -> {args.out}")
    print(f"Missed list -> {missed_path}")


if __name__ == "__main__":
    main()
