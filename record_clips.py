"""
Record our own real-world test set from the webcam.

For each sign it shows a prompt; press SPACE, get a 3-second countdown, then it
saves a burst of frames to test_data/<person>/<label>/.

Usage:
    python record_clips.py --person jack
    python record_clips.py --person hafsah --labels A,B,M,N,S,T   # redo specific signs

Keys: SPACE = record   S = skip   Q = quit
Tip: record in a few different spots (lighting, background) to get a fair test.
"""

import argparse
import os
import string
import time

import cv2

from camera import FrameReader, open_camera

DEFAULT_LABELS = list(string.ascii_uppercase) + ["space", "del", "nothing"]


def put_text(img, text, y, scale=0.9, color=(255, 255, 255)):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="your name, used as the folder name")
    ap.add_argument("--labels", default="", help="comma-separated subset, e.g. A,B,C")
    ap.add_argument("--frames", type=int, default=30, help="frames saved per sign")
    ap.add_argument("--every", type=int, default=2, help="save every Nth frame (adds variety)")
    ap.add_argument("--out", default="test_data")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()] or DEFAULT_LABELS
    cap = open_camera(args.camera)
    reader = FrameReader(cap)

    i = 0
    while i < len(labels):
        label = labels[i]
        folder = os.path.join(args.out, args.person, label)
        existing = len(os.listdir(folder)) if os.path.isdir(folder) else 0

        # Wait for SPACE / S / Q
        while True:
            frame = reader.read()
            if frame is None:
                continue
            view = cv2.flip(frame, 1)
            put_text(view, f"Sign: {label}   ({i + 1}/{len(labels)})", 40, 1.1, (177, 199, 145))
            put_text(view, "SPACE record   S skip   Q quit", 80, 0.7)
            if existing:
                put_text(view, f"{existing} frames already saved (new ones are added)", 115, 0.6)
            cv2.imshow("Record test clips", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord("s"):
                break
            if key == ord(" "):
                record(reader, label, folder, args, existing)
                break
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Frames are in {os.path.join(args.out, args.person)}/")


def record(reader, label, folder, args, start_index):
    os.makedirs(folder, exist_ok=True)

    # 3-second countdown so you can get your hand in position
    end = time.time() + 3
    while time.time() < end:
        frame = reader.read()
        if frame is None:
            continue
        view = cv2.flip(frame, 1)
        put_text(view, f"{label}: starting in {int(end - time.time()) + 1}", 40, 1.1, (0, 220, 255))
        cv2.imshow("Record test clips", view)
        cv2.waitKey(1)

    saved, n = 0, 0
    while saved < args.frames:
        frame = reader.read()
        if frame is None:
            continue
        if n % args.every == 0:
            # save the raw (unmirrored) frame, same as what the app's detector sees
            name = f"{args.person}_{label}_{start_index + saved:04d}.jpg"
            cv2.imwrite(os.path.join(folder, name), frame)
            saved += 1
        n += 1
        view = cv2.flip(frame, 1)
        put_text(view, f"Recording {label}: {saved}/{args.frames}", 40, 1.1, (80, 80, 255))
        cv2.imshow("Record test clips", view)
        cv2.waitKey(1)
    print(f"{label}: saved {saved} frames -> {folder}")


if __name__ == "__main__":
    main()
