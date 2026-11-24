"""
Hand detection with MediaPipe Hand Landmarker.

Shared by the live app, the dataset landmark extractor, and the evaluation
script, so every part of the project detects, crops and normalizes hands
exactly the same way.
"""

import math
import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Pairs of landmark indices to draw the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]


@dataclass
class Hand:
    landmarks: list      # 21 (x, y, z) tuples, x/y in 0-1 image coordinates
    normalized: list     # 21 (x, y, z) tuples, wrist-relative and scale-invariant
    bbox: tuple          # (x1, y1, x2, y2) in pixels, padded and square
    handedness: str      # "Left" or "Right" as MediaPipe reports it
    score: float         # handedness confidence

    def features(self):
        """Flat list of 63 normalized values, the input to the landmark classifier."""
        return [v for pt in self.normalized for v in pt]


def get_model_path():
    """Download the MediaPipe hand model the first time it's needed."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task (one time) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def normalize(lms):
    """Move the wrist to the origin and scale by hand size, so the hand's
    position and distance from the camera don't affect the features."""
    wx, wy, wz = lms[0]
    shifted = [(x - wx, y - wy, z - wz) for x, y, z in lms]
    # hand size = distance from wrist (0) to middle-finger knuckle (9)
    scale = math.dist((0, 0), shifted[9][:2]) or 1.0
    return [(x / scale, y / scale, z / scale) for x, y, z in shifted]


def landmarks_to_bbox(lms, width, height, pad=0.25):
    """Square pixel bounding box around the landmarks with padding, clamped to the image."""
    xs = [p[0] * width for p in lms]
    ys = [p[1] * height for p in lms]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * pad)
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(width, cx + side / 2))
    y2 = int(min(height, cy + side / 2))
    return x1, y1, x2, y2


def crop_hand(image, hand):
    """Crop the hand from an image (numpy array) using its bounding box."""
    x1, y1, x2, y2 = hand.bbox
    return image[y1:y2, x1:x2]


class HandDetector:
    """Wraps MediaPipe's HandLandmarker.

    mode="video" for webcam frames (uses tracking between frames, faster)
    mode="image" for independent photos (dataset images, recorded test frames)
    """

    def __init__(self, mode="video", min_detection_confidence=0.5):
        self.mode = mode
        running_mode = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path()),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb_image):
        """Detect one hand in an RGB numpy image. Returns a Hand or None."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        if self.mode == "video":
            ts = int(time.monotonic() * 1000)
            ts = max(ts, self._last_ts + 1)   # timestamps must strictly increase
            self._last_ts = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)
        return self._to_hand(result, rgb_image.shape[1], rgb_image.shape[0])

    @staticmethod
    def _to_hand(result, width, height):
        if not result.hand_landmarks:
            return None
        lms = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
        cat = result.handedness[0][0]
        return Hand(
            landmarks=lms,
            normalized=normalize(lms),
            bbox=landmarks_to_bbox(lms, width, height),
            handedness=cat.category_name,
            score=cat.score,
        )

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_hand(image, hand, color=(145, 199, 177)):
    """Draw the skeleton and bounding box on an image in place."""
    h, w = image.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(image, p, 4, (255, 255, 255), -1)
        cv2.circle(image, p, 4, color, 1)
    x1, y1, x2, y2 = hand.bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image
