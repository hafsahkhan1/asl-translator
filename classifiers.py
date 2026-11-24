"""
Loads the trained letter classifiers and gives them one common interface:
    label, confidence = classifier.predict(hand, rgb_frame)

Expected model files (see README, "Model export format"):
    models/landmark_model.joblib   scikit-learn model trained on the 63 normalized
                                   landmark features (nx0..nz20), with predict_proba
    models/cnn_model.pt            TorchScript CNN taking a 1x3x224x224 hand crop
    models/cnn_labels.txt          class names, one per line, in the CNN's output order

Whichever files exist get loaded; missing ones are skipped.
"""

import os

import cv2
import numpy as np

from hand_detector import crop_hand

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LANDMARK_MODEL = os.path.join(MODELS_DIR, "landmark_model.joblib")
CNN_MODEL = os.path.join(MODELS_DIR, "cnn_model.pt")
CNN_LABELS = os.path.join(MODELS_DIR, "cnn_labels.txt")

CNN_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LandmarkClassifier:
    name = "Landmarks"

    def __init__(self, path=LANDMARK_MODEL):
        import joblib
        self.model = joblib.load(path)
        # parallel prediction (n_jobs=-1) is much slower for one sample at a time
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

    def predict(self, hand, rgb_frame=None):
        feats = np.asarray(hand.features(), dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(feats)[0]
        i = int(np.argmax(proba))
        return str(self.model.classes_[i]), float(proba[i])


class CNNClassifier:
    name = "CNN"

    def __init__(self, path=CNN_MODEL, labels_path=CNN_LABELS):
        import torch
        self.torch = torch
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()
        with open(labels_path) as f:
            self.labels = [line.strip() for line in f if line.strip()]

    @staticmethod
    def preprocess(crop_rgb):
        img = cv2.resize(crop_rgb, (CNN_SIZE, CNN_SIZE)).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(img, (2, 0, 1))[None]          # 1x3xHxW

    def predict(self, hand, rgb_frame):
        crop = crop_hand(rgb_frame, hand)
        if crop.size == 0:
            return None, 0.0
        x = self.torch.from_numpy(self.preprocess(crop))
        with self.torch.no_grad():
            proba = self.torch.softmax(self.model(x), dim=1)[0].numpy()
        i = int(np.argmax(proba))
        return self.labels[i], float(proba[i])


def load_classifiers(verbose=True):
    """Return {name: classifier} for every model file that exists and loads."""
    found = {}
    for cls, needed in ((LandmarkClassifier, [LANDMARK_MODEL]),
                        (CNNClassifier, [CNN_MODEL, CNN_LABELS])):
        if not all(os.path.exists(p) for p in needed):
            if verbose:
                print(f"[{cls.name}] not found ({', '.join(os.path.basename(p) for p in needed)})")
            continue
        try:
            found[cls.name] = cls(*needed)
            if verbose:
                print(f"[{cls.name}] loaded")
        except Exception as e:
            print(f"[{cls.name}] failed to load: {e}")
    return found
