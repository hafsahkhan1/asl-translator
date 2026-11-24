# ASL Translator
Final project for CS 415 Computer Vision I at UIC, Fall 2026

Live ASL fingerspelling translator: webcam → MediaPipe hand detection → letter classifier → smoothing → text.

## Running the application:
### 1. Download and activate virtual environment:
```
python -m venv .venv
```
Windows: 
```
.\.venv\Scripts\activate
```
macOS/Linux: 
```
source .venv/bin/activate
```

### 2. Download project requirements:
```
pip install -r requirements.txt
```
If you installed the old requirements before, remove the old OpenCV first: `pip uninstall opencv-python`.

### 3. Run Project Application:
```
python app.py
```
Click **Begin Translator**. The app draws the detected hand (skeleton + bounding box), shows the current prediction, and types a letter once it holds steady for ~7 of the last 10 frames. To type the same letter twice, drop your hand or change signs in between. The "space" and "del" signs add a space and delete. Pick the model (Landmarks / CNN) from the dropdown.

Without a trained model in `models/`, the app runs detection only.

## Files
| File | What it does |
| --- | --- |
| `app.py` | Live Tkinter app |
| `hand_detector.py` | MediaPipe hand detection, landmark normalization, cropping, drawing (shared by everything) |
| `classifiers.py` | Loads the trained models from `models/` behind one `predict()` interface |
| `camera.py` | Opens the webcam (falls back to DirectShow on Windows), clear error if it's in use |
| `smoother.py` | Majority vote over recent frames → text |
| `extract_landmarks.py` | Runs MediaPipe over the Kaggle dataset → landmark CSV (+ optional hand crops) |
| `train_landmark_baseline.py` | Quick random-forest landmark model for testing the pipeline |
| `record_clips.py` | Records our own webcam test set |
| `evaluate.py` | Accuracy, per-letter report, confusion matrix, speed for every model |

## Workflow
### 1. Extract landmarks from the dataset
Download [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and point `--data` at the folder containing `A/`, `B/`, … Test on a small sample first:
```
python extract_landmarks.py --data path/to/asl_alphabet_train/asl_alphabet_train --limit 100
```
Full run, also saving hand crops for CNN training:
```
python extract_landmarks.py --data path/to/asl_alphabet_train/asl_alphabet_train --save-crops crops
```
Output: `asl_landmarks.csv` (train on the `nx0…nz20` columns) and `asl_landmarks_missed.txt` (images with no hand found; `nothing/` should be ~0%).

### 2. Train models
Quick baseline to test the app end to end:
```
python train_landmark_baseline.py --csv asl_landmarks.csv
```

### 3. Record our own test set
```
python record_clips.py --person jack
python record_clips.py --person hafsah
```
Saves 30 frames per sign to `test_data/<person>/<sign>/`. SPACE = record, S = skip, Q = quit. Use `--labels M,N,S,T` to redo specific signs.

### 4. Evaluate
```
python evaluate.py --data test_data --tag real_world
python evaluate.py --data path/to/held_out_dataset_split --tag dataset --limit 200
```
Writes `results/<tag>/summary.csv`, `<model>_report.txt` and `<model>_confusion.png`. Comparing the two tags gives the dataset vs. real-world accuracy gap. If accuracy is oddly low on our recordings, try `--flip` (the dataset may have been mirrored relative to our webcam).

## Model export format
The app and evaluation load whatever is in `models/`:

**Landmark model** → `models/landmark_model.joblib`
- Any scikit-learn model with `predict_proba` and `classes_` (random forest, MLP, SVM with `probability=True`, a `Pipeline`)
- Input: the 63 normalized columns `nx0, ny0, nz0, …, nx20, ny20, nz20` in that order
- Save with `joblib.dump(model, "models/landmark_model.joblib")`

**CNN** → `models/cnn_model.pt` + `models/cnn_labels.txt`
- TorchScript, so the app doesn't need the training code:
  ```python
  model.eval()
  torch.jit.trace(model, torch.randn(1, 3, 224, 224)).save("models/cnn_model.pt")
  open("models/cnn_labels.txt", "w").write("\n".join(class_names))
  ```
- Input: 1×3×224×224 RGB hand crop, scaled to 0–1, normalized with ImageNet mean/std
- Train on the crops from `extract_landmarks.py --save-crops` so training crops match what the app feeds in

## Credits
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Google), hand detection and landmarks
- [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle, grassknoted)
- OpenCV, scikit-learn, PyTorch, Pillow, matplotlib
