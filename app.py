"""
ASL Translator: live webcam app.

Webcam (OpenCV) -> hand detection (MediaPipe) -> letter classifier
-> smoothing over frames -> running text.
"""

import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from camera import FrameReader, open_camera
from classifiers import load_classifiers
from hand_detector import HandDetector, draw_hand
from smoother import LetterSmoother, apply_token

BG = "#F1F7ED"
ACCENT = "#91C7B1"
DISPLAY_WIDTH = 560          # webcam preview width in the window
MIRROR = True                # show a mirror image (feels natural); detection uses the raw frame

cap = open_camera(0)
reader = FrameReader(cap)
detector = HandDetector(mode="video")
classifiers = load_classifiers()
smoother = LetterSmoother(window=10, min_votes=7, min_conf=0.6)

state = {"text": "", "started": False, "last_time": time.time(), "fps": 0.0}


def on_model_change(*_):
    smoother.reset()


def clear_text():
    state["text"] = ""
    text_label.configure(text="")
    smoother.reset()


def start():
    if not state["started"]:
        state["started"] = True
        button.pack_forget()
        controls.pack(pady=6)
        show_frame()


def show_frame():
    try:
        frame = reader.read()
    except SystemExit as e:
        status_label.configure(text=str(e), wraplength=560)
        return
    if frame is None:
        status_label.configure(text="Can't read from the webcam")
        window.after(100, show_frame)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand = detector.detect(rgb)

    label, conf = None, 0.0
    clf = classifiers.get(model_var.get())
    if hand is not None:
        if clf is not None:
            label, conf = clf.predict(hand, rgb)
        draw_hand(rgb, hand)

    token = smoother.update(label, conf)
    if token:
        state["text"] = apply_token(state["text"], token)
        text_label.configure(text=state["text"] or " ")

    # FPS, smoothed so the number doesn't jump around
    now = time.time()
    state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / max(now - state["last_time"], 1e-6))
    state["last_time"] = now

    if hand is None:
        pred_text = "No hand detected"
    elif clf is None:
        pred_text = "Hand detected (no model loaded)"
    else:
        pred_text = f"Prediction: {label}  ({conf:.0%})"
    pred_label.configure(text=pred_text)
    status_label.configure(text=f"{state['fps']:.0f} FPS")

    display = cv2.flip(rgb, 1) if MIRROR else rgb
    h, w = display.shape[:2]
    display = cv2.resize(display, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(display))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    window.after(10, show_frame)


def on_close():
    cap.release()
    detector.close()
    window.destroy()


window = tk.Tk()
window.geometry("620x760")
window.config(background=BG)
window.title("ASL Translator")
window.protocol("WM_DELETE_WINDOW", on_close)

video_label = tk.Label(window, bg=BG)
video_label.pack(pady=(10, 4))

pred_label = tk.Label(window, text="", bg=BG, font=("Helvetica", 16, "bold"))
pred_label.pack()

text_label = tk.Label(window, text=" ", bg="white", font=("Helvetica", 22),
                      width=32, height=2, anchor="w", padx=10, relief="groove",
                      wraplength=540, justify="left")
text_label.pack(pady=6)

# Controls appear after Begin: model picker + clear button
controls = tk.Frame(window, bg=BG)
model_names = list(classifiers) or ["(no model)"]
model_var = tk.StringVar(value=model_names[0])
model_var.trace_add("write", on_model_change)
tk.Label(controls, text="Model:", bg=BG).pack(side="left")
tk.OptionMenu(controls, model_var, *model_names).pack(side="left", padx=6)
tk.Button(controls, text="Clear text", bg=ACCENT, command=clear_text).pack(side="left", padx=6)

status_label = tk.Label(window, text="", bg=BG, fg="#555")
status_label.pack(side="bottom", pady=4)

button = tk.Button(window, text="Begin Translator", bg=ACCENT, command=start)
button.pack()

if not classifiers:
    status_label.configure(text="No trained model in models/ yet: detection only")

window.mainloop()
