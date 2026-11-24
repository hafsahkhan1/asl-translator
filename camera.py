"""
Opens the webcam reliably on Windows/macOS/Linux.

On Windows, OpenCV's default backend (MSMF) sometimes fails to grab frames,
so this falls back to DirectShow. It also checks that a frame can actually
be read before returning.
"""

import sys

import cv2

IN_USE_HINT = ("Can't read from the webcam. Close anything else using it "
               "(app.py in another terminal, Discord, Teams, Zoom, browser tabs) and try again.")


def open_camera(index=0):
    """Return an opened cv2.VideoCapture that can read frames, or raise SystemExit."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(10):          # some cameras need a few tries to warm up
                ok, _frame = cap.read()
                if ok:
                    return cap
        cap.release()
    raise SystemExit(IN_USE_HINT)


class FrameReader:
    """Wraps cap.read() and gives up with a clear message after repeated failures
    instead of spinning forever."""

    def __init__(self, cap, max_failures=50):
        self.cap = cap
        self.max_failures = max_failures
        self.failures = 0

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.failures = 0
            return frame
        self.failures += 1
        if self.failures >= self.max_failures:
            raise SystemExit(IN_USE_HINT)
        return None
