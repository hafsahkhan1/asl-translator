"""
Turns noisy per-frame predictions into clean text.

A letter is only typed once it wins a majority vote over the last few frames.
To type the same letter twice (e.g. "LL"), drop your hand or change the sign
in between, so the same held sign doesn't repeat.
"""

from collections import Counter, deque

SPACE = "space"
DELETE = "del"
NOTHING = "nothing"


class LetterSmoother:
    def __init__(self, window=10, min_votes=7, min_conf=0.6):
        """
        window     how many recent frames vote
        min_votes  how many of them must agree before a letter is typed
        min_conf   per-frame predictions below this confidence count as "no hand"
        """
        self.window = window
        self.min_votes = min_votes
        self.min_conf = min_conf
        self.history = deque(maxlen=window)
        self.last_committed = None

    def update(self, label, conf=1.0):
        """Feed one frame's prediction (label=None if no hand).
        Returns a token to apply to the text, or None."""
        if label is None or conf < self.min_conf:
            label = None
        self.history.append(label)
        if len(self.history) < self.window:
            return None

        top, count = Counter(self.history).most_common(1)[0]
        if count < self.min_votes:
            return None
        if top is None or top == NOTHING:
            self.last_committed = None      # hand released: the next letter can repeat
            return None
        if top == self.last_committed:
            return None                     # still holding the same sign
        self.last_committed = top
        return top

    def stable_label(self):
        """The label currently winning the vote (for display), or None."""
        if not self.history:
            return None
        top, count = Counter(self.history).most_common(1)[0]
        return top if count >= self.min_votes else None

    def reset(self):
        self.history.clear()
        self.last_committed = None


def apply_token(text, token):
    """Apply a committed token to the running text."""
    if token == SPACE:
        return text + " "
    if token == DELETE:
        return text[:-1]
    if token == NOTHING or token is None:
        return text
    return text + token
