"""Offline stand-in for Hugging Face `datasets`, used only by the test suite.

The tests put this folder first on PYTHONPATH, so `from datasets import load_dataset` in the
course scripts gets small synthetic data instead of downloading TinyStories or CIFAR-10.
It implements only what the scripts use: column access, map(batched=True) and iter().
"""
import random
from PIL import Image, ImageDraw

TOPICS = ["dog", "cat", "ball", "tree", "bird", "fish", "car", "cake", "boat", "flower",
          "dragon", "star", "moon", "bear", "frog", "train", "kite", "rabbit", "princess", "robot"]
NAMES = ["Lily", "Max", "Tom", "Sue", "Ben", "Mia", "Sam", "Nora", "Lucy", "Anna", "Tim", "Kate"]
ACTS = ["played in the park", "went to the beach", "sat under a big tree", "ran very fast",
        "sang a happy song", "had a nap in the sun", "looked at the sky", "ate a red apple",
        "jumped in the mud", "found a shiny rock"]
ENDS = ["They were very happy.", "It was the best day ever.", "Then they went home to sleep.",
        "Everyone smiled."]
COLORS = [(230, 60, 60), (60, 200, 60), (60, 90, 230), (230, 200, 40), (200, 60, 200),
          (40, 200, 200), (250, 140, 0), (150, 150, 150), (255, 255, 255), (120, 70, 20)]

class Dataset:
    def __init__(self, cols):
        self.cols = cols
    def __len__(self):
        return len(next(iter(self.cols.values())))
    def __getitem__(self, key):
        return self.cols[key]
    def map(self, fn, batched=True, remove_columns=(), num_proc=None, **kw):
        out = {k: v for k, v in self.cols.items() if k not in remove_columns}
        out.update(fn(self.cols))
        return Dataset(out)
    def iter(self, batch_size):
        for i in range(0, len(self), batch_size):
            yield {k: v[i:i + batch_size] for k, v in self.cols.items()}

def _story(r):
    w, n = r.choice(TOPICS), r.choice(NAMES)
    return (f"Once upon a time, {n} had a little {w}. {n} and the {w} {r.choice(ACTS)}. "
            f"Then they {r.choice(ACTS)}. {r.choice(ENDS)}")

def _image(r, c):
    im = Image.new("RGB", (32, 32), tuple(r.randrange(0, 80) for _ in range(3)))
    x, y, s = r.randrange(2, 14), r.randrange(2, 14), r.randrange(10, 17)
    (ImageDraw.Draw(im).ellipse if c % 2 else ImageDraw.Draw(im).rectangle)([x, y, x + s, y + s], fill=COLORS[c])
    return im

def load_dataset(name, split="train", **kw):
    val = split.startswith("validation") or split.startswith("test")
    r = random.Random(1 if val else 0)
    if "cifar" in name.lower():
        labels = [i % 10 for i in range(100 if val else 600)]
        return Dataset({"img": [_image(r, c) for c in labels], "label": labels})
    return Dataset({"text": [_story(r) for _ in range(300 if val else 3000)]})
