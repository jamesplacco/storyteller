import json, random, re
from datasets import load_dataset

random.seed(0)
ds = load_dataset("roneneldan/TinyStories", split="train[:60000]")
templates = ["Write a short story about a {w}.",
             "Tell me a bedtime story with a {w} in it.",
             "Can you make up a story about a {w}?",
             "I want a happy story about a {w}."]
topics = ["dog", "cat", "ball", "tree", "bird", "fish", "car", "cake", "boat", "flower",
          "dragon", "star", "moon", "bear", "frog", "train", "kite", "rabbit", "princess", "robot"]

out = []
for story in ds["text"]:
    words = set(re.findall(r"[a-z]+", story.lower()))
    hits = [w for w in topics if w in words]
    if hits and len(story) < 900:
        w = random.choice(hits)
        out.append({"prompt": random.choice(templates).format(w=w), "response": story.strip(), "word": w})

random.shuffle(out)
json.dump(out, open("sft.json", "w"))
print(f"{len(out):,} pairs. Example:\n{out[0]['prompt']}\n{out[0]['response'][:200]}...")