import re, pickle
from collections import Counter

PAT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")
SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]

def get_stats(ids, counts, weight=1):
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts

def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

class BPETokenizer:
    def __init__(self):
        self.merges = {}                                   # (a, b) -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes
        self.special, self._cache = {}, {}

    def train(self, text, vocab_size, verbose=False):
        chunks = Counter(tuple(c.encode("utf-8")) for c in PAT.findall(text))
        for i in range(vocab_size - 256 - len(SPECIALS)):
            stats = {}
            for ids, n in chunks.items():
                get_stats(ids, stats, n)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = Counter({tuple(merge(list(ids), pair, idx)): n for ids, n in chunks.items()})
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and i % 250 == 0:
                print(f"merge {i}: {self.vocab[idx]!r} ({stats[pair]:,} uses)")
        self._add_specials()

    def _add_specials(self):
        base = 256 + len(self.merges)
        self.special = {s: base + i for i, s in enumerate(SPECIALS)}
        for s, i in self.special.items():
            self.vocab[i] = s.encode("utf-8")

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids, {})
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))   # earliest merge first
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        return [i for chunk in PAT.findall(text) for i in self._encode_chunk(chunk)]

    def encode(self, text):
        pat = "(" + "|".join(re.escape(s) for s in self.special) + ")"
        out = []
        for part in re.split(pat, text):
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.merges, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.merges = pickle.load(f)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self._add_specials()
        self._cache = {}
        return self