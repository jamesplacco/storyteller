import numpy as np
from datasets import load_dataset
from bpe import BPETokenizer

tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]

def encode(batch):
    ids = [tok.encode_ordinary(t) + [EOT] for t in batch["text"]]
    return {"ids": ids, "n": [len(x) for x in ids]}

if __name__ == "__main__":           # needed on macOS: num_proc workers re-import this file
    for split, name in [("train", "train[:25%]"), ("val", "validation")]:
        ds = load_dataset("roneneldan/TinyStories", split=name)
        ds = ds.map(encode, batched=True, remove_columns=["text"], num_proc=8)
        total = sum(ds["n"])
        arr = np.memmap(f"{split}.bin", dtype=np.uint16, mode="w+", shape=(total,))
        i = 0
        for batch in ds.iter(batch_size=1024):
            for ids in batch["ids"]:
                arr[i:i + len(ids)] = ids
                i += len(ids)
        arr.flush()
        print(f"{split}: {total:,} tokens")
