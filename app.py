"""The Storyteller web app (Chapters 16 and 17).

Streams a story token by token with the KV-cache, then draws an illustration for it with the
image GPT if you trained one in Chapter 17.

  pip install fastapi uvicorn pillow
  uvicorn app:app --port 8000          # then open http://localhost:8000
"""
import codecs, io, os, random, re, threading, torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
from model import GPT, GPTConfig
from bpe import BPETokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]
CKPT = os.environ.get("CKPT") or next(p for p in ["chat_ppo.pt", "chat_dpo.pt", "chat.pt"] if os.path.exists(p))
ck = torch.load(CKPT, map_location=device)
model = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
model.load_state_dict(ck["model"])
print(f"story model: {CKPT}")
gpu = threading.Lock()                     # requests run on several threads; the GPU takes one at a time

# ---- optional illustrator (Chapter 17) ----
illustrator = None
if os.path.exists("image_gpt.pt") and os.path.exists("vqvae.pt"):
    from vqvae import CLASSES, load_vqvae, to_images
    from image_gpt import load_image_gpt, generate_codes
    illustrator = (load_image_gpt("image_gpt.pt", device), load_vqvae("vqvae.pt", device))
    print("illustrator: on")
WORD_TO_CLASS = {"dog": "dog", "puppy": "dog", "cat": "cat", "kitten": "cat", "bird": "bird", "frog": "frog",
                 "horse": "horse", "pony": "horse", "deer": "deer", "boat": "ship", "ship": "ship",
                 "car": "automobile", "truck": "truck", "plane": "airplane", "airplane": "airplane"}

@torch.no_grad()
def stream_tokens(ids, max_new=300, temperature=0.8, top_k=50):
    """Chapter 12's KV-cache loop, yielding each token as soon as it is sampled."""
    pasts, x_in, pos0 = [None] * len(model.blocks), torch.tensor([ids], device=device), 0
    for _ in range(max_new):
        T = x_in.size(1)
        if pos0 + T > model.cfg.block_size:
            break
        x = model.wte(x_in) + model.wpe(torch.arange(pos0, pos0 + T, device=device))
        for i, block in enumerate(model.blocks):
            x, pasts[i] = block(x, pasts[i])
        logits = model.lm_head(model.ln_f(x[:, -1, :]))[0] / max(temperature, 1e-3)
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        if nxt.item() >= EOT:                  # end of story, or the model starting a new chat turn
            break
        yield nxt.item()
        pos0, x_in = pos0 + T, nxt.view(1, 1)

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.8

app = FastAPI()

@app.post("/chat")
def chat(req: ChatRequest):
    ids = tok.encode(f"<|user|>{req.prompt}<|assistant|>")
    if len(ids) >= model.cfg.block_size - 16:
        raise HTTPException(400, "That request is too long for this model's context. Try a shorter one.")
    def text():
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")   # holds back half a character
        steps = stream_tokens(ids, temperature=req.temperature)
        while True:
            with gpu:
                t = next(steps, None)
            if t is None:
                break
            piece = dec.decode(tok.vocab[t])
            if piece:
                yield piece
        yield dec.decode(b"", final=True)
    return StreamingResponse(text(), media_type="text/plain")

@app.get("/illustrate")
def illustrate(text: str = ""):
    if illustrator is None:
        return Response(status_code=404)
    words = re.findall(r"[a-z]+", text.lower())
    hits = [WORD_TO_CLASS[w] for w in words if w in WORD_TO_CLASS]
    cls = CLASSES.index(hits[0] if hits else random.choice(["bird", "cat", "dog", "frog", "horse"]))
    gpt, vq = illustrator
    with gpu, torch.no_grad():
        img = to_images(vq.decode_codes(generate_codes(gpt, cls, 1)))[0].resize((192, 192))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Storyteller</title>
<style>
body{font:17px/1.6 Georgia,serif;max-width:640px;margin:40px auto;padding:0 16px;background:#f6f3ee;color:#222}
h1{font-weight:normal} .you{color:#777;font-style:italic} .story{white-space:pre-wrap}
img{image-rendering:pixelated;border-radius:8px;display:block;margin:8px 0}
form{display:flex;gap:8px;position:sticky;bottom:0;background:#f6f3ee;padding:12px 0}
input{flex:1;font:inherit;padding:8px;border:1px solid #ccc;border-radius:6px} button{font:inherit;padding:8px 14px}
</style>
<h1>Storyteller</h1><div id="log"></div>
<form id="f"><input id="q" placeholder="Tell me a story about a brave frog" autofocus><button>Send</button></form>
<script>
f.onsubmit = async (e) => {
  e.preventDefault();
  const prompt = q.value.trim() || "Tell me a story."; q.value = "";
  const you = document.createElement("p"); you.className = "you"; you.textContent = prompt; log.append(you);
  const out = document.createElement("p"); out.className = "story"; log.append(out);
  const r = await fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"},
                                  body: JSON.stringify({prompt})});
  const reader = r.body.getReader(), dec = new TextDecoder();
  for (;;) { const {done, value} = await reader.read(); if (done) break;
             out.textContent += dec.decode(value, {stream: true}); window.scrollTo(0, document.body.scrollHeight); }
  const img = new Image();
  img.onload = () => { out.after(img); window.scrollTo(0, document.body.scrollHeight); };
  img.src = "/illustrate?text=" + encodeURIComponent(prompt + " " + out.textContent);
};
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
