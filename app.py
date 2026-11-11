"""The Storyteller web app (Chapters 16 and 17).

Streams a story token by token with the KV-cache, then draws an illustration for it with the
image GPT if you trained one in Chapter 17.

  pip install fastapi uvicorn pillow
  uvicorn app:app --port 8000          # then open http://localhost:8000
"""
import codecs, io, os, random, re, threading, torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from model import GPT, GPTConfig
from bpe import BPETokenizer
from device import pick_device

def fail(msg):
    raise SystemExit(f"\nStoryteller can't start: {msg}\n")

device = pick_device()
if not os.path.exists("tok.pkl"):
    fail("tok.pkl is missing. Train the tokenizer first (Chapter 6): python train_tok.py")
tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]

if os.environ.get("CKPT"):
    CKPT, CHOSEN_BY = os.environ["CKPT"], "set with CKPT"
    if not os.path.exists(CKPT):
        fail(f"CKPT={CKPT} was requested but there is no such file in {os.getcwd()}.")
else:
    found = [p for p in ["chat_ppo.pt", "chat_dpo.pt", "chat.pt"] if os.path.exists(p)]
    if not found:
        fail("no chat model here (chat.pt, chat_dpo.pt or chat_ppo.pt). Make one with Chapter 14: python sft.py")
    CKPT, CHOSEN_BY = found[0], "default: latest tuning stage present, not an evaluation result"
ck = torch.load(CKPT, map_location=device)
if ck["cfg"]["vocab_size"] != len(tok.vocab):
    fail(f"{CKPT} was trained with a {ck['cfg']['vocab_size']:,}-token vocabulary but tok.pkl has {len(tok.vocab):,}. "
         "Use the tok.pkl it was trained with, or retrain from Chapter 6.")
model = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
model.load_state_dict(ck["model"])
print(f"story model: {CKPT} ({CHOSEN_BY}). Compare checkpoints with eval.py; pick one with CKPT=...")
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
    prompt: str = Field(min_length=1, max_length=2000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)

app = FastAPI()

@app.get("/info")
def info():
    return {"model": CKPT, "chosen_by": CHOSEN_BY, "params": sum(p.numel() for p in model.parameters()),
            "context_tokens": model.cfg.block_size, "device": device, "illustrations": illustrator is not None}

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
    name = hits[0] if hits else random.choice(["bird", "cat", "dog", "frog", "horse"])
    cls = CLASSES.index(name)
    gpt, vq = illustrator
    with gpu, torch.no_grad():
        img = to_images(vq.decode_codes(generate_codes(gpt, cls, 1)))[0].resize((192, 192))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"X-Class": name, "X-Class-Source": "story" if hits else "random"})

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Storyteller</title>
<style>
:root{--bg:#fff;--ink:#0a0b0d;--muted:#5c6169;--line:#e1e3e7;--sunk:#f3f4f6;--red:#d0202f;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#000;--ink:#f2f3f5;--muted:#9aa0a8;--line:#24272c;--sunk:#0c0d0f;--red:#ff3b4a;color-scheme:dark}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:#000;color:#fff;padding:18px 20px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}
header b{letter-spacing:.32em;font-size:15px} header span{color:#9aa0a8;font-size:13px}
main{max-width:680px;margin:0 auto;padding:24px 16px 120px}
.you{color:var(--muted);font-style:italic;margin:28px 0 6px} .story{white-space:pre-wrap;margin:0}
.err{color:var(--red);border-left:3px solid var(--red);padding:4px 0 4px 12px}
figure{margin:12px 0 0} img{image-rendering:pixelated;display:block;width:192px;height:192px;border:1px solid var(--line)}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
form{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 16px;display:flex;gap:8px;justify-content:center}
input{flex:1;max-width:560px;font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:2px;background:var(--sunk);color:var(--ink)}
button{font:600 13px/1 system-ui,sans-serif;letter-spacing:.16em;text-transform:uppercase;padding:0 18px;border-radius:2px;border:1px solid var(--red);background:var(--red);color:#fff;cursor:pointer}
button.stop{background:transparent;color:var(--ink);border-color:var(--ink)}
</style>
<header><b>STORYTELLER</b><span id="info">loading model info…</span></header>
<main id="log"></main>
<form id="f"><input id="q" placeholder="Tell me a story about a brave frog" maxlength="2000" autofocus><button id="go">Send</button></form>
<script>
let busy = null;
fetch("/info").then(r => r.json()).then(i => {
  info.textContent = `${i.model} (${i.chosen_by}) · ${(i.params / 1e6).toFixed(1)}M parameters · ${i.context_tokens}-token context · ${i.device}` +
    (i.illustrations ? " · illustrations on" : " · illustrations off (train Chapter 17)");
}).catch(() => { info.textContent = "could not reach the server"; });
function add(tag, cls, text) { const el = document.createElement(tag); if (cls) el.className = cls; if (text) el.textContent = text; log.append(el); return el; }
f.onsubmit = async (e) => {
  e.preventDefault();
  if (busy) { busy.abort(); return; }
  const prompt = q.value.trim(); if (!prompt) return; q.value = "";
  add("p", "you", prompt);
  const out = add("p", "story");
  busy = new AbortController(); go.textContent = "Stop"; go.className = "stop";
  try {
    const r = await fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"},
                                    body: JSON.stringify({prompt}), signal: busy.signal});
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      out.className = "err";
      out.textContent = typeof body.detail === "string" ? body.detail
        : "The request was not accepted. Keep it between 1 and 2,000 characters.";
      return;
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    for (;;) { const {done, value} = await reader.read(); if (done) break;
               out.textContent += dec.decode(value, {stream: true}); window.scrollTo(0, document.body.scrollHeight); }
    if (!out.textContent.trim()) { out.className = "err"; out.textContent = "The model ended without writing anything. Try again."; return; }
    const pic = await fetch("/illustrate?text=" + encodeURIComponent(prompt + " " + out.textContent));
    if (pic.ok) {
      const fig = document.createElement("figure"), img = new Image(), cap = document.createElement("figcaption");
      img.src = URL.createObjectURL(await pic.blob()); img.alt = "Generated picture of a " + pic.headers.get("X-Class");
      cap.textContent = `Class-conditioned 32×32 picture of a ${pic.headers.get("X-Class")} ` +
        (pic.headers.get("X-Class-Source") === "story" ? "(word found in the story)" : "(no matching word, picked at random)") +
        ". It does not depict this specific story.";
      fig.append(img, cap); out.after(fig); window.scrollTo(0, document.body.scrollHeight);
    }
  } catch (err) {
    if (err.name === "AbortError") { out.textContent += out.textContent ? " [stopped]" : "[stopped]"; }
    else { out.className = "err"; out.textContent = "Lost the connection to the server. Is uvicorn still running?"; }
  } finally { busy = null; go.textContent = "Send"; go.className = ""; }
};
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
