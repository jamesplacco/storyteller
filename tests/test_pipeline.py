"""End-to-end tests for all 17 chapters, at toy scale, with no downloads.

  pip install pytest httpx
  pytest -x -v          # 3 to 10 minutes on a laptop CPU; also checks your Mac's GPU if it has one

The course scripts run unchanged in a temporary folder, in chapter order, each feeding the next.
tests/fakes/datasets.py stands in for Hugging Face, so TinyStories and CIFAR-10 are replaced by
small synthetic data. Each test checks a claim the course makes, not only that a script ran.
"""
import math, os, re, shutil, subprocess, sys
from pathlib import Path
import pytest, torch

ROOT = Path(__file__).resolve().parents[1]
FAKES = Path(__file__).resolve().parent / "fakes"
TINY = ["--n_layer", "2", "--n_head", "2", "--n_embd", "64", "--block_size", "128", "--batch_size", "8"]

@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("storyteller")
    for f in ROOT.iterdir():
        if f.suffix in (".py", ".c"):
            shutil.copy(f, d)
    sys.path.insert(0, str(d))                      # lets tests import the copied modules
    old = os.getcwd(); os.chdir(d)
    yield d
    os.chdir(old); sys.path.remove(str(d))

def run(work, *cmd, timeout=1200):
    env = {**os.environ, "PYTHONPATH": str(FAKES), "OMP_NUM_THREADS": "4"}
    r = subprocess.run([str(c) for c in cmd], cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f"{' '.join(map(str, cmd))} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout

def py(work, script, *args):
    return run(work, sys.executable, script, *args)

def torchrun(work, script, *args, n=2):
    return run(work, sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}", script, *args)

def floats(pattern, text):
    found = []
    for m in re.findall(pattern, text):                # patterns with several groups give tuples
        found.extend(m if isinstance(m, tuple) else [m])
    return [float(x) for x in found]

def load_cpu(path):
    from model import GPT, GPTConfig
    ck = torch.load(path, map_location="cpu")
    m = GPT(GPTConfig(**ck["cfg"])).eval()
    m.load_state_dict(ck["model"])
    return m

# ---------------- Part 1: the model ----------------
def test_ch01_bigram_beats_uniform(work):
    nll, uniform = floats(r"avg NLL ([\d.]+) \| uniform would be ([\d.]+)", py(work, "bigram.py"))[:2]
    assert nll < uniform

def test_ch02_micrograd_learns(work):
    out = py(work, "micrograd.py")
    losses = floats(r"loss ([\d.]+)", out)
    assert losses[-1] < losses[0] / 5
    preds = floats(r"-?\d+\.\d+", out.splitlines()[-1])
    assert [p > 0 for p in preds] == [True, False, False, True]

def test_ch03_mlp_learns(work):
    losses = floats(r"loss ([\d.]+)", py(work, "mlp.py", "--steps", "500"))
    assert losses[-1] < losses[0] - 1

def test_ch04_attention_is_causal(work):
    from attention import Head
    torch.manual_seed(0)
    h = Head(32, 16, 6)
    out = h(torch.randn(2, 6, 32))
    att = h.last_att
    assert out.shape == (2, 6, 16)
    assert torch.allclose(att.sum(-1), torch.ones(2, 6))
    assert torch.all(att.triu(1) == 0)

def test_ch05_parameter_count(work):
    from model import GPT, GPTConfig
    assert sum(p.numel() for p in GPT(GPTConfig()).parameters()) == 12_318_720

def test_ch06_tokenizer(work):
    py(work, "train_tok.py")                        # asserts its own round trip
    py(work, "prepare.py")
    assert (work / "train.bin").stat().st_size > 0 and (work / "val.bin").stat().st_size > 0
    from bpe import BPETokenizer
    tok = BPETokenizer().load("tok.pkl")
    s = "Max ate cake 🍰 at the café.\n\n  Twice!"
    assert tok.decode(tok.encode(s)) == s
    ids = tok.encode("<|user|>hi<|assistant|>")
    assert ids[0] == tok.special["<|user|>"] and ids[-1] == tok.special["<|assistant|>"]

def test_ch07_adamw_by_hand_matches_torch(work):
    from adamw_by_hand import adamw_step
    torch.manual_seed(0)
    p1 = torch.randn(50); p2 = p1.clone().requires_grad_()
    opt = torch.optim.AdamW([p2], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    m, v = torch.zeros(50), torch.zeros(50)
    for t in range(1, 6):
        g = torch.randn(50)
        adamw_step(p1, g, m, v, t)
        p2.grad = g.clone(); opt.step()
    assert torch.allclose(p1, p2.detach(), atol=1e-6)

def test_ch07_pretrain(work):
    from bpe import BPETokenizer
    V = len(BPETokenizer().load("tok.pkl").vocab)
    out = py(work, "train.py", *TINY, "--max_steps", "400", "--warmup", "30", "--eval_every", "200")
    losses = floats(r"\| loss ([\d.]+)", out)
    assert abs(losses[0] - math.log(V)) < 0.5, "first loss should be near ln(vocab): a uniform guess"
    assert losses[-1] < losses[0] - 2
    assert len(py(work, "sample.py", "--n", "30").strip()) > 0

# ---------------- Part 2: speed ----------------
def test_ch08_strides(work):
    assert "False" in py(work, "strides.py")         # the transpose is not contiguous

def test_ch08_matmul_c(work):
    cc = shutil.which("clang") or shutil.which("gcc") or pytest.skip("no C compiler")
    run(work, cc, "-O3", "-ffast-math", "-o", "matmul", "matmul.c", "-lm")
    diffs = floats(r"max diff vs ijk ([\d.e+-]+)", run(work, "./matmul"))
    assert len(diffs) == 2 and max(diffs) < 1e-2

def test_ch08_c_engine_matches_pytorch(work):
    cc = shutil.which("clang") or shutil.which("gcc") or pytest.skip("no C compiler")
    py(work, "export.py")
    run(work, cc, "-O3", "-ffast-math", "-o", "run", "run.c", "-lm")
    c_text = run(work, "./run", "model.bin", "tok.bin", "-t", "0", "-n", "40", "-i", "Once upon a time").rstrip("\n")
    from bpe import BPETokenizer
    tok = BPETokenizer().load("tok.pkl"); EOT = tok.special["<|endoftext|>"]
    ids = tok.encode("Once upon a time")
    gen = load_cpu("ckpt.pt").generate_cached(torch.tensor([ids]), 40, top_k=1)[0, len(ids):].tolist()
    if EOT in gen:
        gen = gen[:gen.index(EOT)]
    assert c_text == "Once upon a time" + tok.decode(gen)

def test_ch08_c_engine_edge_cases(work):
    if not (work / "run").exists():
        pytest.skip("no C engine built")
    env = {**os.environ}
    r = subprocess.run(["./run", "model.bin", "tok.bin", "-i", "", "-n", "5"], cwd=work, capture_output=True, text=True, env=env)
    assert r.returncode == 0                         # empty prompt starts a new story
    r = subprocess.run(["./run", "model.bin", "tok.bin", "-i", "word " * 2000], cwd=work, capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "prompt too long" in r.stderr   # refused cleanly, no overflow

def test_ch09_precision(work):
    out = py(work, "precision.py")
    assert "bf16: 1.0 plus 1,000 updates of 1e-4 = 1.0000" in out
    assert re.search(r"fp32: 1\.0 plus 1,000 updates of 1e-4 = 1\.(0999|1000)", out)
    unscaled, scaled = floats(r"relative error ([\d.]+)", out)
    assert scaled < unscaled

@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_ch09_mixed_precision_training(work, dtype):
    out = py(work, "train.py", *TINY, "--device", "cpu", "--dtype", dtype, "--max_steps", "60",
             "--warmup", "10", "--eval_every", "1000", "--out", f"ck_{dtype}.pt")
    losses = floats(r"\| loss ([\d.]+)", out)
    assert losses[-1] < losses[0]

def test_ch10_ddp_replicas_identical(work):
    assert "replicas identical: True" in torchrun(work, "ddp_demo.py")

def test_ch10_train_ddp_with_accumulation(work):
    out = torchrun(work, "train.py", *TINY, "--device", "cpu", "--max_steps", "20", "--warmup", "5",
                   "--eval_every", "1000", "--grad_accum", "2", "--out", "ck_ddp.pt")
    assert "2 process(es)" in out and "4,096 tokens per step" in out

def test_ch10_zero1_matches_adamw(work):
    out = torchrun(work, "zero1.py")
    assert floats(r"AdamW weights\| = ([\d.e+-]+)", out)[0] < 1e-4
    zero, full = floats(r"([\d.]+) MB", out)
    assert zero < 0.6 * full

def test_ch10_fsdp_shards(work):
    out = torchrun(work, "fsdp_demo.py")
    assert "this rank stores (256, 128)" in out

# ---------------- Part 3: data, inference, finetuning ----------------
def test_ch11_sft_data(work):
    import json
    py(work, "make_sft.py")
    data = json.load(open("sft.json"))
    assert len(data) > 100 and all(ex["word"] in ex["response"].lower() for ex in data)

def test_ch12_kv_cache_matches(work):          # within the context window
    m = load_cpu("ckpt.pt")
    idx = torch.randint(0, m.cfg.vocab_size, (2, 8), generator=torch.Generator().manual_seed(0))
    assert torch.equal(m.generate(idx, 50, top_k=1), m.generate_cached(idx, 50, top_k=1))
    py(work, "bench_kv.py")

def test_ch13_quantize(work):
    out = py(work, "quantize.py")
    (fp_mb, fp_loss), (q_mb, q_loss) = [tuple(map(float, t)) for t in re.findall(r"([\d.]+) MB\s+val ([\d.]+)", out)]
    assert q_mb < 0.7 * fp_mb and abs(q_loss - fp_loss) < 0.05

def test_ch14_sft(work):
    out = py(work, "sft.py", "--steps", "60")
    trainable, total = [int(x.replace(",", "")) for x in re.findall(r"trainable ([\d,]+) of ([\d,]+)", out)[0]]
    assert trainable < 0.2 * total and (work / "chat.pt").exists()
    from bpe import BPETokenizer
    special = list(BPETokenizer().load("tok.pkl").special.values())
    before, after = load_cpu("ckpt.pt").wte.weight, load_cpu("chat.pt").wte.weight
    keep = torch.ones(len(before), dtype=torch.bool); keep[special] = False
    assert torch.equal(before[keep], after[keep])    # LoRA left the pretrained embedding alone
    assert not torch.equal(before[special], after[special])

def test_ch15_reward_model(work):
    out = py(work, "reward.py", "--steps", "40")
    assert floats(r"held-out accuracy ([\d.]+)", out) and (work / "reward.pt").exists()

def test_ch15_dpo_starts_at_ln2(work):
    out = py(work, "dpo.py", "--steps", "10")
    assert abs(floats(r"loss ([\d.]+)", out)[0] - math.log(2)) < 0.01

def test_ch15_ppo(work):
    py(work, "ppo.py", "--iters", "3")
    assert (work / "chat_ppo.pt").exists()

def test_eval_table(work):
    out = py(work, "eval.py", "--n", "4", "--max_new", "40")
    for name in ["ckpt.pt", "chat.pt", "chat_ppo.pt"]:
        assert name in out

# ---------------- Part 4: pictures and the app ----------------
def test_ch17_vqvae(work):
    from vqvae import VectorQuantizer
    z = torch.randn(2, 10, 64, requires_grad=True)
    zq, idx, loss = VectorQuantizer()(z)
    (zq.sum() + loss).backward()
    assert z.grad is not None and idx.shape == (2, 10)          # straight-through gradient
    py(work, "vqvae.py", "--steps", "40")
    assert (work / "image_codes.pt").exists() and (work / "recon.png").exists()

def test_ch17_image_gpt(work):
    py(work, "image_gpt.py", "--steps", "10")
    assert (work / "samples.png").exists()

def test_ch17_dit(work):
    from dit import patchify, unpatchify
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(unpatchify(patchify(x)), x)
    py(work, "dit.py", "--steps", "3")
    assert (work / "dit_samples.png").exists()

def test_ch16_app_streams_and_illustrates(work):
    pytest.importorskip("fastapi"); pytest.importorskip("httpx")
    code = ("from fastapi.testclient import TestClient\nimport app\nc = TestClient(app.app)\n"
            "r = c.post('/chat', json={'prompt': 'Tell me a story about a dog.'}); assert r.status_code == 200\n"
            "r = c.get('/illustrate', params={'text': 'a frog'}); assert r.status_code == 200\n"
            "assert r.content[:4] == b'\\x89PNG'\n"
            "assert '</script>' in c.get('/').text\nprint('ok')")
    assert "ok" in run(work, sys.executable, "-c", code)

# ---------------- Your Mac ----------------
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_mps_matches_cpu(work):
    m = load_cpu("ckpt.pt")
    x = torch.randint(0, m.cfg.vocab_size, (2, 64), generator=torch.Generator().manual_seed(0))
    cpu_logits = m(x)[0]
    m_mps = m.to("mps")
    mps_logits = m_mps(x.to("mps"))[0].cpu()
    assert (cpu_logits - mps_logits).abs().max() < 1e-3
    _, loss = m_mps(x.to("mps"), x.to("mps"))
    loss.backward()                                  # training works on the Apple GPU too
