"""RLHF with PPO (Chapter 15). The second half: optimize the chat model against the reward model.

Each iteration:
  1. Rollout   sample G stories for each of P prompts from the current policy
  2. Score     reward model score, minus a KL penalty for drifting from the SFT model
  3. Advantage compare each story to the others for the same prompt (group baseline)
  4. Update    a few steps of PPO's clipped objective on those samples

Classic PPO trains a value network as the baseline. Comparing within a group of samples
instead (as in GRPO) removes that fourth model and works well for whole-response rewards.

  python ppo.py           # needs chat.pt, reward.pt, sft.json -> chat_ppo.pt
"""
import argparse, copy, json, random, torch, torch.nn.functional as F
from model import GPT, GPTConfig
from bpe import BPETokenizer
from reward import load_reward_model

device = "mps" if torch.backends.mps.is_available() else "cpu"
random.seed(0); torch.manual_seed(0)
P, G = 4, 4                  # prompts per iteration, samples per prompt
p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=150)
args = p.parse_args()
iters, max_new = args.iters, 200
ppo_epochs, clip, kl_coef, lr = 2, 0.2, 0.05, 1e-5

tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]
ck = torch.load("chat.pt", map_location=device)
policy = GPT(GPTConfig(**ck["cfg"])).to(device)
policy.load_state_dict(ck["model"])
ref = copy.deepcopy(policy).eval()                       # frozen SFT model, the KL anchor
rm = load_reward_model("reward.pt", device)
for p in list(ref.parameters()) + list(rm.parameters()):
    p.requires_grad = False
opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
prompts = sorted({ex["prompt"] for ex in json.load(open("sft.json"))})

def token_logprobs(model, seq):
    """log p(seq[t+1] | seq[:t+1]) for every position: (N, L-1)."""
    logits, _ = model(seq[:, :-1])
    return torch.gather(F.log_softmax(logits.float(), -1), 2, seq[:, 1:].unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def rollout():
    seqs, masks, lasts, groups = [], [], [], []
    policy.eval()
    for gi, prompt in enumerate(random.sample(prompts, P)):
        p_ids = tok.encode(f"<|user|>{prompt}<|assistant|>")
        n_new = min(max_new, policy.cfg.block_size - len(p_ids))       # stay inside the context
        out = policy.generate_cached(torch.tensor([p_ids] * G, device=device), n_new, temperature=1.0)
        for row in out.tolist():
            resp = row[len(p_ids):]
            if EOT in resp:
                resp = resp[:resp.index(EOT) + 1]        # keep the EOT, drop anything after it
            seqs.append(p_ids + resp)
            masks.append([0] * (len(p_ids) - 1) + [1] * len(resp))   # aligned with targets seq[1:]
            lasts.append(len(p_ids) + len(resp) - 1)
            groups.append(gi)
    policy.train()
    L = max(len(s) for s in seqs)
    seq = torch.tensor([s + [EOT] * (L - len(s)) for s in seqs], device=device)
    mask = torch.tensor([m + [0] * (L - 1 - len(m)) for m in masks], device=device, dtype=torch.float)
    return seq, mask, torch.tensor(lasts, device=device), torch.tensor(groups, device=device)

for it in range(iters):
    seq, mask, last, group = rollout()
    with torch.no_grad():
        old_lp = token_logprobs(policy, seq)
        ref_lp = token_logprobs(ref, seq)
        score = rm(seq, last)                                    # reward model's opinion
        kl = ((old_lp - ref_lp) * mask).sum(-1)                  # how far each sample drifted
        R = score - kl_coef * kl
        adv = torch.zeros_like(R)
        for g in group.unique():                                 # group-relative advantage
            sel = group == g
            adv[sel] = (R[sel] - R[sel].mean()) / (R[sel].std() + 1e-6)

    for _ in range(ppo_epochs):
        new_lp = token_logprobs(policy, seq)
        ratio = torch.exp(new_lp - old_lp)
        A = adv.unsqueeze(-1)
        surrogate = torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A)
        loss = -(surrogate * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    if it % 10 == 0:
        frac = (((ratio - 1).abs() > clip).float() * mask).sum() / mask.sum()
        print(f"iter {it:3d} | reward {score.mean().item():+.3f} | KL {kl.mean().item():6.2f} | "
              f"clipped {frac.item():.2f} | length {mask.sum(-1).mean().item():.0f}")
    if it % 50 == 0:
        idx = torch.tensor([tok.encode("<|user|>Tell me a story about a cat.<|assistant|>")], device=device)
        policy.eval()
        out = tok.decode(policy.generate_cached(idx, 120, temperature=0.8, top_k=50)[0].tolist())
        policy.train()
        print("   sample:", out.split("<|assistant|>")[1].split("<|endoftext|>")[0][:300])

torch.save({"model": policy.state_dict(), "cfg": ck["cfg"]}, "chat_ppo.pt")
