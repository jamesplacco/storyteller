import torch

def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    p.mul_(1 - lr * wd)                    # decoupled weight decay
    m.mul_(b1).add_(g, alpha=1 - b1)       # 1st moment: momentum
    v.mul_(b2).addcmul_(g, g, value=1 - b2) # 2nd moment: gradient scale
    m_hat = m / (1 - b1 ** t)              # bias correction for early steps
    v_hat = v / (1 - b2 ** t)
    p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)