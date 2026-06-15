"""
Fast architecture-screening driver (NOT the faithful production run).

Trains each given (dim,blocks,expand) config on a short compressed schedule
and records the best test MSE, to rank architectures cheaply. Winners get
re-run faithfully via validate.py (full 10000-epoch schedule + seed sweep).

NOTE: a fast pre-filter only. Its compressed schedule cannot reach the late
LR-restart deep minimum, so its ranking is NOT trustworthy for final decisions
on this problem (see experiments.md) -- use validate.py for those. Outputs go
to the gitignored dev/ directory.

Usage (run from autoresearch/):
    CUDA_VISIBLE_DEVICES=0 python screen.py --out dev/screen_g0.tsv \
        16,4,4 24,6,4 32,6,4
"""
import argparse, copy, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import IN_DIM, OUT_DIM, N_TRAIN, x_train, y_train, evaluate_test_mse

# --- compressed screening schedule (relative ranking only) ---
SCHED_EPOCHS = 1200
N_RESTARTS   = 1          # single cosine decay over the screen
GAMMA        = 0.85
LR           = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC  = 0.05
FINAL_LR_FRAC = 0.01
GRAD_CLIP    = 1.0
BATCH        = 256
EVAL_EVERY   = 200

dev = x_train.device


class ResBlock(nn.Module):
    def __init__(self, d, e):
        super().__init__(); self.fc1 = nn.Linear(d, d * e); self.fc2 = nn.Linear(d * e, d)
    def forward(self, x): return x + self.fc2(F.silu(self.fc1(x)))


class Net(nn.Module):
    def __init__(self, i, o, h, n, e):
        super().__init__()
        self.pi = nn.Linear(i, h)
        self.bl = nn.ModuleList([ResBlock(h, e) for _ in range(n)])
        self.po = nn.Linear(h, o)
    def forward(self, x):
        x = F.silu(self.pi(x))
        for b in self.bl: x = b(x)
        return self.po(x)


def lr_mult(epoch):
    p = epoch / SCHED_EPOCHS
    if p < WARMUP_FRAC:
        return p / WARMUP_FRAC
    pw = (p - WARMUP_FRAC) / (1 - WARMUP_FRAC)
    c = min(int(pw * N_RESTARTS), N_RESTARTS - 1)
    cp = (pw * N_RESTARTS) % 1.0
    peak = GAMMA ** c
    return peak * (FINAL_LR_FRAC + 0.5 * (1 - FINAL_LR_FRAC) * (1 + math.cos(math.pi * cp)))


def make_opt(model):
    try:
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    except Exception:
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def screen(dim, blocks, expand, seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = Net(IN_DIM, OUT_DIM, dim, blocks, expand).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_opt(model)
    loss_fn = nn.MSELoss()
    best = float("inf")
    t0 = time.time()
    for epoch in range(1, SCHED_EPOCHS + 1):
        model.train()
        m = LR * lr_mult(epoch)
        for g in opt.param_groups: g["lr"] = m
        perm = torch.randperm(N_TRAIN, device=dev)
        for st in range(0, N_TRAIN, BATCH):
            idx = perm[st:st + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        if epoch % EVAL_EVERY == 0:
            cur = evaluate_test_mse(model)
            if cur < best: best = cur
    return best, n_params, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("configs", nargs="+", help="dim,blocks,expand tokens")
    args = ap.parse_args()

    new = not os.path.exists(args.out)
    f = open(args.out, "a")
    if new:
        f.write("dim\tblocks\texpand\tparams\tbest_mse\tseconds\tseed\n"); f.flush()
    for tok in args.configs:
        dim, blocks, expand = (int(v) for v in tok.split(","))
        best, n_params, secs = screen(dim, blocks, expand, args.seed)
        f.write(f"{dim}\t{blocks}\t{expand}\t{n_params}\t{best:.6e}\t{secs:.1f}\t{args.seed}\n"); f.flush()
        print(f"[{tok}] params={n_params:6d}  best_mse={best:.3e}  {secs:.0f}s", flush=True)
    f.close()
