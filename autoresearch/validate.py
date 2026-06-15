"""
Faithful validation driver: full production schedule (10000 epochs, 10 cosine
warm restarts, gamma=0.85, warmup 0.05, batch 256, AdamW wd 1e-5, grad clip 1.0)
mirroring the production train_nn.py schedule. Fast index-perm loop + fused AdamW are
math-neutral. Runs the FULL schedule (no early stop) and records best test_mse
per (config, seed) for a clean seed-sweep comparison.

This is the AUTHORITATIVE architecture-comparison harness; screen.py is only a
fast pre-filter (it mis-ranks here, see experiments.md). Outputs go to the
gitignored dev/ directory.

Usage (from autoresearch/):
    CUDA_VISIBLE_DEVICES=0 python validate.py --out dev/validate.tsv 12,6,4,42
"""
import argparse, math, os, sys, time
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import TOTAL_EPOCHS, IN_DIM, OUT_DIM, N_TRAIN, x_train, y_train, evaluate_test_mse
from model import ResNetRegressor

# --- faithful production schedule (matches train_nn.py: scheduler_type=lambda) ---
SCHED_EPOCHS  = TOTAL_EPOCHS   # 10000
N_RESTARTS    = 10
GAMMA         = 0.85
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
WARMUP_FRAC   = 0.05
FINAL_LR_FRAC = 0.01
GRAD_CLIP     = 1.0
BATCH         = 256
EVAL_EVERY    = 500

dev = x_train.device


def lr_mult(epoch):
    p = epoch / SCHED_EPOCHS
    if p < WARMUP_FRAC:
        return p / WARMUP_FRAC
    pw = (p - WARMUP_FRAC) / (1 - WARMUP_FRAC)
    c = min(int(pw * N_RESTARTS), N_RESTARTS - 1)
    cp = (pw * N_RESTARTS) % 1.0
    return GAMMA ** c * (FINAL_LR_FRAC + 0.5 * (1 - FINAL_LR_FRAC) * (1 + math.cos(math.pi * cp)))


def make_opt(model):
    try:
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    except Exception:
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def run(dim, blocks, expand, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = ResNetRegressor(IN_DIM, OUT_DIM, hidden_dim=dim, n_hidden=blocks, expand=expand).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_opt(model); loss_fn = nn.MSELoss()
    best = float("inf"); best_epoch = 0; t0 = time.time()
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
            if cur < best:
                best, best_epoch = cur, epoch
            print(f"  [{dim}/{blocks}/{expand} s{seed}] epoch {epoch}: {cur:.3e} (best {best:.3e}) "
                  f"| {time.time()-t0:.0f}s", flush=True)
    return best, best_epoch, n_params, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("configs", nargs="+", help="dim,blocks,expand,seed tokens")
    args = ap.parse_args()
    new = not os.path.exists(args.out)
    f = open(args.out, "a")
    if new:
        f.write("dim\tblocks\texpand\tseed\tparams\tbest_mse\tbest_epoch\tseconds\n"); f.flush()
    for tok in args.configs:
        dim, blocks, expand, seed = (int(v) for v in tok.split(","))
        best, be, n_params, secs = run(dim, blocks, expand, seed)
        f.write(f"{dim}\t{blocks}\t{expand}\t{seed}\t{n_params}\t{best:.6e}\t{be}\t{secs:.1f}\n"); f.flush()
        print(f"DONE [{tok}] params={n_params} best={best:.3e}@{be} {secs:.0f}s", flush=True)
    f.close()
