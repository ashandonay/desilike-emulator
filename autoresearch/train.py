"""
Cosmological emulator training script. Single-GPU, epoch-based.
Trains a ResNet MLP with cosine warm restarts and early stopping.
Usage: EMULATOR_DATA_DIR=... CUDA_VISIBLE_DEVICES=... uv run train.py
"""

import math
import os
import time
import copy
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TOTAL_EPOCHS, IN_DIM, OUT_DIM, N_TRAIN, evaluate_test_mse, make_dataloader, x_train, y_train

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0, expand=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class NNRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden, dropout=0.0, expand=4):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout, expand) for _ in range(n_hidden)])
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.silu(self.proj_in(x))
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM = 24
N_HIDDEN = 6
DROPOUT = 0.0
EXPAND = 4
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC = 0.05       # fraction of TOTAL_EPOCHS for warmup
FINAL_LR_FRAC = 0.01     # minimum LR as fraction of peak
N_RESTARTS = 10           # cosine warm restart cycles
LR_GAMMA = 0.85          # decay factor for peak LR each restart cycle
GRAD_CLIP = 1.0
SEED = 42

# Early stopping
EVAL_EVERY = 1000         # evaluate every N epochs
PATIENCE = 3              # stop after this many evals with no improvement over own best
GLOBAL_PATIENCE = 2       # stop after this many evals without beating global best from prior experiments

# Experiment tracking
EXPERIMENT = "final_24dim_6blocks_gamma085_v2"
DESCRIPTION = "v2-validated final: 24dim/6blocks/expand4 knee (mean 1.96e-7, 3-seed sweep)"

# ---------------------------------------------------------------------------
# LR schedule (epoch-based cosine with warm restarts + decaying peaks)
# ---------------------------------------------------------------------------

def get_lr(epoch):
    """Returns LR multiplier given current epoch."""
    progress = epoch / TOTAL_EPOCHS
    if progress < WARMUP_FRAC:
        return progress / WARMUP_FRAC if WARMUP_FRAC > 0 else 1.0
    post_warmup = (progress - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)
    cycle_idx = int(post_warmup * N_RESTARTS)
    cycle_idx = min(cycle_idx, N_RESTARTS - 1)
    cycle_progress = (post_warmup * N_RESTARTS) % 1.0
    peak = LR_GAMMA ** cycle_idx
    cosine = FINAL_LR_FRAC + 0.5 * (1.0 - FINAL_LR_FRAC) * (1.0 + math.cos(math.pi * cycle_progress))
    return peak * cosine

# ---------------------------------------------------------------------------
# Read global best from results.tsv (prior experiments)
# ---------------------------------------------------------------------------

def get_global_best(results_file="results.tsv", exclude_experiment=None):
    """Read the best test_mse from results.tsv, excluding the current experiment."""
    if not os.path.exists(results_file):
        return float('inf')
    best = float('inf')
    with open(results_file, "r") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if exclude_experiment and row.get("experiment") == exclude_experiment:
                continue
            try:
                mse = float(row["test_mse"])
                if mse < best:
                    best = mse
            except (ValueError, KeyError):
                continue
    return best

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNRegressor(IN_DIM, OUT_DIM, HIDDEN_DIM, N_HIDDEN, DROPOUT, EXPAND).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {N_HIDDEN} hidden layers, {HIDDEN_DIM} dim, expand={EXPAND}, {num_params:,} params")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()
# Data already lives on-device (prepare.py), so we minibatch with a per-epoch
# GPU index permutation instead of a DataLoader — avoids per-batch host<->device
# overhead that dominates wall-time for this tiny model.

global_best = get_global_best(exclude_experiment=EXPERIMENT)
print(f"Total epochs: {TOTAL_EPOCHS}, Eval every: {EVAL_EVERY}, Patience: {PATIENCE}, Global patience: {GLOBAL_PATIENCE}")
print(f"Batch size: {BATCH_SIZE}, LR: {LR}, LR_GAMMA: {LR_GAMMA}, N_RESTARTS: {N_RESTARTS}")
print(f"Global best from prior experiments: {global_best:.6f}" if global_best < float('inf') else "No prior experiments")

# ---------------------------------------------------------------------------
# Training loop (epoch-based with early stopping)
# ---------------------------------------------------------------------------

t_start = time.time()
best_mse = float('inf')
best_state = None
no_improve_count = 0
no_global_improve_count = 0

_results_header = "experiment\tepoch\ttest_mse\tbest_mse\thidden_dim\tn_hidden\texpand\tlr\tlr_gamma\tn_restarts\tbatch_size\tdescription\n"
_results_file = "results.tsv"
if not os.path.exists(_results_file):
    with open(_results_file, "w") as f:
        f.write(_results_header)
results_log = open(_results_file, "a")

for epoch in range(1, TOTAL_EPOCHS + 1):
    model.train()
    epoch_loss = torch.zeros((), device=device)
    n_batches = 0

    lr_mult = get_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = LR * lr_mult

    perm = torch.randperm(N_TRAIN, device=device)
    for start in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        xb, yb = x_train[idx], y_train[idx]

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.detach()
        n_batches += 1

    # Single host sync per epoch (not per batch): cheap NaN guard + logging value.
    avg_loss = (epoch_loss / n_batches).item()
    if not math.isfinite(avg_loss):
        print(f"FAIL: NaN/Inf loss at epoch {epoch}")
        exit(1)

    if epoch % 100 == 0:
        elapsed = time.time() - t_start
        print(f"epoch {epoch:5d}/{TOTAL_EPOCHS} | train_loss: {avg_loss:.6f} | lr: {LR * lr_mult:.2e} | elapsed: {elapsed:.0f}s")

    # Evaluate and check early stopping
    if epoch % EVAL_EVERY == 0:
        cur_mse = evaluate_test_mse(model)
        elapsed = time.time() - t_start
        print(f"  >> EVAL epoch {epoch}: test_mse={cur_mse:.3e} (best={best_mse:.3e}, global={global_best:.3e}) | elapsed: {elapsed:.0f}s")
        results_log.write(f"{EXPERIMENT}\t{epoch}\t{cur_mse:.6e}\t{min(best_mse, cur_mse):.6e}\t{HIDDEN_DIM}\t{N_HIDDEN}\t{EXPAND}\t{LR}\t{LR_GAMMA}\t{N_RESTARTS}\t{BATCH_SIZE}\t{DESCRIPTION}\n")
        results_log.flush()

        # Within-run early stopping
        if cur_mse < best_mse:
            best_mse = cur_mse
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            print(f"  >> new best: {best_mse:.3e}")
        else:
            no_improve_count += 1
            print(f"  >> no improvement ({no_improve_count}/{PATIENCE})")

        if no_improve_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} evals)")
            break

        # Global best early stopping
        if global_best < float('inf'):
            if cur_mse < global_best:
                no_global_improve_count = 0
                print(f"  >> beating global best!")
            else:
                no_global_improve_count += 1
                print(f"  >> below global best ({no_global_improve_count}/{GLOBAL_PATIENCE})")

            if no_global_improve_count >= GLOBAL_PATIENCE:
                print(f"Early stopping at epoch {epoch} (not beating global best after {GLOBAL_PATIENCE} evals)")
                break

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Loaded best checkpoint (mse={best_mse:.3e})")

test_mse = evaluate_test_mse(model)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

print("---")
print(f"test_mse:         {test_mse:.6e}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"total_epochs:     {epoch}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_params:       {num_params}")
print(f"hidden_dim:       {HIDDEN_DIM}")
print(f"n_hidden:         {N_HIDDEN}")
print(f"lr_gamma:         {LR_GAMMA}")
