"""Compare scaled vs unscaled emulator MSE in physical space across tracer bins."""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from util import latin_hypercube_samples, get_pipeline, build_model
from scale_data import eval_scale_expression

MLRUNS = os.path.expandvars(
    "$SCRATCH/bedcosmo/num_tracers/emulator/mlruns"
)

# Unscaled experiment (922126930265883867)
UNSCALED = {
    "BGS":       "da6e5c78ebfb4eb8ae930fd18d75fa9b",
    "LRG1":      "d065442a3db647f6833440fae4dbf196",
    "LRG2":      "9bc7d9a212b44fb4a73a492329684f9a",
    "LRG3_ELG1": "538cfb0a223e45388da4ff2bf071dbb5",
    "ELG2":      "f315632e2b794aae895986b33f8b9654",
    "QSO":       "537c6194b8d143da945d4c2b254c96ec",
    "Lya_QSO":   "42e6e0b2892641c2a00ec6a05299cb02",
}
UNSCALED_EXP = "922126930265883867"

# Scaled experiment (428057621105644347)
SCALED = {
    "BGS":       "b1fb7689ccc94e34b3fcac132ed72cf1",
    "LRG1":      "623789281fc14b05ac1bf586d2430ce9",
    "LRG2":      "263a669949104686a0bbde2ae0a57302",
    "LRG3_ELG1": "c6f9d8d710034fd4a8823496a99baece",
    "ELG2":      "cbf4ce4f74344c6b828f533a2cc70f81",
    "QSO":       "3d39ef80192c444dbb66c4991d98d226",
    "Lya_QSO":   "7561446959aa4e3f88f977cde0d24c97",
}
SCALED_EXP = "428057621105644347"

TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO", "Lya_QSO"]
N_SAMPLES = 100
SEED = 123  # different from training eval seed (42)


def find_model(exp_id, run_id):
    base = os.path.join(MLRUNS, exp_id, run_id, "artifacts")
    for p in [
        os.path.join(base, "checkpoints", "model_best.pt"),
        os.path.join(base, "model_best.pt"),
        os.path.join(base, "model.pt"),
    ]:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No model found for {run_id}")


def predict(model_path, x_raw, param_names, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(
        analysis=ckpt.get("analysis", "bao"),
        architecture=ckpt.get("architecture", "resnet"),
        in_dim=len(ckpt["param_names"]),
        out_dim=len(ckpt["target_names"]),
        hidden_dim=ckpt["hidden_dim"],
        n_hidden=ckpt["n_hidden"],
        dropout=ckpt.get("dropout", 0.0),
        expand=ckpt.get("expand", 4),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mu = ckpt["x_mu"].cpu().numpy()
    x_sigma = ckpt["x_sigma"].cpu().numpy()
    y_mu = ckpt["y_mu"].cpu().numpy()
    y_sigma = ckpt["y_sigma"].cpu().numpy()
    log_normalize = ckpt.get("log_normalize", False)
    y_linthresh = ckpt["y_linthresh"].cpu().numpy() if ckpt.get("y_linthresh") is not None else None

    x_norm = (x_raw - x_mu) / x_sigma
    with torch.no_grad():
        y_pred_norm = model(torch.from_numpy(x_norm).to(device)).cpu().numpy()
    y_pred = y_pred_norm * y_sigma + y_mu
    if log_normalize and y_linthresh is not None:
        y_pred = np.sign(y_pred) * y_linthresh * np.expm1(np.abs(y_pred))

    # Undo scaling if present
    scale_expressions = ckpt.get("scale_expressions")
    if scale_expressions:
        env = {name: x_raw[:, i].astype(np.float64) for i, name in enumerate(param_names)}
        scale_factor = np.ones(len(x_raw), dtype=np.float64)
        for expr in scale_expressions:
            scale_factor *= np.asarray(eval_scale_expression(expr, env), dtype=np.float64).reshape(-1)
        y_pred = y_pred / scale_factor[:, None].astype(np.float32)
        print(f"    Inverted scaling: {' * '.join(scale_expressions)}")

    return y_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mse_unscaled = {}
    mse_scaled = {}

    for tracer_bin in TRACERS:
        print(f"\n--- {tracer_bin} ---")

        # Generate ground truth samples
        np.random.seed(SEED)
        _, target_names, ground_truth_fn, setup = get_pipeline("bao", "covar", tracer_bin=tracer_bin)

        # Use the param_names from the unscaled checkpoint to ensure consistency
        unscaled_path = find_model(UNSCALED_EXP, UNSCALED[tracer_bin])
        ckpt_tmp = torch.load(unscaled_path, map_location="cpu", weights_only=False)
        param_names = ckpt_tmp["param_names"]
        default_priors, _, _, _ = get_pipeline("bao", "covar", tracer_bin=tracer_bin)

        true_rows = []
        param_rows = []
        skipped = 0
        batch_seed = SEED
        while len(true_rows) < N_SAMPLES:
            remaining = N_SAMPLES - len(true_rows)
            draws = latin_hypercube_samples(default_priors, n_samples=remaining, seed=batch_seed)
            batch_seed += 1
            for sample in draws:
                try:
                    targets = ground_truth_fn(setup, sample)
                    vals = [targets[t] for t in target_names]
                    if not all(np.isfinite(vals)):
                        skipped += 1
                        continue
                    true_rows.append(vals)
                    param_rows.append([sample[p] for p in param_names])
                except Exception:
                    skipped += 1
                    continue
                if len(true_rows) >= N_SAMPLES:
                    break

        y_true = np.array(true_rows, dtype=np.float32)
        x_raw = np.array(param_rows, dtype=np.float32)
        print(f"  {len(true_rows)} samples ({skipped} rejected)")

        # Unscaled predictions
        print(f"  Unscaled model: {unscaled_path}")
        y_pred_unscaled = predict(unscaled_path, x_raw, param_names, device)
        mse_u = float(np.mean((y_pred_unscaled - y_true) ** 2))
        mse_unscaled[tracer_bin] = mse_u
        print(f"  Unscaled MSE: {mse_u:.6e}")

        # Scaled predictions
        scaled_path = find_model(SCALED_EXP, SCALED[tracer_bin])
        print(f"  Scaled model: {scaled_path}")
        y_pred_scaled = predict(scaled_path, x_raw, param_names, device)
        mse_s = float(np.mean((y_pred_scaled - y_true) ** 2))
        mse_scaled[tracer_bin] = mse_s
        print(f"  Scaled MSE: {mse_s:.6e}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TRACERS))
    w = 0.35
    bars1 = ax.bar(x - w / 2, [mse_unscaled[t] for t in TRACERS], w, label="Unscaled", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, [mse_scaled[t] for t in TRACERS], w, label="Scaled", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(TRACERS, rotation=30, ha="right")
    ax.set_ylabel("MSE (physical space)")
    ax.set_yscale("log")
    ax.set_title("Emulator MSE: Unscaled vs Scaled (N_tracers * 1/exp(Om) * hrdrag)")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(MLRUNS, "compare_scaled_unscaled_mse.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
