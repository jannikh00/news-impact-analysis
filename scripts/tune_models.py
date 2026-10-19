"""Milestone D: choose the MLP training settings on the validation tail.

The vectorizers used by 3_classify.py and 4_regression.py are fit on all
training text (inner part plus validation tail). For tuning, that would let the
validation text shape the vocabulary and IDF weights, so this script refits
both vectorizers on the inner-training text only, with the same settings as
2_sampling_vectorize.py, and transforms the validation text with them. The
test partition is never read.

Every grid point is trained for both representations under identical
conditions (seed, batch size, epochs, early stopping). Per task, the selected
setting is the one with the lowest validation loss averaged over DTM and
TF-IDF, so both representations keep one shared configuration.

Writes ``results/model_comparison/tuning_{classification,regression}.csv``
and ``tuning_selection.json``.

Usage:
    python scripts/tune_models.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline" if (ROOT / "pipeline").exists() else ROOT  # repo layout, or the course's Assignment_3/
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(ROOT / "tests"))
from conftest import load_script  # noqa: E402
from utils import DEFAULT_OUT_DIR, TrainConfig, build_mlp, set_seed, train_mlp, write_json  # noqa: E402

GRID = {"lr": [1e-3, 3e-4], "dropout": [0.3, 0.5], "weight_decay": [0.0, 1e-5, 1e-4]}
CLASS_OFFSET = 3


def inner_only_vectors(events_path):
    """Inner-training and validation matrices from vectorizers fit on inner text only."""
    events = pd.read_csv(events_path, usecols=["symbol", "event_date", "text", "IIS", "MIV", "partition"],
                         dtype={"symbol": str}, keep_default_na=False, na_values={"IIS": [""], "MIV": [""]})
    events = events[events["partition"].isin(["inner_train", "validation"])].sort_values(["event_date", "symbol"])
    inner, val = events[events["partition"] == "inner_train"], events[events["partition"] == "validation"]
    vectors = load_script("2_sampling_vectorize.py").vectorize(inner["text"], val["text"])
    return inner, val, {name: (X_inner.astype(np.float32), X_val.astype(np.float32))
                        for name, (_, X_inner, X_val) in vectors.items()}


def targets(task, inner, val):
    if task == "classification":
        return [(df["IIS"].to_numpy() + CLASS_OFFSET).astype(np.int64) for df in (inner, val)]
    mu, sd = inner["MIV"].mean(), inner["MIV"].std()
    return [((df["MIV"].to_numpy() - mu) / sd).astype(np.float32).reshape(-1, 1) for df in (inner, val)]


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="pipeline output folder (artifacts)")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "model_comparison")
    args = ap.parse_args()

    inner, val, vectors = inner_only_vectors(args.out_dir / "artifacts" / "events.csv")
    print(f"inner {len(inner)} rows, validation {len(val)} rows", flush=True)
    selection = {}
    for task, n_out, loss_fn in [("classification", 7, torch.nn.CrossEntropyLoss()),
                                 ("regression", 1, torch.nn.MSELoss())]:
        y_inner, y_val = targets(task, inner, val)
        rows = []
        for lr, dropout, wd in itertools.product(*GRID.values()):
            config = replace(TrainConfig(), lr=lr, dropout=dropout, weight_decay=wd)
            for rep, (X_inner, X_val) in vectors.items():
                start = time.time()
                set_seed(config.seed)
                model = build_mlp(X_inner.shape[1], n_out, config)
                hist = train_mlp(model, loss_fn, X_inner, y_inner, X_val, y_val, config)
                best = hist.loc[hist["val_loss"].idxmin()]
                rows.append({"lr": lr, "dropout": dropout, "weight_decay": wd, "representation": rep,
                             "best_epoch": int(best["epoch"]), "epochs_run": len(hist),
                             "val_loss": best["val_loss"], "seconds": round(time.time() - start, 1)})
                print(f"  {task} {rep} lr={lr} dropout={dropout} wd={wd}: val_loss={best['val_loss']:.5f} "
                      f"(epoch {int(best['epoch'])})", flush=True)
        grid = pd.DataFrame(rows)
        mean = grid.groupby(list(GRID), as_index=False)["val_loss"].mean()
        best = mean.loc[mean["val_loss"].idxmin()]
        selection[task] = {k: float(best[k]) for k in GRID} | {"mean_val_loss": float(best["val_loss"])}
        args.results_dir.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.results_dir / f"tuning_{task}.csv", index=False)
        print(f"{task}: selected {selection[task]}", flush=True)

    write_json({"grid": GRID, "criterion": "lowest validation loss averaged over DTM and TF-IDF",
                "vectorizers": "refit on inner-training text only", "selected": selection},
               args.results_dir / "tuning_selection.json")


if __name__ == "__main__":
    main()
