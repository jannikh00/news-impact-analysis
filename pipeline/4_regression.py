"""Monetary Information Value regression with PyTorch MLPs on DTM and TF-IDF vectors.

Same layout and training conditions as 3_classify.py, with a single linear
output (positive and negative predictions allowed) and mean-squared-error
loss: input -> 256 -> ReLU -> dropout -> 64 -> ReLU -> 1, Adam, batch size 64,
up to 100 epochs with early stopping on validation loss (patience 10), seed 42.
Learning rate, dropout and weight decay are REGRESS_CONFIG in utils.py, chosen
on the validation tail by scripts/tune_models.py. MIV is standardized with the
mean and standard deviation of the inner-training targets only; predictions
are converted back to dollars before any metric is computed.

Evaluation on the test set: RMSE (dollars), R^2 and direction accuracy (share
of events whose predicted MIV has the observed sign) overall, a training-mean
baseline, a supplementary ridge regression on dollars (alpha RIDGE_ALPHA, fit
on all training rows), per-company metrics for every firm, the ten companies
with the most test observations (same selection as the classifier), the
largest absolute errors, and supplementary tables ranking firms with at least
MIN_COMPANY_N test rows by RMSE (lowest first) and by R^2 (highest first). R^2
is undefined for a company with fewer than 2 test rows or constant observed MIV.

Usage:
    python 4_regression.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from utils import (DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, REGRESS_CONFIG, build_mlp, load_vector_split,
                   plot_learning_curves, predict, set_seed, top_companies_by_count, train_mlp, write_json)

REPRESENTATIONS = ["dtm", "tfidf"]
MIN_COMPANY_N = 10  # minimum test rows for the metric-ranked supplementary tables
RIDGE_ALPHA = 10.0
N_LARGEST_ERRORS = 20


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    """Coefficient of determination; NaN when it is undefined (n < 2 or constant y)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if len(y) < 2 or ss_tot == 0:
        return float("nan")
    return float(1 - np.sum((y - p) ** 2) / ss_tot)


def direction_accuracy_pct(y, p):
    return float(100 * np.mean(np.sign(np.asarray(y)) == np.sign(np.asarray(p))))


def train_and_predict(representation, data_dir, artifacts_dir, config):
    import torch

    frames, X, _ = load_vector_split(data_dir, representation, artifacts_dir)
    inner, val, test = frames["inner"], frames["val"], frames["test"]
    mu, sd = inner["MIV"].mean(), inner["MIV"].std()
    y = {name: ((df["MIV"].to_numpy() - mu) / sd).astype(np.float32).reshape(-1, 1)
         for name, df in [("inner", inner), ("val", val)]}

    set_seed(config.seed)
    model = build_mlp(X["inner"].shape[1], 1, config)
    history = train_mlp(model, torch.nn.MSELoss(), X["inner"], y["inner"], X["val"], y["val"], config)
    pred = predict(model, X["test"]).ravel() * sd + mu
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(sp.vstack([X["inner"], X["val"]]), pd.concat([inner, val])["MIV"])
    info = {"input_dim": int(X["inner"].shape[1]), "rows": {k: v.shape[0] for k, v in X.items()},
            "epochs_run": len(history), "best_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"]),
            "target_scaling": {"mean": mu, "std": sd, "fit_on": "inner training targets"},
            "ridge": {"alpha": RIDGE_ALPHA, "fit_on": "inner training + validation rows, MIV in dollars"}}
    train_mean = pd.concat([inner, val])["MIV"].mean()
    out = test[["date", "symbol", "MIV"]].assign(pred=pred, pred_ridge=ridge.predict(X["test"]))
    return out, history, info, train_mean


def company_metrics(predictions, models):
    rows = []
    for symbol, g in predictions.groupby("symbol"):
        row = {"symbol": symbol, "n_test": len(g), "miv_std": g["MIV"].std(),
               "r2_defined": len(g) >= 2 and g["MIV"].nunique() > 1}
        for model in models:
            row[f"rmse_{model}"] = rmse(g["MIV"], g[f"pred_{model}"])
            row[f"r2_{model}"] = r2(g["MIV"], g[f"pred_{model}"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_predicted_vs_observed(predictions, filepath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(REPRESENTATIONS), figsize=(5 * len(REPRESENTATIONS), 4.5), squeeze=False)
    obs = predictions["MIV"] / 1e9
    lim = np.nanpercentile(np.abs(obs), 99.5)
    for ax, rep in zip(axes[0], REPRESENTATIONS):
        ax.scatter(obs, predictions[f"pred_{rep}"] / 1e9, s=6, alpha=0.4)
        ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle=":")
        ax.set(title=rep.upper(), xlabel="observed MIV ($B)", ylabel="predicted MIV ($B)",
               xlim=(-lim, lim), ylim=(-lim, lim))
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    config = REGRESS_CONFIG
    metrics_dir, figures_dir = args.out_dir / "metrics", args.out_dir / "figures"
    for d in (metrics_dir, figures_dir, args.out_dir / "predictions"):
        d.mkdir(parents=True, exist_ok=True)

    predictions, histories, run_info = None, {}, {}
    for rep in REPRESENTATIONS:
        print(f"training {rep} regressor ...", flush=True)
        pred, history, info, train_mean = train_and_predict(rep, args.data_dir, args.out_dir / "artifacts", config)
        histories[rep], run_info[rep] = history, info
        pred = pred.rename(columns={"pred": f"pred_{rep}", "pred_ridge": f"pred_{rep}_ridge"})
        if predictions is None:
            predictions = pred.assign(pred_train_mean=train_mean)
        else:
            assert pred[["date", "symbol"]].equals(predictions[["date", "symbol"]])  # identical test rows
            predictions[[f"pred_{rep}", f"pred_{rep}_ridge"]] = pred[[f"pred_{rep}", f"pred_{rep}_ridge"]].values

    models = {"train_mean": ("training_mean", "baseline"), "dtm": ("mlp_dtm", "main"),
              "tfidf": ("mlp_tfidf", "main"), "dtm_ridge": ("ridge_dtm", "supplementary"),
              "tfidf_ridge": ("ridge_tfidf", "supplementary")}
    overall = pd.DataFrame([{"model": name, "role": role,
                             "rmse_usd": rmse(predictions["MIV"], predictions[f"pred_{m}"]),
                             "r2": r2(predictions["MIV"], predictions[f"pred_{m}"]),
                             "direction_accuracy_pct": direction_accuracy_pct(predictions["MIV"],
                                                                              predictions[f"pred_{m}"])}
                            for m, (name, role) in models.items()])
    by_company = company_metrics(predictions, list(models))
    main_cols = ["symbol", "n_test", "miv_std", "r2_defined",
                 *[f"{metric}_{m}" for metric in ["rmse", "r2"] for m in ["train_mean", *REPRESENTATIONS]]]
    top10 = top_companies_by_count(predictions)[["symbol"]].merge(by_company[main_cols], on="symbol")
    eligible = by_company[by_company["n_test"] >= MIN_COMPANY_N][main_cols]
    for rep in REPRESENTATIONS:
        eligible.nsmallest(10, f"rmse_{rep}").to_csv(metrics_dir / f"regression_top10_by_rmse_{rep}.csv", index=False)
        eligible[eligible["r2_defined"]].nlargest(10, f"r2_{rep}").to_csv(
            metrics_dir / f"regression_top10_by_r2_{rep}.csv", index=False)
    errors = predictions.assign(abs_error_dtm=(predictions["MIV"] - predictions["pred_dtm"]).abs(),
                                abs_error_tfidf=(predictions["MIV"] - predictions["pred_tfidf"]).abs())
    largest = errors.nlargest(N_LARGEST_ERRORS, "abs_error_tfidf")[
        ["date", "symbol", "MIV", "pred_dtm", "pred_tfidf", "abs_error_dtm", "abs_error_tfidf"]]
    share = float((largest["abs_error_tfidf"] ** 2).sum() / (errors["abs_error_tfidf"] ** 2).sum())

    overall.to_csv(metrics_dir / "regression_overall.csv", index=False)
    by_company.sort_values(["n_test", "symbol"], ascending=[False, True]).to_csv(
        metrics_dir / "regression_by_company.csv", index=False)
    top10.to_csv(metrics_dir / "regression_top10.csv", index=False)
    largest.to_csv(metrics_dir / "regression_largest_errors.csv", index=False)
    errors.to_csv(args.out_dir / "predictions" / "regression_test_predictions.csv", index=False)
    for rep, hist in histories.items():
        hist.to_csv(metrics_dir / f"regression_history_{rep}.csv", index=False)
    write_json({"config": config.to_dict(), "loss": "MSE on standardized MIV", "optimizer": "Adam",
                "min_company_n_for_ranked_tables": MIN_COMPANY_N, "runs": run_info,
                f"squared_error_share_of_{N_LARGEST_ERRORS}_largest_tfidf_errors": share},
               metrics_dir / "regression_config.json")
    plot_learning_curves(histories, figures_dir / "regression_learning_curves.png", "MIV regressor loss (standardized)")
    plot_predicted_vs_observed(predictions, figures_dir / "regression_predicted_vs_observed.png")

    fmt = overall.assign(rmse_musd=overall["rmse_usd"] / 1e6).drop(columns="rmse_usd")
    print(fmt.round(3).to_string(index=False))
    print(top10[["symbol", "n_test", "rmse_dtm", "rmse_tfidf", "r2_dtm", "r2_tfidf"]]
          .assign(rmse_dtm=lambda d: d["rmse_dtm"] / 1e6, rmse_tfidf=lambda d: d["rmse_tfidf"] / 1e6)
          .round(3).to_string(index=False))
    print(f"{N_LARGEST_ERRORS} largest TF-IDF errors = {100 * share:.1f}% of the squared error")


if __name__ == "__main__":
    main()
