"""Information Impact Score classification with PyTorch MLPs on DTM and TF-IDF vectors.

Both representations get the same model and training conditions:
input -> 256 -> ReLU -> dropout(0.3) -> 64 -> ReLU -> 7 logits, cross-entropy
loss, Adam (lr 1e-3), batch size 64, up to 100 epochs with early stopping on
validation loss (patience 10, best weights restored), PyTorch default
initialization, seed 42. IIS -3 .. +3 is mapped to class indices 0 .. 6 and
back for reporting.

Evaluation on the test set: % accuracy and macro-F1 overall, a majority-class
baseline (most frequent training IIS), confusion matrices, and accuracy for
the ten companies with the most test observations.

Usage:
    python 3_classify.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from utils import (DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, IIS_CLASSES, TrainConfig, build_mlp, load_vector_split,
                   parse_vectors, plot_learning_curves, predict, set_seed, top_companies_by_count,
                   train_mlp, write_json)

REPRESENTATIONS = ["dtm", "tfidf"]
CLASS_OFFSET = 3  # IIS -3 .. +3 <-> class 0 .. 6


def scores(y_true, y_pred):
    return {"accuracy_pct": 100 * accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=IIS_CLASSES, average="macro", zero_division=0)}


def train_and_predict(representation, data_dir, config):
    import torch

    inner, val, test, _ = load_vector_split(data_dir, representation)
    X = {name: parse_vectors(df["news_vector"]) for name, df in [("inner", inner), ("val", val), ("test", test)]}
    y = {name: (df["IIS"].to_numpy() + CLASS_OFFSET).astype(np.int64) for name, df in
         [("inner", inner), ("val", val)]}

    set_seed(config.seed)
    model = build_mlp(X["inner"].shape[1], len(IIS_CLASSES), config)
    history = train_mlp(model, torch.nn.CrossEntropyLoss(), X["inner"], y["inner"], X["val"], y["val"], config)
    pred = predict(model, X["test"]).argmax(axis=1) - CLASS_OFFSET
    info = {"input_dim": int(X["inner"].shape[1]), "rows": {k: len(v) for k, v in X.items()},
            "epochs_run": len(history), "best_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"])}
    majority = int(pd.concat([inner, val])["IIS"].mode().iloc[0])
    return test[["date", "symbol", "IIS"]].assign(pred=pred), history, info, majority


def plot_confusions(matrices, filepath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.5), squeeze=False)
    for ax, (name, cm) in zip(axes[0], matrices.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(len(IIS_CLASSES)):
            for j in range(len(IIS_CLASSES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(title=name.upper(), xlabel="predicted IIS", ylabel="observed IIS",
               xticks=range(7), yticks=range(7), xticklabels=IIS_CLASSES, yticklabels=IIS_CLASSES)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    config = TrainConfig()
    metrics_dir, figures_dir = args.out_dir / "metrics", args.out_dir / "figures"
    for d in (metrics_dir, figures_dir, args.out_dir / "predictions"):
        d.mkdir(parents=True, exist_ok=True)

    predictions, histories, run_info, matrices, overall = None, {}, {}, {}, []
    for rep in REPRESENTATIONS:
        print(f"training {rep} classifier ...", flush=True)
        pred, history, info, majority = train_and_predict(rep, args.data_dir, config)
        histories[rep], run_info[rep] = history, info
        if predictions is None:
            predictions = pred.rename(columns={"pred": f"pred_{rep}"})
            predictions["pred_majority"] = majority
            overall.append({"model": "majority_class", "majority_iis": majority,
                            **scores(pred["IIS"], predictions["pred_majority"])})
        else:
            assert pred[["date", "symbol"]].equals(predictions[["date", "symbol"]])  # identical test rows
            predictions[f"pred_{rep}"] = pred["pred"].values
        overall.append({"model": f"mlp_{rep}", **scores(pred["IIS"], pred["pred"])})
        matrices[rep] = confusion_matrix(pred["IIS"], pred["pred"], labels=IIS_CLASSES)
        pd.DataFrame(matrices[rep], index=pd.Index(IIS_CLASSES, name="observed"),
                     columns=pd.Index(IIS_CLASSES, name="predicted")).to_csv(
            metrics_dir / f"classification_confusion_{rep}.csv")

    by_company = []
    for symbol, g in predictions.groupby("symbol"):
        row = {"symbol": symbol, "n_test": len(g)}
        for model in ["majority", *REPRESENTATIONS]:
            row[f"accuracy_pct_{model}"] = 100 * accuracy_score(g["IIS"], g[f"pred_{model}"])
        by_company.append(row)
    by_company = pd.DataFrame(by_company)
    top10 = top_companies_by_count(predictions)[["symbol"]].merge(by_company, on="symbol")

    pd.DataFrame(overall).to_csv(metrics_dir / "classification_overall.csv", index=False)
    by_company.sort_values("n_test", ascending=False).to_csv(metrics_dir / "classification_by_company.csv", index=False)
    top10.to_csv(metrics_dir / "classification_top10.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions" / "classification_test_predictions.csv", index=False)
    for rep, hist in histories.items():
        hist.to_csv(metrics_dir / f"classification_history_{rep}.csv", index=False)
    write_json({"config": config.to_dict(), "loss": "cross-entropy", "optimizer": "Adam", "runs": run_info,
                "test_class_distribution": predictions["IIS"].value_counts().sort_index().to_dict()},
               metrics_dir / "classification_config.json")
    plot_learning_curves(histories, figures_dir / "classification_learning_curves.png", "IIS classifier loss")
    plot_confusions(matrices, figures_dir / "classification_confusion.png")

    print(pd.DataFrame(overall).round(3).to_string(index=False))
    print(top10.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
