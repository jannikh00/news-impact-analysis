"""Information Impact Score classification with PyTorch MLPs on DTM and TF-IDF vectors.

Both representations get the same model and training conditions:
input -> 256 -> ReLU -> dropout -> 64 -> ReLU -> 7 logits, cross-entropy loss,
Adam, batch size 64, up to 100 epochs with early stopping on validation loss
(patience 10, best weights restored), PyTorch default initialization, seed 42.
Learning rate, dropout and weight decay are CLASSIFY_CONFIG in utils.py, chosen
on the validation tail by scripts/tune_models.py. IIS -3 .. +3 is mapped to
class indices 0 .. 6 and back for reporting.

Main comparison on the test set: % accuracy, macro-F1 and balanced accuracy
overall against a majority-class baseline (most frequent training IIS),
confusion matrices, and accuracy for the ten companies with the most test
observations. Supplementary models on the same rows: the same MLP trained with
inverse-frequency class weights (targets macro-F1 instead of accuracy), and an
unweighted multinomial logistic regression (C = 1, fit on all training rows).
For every model with probabilities, binary ROC AUCs measure ranking signal for
direction (IIS > 0, IIS < 0) and unusualness (IIS != 0).

Usage:
    python 3_classify.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from utils import (CLASSIFY_CONFIG, DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, IIS_CLASSES, build_mlp, load_vector_split,
                   plot_learning_curves, predict, set_seed, top_companies_by_count, train_mlp, write_json)

REPRESENTATIONS = ["dtm", "tfidf"]
CLASS_OFFSET = 3  # IIS -3 .. +3 <-> class 0 .. 6
LOGREG_C = 1.0
BINARY_TARGETS = {"up": lambda iis: iis > 0, "down": lambda iis: iis < 0, "unusual": lambda iis: iis != 0}


def scores(y_true, y_pred, proba=None):
    """Multiclass scores; with class probabilities (columns = IIS -3 .. +3) also the binary AUCs."""
    out = {"accuracy_pct": 100 * accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, labels=IIS_CLASSES, average="macro", zero_division=0),
           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
    if proba is not None:
        p = {"up": proba[:, CLASS_OFFSET + 1:].sum(axis=1), "down": proba[:, :CLASS_OFFSET].sum(axis=1),
             "unusual": 1 - proba[:, CLASS_OFFSET]}
        out |= {f"auc_{name}": roc_auc_score(target(np.asarray(y_true)), p[name])
                for name, target in BINARY_TARGETS.items()}
    return out


def balanced_class_weights(y, n_classes):
    """n / (k * count_c): the scikit-learn 'balanced' weights from training labels."""
    counts = np.bincount(y, minlength=n_classes)
    return len(y) / (n_classes * np.maximum(counts, 1))


def train_mlp_classifier(X, y, config, class_weights=None):
    import torch

    weight = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
    set_seed(config.seed)
    model = build_mlp(X["inner"].shape[1], len(IIS_CLASSES), config)
    history = train_mlp(model, torch.nn.CrossEntropyLoss(weight=weight), X["inner"], y["inner"], X["val"],
                        y["val"], config)
    proba = torch.softmax(torch.as_tensor(predict(model, X["test"])), dim=1).numpy()
    info = {"epochs_run": len(history), "best_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"])}
    return proba, history, info


def run_representation(representation, data_dir, artifacts_dir, config):
    """Main and supplementary classifiers for one representation; returns test predictions and run info."""
    frames, X, _ = load_vector_split(data_dir, representation, artifacts_dir)
    y = {name: (frames[name]["IIS"].to_numpy() + CLASS_OFFSET).astype(np.int64) for name in ["inner", "val"]}
    weights = balanced_class_weights(y["inner"], len(IIS_CLASSES))

    proba = {}
    proba["mlp"], history, info = train_mlp_classifier(X, y, config)
    proba["mlp_weighted"], history_w, info_w = train_mlp_classifier(X, y, config, weights)
    logreg = LogisticRegression(C=LOGREG_C, max_iter=1000).fit(
        sp.vstack([X["inner"], X["val"]]), np.concatenate([y["inner"], y["val"]]))
    proba["logreg"] = logreg.predict_proba(X["test"])

    info = {"input_dim": int(X["inner"].shape[1]), "rows": {k: v.shape[0] for k, v in X.items()},
            "mlp": info, "mlp_weighted": {**info_w, "class_weights": dict(zip(IIS_CLASSES, weights.round(4)))},
            "logreg": {"C": LOGREG_C, "fit_on": "inner training + validation rows", "n_iter": int(logreg.n_iter_[0])}}
    majority = int(pd.concat([frames["inner"], frames["val"]])["IIS"].mode().iloc[0])
    return frames["test"][["date", "symbol", "IIS"]], proba, {"mlp": history, "mlp_weighted": history_w}, info, majority


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
    config = CLASSIFY_CONFIG
    metrics_dir, figures_dir = args.out_dir / "metrics", args.out_dir / "figures"
    for d in (metrics_dir, figures_dir, args.out_dir / "predictions"):
        d.mkdir(parents=True, exist_ok=True)

    predictions, histories, run_info, matrices, overall = None, {}, {}, {}, []
    for rep in REPRESENTATIONS:
        print(f"training {rep} classifiers ...", flush=True)
        test, proba, hists, info, majority = run_representation(rep, args.data_dir, args.out_dir / "artifacts",
                                                                config)
        histories[rep], run_info[rep] = hists["mlp"], info
        histories[f"{rep}_weighted"] = hists["mlp_weighted"]
        if predictions is None:
            predictions = test.assign(pred_majority=majority)
            overall.append({"model": "majority_class", "role": "baseline", "representation": "",
                            **scores(test["IIS"], predictions["pred_majority"])})
        else:
            assert test[["date", "symbol"]].equals(predictions[["date", "symbol"]])  # identical test rows
        for model, role in [("mlp", "main"), ("mlp_weighted", "supplementary"), ("logreg", "supplementary")]:
            name = rep if model == "mlp" else f"{rep}_{model.removeprefix('mlp_')}"
            pred = proba[model].argmax(axis=1) - CLASS_OFFSET
            predictions[f"pred_{name}"] = pred
            overall.append({"model": f"{model}_{rep}", "role": role, "representation": rep,
                            **scores(test["IIS"], pred, proba[model])})
            if model == "mlp":
                matrices[rep] = confusion_matrix(test["IIS"], pred, labels=IIS_CLASSES)
                pd.DataFrame(matrices[rep], index=pd.Index(IIS_CLASSES, name="observed"),
                             columns=pd.Index(IIS_CLASSES, name="predicted")).to_csv(
                    metrics_dir / f"classification_confusion_{rep}.csv")

    by_company = []
    pred_cols = [c.removeprefix("pred_") for c in predictions.columns if c.startswith("pred_")]
    for symbol, g in predictions.groupby("symbol"):
        row = {"symbol": symbol, "n_test": len(g)}
        for model in pred_cols:
            row[f"accuracy_pct_{model}"] = 100 * accuracy_score(g["IIS"], g[f"pred_{model}"])
        by_company.append(row)
    by_company = pd.DataFrame(by_company)
    main_cols = ["symbol", "n_test", "accuracy_pct_majority", *[f"accuracy_pct_{r}" for r in REPRESENTATIONS]]
    top10 = top_companies_by_count(predictions)[["symbol"]].merge(by_company[main_cols], on="symbol")

    overall = pd.DataFrame(overall)
    overall.to_csv(metrics_dir / "classification_overall.csv", index=False)
    by_company.sort_values(["n_test", "symbol"], ascending=[False, True]).to_csv(
        metrics_dir / "classification_by_company.csv", index=False)
    top10.to_csv(metrics_dir / "classification_top10.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions" / "classification_test_predictions.csv", index=False)
    for name, hist in histories.items():
        hist.to_csv(metrics_dir / f"classification_history_{name}.csv", index=False)
    write_json({"config": config.to_dict(), "loss": "cross-entropy", "optimizer": "Adam", "runs": run_info,
                "test_class_distribution": predictions["IIS"].value_counts().sort_index().to_dict()},
               metrics_dir / "classification_config.json")
    plot_learning_curves({r: histories[r] for r in REPRESENTATIONS}, figures_dir / "classification_learning_curves.png",
                         "IIS classifier loss")
    plot_confusions(matrices, figures_dir / "classification_confusion.png")

    print(overall.drop(columns="representation").round(3).to_string(index=False))
    print(top10.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
