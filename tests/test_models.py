import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from conftest import load_script
from utils import TrainConfig, VECTORS_SCHEMA, build_mlp, format_vector, load_vector_matrix, set_seed, train_mlp

classify = load_script("3_classify.py")
regression = load_script("4_regression.py")


def _vector_csv(tmp_path, X):
    df = pd.DataFrame({"date": "2018-01-02", "symbol": [f"S{i}" for i in range(len(X))],
                       "news_vector": [format_vector(r) for r in X], "IIS": 0, "MIV": 1.0})
    path = tmp_path / "v.csv"
    VECTORS_SCHEMA.validate(df).to_csv(path, index=False)
    return path


def test_npz_copy_is_used_after_matching_the_csv(tmp_path):
    X = np.random.default_rng(0).integers(0, 3, size=(30, 5))
    csv = _vector_csv(tmp_path, X)
    sp.save_npz(tmp_path / "v.npz", sp.csr_matrix(X))
    keys, M = load_vector_matrix(csv, tmp_path / "v.npz")
    assert list(keys.columns) == ["date", "symbol", "IIS", "MIV"]
    assert M.dtype == np.float32 and np.array_equal(M.toarray(), X)


def test_npz_that_differs_from_the_csv_is_rejected(tmp_path):
    X = np.random.default_rng(0).integers(0, 3, size=(30, 5))
    csv = _vector_csv(tmp_path, X)
    sp.save_npz(tmp_path / "v.npz", sp.csr_matrix(X[::-1]))  # same shape, wrong row order
    with pytest.raises(ValueError, match="does not match"):
        load_vector_matrix(csv, tmp_path / "v.npz")


def test_missing_npz_falls_back_to_parsing_the_csv(tmp_path):
    X = np.eye(4, dtype=int)
    _, M = load_vector_matrix(_vector_csv(tmp_path, X), tmp_path / "absent.npz")
    assert np.array_equal(M.toarray(), X)


def test_sparse_training_is_deterministic():
    rng = np.random.default_rng(1)
    X = sp.random(200, 20, density=0.2, format="csr", dtype=np.float32, random_state=1)
    y = rng.integers(0, 7, 200)
    config = TrainConfig(max_epochs=3)

    def run():
        import torch
        set_seed(config.seed)
        model = build_mlp(20, 7, config)
        hist = train_mlp(model, torch.nn.CrossEntropyLoss(), X[:150], y[:150], X[150:], y[150:], config)
        return hist, [p.detach().numpy().copy() for p in model.parameters()]

    (h1, p1), (h2, p2) = run(), run()
    pd.testing.assert_frame_equal(h1, h2)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))


def test_balanced_class_weights():
    w = classify.balanced_class_weights(np.array([3, 3, 3, 0]), 7)
    assert w[3] == pytest.approx(4 / (7 * 3)) and w[0] == pytest.approx(4 / 7)
    assert w[1] == pytest.approx(4 / 7)  # absent classes treated as count 1


def test_binary_aucs_from_class_probabilities():
    iis = np.array([-2, 0, 1, 3])
    proba = np.zeros((4, 7))
    proba[np.arange(4), iis + 3] = 1.0  # perfect, confident predictions
    s = classify.scores(iis, iis, proba)
    assert s["accuracy_pct"] == 100 and s["auc_up"] == s["auc_down"] == s["auc_unusual"] == 1.0


def test_regression_metrics():
    assert regression.direction_accuracy_pct([1, -2, 3, -4], [5, 5, -1, -1]) == 50.0
    assert np.isnan(regression.r2([2.0, 2.0], [1.0, 3.0]))
    assert regression.r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
