"""Shared settings, schemas and helpers for the four pipeline scripts."""

import html
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

# --- Paths and settings -------------------------------------------------------

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PIPELINE_DIR / "datasets"
DEFAULT_OUT_DIR = PIPELINE_DIR / "outputs"

SEED = 42
MARKET_SYMBOL = "^sp500"
MARKET_TZ = "America/New_York"
MARKET_CLOSE_MINUTES = 16 * 60

# Event-study windows, in trading days relative to event day t.
ESTIMATION_DAYS = 120   # OLS sample for alpha/beta: days t-139 .. t-20 (inclusive)
ESTIMATION_GAP = 20     # the estimation window ends on day t-20
VOLATILITY_DAYS = 60    # rolling std of returns t-59 .. t
CAR_DAYS = 3            # event window t, t+1, t+2
MIN_CAR_STD = 1e-6      # below this the Z-score is treated as undefined
IIS_CLASSES = list(range(-3, 4))

TEXT_SEPARATOR = " | "  # joins headlines of one company-day event

# --- CSV schemas --------------------------------------------------------------

NEWS_COLUMNS = ["date", "symbol", "headline", "URL", "publisher", "article"]
PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "market_cap"]
IMPACT_COLUMNS = PRICE_COLUMNS + ["daily_return", "daily_volatility", "market_return",
                                  "alpha", "beta", "AR", "CAR", "Z_CAR", "IIS", "MIV"]
VECTOR_COLUMNS = ["date", "symbol", "news_vector", "IIS", "MIV"]

_ISO_DATE = pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"))
_SYMBOL = pa.Column(str, pa.Check.str_length(min_value=1))


def _float(nullable=True):
    return pa.Column(float, nullable=nullable, coerce=True)


def _schema(columns, **overrides):
    return pa.DataFrameSchema({c: overrides.get(c, _float()) for c in columns}, strict=True, ordered=True)


NEWS_SCHEMA = _schema(
    NEWS_COLUMNS,
    date=pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2})?)?$")),
    symbol=_SYMBOL,
    headline=pa.Column(str),
    URL=pa.Column(str, nullable=True),
    publisher=pa.Column(str, nullable=True),
    article=pa.Column(nullable=True),
)
PRICES_SCHEMA = _schema(PRICE_COLUMNS, date=_ISO_DATE, symbol=_SYMBOL, adj_close=_float(nullable=False))
IMPACTS_SCHEMA = _schema(
    IMPACT_COLUMNS, date=_ISO_DATE, symbol=_SYMBOL, adj_close=_float(nullable=False),
    IIS=pa.Column(float, pa.Check.isin(IIS_CLASSES), nullable=True, coerce=True),
)
VECTORS_SCHEMA = _schema(
    VECTOR_COLUMNS, date=_ISO_DATE, symbol=_SYMBOL,
    news_vector=pa.Column(str, pa.Check.str_startswith("[")),
    IIS=pa.Column(int, pa.Check.isin(IIS_CLASSES), coerce=True),
    MIV=_float(nullable=False),
)


def load_and_validate_csv(filepath, schema):
    df = pd.read_csv(filepath, dtype={"symbol": str})
    return schema.validate(df, lazy=True)


def write_validated_csv(df, filepath, schema):
    df = schema.validate(df, lazy=True)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def write_json(obj, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(json.dumps(obj, indent=2, default=str))


# --- Trading calendar and event dates -------------------------------------------

def trading_days(start, end):
    """NYSE trading dates (incl. special closures such as Hurricane Sandy, 2012-10-29/30)."""
    import pandas_market_calendars as mcal
    return mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end).index.normalize()


def map_to_event_date(timestamps, trading_days):
    """Map raw news timestamps to the first trading day the market could react.

    timestamps: strings as they appear in the news files. Values with a UTC
        offset (e.g. ``2020-06-05 10:30:54-04:00``) are treated as exact
        instants and converted to US Eastern time. Values at exactly midnight
        without an offset carry no time of day and are treated as date-only.
    trading_days: sorted DatetimeIndex of exchange trading dates (no time).

    Rules: pre-open and regular-session news -> same trading day; news at or
    after 16:00 ET, on weekends, or on holidays -> next trading day. Date-only
    news on a trading day -> that day (the 3-day event window still covers a
    possible after-close reaction on day +1).

    Returns a DataFrame with ``event_date``, ``local_date`` (calendar date in
    New York) and ``time_precision`` (``"exact"`` or ``"date"``); event_date is
    NaT beyond the last trading day.
    """
    s = pd.Series(timestamps).astype("string").reset_index(drop=True)
    days = pd.DatetimeIndex(trading_days).normalize()

    has_offset = s.str.contains(r"[+-]\d{2}:\d{2}$", regex=True, na=False)
    exact = pd.Series(pd.NaT, index=s.index, dtype=f"datetime64[ns, {MARKET_TZ}]")
    if has_offset.any():
        exact[has_offset] = pd.to_datetime(s[has_offset], utc=True).dt.tz_convert(MARKET_TZ)

    local_date = pd.to_datetime(s.str.slice(0, 10), errors="coerce")
    local_date[has_offset] = exact[has_offset].dt.tz_localize(None).dt.normalize()

    after_close = pd.Series(False, index=s.index)
    minutes = exact.dt.hour * 60 + exact.dt.minute
    after_close[has_offset] = minutes[has_offset] >= MARKET_CLOSE_MINUTES

    # First trading day >= local date; strictly after it when news came after the close.
    pos = days.searchsorted(local_date.to_numpy(), side="left")
    on_trading_day = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == local_date.to_numpy())
    pos = pos + (after_close.to_numpy() & on_trading_day)
    event = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    valid = local_date.notna().to_numpy() & (pos < len(days))
    event[valid] = days[pos[valid]]

    return pd.DataFrame({
        "event_date": event,
        "local_date": local_date,
        "time_precision": np.where(has_offset, "exact", "date"),
    })


# --- Text preprocessing ---------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_CASHTAG = re.compile(r"\$[A-Za-z]{1,5}(?:\.[A-Za-z])?\b")
_EXCHANGE_TICKER = re.compile(r"\((?:NYSE|NASDAQ|Nasdaq|NYSEARCA|NYSEArca|AMEX)\s?:\s?[A-Z.]{1,6}\)")
_MONEY = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|mm|b|bn|billion|million|thousand)\b)?", re.I)
_PERCENT = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|pct\b)", re.I)
_QUARTER = re.compile(r"\b[Qq][1-4]\b")
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th|s|k|m|mm|b|bn)?\b", re.I)
_NEGATIONS = [(re.compile(r"\bwon't\b"), "will not"), (re.compile(r"\bcan't\b"), "can not"),
              (re.compile(r"n't\b"), " not")]


def clean_text(text, symbol=None):
    """Normalize one headline into lowercase word tokens for the vectorizers.

    HTML tags and entities, and URLs are removed. Ticker mentions (``$KR``,
    ``(NYSE: KR)``, or the event's own symbol in parentheses) become
    ``tickertoken``; currency amounts ``moneytoken``; percentages ``pcttoken``;
    fiscal quarters (``Q2``) ``quarter``; other numbers ``numtoken``. Contracted
    negations are expanded (``don't`` -> ``do not``) and no stop words are
    removed, so negation survives. Hyphenated terms such as
    ``better-than-expected`` stay one token.
    """
    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _CASHTAG.sub(" tickertoken ", text)
    text = _EXCHANGE_TICKER.sub(" tickertoken ", text)
    if symbol:
        text = re.sub(r"\(\s?" + re.escape(symbol) + r"\s?\)", " tickertoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _PERCENT.sub(" pcttoken ", text)
    text = _QUARTER.sub(" quarter ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = text.lower().replace("’", "'")
    for pattern, repl in _NEGATIONS:
        text = pattern.sub(repl, text)
    text = re.sub(r"[^a-z\-\s]", " ", text)
    text = re.sub(r"(?<![a-z])-+|-+(?![a-z])", " ", text)  # drop hyphens not inside a word
    return re.sub(r"\s+", " ", text).strip()


TOKEN_PATTERN = r"\b[a-z][a-z-]+\b"  # words of 2+ characters, hyphenated compounds kept whole


# --- Vectors ------------------------------------------------------------------------

def format_vector(row, decimals=None):
    """Compact JSON list for the required ``news_vector`` CSV field."""
    if decimals is None:
        return "[" + ",".join(str(int(v)) for v in row) + "]"
    return "[" + ",".join("0" if v == 0 else f"{v:.{decimals}g}" for v in row) + "]"


def parse_vectors(series):
    return np.asarray([json.loads(v) for v in series], dtype=np.float32)


VECTOR_KEYS_SCHEMA = VECTORS_SCHEMA.remove_columns(["news_vector"])
NPZ_CHECK_ROWS = 200  # CSV rows compared with the .npz copy on every load


def load_vector_matrix(csv_path, npz_path):
    """Keys/targets of a vector CSV and its feature matrix as float32 CSR.

    The full sample's dense JSON lists need several GB once parsed, so the
    sparse ``.npz`` copy written alongside by 2_sampling_vectorize.py is used
    when present, after checking its shape and a random sample of rows against
    the CSV. Without it, the CSV vectors are parsed directly.
    """
    import scipy.sparse as sp
    if not Path(npz_path).exists():
        df = load_and_validate_csv(csv_path, VECTORS_SCHEMA)
        return df.drop(columns="news_vector"), sp.csr_matrix(parse_vectors(df["news_vector"]))
    keys = VECTOR_KEYS_SCHEMA.validate(
        pd.read_csv(csv_path, usecols=lambda c: c != "news_vector", dtype={"symbol": str}), lazy=True)
    X = sp.load_npz(npz_path).tocsr().astype(np.float32)
    if X.shape[0] != len(keys):
        raise ValueError(f"{npz_path} has {X.shape[0]} rows, {csv_path} has {len(keys)}")
    rows = np.sort(np.random.default_rng(SEED).choice(len(keys), min(NPZ_CHECK_ROWS, len(keys)), replace=False))
    wanted = set(rows + 1)  # file line numbers (header is line 0)
    sample = pd.read_csv(csv_path, usecols=["news_vector"], skiprows=lambda i: i > 0 and i not in wanted)
    if not np.allclose(parse_vectors(sample["news_vector"]), X[rows].toarray(), rtol=1e-5, atol=1e-7):
        raise ValueError(f"{npz_path} does not match the vectors in {csv_path}")
    return keys, X


def load_vector_split(data_dir, representation, artifacts_dir=None):
    """Inner-training, validation and test rows for ``dtm`` or ``tfidf``.

    Returns ``(frames, X, manifest)``: ``frames`` maps ``inner``/``val``/``test``
    to DataFrames of date, symbol, IIS and MIV, and ``X`` maps the same names to
    the matching float32 CSR feature matrices. The validation tail is the part
    of the training CSV on or after the ``validation.start`` date recorded by
    2_sampling_vectorize.py; ``.npz`` copies are read from ``artifacts_dir``.
    """
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_OUT_DIR / "artifacts"
    manifest = json.loads((data_dir / "sampling_manifest.json").read_text())
    train, X_train = load_vector_matrix(data_dir / f"news_{representation}_train.csv",
                                        artifacts_dir / f"{representation}_train.npz")
    test, X_test = load_vector_matrix(data_dir / f"news_{representation}_test.csv",
                                      artifacts_dir / f"{representation}_test.npz")
    is_val = (train["date"] >= manifest["validation"]["start"]).to_numpy()
    if is_val.any() and not is_val[np.argmax(is_val):].all():
        raise ValueError("training rows are not sorted by date")
    frames = {"inner": train[~is_val].reset_index(drop=True), "val": train[is_val].reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    X = {"inner": X_train[~is_val], "val": X_train[is_val], "test": X_test}
    return frames, X, manifest


def top_companies_by_count(df, n=10):
    """Symbols with the most rows, ties broken alphabetically for a stable order."""
    counts = df["symbol"].value_counts().rename_axis("symbol").reset_index(name="n")
    return counts.sort_values(["n", "symbol"], ascending=[False, True]).head(n).reset_index(drop=True)


# --- PyTorch models -------------------------------------------------------------------

@dataclass
class TrainConfig:
    hidden: tuple = (256, 64)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10      # early stopping on validation loss
    seed: int = SEED

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


# Settings chosen on the validation tail by scripts/tune_models.py (results/model_comparison/tuning_*.csv),
# shared by the DTM and TF-IDF models of each task.
CLASSIFY_CONFIG = TrainConfig(lr=1e-3, dropout=0.5, weight_decay=1e-4)
REGRESS_CONFIG = TrainConfig(lr=1e-3, dropout=0.5, weight_decay=1e-5)  # grid is flat: all settings within 1%


def set_seed(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_mlp(n_inputs, n_outputs, config):
    """input -> 256 -> ReLU -> dropout -> 64 -> ReLU -> n_outputs (PyTorch default init)."""
    import torch.nn as nn
    first, second = config.hidden
    return nn.Sequential(
        nn.Linear(n_inputs, first), nn.ReLU(), nn.Dropout(config.dropout),
        nn.Linear(first, second), nn.ReLU(),
        nn.Linear(second, n_outputs),
    )


def _dense(X):
    """float32 tensor from a dense array or a scipy sparse matrix (densified here, batch by batch)."""
    import torch
    return torch.as_tensor(np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32))


def train_mlp(model, loss_fn, X_train, y_train, X_val, y_val, config):
    """Mini-batch Adam with early stopping; restores the best validation-loss weights.

    X_train/X_val may be dense arrays or scipy sparse matrices; sparse rows are
    densified one mini-batch at a time. Returns the per-epoch history as a
    DataFrame (epoch, train_loss, val_loss).
    """
    import torch

    gen = torch.Generator().manual_seed(config.seed)
    y_train, n = torch.as_tensor(y_train), X_train.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history, best_loss, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(n, generator=gen).split(config.batch_size):
            xb, yb = _dense(X_train[idx.numpy()]), y_train[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_loss = float(loss_fn(torch.as_tensor(predict(model, X_val)), torch.as_tensor(y_val)))
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


def predict(model, X, chunk=8192):
    """Model outputs for dense or sparse X, evaluated in chunks."""
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(_dense(X[i:i + chunk])).numpy() for i in range(0, X.shape[0], chunk)])


# --- Figures ------------------------------------------------------------------------

def plot_learning_curves(histories, filepath, title):
    """One panel per representation: training and validation loss by epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 3.5), squeeze=False)
    for ax, (name, hist) in zip(axes[0], histories.items()):
        ax.plot(hist["epoch"], hist["train_loss"], label="training")
        ax.plot(hist["epoch"], hist["val_loss"], label="validation")
        best = hist.loc[hist["val_loss"].idxmin()]
        ax.axvline(best["epoch"], color="grey", linestyle=":", label=f"best epoch {int(best['epoch'])}")
        ax.set(title=name.upper(), xlabel="epoch", ylabel="loss")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
