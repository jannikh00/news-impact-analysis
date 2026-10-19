"""Figures for the technical report that the pipeline scripts don't produce.

- ``pipeline.png``        data flow from the raw inputs through the four numbered scripts
- ``split_timeline.png``  company-day events per month in the seven-year period, by partition

Writes to ``results/figures/``.

Usage:
    python scripts/report_figures.py [--out-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline" if (ROOT / "pipeline").exists() else ROOT  # repo layout, or the course's Assignment_3/
sys.path.insert(0, str(PIPELINE))
from utils import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR  # noqa: E402

FIGURES = ROOT / "results" / "figures"
PARTITIONS = [("inner_train", "inner training", "#4C72B0"), ("validation", "validation tail", "#DD8452"),
              ("test", "testing", "#55A868")]


def box(ax, x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="#333333", linewidth=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, fontweight="bold" if bold else "normal")


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 1})


def pipeline_figure(path):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set(xlim=(0, 11), ylim=(0, 6.2))
    ax.axis("off")
    raw, script, data = "#EEEEEE", "#C9DAF8", "#FFF2CC"
    for y, label in [(5.6, "news_datasets.zip\n(3.25M headlines)"), (4.7, "sp500.csv\n(503 companies)"),
                     (3.8, "Yahoo Finance\n(daily prices)"), (2.9, "SEC EDGAR\n(share counts)")]:
        box(ax, 1.0, y, 1.7, 0.7, label, raw)
        arrow(ax, (1.85, y), (2.75, 4.25))
    box(ax, 3.5, 4.25, 1.5, 0.8, "prepare_data.py", script, bold=True)
    box(ax, 5.4, 5.3, 1.7, 0.6, "news.csv", data)
    box(ax, 5.4, 3.2, 1.7, 0.6, "historical_prices.csv", data)
    arrow(ax, (4.25, 4.45), (4.55, 5.3))
    arrow(ax, (4.25, 4.05), (4.55, 3.2))
    box(ax, 5.4, 2.0, 2.2, 0.7, "1_calculate_impacts_values.py\nmarket model, CAR, Z, IIS, MIV", script, bold=True)
    arrow(ax, (5.4, 2.9), (5.4, 2.35))
    box(ax, 8.0, 3.7, 2.4, 0.9, "2_sampling_vectorize.py\nevents, 2012-2018 period,\n80/20 split, DTM + TF-IDF",
        script, bold=True)
    arrow(ax, (6.25, 5.3), (7.2, 4.15))
    arrow(ax, (6.5, 2.0), (7.2, 3.25))
    box(ax, 8.0, 5.4, 2.4, 0.6, "news_{dtm,tfidf}_{train,test}.csv", data)
    arrow(ax, (8.0, 4.15), (8.0, 5.1))
    box(ax, 7.7, 0.75, 1.9, 0.9, "3_classify.py\nIIS: accuracy, macro-F1,\nconfusion, top 10", script, bold=True)
    box(ax, 9.9, 0.75, 1.9, 0.9, "4_regression.py\nMIV: RMSE, R²,\ntop 10", script, bold=True)
    arrow(ax, (7.7, 3.25), (7.7, 1.2))
    arrow(ax, (8.3, 3.25), (9.9, 1.2))
    for x, y, label, color in [(1.0, 0.9, "external data", raw), (2.7, 0.9, "script", script),
                               (4.4, 0.9, "generated CSV", data)]:
        box(ax, x, y, 1.4, 0.45, label, color)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def split_figure(events_path, manifest, path):
    events = pd.read_csv(events_path, usecols=["event_date", "partition"], parse_dates=["event_date"])
    events = events[events["partition"].isin([p for p, _, _ in PARTITIONS])]
    monthly = events.groupby([events["event_date"].dt.to_period("M"), "partition"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = monthly.index.to_timestamp()
    bottom = 0
    for key, label, color in PARTITIONS:
        values = monthly.get(key, 0)
        ax.bar(x, values, width=25, bottom=bottom, color=color, label=f"{label} ({int(events['partition'].eq(key).sum()):,})")
        bottom = bottom + values
    for date, text, side in [(manifest["validation"]["start"], "validation starts", "right"),
                             (manifest["test"]["start"], "test starts", "left")]:
        ax.axvline(pd.Timestamp(date), color="#333333", linestyle=":", linewidth=1)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.97, f" {text} \n {date} ", fontsize=8, va="top", ha=side)
    ax.set(ylabel="company-day events per month", xlim=(pd.Timestamp("2011-12-01"), pd.Timestamp("2019-01-31")))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    pipeline_figure(FIGURES / "pipeline.png")
    manifest = json.loads((args.data_dir / "sampling_manifest.json").read_text())
    split_figure(args.out_dir / "artifacts" / "events.csv", manifest, FIGURES / "split_timeline.png")
    print(f"wrote {FIGURES}/pipeline.png and split_timeline.png")


if __name__ == "__main__":
    main()
