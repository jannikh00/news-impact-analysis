"""Evidence for the five financially informative terms of report section 5.

For each candidate term (matched as a phrase on the cleaned event text, so
"raises pt" and "raises price target" both reflect clean_text's output), this
script measures on the training events (inner training + validation tail) how
often events containing the term are followed by a positive (IIS > 0), a
negative (IIS < 0) or an extreme (|IIS| >= 2) reaction, relative to all
training events. The selected terms are fixed here from the training numbers;
the test events (2018) are reported only as an out-of-sample check.

Also writes context pairs (the same word in different phrases) and
before/after examples of the preprocessing for raw headlines.

Writes ``results/text/term_evidence.csv``, ``term_context.csv``,
``term_examples.csv`` and ``preprocessing_examples.csv``.

Usage:
    python scripts/term_evidence.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline" if (ROOT / "pipeline").exists() else ROOT
sys.path.insert(0, str(PIPELINE))
from utils import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, clean_text  # noqa: E402

RESULTS = ROOT / "results" / "text"
# term -> regex on cleaned text; analyst headlines abbreviate "price target" as "pt"
SELECTED = {
    "upgrades": r"\bupgrades\b",
    "downgrades": r"\bdowngrades\b",
    "raises price target": r"\braises (?:pt|price target)\b",
    "lowers price target": r"\blowers (?:pt|price target)\b",
    "misses": r"\bmiss(?:es)?\b",
}
OTHER_CANDIDATES = {
    "beats": r"\bbeats?\b", "better-than-expected": r"\bbetter-than-expected\b", "guidance": r"\bguidance\b",
    "loss": r"\bloss\b", "weak": r"\bweak\b", "strong": r"\bstrong\b", "gain": r"\bgain\b",
}
CONTEXT_PAIRS = {
    "upgraded": r"\bupgraded\b", "downgraded": r"\bdowngraded\b",
    "narrower loss": r"\bnarrower loss\b", "wider loss": r"\bwider loss\b",
    "raises guidance": r"\braises guidance\b", "lowers guidance": r"\b(?:lowers|cuts) guidance\b",
}
PREPROCESSING_HEADLINES = 6
EXAMPLES_PER_TERM = 3


def term_stats(events, patterns, base):
    rows = []
    for term, pattern in patterns.items():
        hit = events[events["text"].str.contains(pattern, regex=True)]
        row = {"term": term, "events": len(hit)}
        if len(hit):
            row |= {"share_up": (hit["IIS"] > 0).mean(), "share_down": (hit["IIS"] < 0).mean(),
                    "share_extreme": (hit["IIS"].abs() >= 2).mean(), "mean_iis": hit["IIS"].mean()}
            row |= {"lift_up": row["share_up"] / base["share_up"], "lift_down": row["share_down"] / base["share_down"],
                    "lift_extreme": row["share_extreme"] / base["share_extreme"]}
        rows.append(row)
    return pd.DataFrame(rows)


def base_rates(events):
    return {"share_up": (events["IIS"] > 0).mean(), "share_down": (events["IIS"] < 0).mean(),
            "share_extreme": (events["IIS"].abs() >= 2).mean(), "events": len(events)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    events = pd.read_csv(args.out_dir / "artifacts" / "events.csv", dtype={"symbol": str},
                         usecols=["symbol", "event_date", "text", "IIS", "partition"])
    events["text"] = events["text"].fillna("")
    parts = {"train": events[events["partition"].isin(["inner_train", "validation"])],
             "test": events[events["partition"] == "test"]}
    bases = {name: base_rates(df) for name, df in parts.items()}

    tables = []
    for name, df in parts.items():
        for group, patterns in [("selected", SELECTED), ("other_candidate", OTHER_CANDIDATES),
                                ("context", CONTEXT_PAIRS)]:
            tables.append(term_stats(df, patterns, bases[name]).assign(split=name, group=group))
    stats = pd.concat(tables, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stats[stats["group"] != "context"].to_csv(RESULTS / "term_evidence.csv", index=False)
    stats[stats["group"] == "context"].to_csv(RESULTS / "term_context.csv", index=False)

    # Raw headline examples from the training period, matched on their cleaned text.
    manifest = json.loads((args.data_dir / "sampling_manifest.json").read_text())
    news = pd.read_csv(args.data_dir / "news.csv", usecols=["date", "symbol", "headline"], dtype={"symbol": str})
    day = news["date"].str.slice(0, 10)
    news = news[(day >= manifest["train"]["start"]) & (day <= manifest["train"]["end"])].copy()
    news["clean"] = [clean_text(h, s) for h, s in zip(news["headline"], news["symbol"])]
    examples = []
    for term, pattern in {**SELECTED, **CONTEXT_PAIRS}.items():
        hit = news[news["clean"].str.contains(pattern, regex=True) & news["headline"].str.len().between(30, 110)]
        for _, r in hit.sample(min(EXAMPLES_PER_TERM, len(hit)), random_state=42).iterrows():
            examples.append({"term": term, "date": r["date"][:10], "symbol": r["symbol"], "headline": r["headline"]})
    pd.DataFrame(examples).to_csv(RESULTS / "term_examples.csv", index=False)

    tricky = news[news["headline"].str.contains(r"\$|%|\(|n't|&#|<", regex=True)]
    pre = tricky.sample(PREPROCESSING_HEADLINES, random_state=7)[["symbol", "headline", "clean"]]
    pre.rename(columns={"clean": "cleaned"}).to_csv(RESULTS / "preprocessing_examples.csv", index=False)

    pd.Series({f"{k}_{m}": v for k, b in bases.items() for m, v in b.items()}).to_csv(RESULTS / "term_base_rates.csv",
                                                                                     header=["value"])
    show = stats[stats["split"] == "train"].round(3)
    print(json.dumps(bases, indent=1, default=float))
    print(show.drop(columns="split").to_string(index=False))
    print(stats[(stats["split"] == "test") & (stats["group"] == "selected")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
