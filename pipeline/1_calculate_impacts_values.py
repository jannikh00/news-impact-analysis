"""Event-study targets for every company trading day: historical_impacts_values.csv.

Each row t of historical_prices.csv is treated as a potential event day:

- daily_return      R[t] = adj_close[t] / adj_close[t-1] - 1 (decimal)
- daily_volatility  sample std (ddof=1) of R over days t-59 .. t
- market_return     S&P 500 (^sp500) return over the same interval
- alpha, beta       OLS of R on market_return over the 120 trading days
                    t-139 .. t-20 (the estimation window ends 20 trading days
                    before t, so it never overlaps the event window)
- AR                R[t] - (alpha + beta * market_return[t])
- CAR               AR summed over t, t+1, t+2 with the same alpha and beta
- Z_CAR             (CAR - mean) / std of the 118 overlapping three-day residual
                    sums that lie entirely inside the estimation window (ddof=1)
- IIS               Z_CAR mapped to -3 .. +3 (see iis_from_z)
- MIV               market_cap[t-1] * CAR, in dollars (pre-event cap proxy: the
                    previous trading day's close)

Targets are left blank when they cannot be computed; the reasons are counted
in impacts_manifest.json. A missing market cap blanks MIV but keeps IIS.

Usage:
    python 1_calculate_impacts_values.py [--data-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (CAR_DAYS, DEFAULT_DATA_DIR, ESTIMATION_DAYS, ESTIMATION_GAP, IMPACT_COLUMNS, IMPACTS_SCHEMA,
                   MARKET_SYMBOL, MIN_CAR_STD, PRICES_SCHEMA, VOLATILITY_DAYS, load_and_validate_csv, write_json)

HIST_CAR_COUNT = ESTIMATION_DAYS - CAR_DAYS + 1  # 118 complete three-day sums inside the window


def iis_from_z(z):
    """Z <= -3 -> -3; -3 < Z <= -2 -> -2; -2 < Z <= -1 -> -1; -1 < Z < 1 -> 0;
    1 <= Z < 2 -> +1; 2 <= Z < 3 -> +2; Z >= 3 -> +3; NaN stays NaN."""
    z = np.asarray(z, dtype=float)
    conditions = [z <= -3, z <= -2, z <= -1, z < 1, z < 2, z < 3, z >= 3]
    return np.select(conditions, [-3, -2, -1, 0, 1, 2, 3], default=np.nan)


def forward_sum(s, n):
    """s[t] + ... + s[t+n-1], NaN when any term is missing."""
    return s.rolling(n).sum().shift(-(n - 1))


def event_study(company, market_adj_close):
    """Target columns for one company, rows sorted by date.

    company: DataFrame with date (Timestamp), adj_close, market_cap.
    market_adj_close: index adj_close Series indexed by date.
    """
    out = pd.DataFrame(index=company.index)
    market = market_adj_close.reindex(company["date"]).to_numpy()
    y = company["adj_close"].pct_change()
    # Market return over the same interval as the company return (also across company gaps).
    x = pd.Series(market, index=company.index).pct_change()
    both = y.notna() & x.notna()
    xv, yv = x.where(both), y.where(both)

    out["daily_return"] = y
    out["daily_volatility"] = y.rolling(VOLATILITY_DAYS).std()
    out["market_return"] = x

    # Market model on days t-139 .. t-20: rolling stats ending at row t-20, shifted onto row t.
    beta = (xv.rolling(ESTIMATION_DAYS).cov(yv) / xv.rolling(ESTIMATION_DAYS).var()).shift(ESTIMATION_GAP)
    alpha = (yv.rolling(ESTIMATION_DAYS).mean().shift(ESTIMATION_GAP)
             - beta * xv.rolling(ESTIMATION_DAYS).mean().shift(ESTIMATION_GAP))
    out["alpha"], out["beta"] = alpha, beta
    out["AR"] = y - (alpha + beta * x)
    out["CAR"] = forward_sum(y, CAR_DAYS) - CAR_DAYS * alpha - beta * forward_sum(x, CAR_DAYS)

    # Historical three-day residual sums c_j = Fy_j - 3*alpha - beta*Fx_j for start days
    # j = t-139 .. t-22, so every sum lies inside the estimation window. Their mean and
    # variance follow from rolling moments of the forward sums Fy, Fx over those 118 starts.
    fy, fx = forward_sum(yv, CAR_DAYS), forward_sum(xv, CAR_DAYS)
    lag = ESTIMATION_GAP + CAR_DAYS - 1
    roll = lambda s: s.rolling(HIST_CAR_COUNT)
    mean_fy, mean_fx = roll(fy).mean().shift(lag), roll(fx).mean().shift(lag)
    var_fy, var_fx = roll(fy).var().shift(lag), roll(fx).var().shift(lag)
    cov_fyx = roll(fy).cov(fx).shift(lag)
    hist_mean = mean_fy - CAR_DAYS * alpha - beta * mean_fx
    hist_std = np.sqrt((var_fy - 2 * beta * cov_fyx + beta ** 2 * var_fx).clip(lower=0))

    z = (out["CAR"] - hist_mean) / hist_std
    out["Z_CAR"] = z.where(hist_std >= MIN_CAR_STD)
    out["IIS"] = iis_from_z(out["Z_CAR"])
    out["MIV"] = company["market_cap"].shift(1) * out["CAR"]

    reasons = pd.Series("valid", index=company.index)
    reasons[out["MIV"].isna()] = "missing_market_cap"
    reasons[(hist_std < MIN_CAR_STD)] = "near_zero_car_std"
    reasons[out["CAR"].isna() & alpha.notna()] = "incomplete_event_window"
    reasons[alpha.isna() | beta.isna()] = "insufficient_history"
    return out, reasons


def calculate(prices):
    prices = prices.assign(date=pd.to_datetime(prices["date"])).sort_values(["symbol", "date"], kind="stable")
    market = prices[prices["symbol"] == MARKET_SYMBOL].set_index("date")["adj_close"]
    if market.empty:
        raise SystemExit(f"{MARKET_SYMBOL} rows missing from historical_prices.csv")

    frames, reason_frames = [], []
    for symbol, company in prices.groupby("symbol", sort=True):
        if symbol == MARKET_SYMBOL:
            r = company["adj_close"].pct_change()
            targets = pd.DataFrame({"daily_return": r, "daily_volatility": r.rolling(VOLATILITY_DAYS).std(),
                                    "market_return": r}, index=company.index)
        else:
            targets, reasons = event_study(company, market)
            reason_frames.append(reasons.rename("reason").to_frame().assign(symbol=symbol))
        frames.append(company.join(targets))

    impacts = pd.concat(frames).reindex(columns=IMPACT_COLUMNS)
    impacts["date"] = impacts["date"].dt.strftime("%Y-%m-%d")
    reasons = pd.concat(reason_frames)
    return impacts.reset_index(drop=True), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    prices = load_and_validate_csv(args.data_dir / "historical_prices.csv", PRICES_SCHEMA)
    impacts, reasons = calculate(prices)
    impacts = IMPACTS_SCHEMA.validate(impacts, lazy=True)
    impacts.assign(IIS=impacts["IIS"].astype("Int64")).to_csv(
        args.data_dir / "historical_impacts_values.csv", index=False)

    company = impacts[impacts["symbol"] != MARKET_SYMBOL]
    manifest = {
        "windows": {"estimation_days": ESTIMATION_DAYS, "estimation_ends_days_before_event": ESTIMATION_GAP,
                    "volatility_days": VOLATILITY_DAYS, "car_days": CAR_DAYS,
                    "historical_car_sums": HIST_CAR_COUNT, "std_ddof": 1, "min_car_std": MIN_CAR_STD},
        "rows": len(impacts),
        "company_rows": len(company),
        "rows_with_iis": int(company["IIS"].notna().sum()),
        "rows_with_iis_and_miv": int((company["IIS"].notna() & company["MIV"].notna()).sum()),
        "reasons": reasons["reason"].value_counts().to_dict(),
        "reasons_by_symbol": reasons.groupby("symbol")["reason"].value_counts().unstack(fill_value=0).to_dict("index"),
        "iis_distribution": company["IIS"].value_counts().sort_index().astype(int).to_dict(),
    }
    write_json(manifest, args.data_dir / "impacts_manifest.json")
    print({k: manifest[k] for k in ["rows", "rows_with_iis", "rows_with_iis_and_miv", "reasons", "iis_distribution"]})


if __name__ == "__main__":
    main()
