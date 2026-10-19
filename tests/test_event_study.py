import numpy as np
import pandas as pd
import pytest

from conftest import load_script

impacts = load_script("1_calculate_impacts_values.py")
N = 320


def synthetic(noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rm = rng.normal(0.0005, 0.01, N)
    r = 0.001 + 1.3 * rm + rng.normal(0, noise, N)
    r[250] += 0.05  # news shock
    dates = pd.bdate_range("2015-01-01", periods=N)
    market = pd.Series(100 * np.cumprod(1 + np.r_[0, rm[1:]]), index=dates)
    company = pd.DataFrame({"date": dates, "adj_close": 50 * np.cumprod(1 + np.r_[0, r[1:]]),
                            "market_cap": np.linspace(10e9, 12e9, N)})
    return company, market


def reference(company, market, t):
    """Slow, literal implementation of the event-study definitions for event row t."""
    y = company["adj_close"].pct_change().to_numpy()
    x = market.pct_change().to_numpy()
    est = np.arange(t - 139, t - 19)                      # t-139 .. t-20 inclusive
    assert len(est) == 120
    beta, alpha = np.polyfit(x[est], y[est], 1)
    resid = lambda i: y[i] - (alpha + beta * x[i])
    car = sum(resid(i) for i in (t, t + 1, t + 2))
    hist = [resid(j) + resid(j + 1) + resid(j + 2) for j in range(est[0], est[-1] - 1)]
    assert len(hist) == 118
    z = (car - np.mean(hist)) / np.std(hist, ddof=1)
    return {"alpha": alpha, "beta": beta, "AR": resid(t), "CAR": car, "Z_CAR": z,
            "MIV": company["market_cap"].iloc[t - 1] * car}


@pytest.mark.parametrize("t", [140, 200, 250, N - 3])
def test_matches_reference(t):
    company, market = synthetic()
    out, reasons = impacts.event_study(company, market)
    ref = reference(company, market, t)
    for col, value in ref.items():
        assert out[col].iloc[t] == pytest.approx(value, rel=1e-9, abs=1e-12), col
    assert out["IIS"].iloc[t] == impacts.iis_from_z([ref["Z_CAR"]])[0]
    assert reasons.iloc[t] == "valid"


def test_shock_is_detected():
    company, market = synthetic()
    out, _ = impacts.event_study(company, market)
    assert out["IIS"].iloc[250] == 3
    assert out["CAR"].iloc[250] == pytest.approx(0.05, abs=0.03)


def test_window_boundaries():
    company, market = synthetic()
    out, reasons = impacts.event_study(company, market)
    assert reasons.iloc[139] == "insufficient_history" and reasons.iloc[140] == "valid"
    assert reasons.iloc[N - 3] == "valid"
    assert (reasons.iloc[N - 2:] == "incomplete_event_window").all()

    # Day t-20 is the last estimation day; day t-19 is in the gap and must not matter.
    t = 200
    for day, should_change in [(t - 20, True), (t - 19, False), (t - 139, True), (t - 140, False)]:
        bumped = company.copy()
        bumped.loc[day:, "adj_close"] *= 1.03  # changes only the return on `day`
        new, _ = impacts.event_study(bumped, market)
        changed = not np.isclose(new["alpha"].iloc[t], out["alpha"].iloc[t], rtol=0, atol=1e-12)
        assert changed == should_change, day


def test_same_model_for_all_three_days():
    company, market = synthetic()
    out, _ = impacts.event_study(company, market)
    y = company["adj_close"].pct_change()
    x = market.pct_change().to_numpy()
    t = 200
    a, b = out["alpha"].iloc[t], out["beta"].iloc[t]
    expected = sum(y.iloc[i] - (a + b * x[i]) for i in (t, t + 1, t + 2))
    assert out["CAR"].iloc[t] == pytest.approx(expected, rel=1e-12)
    # Summing the AR column instead would mix three different rolling models.
    assert out["AR"].iloc[t:t + 3].sum() != pytest.approx(expected, rel=1e-12)


def test_near_zero_variability_is_excluded():
    company, market = synthetic(noise=0.0)
    company["adj_close"] = 50 * np.cumprod(1 + np.r_[0, (0.001 + 1.3 * market.pct_change()).iloc[1:]])
    out, reasons = impacts.event_study(company, market)
    assert out["Z_CAR"].iloc[200:300].isna().all()
    assert out["IIS"].iloc[200:300].isna().all()
    assert (reasons.iloc[200:300] == "near_zero_car_std").all()


def test_missing_cap_keeps_iis():
    company, market = synthetic()
    company.loc[199, "market_cap"] = np.nan
    out, reasons = impacts.event_study(company, market)
    assert np.isnan(out["MIV"].iloc[200]) and not np.isnan(out["IIS"].iloc[200])
    assert reasons.iloc[200] == "missing_market_cap"


@pytest.mark.parametrize("z, iis", [
    (-3.5, -3), (-3.0, -3), (-2.999, -2), (-2.0, -2), (-1.999, -1), (-1.0, -1), (-0.999, 0),
    (0.0, 0), (0.999, 0), (1.0, 1), (1.999, 1), (2.0, 2), (2.999, 2), (3.0, 3), (7.0, 3),
])
def test_iis_boundaries(z, iis):
    assert impacts.iis_from_z([z])[0] == iis


def test_iis_nan():
    assert np.isnan(impacts.iis_from_z([np.nan])[0])


def test_miv_example():
    # $20 billion pre-event cap and a +2.5% CAR -> +$500 million.
    company, market = synthetic()
    company.loc[199, "market_cap"] = 20e9
    out, _ = impacts.event_study(company, market)
    assert out["MIV"].iloc[200] == pytest.approx(20e9 * out["CAR"].iloc[200])
    assert 20e9 * 0.025 == pytest.approx(500e6)
