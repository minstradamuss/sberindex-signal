"""Metrics in original rubles, paired comparisons and month-cluster uncertainty."""
import numpy as np
import pandas as pd


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "wape": np.nan}
    y, p = y[valid], p[valid]
    e = y - p
    sst = np.sum((y - y.mean())**2)
    return {"n": len(y), "mae": float(np.mean(abs(e))), "rmse": float(np.sqrt(np.mean(e**2))),
            "r2": float(1 - np.sum(e**2) / sst) if sst > 0 else np.nan,
            "wape": float(np.sum(abs(e)) / np.sum(abs(y)))}


def summarize(predictions, by=("split", "category", "horizon", "model")):
    rows = []
    for key, g in predictions.groupby(list(by), observed=True):
        rows.append(dict(zip(by, key), **scores(g.actual, g.prediction),
                         target_months=g.target_date.nunique(), origins=g.origin_date.nunique()))
    return pd.DataFrame(rows)


def paired_bootstrap(frame, candidate, baseline, seed=20261001, n_boot=2000):
    keys = ["territory_id", "category", "origin_date", "target_date", "horizon"]
    a = frame[frame.model == candidate].set_index(keys)
    b = frame[frame.model == baseline].set_index(keys)
    x = a[["actual", "prediction"]].join(b.prediction.rename("baseline"), how="inner").dropna()
    x["difference"] = abs(x.actual - x.prediction) - abs(x.actual - x.baseline)
    monthly = x.groupby("target_date").difference.agg(["sum", "count"])
    if len(monthly) < 2:
        return {"candidate": candidate, "baseline": baseline, "paired_n": len(x),
                "target_months": len(monthly), "ci95": None, "reason": "insufficient independent time blocks"}
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(monthly), (n_boot, len(monthly)))
    samples = monthly["sum"].to_numpy()[ids].sum(1) / monthly["count"].to_numpy()[ids].sum(1)
    by_id = x.groupby("territory_id").difference.mean()
    return {"candidate": candidate, "baseline": baseline, "paired_n": len(x),
            "target_months": len(monthly), "mae_difference": float(x.difference.mean()),
            "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "territory_win_rate": float((by_id < 0).mean()),
            "caution": "few target-month clusters; descriptive interval, not proof of generalization"}
