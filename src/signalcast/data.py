"""Data validation. Never interpolate using future observations."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SOURCE_URL = "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


JSON_HASH_FORMAT = "canonical-json-v1"


def json_sha256(path):
    """Hash JSON content independently of key order, spacing and line endings."""
    content = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class Panel:
    values: np.ndarray  # territory x category x calendar month
    ids: np.ndarray
    categories: list
    dates: pd.DatetimeIndex

    def prefix(self, n):
        return Panel(self.values[:, :, :n].copy(), self.ids.copy(), self.categories.copy(), self.dates[:n])


def load_panel(path, limit=None):
    d = pd.read_parquet(path)
    d = d.rename(columns={"consumption": "value"})
    needed = {"territory_id", "category", "date", "value"}
    if not needed.issubset(d.columns):
        raise ValueError(f"Missing columns: {needed - set(d.columns)}")
    d["category"] = d.category.str.strip()
    d["date"] = pd.to_datetime(d.date).dt.to_period("M").dt.to_timestamp()
    if d.duplicated(["territory_id", "category", "date"]).any():
        raise ValueError("Duplicate (territory, category, month) keys")
    if not np.isfinite(d.value).all() or (d.value <= 0).any():
        raise ValueError("Expected finite, positive spending; missing rows remain missing")
    ids = np.sort(d.territory_id.unique())
    if limit:
        ids = ids[:limit]
    cats = sorted(d.category.unique())
    dates = pd.date_range(d.date.min(), d.date.max(), freq="MS")
    index = pd.MultiIndex.from_product([ids, cats, dates], names=["territory_id", "category", "date"])
    a = d.set_index(["territory_id", "category", "date"]).value.reindex(index)
    return Panel(a.to_numpy().reshape(len(ids), len(cats), len(dates)), ids, cats, dates)


def historical_fill(a):
    """Past-only forward fill; leading gaps use the same month's panel median.

    Used for model inputs only. Evaluation always uses the original observation mask.
    """
    a = a.copy()
    for t in range(a.shape[-1]):
        missing = ~np.isfinite(a[:, :, t])
        if t:
            a[:, :, t] = np.where(missing, a[:, :, t - 1], a[:, :, t])
        med = np.nanmedian(a[:, :, t], axis=0)
        med = np.where(np.isfinite(med), med, 1.0)
        a[:, :, t] = np.where(np.isfinite(a[:, :, t]), a[:, :, t], med)
    return a


def write_audit(panel, input_path, output):
    x = panel.values
    audit = {"rows_observed": int(np.isfinite(x).sum()), "municipalities": len(panel.ids),
             "categories": panel.categories, "months": len(panel.dates),
             "start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
             "complete_territories": int(np.isfinite(x).all(axis=(1, 2)).sum()),
             "missing_cells": int(np.isnan(x).sum()), "sha256": sha256(input_path),
             "source_url": SOURCE_URL, "license": "CC BY-SA 4.0",
             "target": "mean cashless consumer spending in current rubles; not aggregate turnover",
             "excluded_covariates": "2024 market access and 2024-12-31 road distances: no historical vintage"}
    Path(output).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
