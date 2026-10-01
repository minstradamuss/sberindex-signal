"""Validate distributed evidence without retraining models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from signalcast.metrics import summarize
from signalcast.reporting import read_predictions


def main():
    out=Path("results/full")
    manifest=json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    actual=hashlib.sha256(Path("data/raw/consumption.parquet").read_bytes()).hexdigest()
    assert actual==manifest["consumption_sha256"],"Data checksum mismatch"
    run=json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(Path("data/external/news.json").read_bytes()).hexdigest()==run["news_sha256"]
    d=read_predictions(out)
    keys=["territory_id","category","origin_date","target_date","horizon","model"]
    assert not d.duplicated(keys).any(),"Duplicate forecasts"
    assert np.isfinite(d.actual).all() and np.isfinite(d.prediction).all()
    assert (d.prediction>0).all()
    origin=pd.to_datetime(d.origin_date);target=pd.to_datetime(d.target_date)
    delta=(target.dt.year-origin.dt.year)*12+target.dt.month-origin.dt.month
    assert (delta==d.horizon).all(),"Horizon mismatch"
    assert set(d.horizon)=={1,3,6,12}
    calc=summarize(d).sort_values(["split","category","horizon","model"]).reset_index(drop=True)
    saved=pd.read_csv(out/"metrics.csv").sort_values(["split","category","horizon","model"]).reset_index(drop=True)
    np.testing.assert_allclose(calc[["mae","r2","wape"]],saved[["mae","r2","wape"]],rtol=1e-8,atol=1e-8)
    assert (calc.n==saved.n).all()
    pair_keys=keys[:-1]
    common=d[d.model=="prophet_default"][pair_keys].merge(d[d.model=="prophet_monthly"][pair_keys],on=pair_keys)
    paired=summarize(d.merge(common,on=pair_keys)).sort_values(["split","category","horizon","model"]).reset_index(drop=True)
    paired_saved=pd.read_csv(out/"metrics_paired_prophet.csv").sort_values(["split","category","horizon","model"]).reset_index(drop=True)
    np.testing.assert_allclose(paired[["mae","r2","wape"]],paired_saved[["mae","r2","wape"]],rtol=1e-8,atol=1e-8)
    assert (paired.n==paired_saved.n).all()
    fail=json.loads((out/"failures.json").read_text(encoding="utf-8"))
    print(f"Verified {len(d):,} forecasts, {len(calc)} metric rows; model failures: {len(fail)}")
    for filename in ["ensemble_weights.csv","adaptive_weights.csv"]:
        weights=pd.read_csv(out/filename)
        assert not weights.duplicated(["origin_date","category","horizon","model"]).any()
        assert (weights.weight>=0).all()
        np.testing.assert_allclose(weights.groupby(["origin_date","category","horizon"]).weight.sum(),1,atol=1e-8)
    partitions=pd.read_csv(out/"detection_split.csv")
    assert not partitions.territory_id.duplicated().any()
    assert set(partitions.partition)=={"calibration","evaluation"}
    print("Weights and detector partitions verified")


if __name__=="__main__":main()
