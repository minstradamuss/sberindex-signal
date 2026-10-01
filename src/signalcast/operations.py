"""An auditable historical deployment snapshot, without future outcomes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import load_panel
from .detection import relative_log
from .news import News


def snapshot(cfg):
    out=Path(cfg["results_dir"])
    panel=load_panel(Path(cfg["data_dir"])/"consumption.parquet")
    category=cfg["primary_category"];ci=panel.categories.index(category)
    observed=np.isfinite(panel.values[:,ci,:]).all(1)
    y=panel.values[observed,ci,:];ids=panel.ids[observed]
    q=relative_log(y);changes=np.diff(q,axis=1);news=News("data/external/news.json")
    cutoff=float(np.quantile(abs(changes[:,:11]),.95))
    def features(t):
        history=np.column_stack([changes[:,t-1],abs(changes[:,t-1]),
            np.mean(changes[:,t-3:t],axis=1),np.std(changes[:,t-6:t],axis=1),
            q[:,t]-np.median(q[:,t-5:t],axis=1)])
        return np.column_stack([history,np.tile(list(news.at(panel.dates[t]).values()),(len(y),1))])
    last=len(panel.dates)-1
    X=np.concatenate([features(t) for t in range(12,last)])
    target=np.concatenate([(abs(changes[:,t])>cutoff).astype(int) for t in range(12,last)])
    future=features(last)
    risk=pd.DataFrame({"territory_id":ids})
    for name,width in [("history",5),("news",X.shape[1])]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=300,random_state=cfg["seed"]))
        model.fit(X[:,:width],target)
        risk[f"next_month_proxy_probability_{name}"]=model.predict_proba(future[:,:width])[:,1]
    forecasts=pd.read_parquet(out/"future_forecasts.parquet")
    forecasts=forecasts[(forecasts.model=="signal_ensemble")&(forecasts.category==category)]
    table=forecasts.pivot(index="territory_id",columns="horizon",values="prediction").rename(columns=lambda h:f"forecast_h{h:02d}").reset_index()
    table=table.merge(risk,on="territory_id",how="left")
    table["origin_month"]=str(panel.dates[-1].date())
    actual=dict(zip(panel.ids,panel.values[:,ci,-1]))
    table["last_observed_spending"]=table.territory_id.map(actual)
    selected_path=out/"selected_detector.json"
    alert_path=out/"real_alerts.csv"
    if selected_path.exists() and alert_path.exists():
        chosen=json.loads(selected_path.read_text(encoding="utf-8"))["model"]
        alerts=pd.read_csv(alert_path)
        latest=alerts[(alerts.detector==chosen)&(pd.to_datetime(alerts.month)==panel.dates[-1])]
        table["selected_detector"]=chosen
        table["alert_in_last_observed_month"]=table.territory_id.isin(latest.territory_id)
    table.to_csv(out/"operational_snapshot.csv",index=False)
    (out/"operational_snapshot_metadata.json").write_text(json.dumps({
        "origin_month":str(panel.dates[-1].date()),"category":category,
        "forecast_model":"signal_ensemble",
        "status":"historical deployment example, not a current 2026 forecast",
        "probability_target":"large relative innovation next month, not verified economic shock",
        "proxy_threshold_log_change":cutoff,
        "probability_missing":"risk model is restricted to complete historical series",
        "future_actuals_available":False},ensure_ascii=False,indent=2),encoding="utf-8")
