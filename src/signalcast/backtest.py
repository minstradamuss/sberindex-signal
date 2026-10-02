import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .data import load_panel, write_audit, sha256, json_sha256, JSON_HASH_FORMAT
from .models import seasonal_experts, ridge_panel, boosting, prophet_forecasts, Foundation
from .ensemble import add_ensemble, residual_interval
from .metrics import summarize, paired_bootstrap
from .news import News


def compact_history(frame,cfg):
    keep=list(cfg["ensemble"]["members"])+["signal_ensemble","pooled_yoy_damped","seasonal_ratio"]
    columns=["territory_id","category","origin_date","target_date","horizon","model","actual","prediction","absolute_error"]
    return frame.loc[frame.model.isin(keep),columns].copy()


def to_frame(forecasts, panel, n, H, test_start):
    N,C,T=panel.values.shape
    rows=[]
    eligible=(np.isfinite(panel.values[...,:n]).sum(-1)>=6)&np.isfinite(panel.values[...,n-1])
    for model,arr in forecasts.items():
        for h in range(1,min(H,T-n)+1):
            actual=panel.values[...,n+h-1]
            ok=eligible&np.isfinite(actual)&np.isfinite(arr[...,h-1])
            i,j=np.where(ok)
            rows.append(pd.DataFrame({"territory_id":panel.ids[i],"category":np.array(panel.categories)[j],
                                     "origin_date":panel.dates[n-1],"target_date":panel.dates[n+h-1],
                                     "horizon":h,"model":model,"actual":actual[i,j],"prediction":arr[i,j,h-1],
                                     "split":"test" if panel.dates[n+h-1]>=pd.Timestamp(test_start) else "development"}))
    d=pd.concat(rows,ignore_index=True)
    d["absolute_error"]=abs(d.actual-d.prediction)
    return d


def run(cfg, limit=None, forecast_only=False):
    out=Path(cfg["results_dir"]);out.mkdir(parents=True,exist_ok=True)
    path=Path(cfg["data_dir"])/"consumption.parquet"
    panel=load_panel(path,limit)
    news_path=Path("data/external/news.json")
    events=News(news_path) if news_path.exists() else None
    foundation=None
    if cfg["models"].get("chronos"):
        foundation=Foundation(cfg["models"]["chronos_id"],Path(".cache/huggingface"),cfg["threads"],cfg["models"].get("chronos_revision"))
    history=[]; timings=[]; weight_logs=[]; importance=[]; failures=[]
    for filename,container in [("timings.csv",timings),("ensemble_weights.csv",weight_logs),("feature_importance.csv",importance)]:
        f=out/filename
        if f.exists() and f.stat().st_size>1:
            container.extend(pd.read_csv(f).to_dict("records"))
    if (out/"failures.json").exists():
        failures=json.loads((out/"failures.json").read_text(encoding="utf-8"))
    if forecast_only:
        history=[compact_history(pd.read_parquet(f),cfg) for f in sorted(out.glob("origin_*.parquet"))]
    # A fixed random ID sample is chosen without looking at future data or errors.
    # Every comparison to Prophet is paired on this sample and its observed targets.
    max_prophet=cfg["models"].get("prophet_max_territories",len(panel.ids))
    prophet_ids=np.sort(np.random.default_rng(cfg["seed"]).choice(len(panel.ids),min(max_prophet,len(panel.ids)),replace=False))
    origins=[len(panel.dates)] if forecast_only else cfg["origins"]
    manifest={"python":platform.python_version(),"platform":platform.platform(),
              "data_sha256":sha256(path),"foundation_revision":foundation.revision if foundation else None,
              "seed":cfg["seed"],"limit":limit,"news_sha256":json_sha256(news_path) if events else None,
              "news_hash_format":JSON_HASH_FORMAT}
    if (out/"run_manifest.json").exists():
        previous=json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))
        for key in ["data_sha256","news_sha256","news_hash_format","limit","foundation_revision","seed"]:
            if previous.get(key)!=manifest.get(key):
                raise ValueError(f"Run inputs changed: {key}. Select a new results_dir.")
    else:
        (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    write_audit(panel,path,out/"data_audit.json")
    (out/"config.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    pd.DataFrame({"territory_id":panel.ids[prophet_ids]}).to_csv(out/"prophet_sample.csv",index=False)
    for n in origins:
        origin=panel.dates[n-1]
        checkpoint=out/f"origin_{n:02d}.parquet"
        if checkpoint.exists() and not forecast_only:
            cached=pd.read_parquet(checkpoint)
            # A checkpoint belongs to the saved run configuration. Resumption is safe
            # only when the caller keeps the configuration fixed (CLI enforces this).
            history.append(compact_history(cached,cfg)); print(f"resume {origin.date()}",flush=True); continue
        # Recomputing a deployment snapshot replaces its audit rows instead of
        # appending duplicate weights or misleading duplicate timing records.
        for records in [timings,weight_logs,importance,failures]:
            records[:]=[r for r in records if str(r.get("origin_date"))!=str(origin.date())]
        start=time.time(); raw=panel.values[...,:n].copy(); H=cfg["max_horizon"]
        hist=pd.concat(history,ignore_index=True) if history else pd.DataFrame()
        pred=seasonal_experts(raw,H)
        pred["ridge_panel"]=ridge_panel(raw,H)
        extra,imp=boosting(raw,H,cfg,events,panel.dates[:n]);pred.update(extra)
        importance.extend(dict(x,origin_date=str(origin.date())) for x in imp)
        if foundation:pred.update(foundation.predict(raw,H,cfg["models"]["chronos_batch"]))
        if cfg["models"].get("prophet"):
            print(f"Prophet {origin.date()} ({len(prophet_ids)} sampled territories)",flush=True)
            extra,fail=prophet_forecasts(raw,panel.dates[:n],H,panel.categories.index(cfg["primary_category"]),cfg["threads"],prophet_ids)
            pred.update(extra); failures.extend(dict(x,origin_date=str(origin.date())) for x in fail)
        weight_logs.extend(add_ensemble(pred,panel,n,hist,cfg))
        if forecast_only:
            from .refinement import fast_experts,adaptive_frame
            pred.update(fast_experts(raw,H))
            rows=[]
            eligible=(np.isfinite(raw).sum(-1)>=6)&np.isfinite(raw[...,-1])
            for model,a in pred.items():
                for h in cfg["horizons"]:
                    i,j=np.where(np.isfinite(a[...,h-1])&eligible)
                    rows.append(pd.DataFrame({"territory_id":panel.ids[i],"category":np.array(panel.categories)[j],
                                             "origin_date":origin,"target_date":origin+pd.DateOffset(months=h),
                                             "horizon":h,"model":model,"prediction":a[i,j,h-1]}))
            future_frame=pd.concat(rows,ignore_index=True)
            future_frame["actual"]=np.nan;future_frame["absolute_error"]=np.nan
            adaptive,ll=adaptive_frame(future_frame,hist,origin,cfg)
            future_frame=pd.concat([future_frame,adaptive],ignore_index=True)
            future_frame.to_parquet(out/"future_forecasts.parquet",index=False,compression="zstd")
        else:
            frame=to_frame(pred,panel,n,H,cfg["test_start"])
            frame["lower80"]=np.nan;frame["upper80"]=np.nan
            for category in panel.categories:
                for h in cfg["horizons"]:
                    mask=(frame.model=="signal_ensemble")&(frame.category==category)&(frame.horizon==h)
                    lo,hi=residual_interval(hist,category,h,origin,frame.loc[mask,"prediction"].to_numpy())
                    frame.loc[mask,"lower80"]=lo;frame.loc[mask,"upper80"]=hi
            frame.to_parquet(checkpoint,index=False,compression="zstd",compression_level=9)
            history.append(compact_history(frame,cfg))
        elapsed=time.time()-start
        timings.append({"origin_date":str(origin.date()),"seconds":elapsed})
        print(f"completed {origin.date()}: {elapsed:.1f}s",flush=True)
        pd.DataFrame(weight_logs).to_csv(out/"ensemble_weights.csv",index=False)
        pd.DataFrame(importance).to_csv(out/"feature_importance.csv",index=False)
        (out/"failures.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(timings).to_csv(out/"timings.csv",index=False)
    if forecast_only:return
    history.clear()
    selected=pd.concat([pd.read_parquet(out/f"origin_{n:02d}.parquet",filters=[("horizon","in",cfg["horizons"])])
                        for n in origins],ignore_index=True)
    for h,g in selected.groupby("horizon"):
        g.to_parquet(out/f"predictions_h{h:02d}.parquet",index=False,compression="zstd",compression_level=9)
    summary=summarize(selected);summary.to_csv(out/"metrics.csv",index=False)
    if "prophet_monthly" in selected.model.unique():
        keys=["territory_id","category","origin_date","target_date","horizon"]
        common=selected[selected.model=="prophet_monthly"][keys].merge(selected[selected.model=="prophet_default"][keys],on=keys)
        paired=selected.merge(common,on=keys)
        summarize(paired).to_csv(out/"metrics_paired_prophet.csv",index=False)
    primary=selected[(selected.category==cfg["primary_category"])&(selected.split=="test")]
    comparisons=[dict(paired_bootstrap(primary[primary.horizon==h],"signal_ensemble",b),horizon=h)
                 for h in cfg["horizons"] for b in ["prophet_default","prophet_monthly","seasonal_naive","seasonal_growth"]
                 if b in primary.model.unique()]
    (out/"paired_comparisons.json").write_text(json.dumps(comparisons,ensure_ascii=False,indent=2),encoding="utf-8")
    covered=selected[(selected.model=="signal_ensemble")&selected.lower80.notna()].copy()
    if len(covered):
        covered["covered"]=(covered.actual>=covered.lower80)&(covered.actual<=covered.upper80)
        covered.groupby(["split","category","horizon"]).agg(coverage=("covered","mean"),n=("covered","size")).to_csv(out/"interval_coverage.csv")
    print(summary[(summary.category==cfg["primary_category"])&(summary.split=="test")].to_string(index=False),flush=True)
