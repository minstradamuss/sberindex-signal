"""Fast causal experts and adaptive selection, preserving the initial ensemble.

This second, documented research iteration was motivated by development errors.
Its fitted weights still use only matured targets at every individual origin.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_panel,historical_fill
from .models import seasonal_experts
from .backtest import to_frame
from .metrics import summarize,paired_bootstrap

MEMBERS=["panel_shrink","seasonal_growth","ridge_panel","chronos_seasonal","catboost","lightgbm","pooled_yoy_damped","seasonal_ratio"]


def fast_experts(raw,H=12):
    y=historical_fill(raw);z=np.log(y);N,C,n=y.shape
    base=seasonal_experts(raw,H)
    pooled=base["panel_shrink"].copy();ratio=pooled.copy()
    if n>12:
        g=z[...,12:]-z[...,:-12]
        common=np.median(g,axis=0)
        local=np.median(g[...,-3:]-common[None,:,-3:],axis=-1)
        w=min(4,n-12)
        if w>1:
            x=np.arange(w);x=x-x.mean()
            slope=(common[:,-w:]@x)/(x@x)
            slope=np.clip(slope,-.025,.025)*.7
        else:slope=np.zeros(C)
        horizon=np.arange(1,H+1)
        projected=common[:,-1,None]+slope[:,None]*(1-.8**horizon)/.2
        growth=.85*local[...,None]+projected[None]
        pooled=base["seasonal_naive"]*np.exp(np.clip(growth,-.5,.8))
        # Blend local and peer seasonal *ratios* while anchoring the latest level.
        old_last=z[...,n-13]
        shape=np.log(base["seasonal_naive"])-old_last[...,None]
        panel_shape=np.median(shape,axis=0)
        season_ratio=.65*shape+.35*panel_shape[None]
        growth_delta=projected-common[:,-1,None]
        ratio=np.exp(z[...,-1:]+season_ratio+growth_delta[None])
    return {"pooled_yoy_damped":pooled,"seasonal_ratio":ratio}


def adaptive_frame(frame,history,origin,config):
    """Choose at most two experts by paired recent matured MAE; no future labels."""
    output=[];logs=[]
    for (cat,h),current in frame[frame.model.isin(MEMBERS)].groupby(["category","horizon"]):
        pivot=current.pivot(index="territory_id",columns="model",values="prediction")
        available=[m for m in MEMBERS if m in pivot and pivot[m].notna().all()]
        prior={"panel_shrink":.5,"pooled_yoy_damped":.5}
        weights={m:prior.get(m,0.) for m in available};reason="structural_prior"
        past=history[(history.category==cat)&(history.horizon==h)&(history.target_date<=origin)&history.model.isin(available)] if len(history) else history
        if len(past) and past.target_date.nunique()>=3:
            # Recency supports adjustment to the current growth regime; all labels
            # still belong to closed months. At least 3 independent time blocks.
            last_months=np.sort(past.target_date.unique())[-4:]
            past=past[past.target_date.isin(last_months)]
            paired=past.pivot_table(index=["territory_id","origin_date","target_date"],columns="model",values="absolute_error").reindex(columns=available).dropna()
            error=paired.mean().sort_values()
            if len(error)>=2:
                weights={m:0. for m in available};weights[error.index[0]]=.7;weights[error.index[1]]=.3
                reason="two_best_matured_recent_paired_mae"
        total=sum(weights.values())
        if total<=0:weights={m:1/len(available) for m in available}
        else:weights={m:w/total for m,w in weights.items()}
        pred=sum(pivot[m]*w for m,w in weights.items())
        base=current[current.model==available[0]].set_index("territory_id").loc[pred.index].reset_index().copy()
        base["model"]="signal_adaptive";base["prediction"]=pred.to_numpy()
        base["absolute_error"]=abs(base.actual-base.prediction)
        base["lower80"]=np.nan;base["upper80"]=np.nan
        output.append(base)
        logs.extend({"origin_date":str(origin.date()),"category":cat,"horizon":int(h),"model":m,"weight":w,"rule":reason} for m,w in weights.items())
    return pd.concat(output,ignore_index=True),logs


def refine(cfg,limit=None):
    out=Path(cfg["results_dir"]);panel=load_panel(Path(cfg["data_dir"])/"consumption.parquet",limit)
    history=[];logs=[];all_selected=[]
    for n in cfg["origins"]:
        path=out/f"origin_{n:02d}.parquet"
        if not path.exists():raise FileNotFoundError(f"Backtest incomplete: {path}")
        frame=pd.read_parquet(path)
        frame=frame[~frame.model.isin(["pooled_yoy_damped","seasonal_ratio","signal_adaptive"])]
        new=to_frame(fast_experts(panel.values[...,:n],cfg["max_horizon"]),panel,n,cfg["max_horizon"],cfg["test_start"])
        frame=pd.concat([frame,new],ignore_index=True)
        past=pd.concat(history,ignore_index=True) if history else pd.DataFrame()
        adaptive,ll=adaptive_frame(frame,past,panel.dates[n-1],cfg)
        frame=pd.concat([frame,adaptive],ignore_index=True)
        logs.extend(ll)
        columns=["territory_id","category","origin_date","target_date","horizon","model","absolute_error"]
        history.append(frame.loc[frame.model.isin(MEMBERS),columns].copy())
        all_selected.append(frame[frame.horizon.isin(cfg["horizons"])])
        frame.to_parquet(path,index=False,compression="zstd",compression_level=12)
        print(f"refined {panel.dates[n-1].date()}",flush=True)
    d=pd.concat(all_selected,ignore_index=True)
    for h,g in d.groupby("horizon"):
        g.to_parquet(out/f"predictions_h{h:02d}.parquet",index=False,compression="zstd",compression_level=12)
    summarize(d).to_csv(out/"metrics.csv",index=False)
    keys=["territory_id","category","origin_date","target_date","horizon"]
    common=d[d.model=="prophet_monthly"][keys].merge(d[d.model=="prophet_default"][keys],on=keys)
    summarize(d.merge(common,on=keys)).to_csv(out/"metrics_paired_prophet.csv",index=False)
    primary=d[(d.category==cfg["primary_category"])&(d.split=="test")]
    comparisons=[dict(paired_bootstrap(primary[primary.horizon==h],a,b),horizon=h)
                 for h in cfg["horizons"] for a in ["signal_ensemble","signal_adaptive"]
                 for b in ["prophet_default","prophet_monthly","seasonal_naive","seasonal_growth","signal_ensemble"] if a!=b]
    (out/"paired_comparisons.json").write_text(json.dumps(comparisons,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(logs).to_csv(out/"adaptive_weights.csv",index=False)
    (out/"research_iterations.json").write_text(json.dumps({"iteration_1":"original equal-prior/exponential ensemble; preserved as signal_ensemble",
       "iteration_2":"pooled damped annual-growth experts and past-only recent model selection",
       "motivation":"weak long-horizon performance seen in development targets through September 2024",
       "no_test_numeric_tuning":True,"caveat":"not a preregistered study; long-horizon results exploratory"},indent=2),encoding="utf-8")
    history.clear();all_selected.clear()
    delayed=[]
    for n in cfg["origins"]:
        f=pd.read_parquet(out/f"origin_{n:02d}.parquet",filters=[("category","==",cfg["primary_category"]),("horizon","in",[2,4,7])])
        f["horizon"]=f.horizon-1
        f["origin_date"]=f.origin_date+pd.offsets.MonthBegin(1)
        delayed.append(f)
    summarize(pd.concat(delayed,ignore_index=True)).to_csv(out/"release_lag_sensitivity.csv",index=False)
