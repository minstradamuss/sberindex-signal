"""Prequential model weights and residual intervals. Only matured errors are used."""
import numpy as np
import pandas as pd


def weights_from_history(history, members, category, horizon, origin_date, cfg):
    # Strict date filter is intentional, even when caller has already sliced history.
    g=history[(history.target_date<=pd.Timestamp(origin_date)) &
              (history.category==category) & (history.horizon==horizon) & history.model.isin(members)] if len(history) else history
    default=np.ones(len(members))/len(members)
    if not len(g) or g.target_date.nunique()<cfg["min_resolved_months"]:
        return default,"equal_prior"
    pivot=g.pivot_table(index=["territory_id","origin_date","target_date"],columns="model",values="absolute_error")
    pivot=pivot.reindex(columns=members).dropna()
    if len(pivot)==0:return default,"equal_prior_no_common_pairs"
    error=pivot.mean().to_numpy()
    scale=max(float(np.median(error)),1.)
    w=np.exp(-(error-error.min())/(scale*cfg["temperature"]))
    w=w/w.sum()
    floor=cfg["floor_weight"]
    w=(1-floor*len(members))*w+floor
    return w,"matured_paired_errors"


def add_ensemble(forecasts, panel, origin, history, cfg):
    n,c,h=next(iter(forecasts.values())).shape
    members=[m for m in cfg["ensemble"]["members"] if m in forecasts]
    if not members:raise ValueError("No ensemble members available")
    result=np.zeros((n,c,h));logs=[]
    for j,category in enumerate(panel.categories):
        for k in range(h):
            w,reason=weights_from_history(history,members,category,k+1,panel.dates[origin-1],cfg["ensemble"])
            result[:,j,k]=sum(wi*forecasts[m][:,j,k] for m,wi in zip(members,w))
            logs.extend({"origin_date":str(panel.dates[origin-1].date()),"category":category,"horizon":k+1,
                         "model":m,"weight":float(wi),"rule":reason} for m,wi in zip(members,w))
    forecasts["signal_ensemble"]=result
    return logs


def residual_interval(history, category, horizon, origin_date, prediction, alpha=.2):
    g=history[(history.model=="signal_ensemble") & (history.category==category) &
              (history.horizon==horizon) & (history.target_date<=pd.Timestamp(origin_date))] if len(history) else history
    if not len(g) or g.target_date.nunique()<3:
        return np.full_like(prediction,np.nan),np.full_like(prediction,np.nan)
    # Scaled cross-sectional residual interval: empirical, NOT exchangeability-guaranteed.
    residual=np.abs(g.actual-g.prediction)/np.maximum(g.prediction,1)
    q=float(np.quantile(residual,1-alpha,method="higher"))
    return np.maximum(0,prediction*(1-q)),prediction*(1+q)
