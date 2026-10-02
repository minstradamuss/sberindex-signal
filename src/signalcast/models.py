"""Forecasting experts. Each entry point receives only a historical prefix."""
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .data import historical_fill


def robust_slope(z):
    """Median of three-month slopes, damped and bounded for short histories."""
    if z.shape[-1] < 4:
        return np.zeros(z.shape[:-1])
    return np.clip(np.median((z[..., 3:] - z[..., :-3]) / 3, axis=-1), -.025, .04)


def seasonal_experts(raw, H=12):
    y = historical_fill(raw)
    z = np.log(y)
    N, C, n = y.shape
    hh = np.arange(1, H + 1)
    naive = np.repeat(y[..., -1:], H, axis=-1)
    seasonal = np.stack([y[..., (n + h - 1 - 12) % n] if n >= 12 else y[..., -1]
                         for h in hh], axis=-1)
    # No second year yet: trend is not identifiable separately from annual seasonality.
    # An explicit, conservative extrapolation is preferable to hidden future information.
    if n > 12:
        growth_hist = z[..., 12:] - z[..., :-12]
        recent = growth_hist[..., -min(3, n - 12):]
        growth = np.median(recent, axis=-1)
        panel_g = np.median(recent, axis=(0, 2))
        v = np.mean((recent - growth[..., None])**2, axis=-1)
        cross = np.median((growth - np.median(growth, axis=0))**2, axis=0)
        reliability = np.clip(cross / (cross + v / recent.shape[-1] + .001), .15, .85)
        shrunk_growth = reliability * growth + (1 - reliability) * panel_g
    else:
        # Exclude the final December peak from a first-year trend estimate.
        short = z[..., :11] if n == 12 else z
        annual = robust_slope(short) * 12
        panel_g = np.clip(np.median(annual, axis=0), -.05, .25)
        growth = np.clip(.5 * annual + .5 * panel_g, -.15, .4)
        shrunk_growth = .25 * growth + .75 * panel_g
    growth = np.clip(growth, -.5, .8)
    shrunk_growth = np.clip(shrunk_growth, -.4, .7)
    sg = seasonal * np.exp(growth[..., None])
    # Shrink the seasonal *shape*, not the spending level, to a robust panel shape.
    # Median log shape prevents rich territories dominating common seasonality.
    if n >= 12:
        last_year = z[..., -12:]
        centered = last_year - np.mean(last_year, axis=-1, keepdims=True)
        common = np.median(centered, axis=0)
        pos = (np.arange(H) % 12)
        pooled_season = np.exp(np.mean(last_year, axis=-1)[..., None] + common[:, pos][None])
        seasonal_stable = np.exp(.75 * np.log(seasonal) + .25 * np.log(pooled_season))
    else:
        seasonal_stable = seasonal
    shrunk = seasonal_stable * np.exp(shrunk_growth[..., None])
    return {"naive": naive, "seasonal_naive": seasonal, "seasonal_growth": sg,
            "panel_shrink": shrunk,
            "damped_trend": naive * np.exp(robust_slope(z)[..., None] * (1 - .85**hh) / .15)}


def ridge_panel(raw, H):
    """Partial pooling: panel calendar + shared trend and shrunken local deviation."""
    z = np.log(historical_fill(raw))
    N, C, n = z.shape
    t = np.arange(n + H)
    # Penalized month effects, linear trend; a ridge prior regularizes the 1-year case.
    X = np.column_stack([t / 12, np.eye(12)[t % 12]])
    pred = np.zeros((N, C, H))
    for c in range(C):
        center = np.mean(z[:, c, :], axis=1)
        f = np.median(z[:, c, :] - center[:, None], axis=0)
        m = Ridge(alpha=.35).fit(X[:n], f)
        common = m.predict(X[n:])
        common_past = m.predict(X[:n])
        resid = z[:, c, :] - common_past
        anchor = np.median(resid[:, -3:], axis=1)
        slope = robust_slope(resid[:, -6:]) * .35
        path = anchor[:, None] + common + slope[:, None] * ((1 - .8**np.arange(1, H + 1)) / .2)
        pred[:, c, :] = np.exp(path)
    return pred


def feature_frame(raw, H, event_features=None):
    y = historical_fill(raw)
    z = np.log(y)
    N, C, n = y.shape
    base = seasonal_experts(raw, H)["panel_shrink"]
    flat = lambda a: np.broadcast_to(a, (N, C, H)).reshape(-1)
    features = {"category": flat(np.arange(C)[None, :, None]),
                "horizon": flat(np.arange(1, H + 1)[None, None, :]),
                "month": flat(((n + np.arange(H)) % 12 + 1)[None, None, :]),
                "history_length": np.full(N*C*H, n),
                "log_base": np.log(base).reshape(-1),
                "log_level": flat(z[..., -1:]),
                "base_change": (np.log(base) - z[..., -1:]).reshape(-1)}
    for lag in [1, 2, 3, 6, 12]:
        delta = z[..., -1] - z[..., max(0, n - 1 - lag)]
        features[f"change_{lag}"] = flat(delta[..., None])
        panel = np.median(delta, axis=0)
        features[f"panel_change_{lag}"] = flat(panel[None, :, None])
        features[f"relative_change_{lag}"] = flat((delta - panel)[..., None])
    for w in [3, 6]:
        features[f"std_{w}"] = flat(np.std(np.diff(z[..., -min(w+1,n):], axis=-1), axis=-1)[...,None])
        features[f"slope_{w}"] = flat(robust_slope(z[..., -min(w,n):])[..., None])
    # All category covariates refer to the SAME origin, never to the forecast month.
    for c in range(C):
        share = z[:, c, -1] - z[:, 0, -1]  # sorted category 0 is all categories
        features[f"category_ratio_{c}"] = flat(share[:, None, None])
        mom = z[:, c, -1] - z[:, c, max(0, n-2)]
        features[f"category_mom_{c}"] = flat(mom[:, None, None])
    if event_features is not None:
        for key, value in event_features.items():
            features[f"news_{key}"] = np.full(N*C*H, value)
    return pd.DataFrame(features).astype(np.float32), base


def boosting(raw, H, cfg, events=None, dates=None):
    """Direct multi-horizon residual learners, with labels <= forecast origin."""
    n = raw.shape[-1]
    Xs, ys, ws = [], [], []
    for k in range(cfg["models"]["min_training_origin"], n):
        horizon = min(H, n - k)
        ef = events.at(dates[k-1]) if events is not None else None
        X, base = feature_frame(raw[..., :k], horizon, ef)
        target = raw[..., k:k+horizon].reshape(-1)
        mask = np.isfinite(target)
        Xs.append(X.loc[mask])
        ys.append(np.log(target[mask] / base.reshape(-1)[mask]))
        # Ruble-weighted absolute log loss approximates absolute ruble error locally.
        ws.append(np.minimum(base.reshape(-1)[mask], 80000) / 10000)
    X = pd.concat(Xs, ignore_index=True)
    target, weights = np.concatenate(ys), np.concatenate(ws)
    if len(X) > cfg["models"]["max_training_rows"]:
        ids = np.random.default_rng(cfg["seed"] + n).choice(len(X), cfg["models"]["max_training_rows"], False)
        X, target, weights = X.iloc[ids], target[ids], weights[ids]
    ef = events.at(dates[n-1]) if events is not None else None
    future, base = feature_frame(raw, H, ef)
    out, importance = {}, []
    args = cfg["models"]
    no_news = [c for c in X.columns if not c.startswith("news_")]
    if args.get("catboost"):
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=args["boosting_iterations"], depth=args["depth"],
                             learning_rate=args["learning_rate"], loss_function="MAE", l2_leaf_reg=10,
                             thread_count=cfg["threads"], random_seed=cfg["seed"],
                             verbose=False, allow_writing_files=False)
        m.fit(X[no_news], target, sample_weight=weights)
        out["catboost"] = base * np.exp(np.clip(m.predict(future[no_news]), -.5, .5).reshape(base.shape))
        importance.extend([{"model":"catboost", "feature":f, "importance":float(v)}
                           for f,v in zip(no_news, m.feature_importances_)])
    if args.get("lightgbm"):
        from lightgbm import LGBMRegressor
        for name, cols in [("lightgbm", no_news)] + ([("lightgbm_news", list(X.columns))] if events else []):
            m = LGBMRegressor(objective="regression_l1", n_estimators=args["boosting_iterations"],
                              learning_rate=args["learning_rate"], num_leaves=15, max_depth=5,
                              min_child_samples=150, reg_lambda=10, n_jobs=cfg["threads"],
                              random_state=cfg["seed"], verbosity=-1, deterministic=True, force_col_wise=True)
            m.fit(X[cols], target, sample_weight=weights)
            out[name] = base * np.exp(np.clip(m.predict(future[cols]), -.5, .5).reshape(base.shape))
            importance.extend([{"model":name,"feature":f,"importance":int(v)} for f,v in zip(cols,m.feature_importances_)])
    return out, importance


def prophet_forecasts(raw, dates, H, primary=0, threads=8, indices=None):
    """Two explicit Prophet baselines on primary target; no synthetic replacement."""
    from prophet import Prophet
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").setLevel(logging.ERROR)
    N, C, n = raw.shape
    future = pd.DataFrame({"ds":pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")})
    out = {k:np.full((N,C,H),np.nan) for k in ["prophet_default", "prophet_monthly"]}
    failures = []
    def one(i):
        y = raw[i, primary]
        ok = np.isfinite(y)
        if ok.sum() < 6:
            return i, {}, "fewer than six observations"
        train = pd.DataFrame({"ds":dates[ok],"y":y[ok]})
        p = {}
        for name in out:
            try:
                m = Prophet(weekly_seasonality=False, daily_seasonality=False, uncertainty_samples=0,
                            **({} if name=="prophet_default" else
                               {"yearly_seasonality":3,"seasonality_mode":"multiplicative","changepoint_prior_scale":.01,"seasonality_prior_scale":1.0}))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(train)
                p[name] = np.maximum(1.0, m.predict(future).yhat.to_numpy())
            except Exception as e:
                failures.append({"territory_index":int(i), "model":name, "error":str(e)})
        return i, p, None
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, p, error in pool.map(one, range(N) if indices is None else indices):
            for name, v in p.items(): out[name][i, primary] = v
            if error: failures.append({"territory_index":int(i),"error":error})
    return out, failures


class Foundation:
    def __init__(self, model_id, cache_dir, threads=8, revision=None):
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(threads)
        self.torch = torch
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32,
                                                          cache_dir=str(cache_dir), revision=revision)
        self.revision = revision or getattr(self.pipeline.model.config, "_commit_hash", None)

    def predict(self, raw, H, batch_size=256):
        y = historical_fill(raw)
        N,C,n = y.shape
        def infer(arr):
            result=[]
            for i in range(0,len(arr),batch_size):
                q = self.pipeline.predict(self.torch.tensor(arr[i:i+batch_size],dtype=self.torch.float32), prediction_length=H)
                result.append(q[:,q.shape[1]//2,:].numpy())
            return np.concatenate(result).reshape(N,C,H)
        raw_prediction = np.maximum(1, infer(y.reshape(N*C,n)))
        # Foundation forecasts the nonseasonal component; only the first historical
        # annual cycle defines the template, hence no future seasonal decomposition.
        z=np.log(y)
        if n >= 12:
            first=z[...,:12]
            slope=robust_slope(first[...,:11])
            template=first-slope[...,None]*np.arange(12)
            template-=template.mean(-1,keepdims=True)
            template=.5*template+.5*np.median(template,axis=0)[None]
            residual=z-template[...,np.arange(n)%12]
            residual_prediction=infer(residual.reshape(N*C,n))
            corrected=np.exp(residual_prediction+template[...,(n+np.arange(H))%12])
        else:
            corrected=raw_prediction.copy()
        return {"chronos_raw":raw_prediction, "chronos_seasonal":corrected}
