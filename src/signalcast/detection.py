"""Causal break monitoring, independent semi-synthetic validation, and early risk.

Real data do not contain ground-truth structural-break annotations. Injected events
measure controlled sensitivity; next-month innovations are an explicitly labelled
proxy task, never represented as validated economic shocks.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from .data import load_panel
from .news import News


def relative_log(y):
    z=np.log(y)
    return z-np.median(z,axis=0,keepdims=True)


def score_detectors(y):
    """N x T matrix -> N x T scores; score at t uses only observations <=t."""
    q=relative_log(y)
    N,T=q.shape
    out={k:np.zeros((N,T)) for k in ["robust_innovation","discounted_cusum","window_glr","bayesian_runlength"]}
    inc=np.diff(q,axis=1)
    # BOCPD Gaussian observation likelihood with a conjugate unknown-mean prior.
    # Fixed noise is estimated from the first six historical observations only.
    initial=np.maximum(1.4826*np.median(abs(inc[:,:5]-np.median(inc[:,:5],axis=1)[:,None]),axis=1),.015)
    norm=(q-q[:,:6].mean(1)[:,None])/initial[:,None]
    logp=np.zeros((N,1));mu=np.zeros((N,1));kappa=np.ones((N,1))*.25
    plus=np.zeros(N);minus=np.zeros(N)
    for t in range(6,T):
        past=inc[:,:t-1]
        center=np.median(past[:,-6:],axis=1)
        local=1.4826*np.median(abs(past-center[:,None]),axis=1)
        pooled=np.median(local)
        sigma=np.maximum(np.sqrt(.5*local**2+.5*pooled**2),.008)
        innovation=(inc[:,t-1]-center)/sigma
        out["robust_innovation"][:,t]=abs(innovation)
        plus=np.maximum(0,.8*plus+innovation-.5)
        minus=np.maximum(0,.8*minus-innovation-.5)
        out["discounted_cusum"][:,t]=np.maximum(plus,minus)
        candidates=[]
        for w in [1,2,3]:
            recent=q[:,t-w+1:t+1].mean(1)
            baseline=q[:,t-w-2:t-w+1].mean(1)
            # Expected separation of centers under the observed prior drift.
            delta=recent-baseline-center*(w+3)/2
            candidates.append(abs(delta)/(sigma*np.sqrt(1/w+1/3)))
        out["window_glr"][:,t]=np.max(candidates,axis=0)
        x=norm[:,t,None]
        variance=1+1/kappa
        predictive=-.5*(np.log(2*np.pi*variance)+(x-mu)**2/variance)
        prior_predictive=-.5*(np.log(2*np.pi*5)+x[:,0]**2/5)
        cp=logsumexp(logp,axis=1)+np.log(.05)+prior_predictive
        growth=logp+np.log(.95)+predictive
        logp=np.column_stack([cp,growth]);logp-=logsumexp(logp,axis=1)[:,None]
        mu=np.column_stack([x[:,0]/1.25,(kappa*mu+x)/(kappa+1)])
        kappa=np.column_stack([np.ones(N)*1.25,kappa+1])
        out["bayesian_runlength"][:,t]=np.exp(logp[:,:min(3,logp.shape[1])]).sum(1)
    return out


def alarms(score,threshold,start=12,cooldown=2):
    result=np.zeros_like(score,dtype=bool)
    next_allowed=np.zeros(score.shape[0],int)
    for t in range(start,score.shape[1]):
        hit=(score[:,t]>threshold)&(t>=next_allowed)
        result[:,t]=hit;next_allowed[hit]=t+cooldown+1
    return result


def choose_threshold(score,target=.03):
    rows=[]
    for quantile in np.linspace(.80,.999,100):
        threshold=float(np.quantile(score[:,12:],quantile))
        far=float(alarms(score,threshold)[:,12:].mean())
        rows.append((threshold,far))
    eligible=[x for x in rows if x[1]<=target]
    return max(eligible,key=lambda x:x[1]) if eligible else rows[-1]


def event_metrics(alerts,times,treated,tolerance=2):
    detected=[];delay=[];true_alerts=0
    for i in np.flatnonzero(treated):
        hits=np.flatnonzero(alerts[i,times[i]:min(alerts.shape[1],times[i]+tolerance+1)])
        detected.append(len(hits)>0)
        if len(hits):delay.append(int(hits[0]));true_alerts+=1
    total=int(alerts[:,12:].sum())
    false_control=float(alerts[~treated,12:].mean())
    recall=float(np.mean(detected))
    precision=true_alerts/max(total,1)
    return {"recall":recall,"precision":precision,"f1":2*precision*recall/max(precision+recall,1e-12),
            "median_delay_detected":float(np.median(delay)) if delay else None,
            "false_alarms_per_100":100*false_control,"events":int(treated.sum()),"alarms":total,
            "random_recall_same_monthly_rate":1-(1-false_control)**(tolerance+1)}


def run_detection(cfg,limit=None):
    out=Path(cfg["results_dir"]);out.mkdir(exist_ok=True,parents=True)
    p=load_panel(Path(cfg["data_dir"])/"consumption.parquet",limit)
    ci=p.categories.index(cfg["primary_category"])
    complete=np.isfinite(p.values[:,ci,:]).all(1)
    y=p.values[complete,ci,:];ids=p.ids[complete]
    # Disjoint calibration/evaluation municipalities prevent same-series threshold tuning.
    rng=np.random.default_rng(cfg["detectors"]["calibration_seed"])
    perm=rng.permutation(len(ids));cut=max(2,len(ids)//3)
    calib,test=perm[:cut],perm[cut:]
    # Holdout series cannot affect even the cross-sectional normalization used
    # for calibration or algorithm selection. Deployment uses the full panel.
    calibration_scores=score_detectors(y[calib])
    thresholds={};curve=[]
    for name,s in calibration_scores.items():
        th,far=choose_threshold(s,cfg["detectors"]["false_alarm_target"])
        thresholds[name]={"threshold":th,"calibration_far":far}
        for q in [.9,.93,.95,.97,.98,.99,.995]:
            t=float(np.quantile(s[:,12:],q))
            curve.append({"model":name,"quantile":q,"threshold":t,"calibration_far":float(alarms(s,t)[:,12:].mean())})
    (out/"detector_thresholds.json").write_text(json.dumps(thresholds,indent=2),encoding="utf-8")
    pd.DataFrame(curve).to_csv(out/"detector_threshold_curve.csv",index=False)
    # Algorithm selection is separate from the evaluation scenarios below.
    selection_rng=np.random.default_rng(cfg["detectors"]["calibration_seed"]+1)
    selection_rows=[]
    for magnitude in cfg["detectors"]["magnitudes"]:
        for kind in ["step","ramp","temporary"]:
            synth=y[calib].copy();treated=selection_rng.random(len(calib))<.5
            times=selection_rng.integers(13,22,len(calib));signs=selection_rng.choice([-1,1],len(calib))
            for j in range(len(calib)):
                if not treated[j]:continue
                t=times[j];a=np.log1p(signs[j]*magnitude)
                if kind=="step":synth[j,t:]*=np.exp(a)
                elif kind=="ramp":synth[j,t:]*=np.exp(a*np.minimum(np.arange(1,synth.shape[1]-t+1)/3,1))
                else:synth[j,t:t+2]*=np.exp(a)
            for name,s in score_detectors(synth).items():
                selection_rows.append(dict(model=name,magnitude=magnitude,kind=kind,
                    **event_metrics(alarms(s,thresholds[name]["threshold"]),times,treated,cfg["detectors"]["delay_tolerance"])))
    selection=pd.DataFrame(selection_rows)
    selection.to_csv(out/"detector_selection_calibration.csv",index=False)
    winner=selection.groupby("model").f1.mean().idxmax()
    (out/"selected_detector.json").write_text(json.dumps({"model":winner,"criterion":"mean event F1 on calibration IDs only",
        "evaluation_ids_used_for_selection":False},indent=2),encoding="utf-8")
    rng=np.random.default_rng(cfg["detectors"]["evaluation_seed"])
    rows=[]
    # Each scenario uses untouched real controls, not zero-noise synthetic controls.
    for rep in range(cfg["detectors"]["repetitions"]):
        for magnitude in cfg["detectors"]["magnitudes"]:
            for kind in ["step","ramp","temporary"]:
                synth=y[test].copy();N=len(test)
                treated=rng.random(N)<.5
                times=rng.integers(13,22,N)
                signs=rng.choice([-1,1],N)
                for j in range(N):
                    if not treated[j]:continue
                    t=times[j];a=np.log1p(signs[j]*magnitude)
                    if kind=="step":synth[j,t:]*=np.exp(a)
                    elif kind=="ramp":synth[j,t:]*=np.exp(a*np.minimum(np.arange(1,synth.shape[1]-t+1)/3,1))
                    else:synth[j,t:t+2]*=np.exp(a)
                scores=score_detectors(synth)
                for name,s in scores.items():
                    alerts=alarms(s,thresholds[name]["threshold"])
                    rows.append(dict(model=name,repeat=rep,magnitude=magnitude,kind=kind,
                                     **event_metrics(alerts,times,treated,cfg["detectors"]["delay_tolerance"])))
    pd.DataFrame(rows).to_csv(out/"detection_metrics.csv",index=False)
    pd.DataFrame({"territory_id":ids,"partition":np.where(np.isin(np.arange(len(ids)),calib),"calibration","evaluation")}).to_csv(out/"detection_split.csv",index=False)
    alerts=[]
    clean_scores=score_detectors(y)
    for name,s in clean_scores.items():
        a=alarms(s,thresholds[name]["threshold"])
        for i,t in zip(*np.where(a)):
            alerts.append({"territory_id":int(ids[i]),"month":str(p.dates[t].date()),"detector":name,
                           "score":float(s[i,t]),"status":"statistical_alert_not_verified_economic_event"})
    pd.DataFrame(alerts).to_csv(out/"real_alerts.csv",index=False)
    risk=early_risk(y,ids,p.dates,News("data/external/news.json"),cfg)
    risk.to_csv(out/"early_warning_metrics.csv",index=False)
    print(pd.DataFrame(rows).groupby("model")[["recall","precision","false_alarms_per_100"]].mean().to_string(),flush=True)


def early_risk(y,ids,dates,news,cfg):
    """Predict next-month large relative innovations, NOT certified future shocks."""
    q=relative_log(y);changes=np.diff(q,axis=1)
    # Cutoff fixed on 2023 only, prior to any test target.
    threshold=float(np.quantile(abs(changes[:,:11]),.95))
    features=[];labels=[];times=[]
    for t in range(6,len(dates)-1):
        past=changes[:,:t]
        x=np.column_stack([changes[:,t-1],abs(changes[:,t-1]),
                           np.mean(changes[:,t-3:t],axis=1),np.std(changes[:,t-6:t],axis=1),
                           q[:,t]-np.median(q[:,t-5:t],axis=1)])
        ef=news.at(dates[t]);nf=np.tile(list(ef.values()),(len(y),1))
        features.append(np.column_stack([x,nf]));labels.append((abs(changes[:,t])>threshold).astype(int));times.append(np.full(len(y),t))
    X=np.concatenate(features);target=np.concatenate(labels);time=np.concatenate(times)
    records=[];predictions=[]
    # Test targets Oct-Dec 2024. Fit uses only labels observed at prediction time.
    for t in [20,21,22]:
        train=(time<t)&(time>=12);test=time==t
        if not test.any() or len(np.unique(target[train]))<2:continue
        for name,columns in [("history_only",5),("history_and_news",X.shape[1]),("constant_prior",0)]:
            if columns:
                m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=300,random_state=cfg["seed"]))
                m.fit(X[train,:columns],target[train]);prob=m.predict_proba(X[test,:columns])[:,1]
            else:prob=np.full(test.sum(),target[train].mean())
            actual=target[test]
            records.append({"model":name,"target_month":str(dates[t+1].date()),"n":len(actual),
                            "prevalence":float(actual.mean()),"average_precision":float(average_precision_score(actual,prob)),
                            "roc_auc":float(roc_auc_score(actual,prob)),"brier":float(brier_score_loss(actual,prob)),
                            "label":"next_month_large_relative_innovation_proxy","threshold_log_change":threshold})
            predictions.extend({"territory_id":int(i),"origin_month":str(dates[t].date()),"target_month":str(dates[t+1].date()),
                                "model":name,"probability":float(pr),"actual_proxy":int(a)} for i,pr,a in zip(ids,prob,actual))
    pd.DataFrame(predictions).to_csv(Path(cfg["results_dir"])/"early_warning_predictions.csv",index=False)
    schedule=[]
    for date in dates:
        schedule.extend(dict(origin_month=str(date.date()),**r) for r in news.upcoming(date+pd.offsets.MonthEnd(1)))
    (Path(cfg["results_dir"])/"preannounced_events.json").write_text(json.dumps(schedule,ensure_ascii=False,indent=2),encoding="utf-8")
    return pd.DataFrame(records)
