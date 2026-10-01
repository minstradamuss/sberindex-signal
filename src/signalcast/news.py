"""Auditable news features aligned by publication, never by future event outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


class News:
    def __init__(self, path):
        records=json.loads(Path(path).read_text(encoding="utf-8"))
        if not records:
            raise ValueError("News registry is empty; cannot silently claim news integration")
        self.records=records
        self.frame=pd.DataFrame(records)
        self.frame["published_at"]=pd.to_datetime(self.frame.published_at)
        self.frame=self.frame.sort_values("published_at")

    def at(self, last_observed_month, publication_lag_months=0):
        # Data release timestamps are absent. This is an explicit month-end
        # information-set convention; sensitivity is supported via a release lag.
        cutoff=pd.Timestamp(last_observed_month).to_period("M").end_time+pd.DateOffset(months=publication_lag_months)
        known=self.frame[self.frame.published_at<=cutoff]
        if len(known)==0:
            return {"rate":7.5,"rate_change_3m":0.,"tightening_signal":0.,"releases_3m":0.}
        last=known.iloc[-1]
        prior=known[known.published_at<=cutoff-pd.DateOffset(months=3)]
        recent=known[known.published_at>cutoff-pd.DateOffset(months=3)]
        return {"rate":float(last.rate),
                "rate_change_3m":float(last.rate-(prior.iloc[-1].rate if len(prior) else 7.5)),
                "tightening_signal":float(last.tightening_signal), "releases_3m":float(len(recent))}

    def upcoming(self, cutoff):
        """Only preannounced event dates may generate a true pre-event alert."""
        out=[]
        for r in self.records:
            if r.get("next_meeting") and pd.Timestamp(r["published_at"])<=pd.Timestamp(cutoff)<pd.Timestamp(r["next_meeting"]):
                out.append({"announced_at":r["published_at"],"event_at":r["next_meeting"],
                            "type":"scheduled_policy_meeting", "known_rate_outcome":False,
                            "source":r["source_url"]})
        return out[-1:]
