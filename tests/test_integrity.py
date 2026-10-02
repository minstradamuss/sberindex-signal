import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from signalcast.data import historical_fill, load_panel
from signalcast.models import seasonal_experts, ridge_panel, feature_frame
from signalcast.metrics import scores
from signalcast.ensemble import weights_from_history
from signalcast.news import News


def test_metrics_known_values():
    s=scores([1,2,3],[2,2,2])
    assert s["mae"]==pytest.approx(2/3)
    assert s["r2"]==0


def test_missing_never_filled_from_future():
    a=np.array([[[2.,np.nan,100.]],[[4.,5.,6.]]])
    assert historical_fill(a)[0,0,1]==2
    np.testing.assert_allclose(historical_fill(a[...,:2]),historical_fill(a)[...,:2])


def test_prefix_invariance():
    rng=np.random.default_rng(8)
    a=np.exp(rng.normal(8,.1,(10,6,24)))
    b=a.copy();b[...,18:]*=100
    for f in [seasonal_experts,ridge_panel]:
        x,y=f(a[...,:18],12),f(b[...,:18],12)
        if isinstance(x,dict):
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        else:np.testing.assert_array_equal(x,y)
    x,_=feature_frame(a[...,:18],12);y,_=feature_frame(b[...,:18],12)
    pd.testing.assert_frame_equal(x,y)


def test_seasonal_alignment():
    a=np.arange(1,19,dtype=float)[None,None,:]
    p=seasonal_experts(a,12)["seasonal_naive"]
    np.testing.assert_equal(p.ravel(),np.arange(7,19))


def test_duplicate_keys_rejected(tmp_path):
    d=pd.DataFrame({"territory_id":[1,1],"category":["x","x"],"date":["2023-01"]*2,"value":[2,3]})
    path=tmp_path/"x.parquet";d.to_parquet(path)
    with pytest.raises(ValueError,match="Duplicate"):load_panel(path)


def test_no_future_expert_selection():
    d=pd.DataFrame({"target_date":pd.to_datetime(["2025-01-01"]*2),"category":["x"]*2,
                    "horizon":[1]*2,"model":["a","b"],"absolute_error":[0.,1e9]})
    w,reason=weights_from_history(d,["a","b"],"x",1,"2024-12-01",
                                 {"min_resolved_months":1,"temperature":.5,"floor_weight":.05})
    np.testing.assert_equal(w,[.5,.5]);assert reason=="equal_prior"


def test_news_publication_cutoff(tmp_path):
    records=[{"published_at":"2024-07-26","rate":18,"tightening_signal":1,"next_meeting":"2024-09-13","source_url":"official"}]
    p=tmp_path/"news.json";p.write_text(json.dumps(records))
    n=News(p)
    assert n.at("2024-06-01")["rate"]==7.5
    assert n.at("2024-07-01")["rate"]==18
    assert n.upcoming("2024-06-30")==[]


def test_all_categories_is_not_sum():
    # The dataset's five named subcategories are not an exhaustive partition.
    p=load_panel(Path(__file__).parents[1]/"data/raw/consumption.parquet",20)
    all_index=p.categories.index("Все категории")
    assert not np.allclose(p.values[:,all_index],np.nansum(np.delete(p.values,all_index,axis=1),axis=1))
