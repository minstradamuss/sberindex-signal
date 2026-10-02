import numpy as np
import pandas as pd
from signalcast.refinement import fast_experts,adaptive_frame


def test_annual_growth_forecasts_use_only_the_prefix():
    rng=np.random.default_rng(3)
    y=np.exp(rng.normal(8,.1,(10,6,24)))
    a=fast_experts(y[...,:17],12)
    y[...,17:]*=10
    b=fast_experts(y[...,:17],12)
    for name in a:
        np.testing.assert_array_equal(a[name],b[name])
        assert np.isfinite(a[name]).all() and (a[name]>0).all()


def test_adaptive_selection_ignores_future_errors():
    current=pd.DataFrame({"territory_id":[1,1],"category":["x"]*2,"horizon":[1]*2,
                          "model":["panel_shrink","pooled_yoy_damped"],"prediction":[100.,200.],"actual":[150.,150.]})
    future=[]
    for date in pd.date_range("2025-01-01",periods=3,freq="MS"):
        for m,e in [("panel_shrink",0.),("pooled_yoy_damped",10000.)]:
            future.append({"territory_id":1,"category":"x","horizon":1,"model":m,"absolute_error":e,
                           "origin_date":date-pd.offsets.MonthBegin(1),"target_date":date})
    result,logs=adaptive_frame(current,pd.DataFrame(future),pd.Timestamp("2024-12-01"),{})
    assert result.prediction.item()==150
    assert all(x["rule"]=="structural_prior" for x in logs)
