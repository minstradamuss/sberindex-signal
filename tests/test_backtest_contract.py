import numpy as np
import pandas as pd
from signalcast.backtest import to_frame
from signalcast.data import Panel


def test_missing_actual_is_never_scored_and_future_gaps_do_not_exclude_history():
    values=np.ones((2,1,24))*100
    values[0,0,15]=np.nan
    p=Panel(values,np.array([1,2]),["x"],pd.date_range("2023-01-01",periods=24,freq="MS"))
    predictions={"m":np.ones((2,1,12))*110}
    d=to_frame(predictions,p,12,12,"2024-10-01")
    assert not ((d.territory_id==1)&(d.horizon==4)).any()
    assert ((d.territory_id==1)&(d.horizon==1)).any()
    assert np.isfinite(d.actual).all()


def test_actuals_at_different_horizons_are_distinct():
    v=np.arange(1,25,dtype=float)[None,None,:]
    p=Panel(v,np.array([1]),["x"],pd.date_range("2023-01-01",periods=24,freq="MS"))
    d=to_frame({"m":np.ones((1,1,12))},p,12,12,"2024-10-01")
    assert d.loc[d.horizon==12,"actual"].item()==24
    assert d.loc[d.horizon==1,"actual"].item()==13
