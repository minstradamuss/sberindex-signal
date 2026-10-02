import numpy as np
from signalcast.detection import score_detectors,alarms,event_metrics


def test_detectors_never_change_the_past():
    rng=np.random.default_rng(40)
    y=np.exp(rng.normal(8,.03,(40,24)))
    a=score_detectors(y)
    y[:,18:]*=100
    b=score_detectors(y)
    for k in a:np.testing.assert_allclose(a[k][:,:18],b[k][:,:18])


def test_refractory_window():
    x=np.ones((1,24))*100
    a=alarms(x,1)
    assert np.flatnonzero(a[0]).tolist()==[12,15,18,21]


def test_delay_is_measured_from_event_not_from_series_start():
    a=np.zeros((2,24),bool);a[0,17]=True
    r=event_metrics(a,np.array([15,15]),np.array([True,False]))
    assert r['recall']==1
    assert r['median_delay_detected']==2
    assert r['false_alarms_per_100']==0
