from tau_rec.metrics.pass_k import pass_at_k, aggregate_pass_k
from tau_rec.metrics.bootstrap import bootstrap_ci

def test_pass_at_1():
    assert abs(pass_at_k(n=16, c=8, k=1) - 0.5) < 1e-9

def test_pass_at_1_all_pass():
    assert abs(pass_at_k(n=16, c=16, k=1) - 1.0) < 1e-9

def test_pass_at_1_none_pass():
    assert abs(pass_at_k(n=16, c=0, k=1) - 0.0) < 1e-9

def test_pass_at_k_higher():
    p1 = pass_at_k(n=16, c=12, k=1)
    p2 = pass_at_k(n=16, c=12, k=2)
    p4 = pass_at_k(n=16, c=12, k=4)
    assert p1 > p2 > p4
    assert p1 == 12 / 16

def test_aggregate_pass_k():
    task_results = {
        "t1": {"n": 16, "c": 16},
        "t2": {"n": 16, "c": 8},
    }
    agg = aggregate_pass_k(task_results, k=1)
    assert abs(agg - 0.75) < 1e-9

def test_bootstrap_ci_basic():
    scores = [1.0] * 50
    lo, hi = bootstrap_ci(scores, confidence=0.95, n_resamples=1000)
    assert lo == 1.0
    assert hi == 1.0

def test_bootstrap_ci_mixed():
    import numpy as np
    np.random.seed(42)
    scores = [1.0] * 25 + [0.0] * 25
    lo, hi = bootstrap_ci(scores, confidence=0.95, n_resamples=5000)
    assert 0.2 < lo < 0.5
    assert 0.5 < hi < 0.8

def test_bootstrap_ci_returns_tuple():
    scores = [0.5, 0.6, 0.7, 0.8]
    result = bootstrap_ci(scores, confidence=0.95, n_resamples=1000)
    assert len(result) == 2
    assert result[0] <= result[1]
