from __future__ import annotations
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

def bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    method: str = "BCa",
) -> tuple[float, float]:
    """Compute BCa bootstrap confidence interval for the mean."""
    data = np.array(scores)
    if len(set(scores)) == 1:
        return (scores[0], scores[0])

    result = scipy_bootstrap(
        (data,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))
