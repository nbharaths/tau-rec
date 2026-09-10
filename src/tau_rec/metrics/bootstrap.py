from __future__ import annotations
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

DEFAULT_SEED = 20260910

def bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    method: str = "BCa",
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Compute BCa bootstrap confidence interval for the mean.

    The resampling RNG is seeded so repeated calls agree exactly. The
    leaderboard renderer must be a pure function of its inputs; an unseeded
    bootstrap would make the "re-render is byte-identical" gate unpassable.
    """
    data = np.array(scores)
    if len(set(scores)) == 1:
        return (scores[0], scores[0])

    result = scipy_bootstrap(
        (data,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
        random_state=np.random.default_rng(seed),
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))
