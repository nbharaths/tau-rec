from __future__ import annotations
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass^k for a single task."""
    if n < k:
        return 0.0
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)

def aggregate_pass_k(task_results: dict[str, dict[str, int]], k: int) -> float:
    """Compute average pass^k across tasks."""
    if not task_results:
        return 0.0
    scores = [pass_at_k(r["n"], r["c"], k) for r in task_results.values()]
    return sum(scores) / len(scores)
