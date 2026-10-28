"""Experiment 3: Tables 2 and 3, one factor at a time.

    du      Table 2.  How much of the reward is hidden: the baseline's error
            should appear as soon as anything is hidden; CoLF's should not.
    delta   Table 3.  delta-soundness: the realised error must sit under delta
            at every level, which is not testable at a single delta.

Every level reuses the same seeds, so the levels are paired: a difference
between two rows is not a difference in which instances were drawn.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from math import comb
from typing import Callable, Dict, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import InstanceConfig, RunConfig  # noqa: E402
from expt.benchmark import run_benchmark  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]


def _levels(factor: str, value, base_cfg: InstanceConfig, run_cfg: RunConfig):
    # `du` is spelled latent_dim in InstanceConfig, where du is derived
    if factor == "du":
        return replace(base_cfg, latent_dim=value), run_cfg
    if factor == "delta":
        return base_cfg, replace(run_cfg, delta=value)
    raise ValueError(f"unknown factor {factor!r}; expected 'du' or 'delta'")


def run_sweep(
    factor: str,
    values: Sequence,
    base_cfg: InstanceConfig,
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
) -> list[dict]:
    """Run the benchmark once per level of `factor` and tag every record."""
    records = []
    for value in values:
        icfg, rcfg = _levels(factor, value, base_cfg, run_cfg)
        for r in run_benchmark([icfg], agent_factories, rcfg):
            r["value"] = value
            records.append(r)
    return records


def _rates(records, value, agent):
    g = [r for r in records if r["value"] == value and r["agent"] == agent]
    return (len(g), sum(not r["correct"] for r in g),
            float(np.mean([r["censored"] for r in g])))


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p): the one-sided test of a realised
    error count against the level it was run at."""
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def binom_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound on a rate, by bisection: a measured
    0/200 is evidence that the rate is below 1.5%, not that it is zero."""
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 1.0 - binom_tail(k + 1, n, mid) > alpha:     # P(X <= k) at rate mid
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def table_du(records: Sequence[dict]) -> list[dict]:
    """Table 2: per (du, agent), the error and censoring rates."""
    rows = []
    for value, agent in dict.fromkeys((r["value"], r["agent"]) for r in records):
        n, k, cens = _rates(records, value, agent)
        rows.append({"du": value, "agent": agent, "n": n,
                     "error_rate": k / n, "censored": cens})
    return rows


def table_delta(records: Sequence[dict]) -> list[dict]:
    """Table 3: per (delta, agent), the error rate against the level it was run
    at, the one-sided binomial p-value of the error count against delta, and
    the 95% upper bound on the rate that the text quotes for CoLF."""
    rows = []
    for value, agent in dict.fromkeys((r["value"], r["agent"]) for r in records):
        n, k, cens = _rates(records, value, agent)
        rows.append({"delta": value, "agent": agent, "n": n,
                     "error_rate": k / n, "err_over_delta": k / n / value,
                     "p_value": binom_tail(k, n, value),
                     "upper95": binom_upper(k, n), "censored": cens})
    return rows


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent
    from config import Scenario

    assert binom_upper(0, 200) < 0.016 and binom_upper(0, 20) > 0.1, "CP upper bound"
    assert 0.05 < binom_upper(5, 100) < 0.12
    assert np.isclose(binom_tail(0, 50, 0.3), 1.0) and binom_tail(50, 50, 0.3) < 1e-20

    factories = {
        "gopt": lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng),
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng),
    }
    base = InstanceConfig(K=10, d=4, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.15)
    run_cfg = RunConfig(n_reps=8, T_max=20_000)

    recs = run_sweep("du", [0, 2, 8], base, factories, run_cfg)
    assert len(recs) == 3 * run_cfg.n_reps * len(factories)
    rows = table_du(recs)
    for r in rows:
        print(f"  du={r['du']:>3} {r['agent']:<5} n={r['n']:>3} "
              f"error={r['error_rate']:.3f} censored={r['censored']:.3f}")
    at0 = {r["agent"]: r for r in rows if r["du"] == 0}
    assert at0["gopt"]["error_rate"] <= 0.25, (
        "with du = 0 nothing is hidden, so the baseline must not fail"
    )

    # levels are paired: the same seeds are drawn at every level
    seeds = {v: sorted({r["seed"] for r in recs if r["value"] == v}) for v in (0, 2, 8)}
    assert seeds[0] == seeds[2] == seeds[8], "levels are not paired on seeds"

    # delta is a run-level factor, not an instance one
    recs_d = run_sweep("delta", [0.2, 0.01], base, factories, replace(run_cfg, n_reps=6))
    T = {(r["value"], r["agent"]): [] for r in recs_d}
    for r in recs_d:
        T[(r["value"], r["agent"])].append(r["T_delta"])
    assert np.median(T[(0.01, "colf")]) >= np.median(T[(0.2, "colf")]), (
        "a smaller delta must not make CoLF stop sooner"
    )
    assert {r["delta"] for r in table_delta(recs_d)} == {0.2, 0.01}

    try:
        run_sweep("nonsense", [1], base, factories, run_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown factor should be rejected")

    print("expt/sweep.py self-test passed")


if __name__ == "__main__":
    _selftest()
