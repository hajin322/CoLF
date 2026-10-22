"""Experiment 3: one factor at a time, holding everything else fixed.

The benchmark answers "does CoLF beat the baseline on three fixed scenarios".
It cannot say *which* property of the problem the difference comes from, because
S1 -> S2 changes du from 0 to d and S2 -> S3 additionally conditions the draw.
This module moves one factor at a time so each of the draft's claims gets its
own curve:

    du       the treatment the paper is about.  The baseline's error should
             grow with how much of the reward is hidden; CoLF's should not.
    K        Theorem 1's leading term.  With the gap made large the
             gap-dependent sum is negligible and T_delta should be a small
             multiple of K, not of anything else.
    delta    delta-soundness.  The realised error must sit under delta at every
             level, which is the claim in the abstract and is not testable at a
             single delta.
    min_gap  the 1/Delta^2 term, and the K floor it sits on top of.

Every level reuses the same seeds, so the levels are paired: a difference
between two points is not a difference in which instances were drawn.  Each
record is a benchmark record plus {"factor", "value"}, so plots.py and
summarize_benchmark work on it unchanged.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Callable, Dict, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import InstanceConfig, RunConfig  # noqa: E402
from expt.benchmark import run_benchmark  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]

# factor name -> which dataclass field it sets.  `du` is spelled latent_dim in
# InstanceConfig because `du` there is a derived property.
_INSTANCE_FIELD = {"du": "latent_dim", "K": "K", "d": "d",
                   "sigma": "sigma", "min_gap": "min_gap"}
_RUN_FIELD = {"delta": "delta", "T_max": "T_max"}


def _levels(factor: str, value, base_cfg: InstanceConfig, run_cfg: RunConfig):
    if factor in _INSTANCE_FIELD:
        return replace(base_cfg, **{_INSTANCE_FIELD[factor]: value}), run_cfg
    if factor in _RUN_FIELD:
        return base_cfg, replace(run_cfg, **{_RUN_FIELD[factor]: value})
    raise ValueError(f"unknown factor {factor!r}; expected one of "
                     f"{sorted(set(_INSTANCE_FIELD) | set(_RUN_FIELD))}")


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
        rows = run_benchmark([icfg], agent_factories, rcfg)
        for r in rows:
            r["factor"] = factor
            r["value"] = value
            r["du"] = icfg.du
            r["delta"] = rcfg.delta
        records.extend(rows)
    return records


def summarize_sweep(records: Sequence[dict]) -> list[dict]:
    """Per (factor, value, agent): error rate, censor rate, median T_delta and
    the median gap, which is what T_delta has to be read against."""
    keys, out = [], []
    for r in records:
        k = (r["factor"], r["value"], r["agent"])
        if k not in keys:
            keys.append(k)
    for factor, value, agent in keys:
        rows = [r for r in records
                if r["factor"] == factor and r["value"] == value and r["agent"] == agent]
        T = np.array([r["T_delta"] for r in rows], dtype=float)
        out.append({
            "factor": factor,
            "value": value,
            "agent": agent,
            "n": len(rows),
            "error_rate": float(np.mean([not r["correct"] for r in rows])),
            "censor_rate": float(np.mean([r["censored"] for r in rows])),
            "T_median": float(np.median(T)),
            "T_q25": float(np.percentile(T, 25)),
            "T_q75": float(np.percentile(T, 75)),
            "Delta_2": float(np.median([r["Delta_2"] for r in rows])),
            "proj_reversal_rate": float(np.mean([r["proj_reversal"] for r in rows])),
        })
    return out


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent
    from config import Scenario

    factories = {
        "gopt": lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng),
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng),
    }
    base = InstanceConfig(K=10, d=4, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.15)
    run_cfg = RunConfig(n_reps=8, T_max=20_000)

    recs = run_sweep("du", [0, 2, 8], base, factories, run_cfg)
    assert len(recs) == 3 * run_cfg.n_reps * len(factories)
    assert {r["value"] for r in recs} == {0, 2, 8}
    assert {r["du"] for r in recs} == {0, 2, 8}

    summ = summarize_sweep(recs)
    for row in summ:
        print(f"  du={row['value']:>3} {row['agent']:<5} n={row['n']:>3} "
              f"error={row['error_rate']:.3f} T_med={row['T_median']:>8.1f} "
              f"proj-rev={row['proj_reversal_rate']:.2f}")

    at0 = {r["agent"]: r for r in summ if r["value"] == 0}
    assert at0["gopt"]["error_rate"] <= 0.25, (
        "with du = 0 nothing is hidden, so the baseline must not fail"
    )
    assert at0["gopt"]["proj_reversal_rate"] == 0.0, (
        "with du = 0 the observed-only limit is the truth"
    )

    # levels are paired: the same seeds are drawn at every level
    seeds = {v: sorted({r["seed"] for r in recs if r["value"] == v}) for v in (0, 2, 8)}
    assert seeds[0] == seeds[2] == seeds[8], "levels are not paired on seeds"

    # delta is a run-level factor, not an instance one
    recs_d = run_sweep("delta", [0.2, 0.01], base, factories,
                       replace(run_cfg, n_reps=6))
    assert {r["delta"] for r in recs_d} == {0.2, 0.01}
    T = {(r["value"], r["agent"]): [] for r in recs_d}
    for r in recs_d:
        T[(r["value"], r["agent"])].append(r["T_delta"])
    assert np.median(T[(0.01, "colf")]) >= np.median(T[(0.2, "colf")]), (
        "a smaller delta must not make CoLF stop sooner"
    )

    try:
        run_sweep("nonsense", [1], base, factories, run_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown factor should be rejected")

    print("expt/sweep.py self-test passed")


if __name__ == "__main__":
    _selftest()
