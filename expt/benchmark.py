"""Experiment 1: Figure 1 and Table 1, error and censoring rates per scenario.

One record per (scenario, agent, instance).  `table` reduces them to the rows
of Table 1, plus the median Delta_2 per scenario that Appendix A quotes to show
the reversed scenario is not the easier one.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import InstanceConfig, RunConfig  # noqa: E402
from env import make_batch  # noqa: E402
from expt.runner import run_single  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]


def run_benchmark(
    instance_cfgs: Sequence[InstanceConfig],
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
) -> list[dict]:
    records = []
    for icfg in instance_cfgs:
        for inst in make_batch(icfg, run_cfg.n_reps, run_cfg.seed0):
            for agent_idx, (agent_name, factory) in enumerate(agent_factories.items()):
                rng = np.random.default_rng((inst.seed, agent_idx))
                agent = factory(inst.observed(), run_cfg.delta, icfg.sigma, rng)
                res = run_single(inst, agent, run_cfg, noise_seed=inst.seed)
                records.append({
                    "scenario": icfg.scenario.value,
                    "agent": agent_name,
                    "seed": inst.seed,
                    "T_delta": res.T_delta,
                    "correct": res.correct,
                    "censored": res.censored,
                    "Delta_2": inst.gap_min,
                })
    return records


def table(records: Sequence[dict]) -> list[dict]:
    """Table 1: per (scenario, agent), the error and censoring rates."""
    keys = list(dict.fromkeys((r["scenario"], r["agent"]) for r in records))
    rows = []
    for scen, agent in keys:
        g = [r for r in records if r["scenario"] == scen and r["agent"] == agent]
        rows.append({
            "scenario": scen,
            "agent": agent,
            "n": len(g),
            "error_rate": float(np.mean([not r["correct"] for r in g])),
            "censored": float(np.mean([r["censored"] for r in g])),
            "median_Delta_2": float(np.median([r["Delta_2"] for r in g])),
        })
    return rows


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent
    from config import DEFAULT_SCENARIOS, Scenario

    run_cfg = RunConfig(delta=0.05, n_reps=10, T_max=30_000)
    factories = {
        "gopt": lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng),
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng),
    }
    instance_cfgs = [
        InstanceConfig(K=10, d=4, scenario=s, sigma=0.1, min_gap=0.15)
        for s in DEFAULT_SCENARIOS
    ]

    records = run_benchmark(instance_cfgs, factories, run_cfg)
    assert len(records) == len(instance_cfgs) * run_cfg.n_reps * len(factories)

    rows = {(r["scenario"], r["agent"]): r for r in table(records)}
    for (scen, agent), r in rows.items():
        print(f"{scen:<3} {agent:<5} n={r['n']:>3} error={r['error_rate']:.3f} "
              f"censored={r['censored']:.3f} median Delta_2={r['median_Delta_2']:.3f}")

    # S1: both should be reasonably sound with nothing hidden.
    assert rows[(Scenario.FULL.value, "gopt")]["error_rate"] <= 0.3
    assert rows[(Scenario.FULL.value, "colf")]["error_rate"] <= 0.3

    # S2/S3: the observed-only baseline must fail far more than CoLF, either
    # by certifying the wrong arm or by never separating it (censoring).
    for scen in (Scenario.PARTIAL.value, Scenario.REVERSED.value):
        fail = {a: rows[(scen, a)]["error_rate"] + rows[(scen, a)]["censored"]
                for a in ("gopt", "colf")}
        assert fail["gopt"] > fail["colf"], f"{scen}: baseline should fail more than CoLF"

    print("expt/benchmark.py self-test passed")


if __name__ == "__main__":
    _selftest()
