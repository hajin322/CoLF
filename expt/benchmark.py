"""Experiment 1: sample complexity + error rate, per (scenario, agent, seed).

The headline is not T_delta alone -- an observed-only agent can stop fast and
wrong, which would make it look good on speed while failing delta-soundness.
Every row therefore carries both the stopping time and enough oracle context
(reversal, obs_correct_prob, Delta_2) to separate "fast and right" from "fast
and wrong" downstream in plots.py.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DesignConfig, InstanceConfig, RunConfig  # noqa: E402
from env import make_batch  # noqa: E402
from expt.runner import run_single  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]


def run_benchmark(
    instance_cfgs: Sequence[InstanceConfig],
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
    dcfg: Optional[DesignConfig] = None,
) -> list[dict]:
    dcfg = dcfg or DesignConfig()
    records = []

    for icfg in instance_cfgs:
        instances = make_batch(icfg, run_cfg.n_reps, run_cfg.seed0, dcfg)
        for inst in instances:
            D = inst.design(dcfg)
            oracle = inst.oracle(D)

            for agent_idx, (agent_name, factory) in enumerate(agent_factories.items()):
                rng = np.random.default_rng((inst.seed, agent_idx))
                agent = factory(inst.observed(), run_cfg.delta, icfg.sigma, rng)
                res = run_single(inst, agent, run_cfg, noise_seed=inst.seed)

                records.append({
                    "scenario": icfg.scenario.value,
                    "agent": agent_name,
                    "seed": inst.seed,
                    "T_delta": res.T_delta,
                    "psi": res.psi,
                    "a_star": res.a_star,
                    "correct": res.correct,
                    "censored": res.censored,
                    "reversal": oracle["reversal"],
                    "obs_correct_prob": oracle["obs_correct_prob"],
                    "Delta_2": inst.gap_min,
                })

    return records


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent
    from config import DEFAULT_SCENARIOS, Scenario
    from plots import summarize_benchmark

    run_cfg = RunConfig(delta=0.05, n_reps=10, T_max=30_000)
    factories = {
        "gopt": lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng),
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng, run_cfg),
    }
    instance_cfgs = [
        InstanceConfig(K=10, d=4, scenario=s, sigma=0.1, min_gap=0.15)
        for s in DEFAULT_SCENARIOS
    ]

    records = run_benchmark(instance_cfgs, factories, run_cfg)
    assert len(records) == len(instance_cfgs) * run_cfg.n_reps * len(factories)
    for key in ("scenario", "agent", "seed", "T_delta", "psi", "a_star", "correct",
                "censored", "reversal", "obs_correct_prob", "Delta_2"):
        assert key in records[0], f"missing column: {key}"

    summary = summarize_benchmark(records)
    for scen, agent in summary:
        s = summary[(scen, agent)]
        print(f"{scen:<3} {agent:<5} n={s['n']:>3} T_median={s['T_median']:>8.1f} "
              f"error={s['error_rate']:.3f} censor={s['censor_rate']:.3f} "
              f"error|reversal={s['error_rate_given_reversal']:.3f}")

    # S1: both should be reasonably sound with nothing hidden.
    assert summary[(Scenario.FULL.value, "gopt")]["error_rate"] <= 0.3
    assert summary[(Scenario.FULL.value, "colf")]["error_rate"] <= 0.3

    # S2/S3: the observed-only baseline must fail far more than CoLF, either
    # by certifying the wrong arm or by never separating it (censoring).
    for scen in (Scenario.HALF.value, Scenario.UNIDENT.value):
        gopt_fail = summary[(scen, "gopt")]["error_rate"] + summary[(scen, "gopt")]["censor_rate"]
        colf_fail = summary[(scen, "colf")]["error_rate"] + summary[(scen, "colf")]["censor_rate"]
        assert gopt_fail > colf_fail, f"{scen}: baseline should fail (wrong or censored) more than CoLF"

    print("expt/benchmark.py self-test passed")


if __name__ == "__main__":
    _selftest()
