"""Experiment 1: sample complexity + error rate, per (scenario, agent, seed).

The headline is not T_delta alone -- an observed-only agent can stop fast and
wrong, which would make it look good on speed while failing delta-soundness.
Every row therefore carries both the stopping time and enough oracle context
to separate "fast and right" from "fast and wrong" downstream in plots.py.

Two different reversal flags, because they are not the same event:

    reversal        argmax_a x_a^T theta_o != a_star.  A property of how the
                    reward decomposes -- the recorded covariates, as they
                    actually enter the reward, favour a suboptimal arm.
    proj_reversal   argmax_a x_a^T theta_proj != a_star, with theta_proj the
                    observed-only least-squares limit of the *whole* reward.
                    A property of what is learnable: no observed-only linear
                    rule can identify a_star here, however long it samples.

Over 300 S3 instances the two agree only 24% of the time, so conditioning an
error rate on the first when the question is about the second measures the
wrong thing.  See agents/gopt.py::observed_only_limit.
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
from agents.gopt import observed_only_limit  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]


def run_benchmark(
    instance_cfgs: Sequence[InstanceConfig],
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
) -> list[dict]:
    records = []

    for icfg in instance_cfgs:
        instances = make_batch(icfg, run_cfg.n_reps, run_cfg.seed0)
        for inst in instances:
            oracle = inst.oracle()
            limit = observed_only_limit(inst.X, inst.means)

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
                    "proj_reversal": limit["proj_reversal"],
                    "shift_spread": oracle["shift_spread"],
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
        # ipw is pinned rather than left at the constructor default: at the
        # draft's w = 1/p the imputation stays in mu_hat, so a width that
        # honestly bounds it never lets CoLF stop and this test fails on
        # censoring alone. See agents/colf.py, "THE IPW WEIGHT".
        "colf": lambda X, delta, sigma, rng: CoLFAgent(
            X, delta, sigma, rng, ipw="round_robin"),
    }
    instance_cfgs = [
        InstanceConfig(K=10, d=4, scenario=s, sigma=0.1, min_gap=0.15)
        for s in DEFAULT_SCENARIOS
    ]

    records = run_benchmark(instance_cfgs, factories, run_cfg)
    assert len(records) == len(instance_cfgs) * run_cfg.n_reps * len(factories)
    for key in ("scenario", "agent", "seed", "T_delta", "psi", "a_star", "correct",
                "censored", "reversal", "proj_reversal", "shift_spread", "Delta_2"):
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
    for scen in (Scenario.PARTIAL.value, Scenario.REVERSED.value):
        gopt_fail = summary[(scen, "gopt")]["error_rate"] + summary[(scen, "gopt")]["censor_rate"]
        colf_fail = summary[(scen, "colf")]["error_rate"] + summary[(scen, "colf")]["censor_rate"]
        assert gopt_fail > colf_fail, f"{scen}: baseline should fail (wrong or censored) more than CoLF"

    print("expt/benchmark.py self-test passed")


if __name__ == "__main__":
    _selftest()
