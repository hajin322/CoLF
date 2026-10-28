"""Experiment 2: Figure 2 and the ablation table of Appendix A.

Two runs over the same ladder of four agents, where adjacent rungs differ by
exactly one thing:

    gopt         G-allocation on the observed features
    colf_noaug   CoLF with `augment=False`: the design, the round-robin
                 schedule, the DR estimator and the two-phase elimination rule
                 are kept, and only the K - d appended directions are removed
    colf         CoLF
    oracle       G-allocation handed z_a = [x_a; u_a], the ceiling on what
                 seeing the latent features would buy

`run_ablation` (Figure 2) drives every agent on one round-robin schedule
directly through `agent.update()`, bypassing `agent.sample()`, and records the
largest predicted-mean error and the agent's own width at log-spaced
checkpoints.  CoLF is built with `eliminate=False` here so its phase-1
estimator keeps running past where it would have switched to phase 2.

`run_stopping_ablation` (the table) runs every agent to its own stopping rule.

READ THE BASELINE'S CURVES AS A STATEMENT ABOUT THE ESTIMATOR, NOT ABOUT THE
ALGORITHM. The schedule is not neutral for it. Measured on 100 S2 instances at
t = 10000, the error of `argmax_a predicted_means` is

    forced round-robin           0.510
    its own pi^G + elimination   0.170

because elimination collapses the allocation -- past its stop it has pulled an
effective 1.1 arms of 20 -- and a d = 8 linear model fits one or two arms
essentially exactly. Its honest error rate is the table's, not the figure's.
What the figure does say is the identifiability claim: given the same data
covering every arm, the observed-only model cannot fit the means at all (the
error plateaus) while CoLF's converges.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import round_robin  # noqa: E402
from config import InstanceConfig, RunConfig  # noqa: E402
from env import make_batch  # noqa: E402
from expt.runner import run_single  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]


def _features(inst, agent_name):
    """"oracle" is the one agent handed the latent features as well."""
    return np.vstack([inst.X, inst.U]) if agent_name == "oracle" else inst.observed()


def run_ablation(
    instance_cfg: InstanceConfig,
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
) -> list[dict]:
    """Figure 2: one row per (agent, instance, checkpoint t)."""
    K = instance_cfg.K
    # log-spaced, snapped up to whole round-robin sweeps
    checkpoints = np.unique(
        K * np.ceil(
            np.geomspace(K, run_cfg.ablation_T, run_cfg.ablation_points) / K
        ).astype(int)
    ).tolist()

    records = []
    for inst in make_batch(instance_cfg, run_cfg.n_reps, run_cfg.seed0):
        for agent_idx, (agent_name, factory) in enumerate(agent_factories.items()):
            rng = np.random.default_rng((inst.seed, agent_idx))
            agent = factory(_features(inst, agent_name), run_cfg.delta,
                            instance_cfg.sigma, rng)
            inst.reset_noise(inst.seed)

            cp_idx = 0
            for t in range(1, run_cfg.ablation_T + 1):
                a = round_robin(t - 1, K)
                agent.update(a, inst.pull(a))
                if cp_idx < len(checkpoints) and t >= checkpoints[cp_idx]:
                    records.append({
                        "agent": agent_name,
                        "seed": inst.seed,
                        "t": t,
                        "eps": float(np.max(np.abs(agent.predicted_means() - inst.means))),
                        "width": agent.confidence_width(),
                    })
                    cp_idx += 1
    return records


def run_stopping_ablation(
    instance_cfg: InstanceConfig,
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
) -> list[dict]:
    """The ablation table: one row per (agent, instance), each agent run to its
    own stopping rule."""
    records = []
    for inst in make_batch(instance_cfg, run_cfg.n_reps, run_cfg.seed0):
        for agent_idx, (agent_name, factory) in enumerate(agent_factories.items()):
            rng = np.random.default_rng((inst.seed, agent_idx))
            agent = factory(_features(inst, agent_name), run_cfg.delta,
                            instance_cfg.sigma, rng)
            res = run_single(inst, agent, run_cfg, noise_seed=inst.seed)
            records.append({
                "agent": agent_name, "seed": inst.seed, "T_delta": res.T_delta,
                "correct": res.correct, "censored": res.censored,
            })
    return records


def table(records: Sequence[dict]) -> list[dict]:
    """Per agent: error and censoring rates, and the median stopping time that
    Appendix A quotes for CoLF with and without the augmentation."""
    rows = []
    for agent in dict.fromkeys(r["agent"] for r in records):
        g = [r for r in records if r["agent"] == agent]
        rows.append({
            "agent": agent,
            "n": len(g),
            "error_rate": float(np.mean([not r["correct"] for r in g])),
            "censored": float(np.mean([r["censored"] for r in g])),
            "T_median": float(np.median([r["T_delta"] for r in g])),
        })
    return rows


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent
    from config import Scenario
    from plots import plot_ablation

    run_cfg = RunConfig(delta=0.05, n_reps=6, ablation_T=8_000, ablation_points=20)
    factories = {
        "gopt": lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng),
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng,
                                                       eliminate=False),
    }
    icfg = InstanceConfig(K=10, d=4, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.10)

    records = run_ablation(icfg, factories, run_cfg)
    ts = sorted({r["t"] for r in records})

    def median_at(agent, t, field):
        return float(np.median([r[field] for r in records
                                if r["agent"] == agent and r["t"] == t]))

    early, late = median_at("colf", ts[0], "eps"), median_at("colf", ts[-1], "eps")
    print(f"colf eps: t={ts[0]} -> {early:.4f}, t={ts[-1]} -> {late:.4f}; "
          f"gopt eps at t={ts[-1]}: {median_at('gopt', ts[-1], 'eps'):.4f}")
    assert late < early, "CoLF's predicted means should converge"
    assert late < median_at("gopt", ts[-1], "eps"), "and end up more accurate"
    assert late < median_at("colf", ts[-1], "width"), "inside its own width"

    fig = plot_ablation(records)
    assert len(fig.axes) == 2

    own_rules = {
        "gopt": factories["gopt"],
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng),
    }
    stop = table(run_stopping_ablation(icfg, own_rules, RunConfig(n_reps=4, T_max=50_000)))
    assert [r["agent"] for r in stop] == ["gopt", "colf"] and stop[0]["n"] == 4
    assert stop[1]["censored"] == 0.0, "CoLF should stop on its own"

    print("expt/ablation.py self-test passed")


if __name__ == "__main__":
    _selftest()
