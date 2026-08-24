"""Experiment 2: estimation-error diagnostics, independent of any stopping rule.

Drives a plain round-robin schedule directly through `agent.update()`, bypassing
`agent.sample()` entirely -- the point is to watch predicted_means()/components()
converge (or not) as t grows to ablation_T, regardless of when either agent's own
stopping rule would have fired. Records only at log-spaced checkpoints to keep
memory bounded over long horizons.

Both agents are driven on the *same* round-robin schedule, which is CoLF's own
but not the baseline's (that would be pi^G rounding with elimination). This is
deliberate: the quantity on trial is estimation error, and holding the schedule
fixed makes the two curves a controlled comparison rather than a confound of
schedule and estimator. The baseline's plateau is a misspecification bias and is
schedule-independent, so the comparison is fair on its own terms.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import round_robin  # noqa: E402
from config import DesignConfig, InstanceConfig, RunConfig  # noqa: E402
from env import make_batch  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]


def _checkpoints(K: int, T: int, n_points: int) -> list[int]:
    pts = np.unique(np.geomspace(K, T, n_points).astype(int))
    return [int(p) for p in pts if 1 <= p <= T]


def run_ablation(
    instance_cfg: InstanceConfig,
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
    dcfg: Optional[DesignConfig] = None,
) -> list[dict]:
    dcfg = dcfg or DesignConfig()
    K = instance_cfg.K
    checkpoints = _checkpoints(K, run_cfg.ablation_T, run_cfg.ablation_points)

    instances = make_batch(instance_cfg, run_cfg.n_reps, run_cfg.seed0, dcfg)
    records = []

    for inst in instances:
        D = inst.design(dcfg)
        oracle = inst.oracle(D)
        true_means = inst.means

        for agent_idx, (agent_name, factory) in enumerate(agent_factories.items()):
            rng = np.random.default_rng((inst.seed, agent_idx))
            agent = factory(inst.observed(), run_cfg.delta, instance_cfg.sigma, rng)
            inst.reset_noise(inst.seed)

            cp_idx = 0
            for t in range(1, run_cfg.ablation_T + 1):
                a = round_robin(t - 1, K)
                agent.update(a, inst.pull(a))

                if cp_idx >= len(checkpoints) or t < checkpoints[cp_idx]:
                    continue

                comp = agent.components()
                pm = agent.predicted_means()
                row = {
                    "agent": agent_name,
                    "seed": inst.seed,
                    "t": t,
                    "eps_base": float(np.max(np.abs(pm - true_means))),
                    "beta": agent.confidence_width(),
                    "sigma_perp": None,
                    "perp_signal": None,
                }
                if comp.get("perp") is not None:
                    perp_hat = D.B @ comp["perp"]
                    row["sigma_perp"] = float(np.max(np.abs(perp_hat - oracle["perp_per_arm"])))
                    row["perp_signal"] = oracle["perp_signal"]
                records.append(row)

                while cp_idx < len(checkpoints) and t >= checkpoints[cp_idx]:
                    cp_idx += 1

    return records


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
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng, run_cfg),
    }
    icfg = InstanceConfig(K=10, d=4, scenario=Scenario.HALF, sigma=0.1, min_gap=0.10)

    records = run_ablation(icfg, factories, run_cfg)
    assert records
    for key in ("agent", "seed", "t", "eps_base", "beta", "sigma_perp", "perp_signal"):
        assert key in records[0]

    colf_rows = sorted([r for r in records if r["agent"] == "colf"], key=lambda r: r["t"])
    gopt_rows = sorted([r for r in records if r["agent"] == "gopt"], key=lambda r: r["t"])
    assert all(r["sigma_perp"] is not None for r in colf_rows)
    assert all(r["sigma_perp"] is None for r in gopt_rows)

    def _median_at_t(rows, t, field):
        vals = [r[field] for r in rows if r["t"] == t]
        return float(np.median(vals))

    ts = sorted({r["t"] for r in colf_rows})
    early_sigma = _median_at_t(colf_rows, ts[0], "sigma_perp")
    late_sigma = _median_at_t(colf_rows, ts[-1], "sigma_perp")
    print(f"colf sigma_perp: t={ts[0]} -> {early_sigma:.4f}, t={ts[-1]} -> {late_sigma:.4f}")
    assert late_sigma < early_sigma, "the orthogonal offset error should shrink over time"

    early_base_gopt = _median_at_t(gopt_rows, ts[0], "eps_base")
    late_base_gopt = _median_at_t(gopt_rows, ts[-1], "eps_base")
    late_base_colf = _median_at_t(colf_rows, ts[-1], "eps_base")
    print(f"eps_base at t={ts[-1]}: gopt={late_base_gopt:.4f} colf={late_base_colf:.4f} "
          f"(gopt at t={ts[0]}: {early_base_gopt:.4f})")
    assert late_base_colf < late_base_gopt, "CoLF's predicted means should end up more accurate"

    fig = plot_ablation(records)
    assert len(fig.axes) == 2

    print("expt/ablation.py self-test passed")


if __name__ == "__main__":
    _selftest()
