"""Experiment 2: the ablation ladder -- estimation-error diagnostics,
independent of any stopping rule.

Drives a plain round-robin schedule directly through `agent.update()`, bypassing
`agent.sample()` entirely -- the point is to watch predicted_means()/components()
converge (or not) as t grows to ablation_T, regardless of when either agent's own
stopping rule would have fired. Records only at log-spaced checkpoints to keep
memory bounded over long horizons.

An agent registered under the name "oracle" is handed z_a = [x_a; u_a]; every
other agent sees x_a only.  With "colf_noaug" -- CoLF with `augment=False`,
which keeps the design, the round-robin schedule, the DR estimator and the
stopping rule and removes only the K - d appended directions -- the three
agents form a ladder where adjacent rungs differ by exactly one thing.

Both agents are driven on the *same* round-robin schedule, which is CoLF's own
but not the baseline's (that would be pi^G rounding with elimination). The
quantity on trial is estimation error, so holding the schedule fixed makes the
two curves a controlled comparison rather than a confound of schedule and
estimator.

READ THE BASELINE'S CURVES AS A STATEMENT ABOUT THE ESTIMATOR, NOT ABOUT THE
ALGORITHM. The schedule is not neutral for it. Measured on 100 S2 instances at
t = 10000, the error of `argmax_a predicted_means` is

    forced round-robin           0.510
    its own pi^G + elimination   0.170

and the gap is not finite-sample: the observed-only least-squares limit names
the wrong arm on 52.0% of instances under a uniform allocation but on 16.0%
under the allocation the agent really produces. The reason is that elimination
collapses the allocation -- past its stop it has pulled an effective 1.1 arms
of 20 -- and a d = 8 linear model fits one or two arms essentially exactly.
The baseline is not recovering the latent structure; it has stopped asking the
question. Its honest error rate is the benchmark's, not this figure's.

What this figure does say, and what the benchmark cannot, is the
identifiability claim: given the same data covering every arm, the
observed-only model cannot fit the means at all (eps plateaus) while CoLF's
converges.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Callable, Dict, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import round_robin  # noqa: E402
from config import InstanceConfig, RunConfig  # noqa: E402
from agents.colf import AugmentedDesign, oracle_mu_star  # noqa: E402
from env import make_batch  # noqa: E402

AgentFactory = Callable[[np.ndarray, float, float, np.random.Generator], object]


def run_ablation(
    instance_cfg: InstanceConfig,
    agent_factories: Dict[str, AgentFactory],
    run_cfg: RunConfig,
) -> list[dict]:
    K = instance_cfg.K
    # Log-spaced record points, so memory stays bounded over long horizons, and
    # snapped up to whole sweeps: CoLF's width is a sweep-level object (the
    # imputation leak is zero only once the sweep telescopes), so a checkpoint
    # taken mid-sweep would read a width the algorithm never acts on.
    checkpoints = np.unique(
        K * np.ceil(
            np.geomspace(K, run_cfg.ablation_T, run_cfg.ablation_points) / K
        ).astype(int)
    ).tolist()

    instances = make_batch(instance_cfg, run_cfg.n_reps, run_cfg.seed0)
    records = []

    for inst in instances:
        D = AugmentedDesign(inst.observed())
        truth = oracle_mu_star(inst.X, inst.U, inst.theta_o, inst.theta_u, D)
        true_means = inst.means

        for agent_idx, (agent_name, factory) in enumerate(agent_factories.items()):
            rng = np.random.default_rng((inst.seed, agent_idx))
            # One name is privileged: "oracle" is handed the full feature
            # vector z_a = [x_a; u_a]. It is not a competitor but the ceiling,
            # the answer to "would seeing the latent features have helped".
            feats = (np.vstack([inst.X, inst.U]) if agent_name == "oracle"
                     else inst.observed())
            agent = factory(feats, run_cfg.delta, instance_cfg.sigma, rng)
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
                    # both agents report a reward-scale width, the same scale
                    # eps_base is on
                    "beta": agent.confidence_width(),
                    # would this agent name the right arm if it had to answer
                    # now?  Recorded so the figure can show the baseline's
                    # error *growing* with data, which its own stopping rule
                    # hides by firing before the bias takes over.
                    "correct": int(int(np.argmax(pm)) == int(np.argmax(true_means))),
                    "sigma_perp": None,
                    "perp_signal": None,
                }
                # The offset panel is about agents built on *this* design, so
                # only they get a row. An agent with the design but no
                # perpendicular block estimates the offset as identically
                # zero, and its error is therefore the whole offset -- which
                # is the point of ablating the augmentation, so record it.
                if hasattr(agent, "D") and agent.D.pi.shape == D.pi.shape:
                    perp = comp.get("perp")
                    perp_hat = D.B @ perp if perp is not None else np.zeros(K)
                    row["sigma_perp"] = float(np.max(np.abs(perp_hat - truth["perp_per_arm"])))
                    row["perp_signal"] = truth["perp_signal"]
                records.append(row)

                while cp_idx < len(checkpoints) and t >= checkpoints[cp_idx]:
                    cp_idx += 1

    return records


# --------------------------------------------------------------------------- #
# the feature contrast: same algorithm, different feature set
# --------------------------------------------------------------------------- #

def run_feature_contrast(
    instance_cfg: InstanceConfig,
    run_cfg: RunConfig,
    deltas: Sequence[float] = (0.2, 0.1, 0.05, 0.01),
) -> list[dict]:
    """Run CoLF twice on each instance: once on the observed features x_a, once
    on the full z_a = [x_a; u_a], across several confidence levels.

    This is the experiment that answers "did CoLF capture what the latent
    features contribute".  Asking it directly is not possible -- Eq. (8)
    reparameterises the split, absorbing the parallel part of the latent
    contribution into mu_star^(par), so neither block of mu_star *is* the
    latent contribution and Table 1 marks theta_star itself unidentifiable.
    What is identifiable is the counterfactual: hand the same algorithm the
    features it was denied and see what changes.  Nothing but the feature set
    differs, so the gap is the price of partial observability and nothing else.
    """
    from agents.colf import CoLFAgent
    from expt.runner import run_single

    instances = make_batch(instance_cfg, run_cfg.n_reps, run_cfg.seed0)
    records = []
    for inst in instances:
        Z = np.vstack([inst.X, inst.U]) if inst.U.size else inst.observed()
        for delta in deltas:
            for name, feats in (("colf", inst.observed()), ("colf_full", Z)):
                rng = np.random.default_rng((inst.seed, name == "colf_full"))
                agent = CoLFAgent(feats, delta, instance_cfg.sigma, rng)
                res = run_single(inst, agent, replace(run_cfg, delta=delta),
                                 noise_seed=inst.seed)
                records.append({
                    "agent": name,
                    "seed": inst.seed,
                    "delta": delta,
                    "T_delta": res.T_delta,
                    "correct": int(res.correct),
                    "censored": int(res.censored),
                })
    return records


def summarize_contrast(records: Sequence[dict]) -> list[dict]:
    """Per (agent, delta): median T_delta with the inter-quartile band, and the
    realised error rate."""
    keys = []
    for r in records:
        if (r["agent"], r["delta"]) not in keys:
            keys.append((r["agent"], r["delta"]))
    out = []
    for agent, delta in keys:
        rows = [r for r in records if r["agent"] == agent and r["delta"] == delta]
        T = np.array([r["T_delta"] for r in rows], dtype=float)
        out.append({
            "agent": agent, "delta": delta, "n": len(rows),
            "T_median": float(np.median(T)),
            "T_q25": float(np.percentile(T, 25)),
            "T_q75": float(np.percentile(T, 75)),
            "error_rate": float(np.mean([not r["correct"] for r in rows])),
        })
    return out


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
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng),
    }
    icfg = InstanceConfig(K=10, d=4, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.10)

    records = run_ablation(icfg, factories, run_cfg)
    assert records
    for key in ("agent", "seed", "t", "eps_base", "beta", "correct",
                "sigma_perp", "perp_signal"):
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
