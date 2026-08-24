"""Plotting for experiment 1 (benchmark) and experiment 2 (ablation).

Neither function talks to the environment, agents, or expt/*; both take plain
records so benchmark.py / ablation.py stay the only place that touches those.

Benchmark record (one row per scenario x agent x seed), matches expt/benchmark.py:
    scenario, agent, seed, T_delta, psi, a_star, correct, censored,
    reversal, obs_correct_prob, Delta_2

Ablation record (one row per agent x seed x checkpoint t), matches expt/ablation.py:
    agent, seed, t, sigma_perp, perp_signal, eps_base, beta
    sigma_perp / beta are None for agents without an augmented block / width.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


# --------------------------------------------------------------------------- #
# experiment 1: benchmark
# --------------------------------------------------------------------------- #

def summarize_benchmark(records: Sequence[dict]) -> dict:
    """Per (scenario, agent): T_delta mean/median/IQR, error rate, censor rate,
    and the error rate conditional on the instance being a ranking reversal."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["scenario"], r["agent"])].append(r)

    out = {}
    for key, rows in groups.items():
        T = np.array([r["T_delta"] for r in rows], dtype=float)
        correct = np.array([r["correct"] for r in rows], dtype=bool)
        censored = np.array([r["censored"] for r in rows], dtype=bool)
        rev = np.array([r["reversal"] for r in rows], dtype=bool)
        out[key] = {
            "n": len(rows),
            "T_mean": float(T.mean()),
            "T_median": float(np.median(T)),
            "T_q25": float(np.percentile(T, 25)),
            "T_q75": float(np.percentile(T, 75)),
            "error_rate": float(np.mean(~correct)),
            "censor_rate": float(np.mean(censored)),
            "error_rate_given_reversal": (
                float(np.mean(~correct[rev])) if rev.any() else float("nan")
            ),
        }
    return out


def plot_benchmark(
    records: Sequence[dict],
    scenarios: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[str]] = None,
    delta: float = 0.05,
    out_path: Optional[str] = None,
):
    """Top row: T_delta box plot per agent. Bottom row: error / censor rate bars
    with the delta target as a reference line. One column per scenario."""
    import matplotlib.pyplot as plt

    scenarios = list(scenarios) if scenarios else sorted({r["scenario"] for r in records})
    agents = list(agents) if agents else sorted({r["agent"] for r in records})

    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.2 * len(scenarios), 6.5), squeeze=False)

    for col, scen in enumerate(scenarios):
        rows = [r for r in records if r["scenario"] == scen]
        ax_t, ax_e = axes[0, col], axes[1, col]

        box_data = [[r["T_delta"] for r in rows if r["agent"] == ag] for ag in agents]
        ax_t.boxplot(box_data, tick_labels=agents, showmeans=True)
        ax_t.set_yscale("log")
        ax_t.set_title(scen)
        if col == 0:
            ax_t.set_ylabel(r"$T_\delta$")

        err, cens = [], []
        for ag in agents:
            sub = [r for r in rows if r["agent"] == ag]
            err.append(float(np.mean([not r["correct"] for r in sub])) if sub else float("nan"))
            cens.append(float(np.mean([r["censored"] for r in sub])) if sub else float("nan"))

        x = np.arange(len(agents))
        w = 0.35
        ax_e.bar(x - w / 2, err, width=w, label="error rate")
        ax_e.bar(x + w / 2, cens, width=w, alpha=0.5, label="censor rate")
        ax_e.axhline(delta, ls="--", color="k", lw=1, label=rf"$\delta$={delta}")
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(agents)
        ax_e.set_ylim(0, max(1.0, max(err + cens + [delta]) * 1.15))
        if col == 0:
            ax_e.set_ylabel("rate")
            ax_e.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# experiment 2: ablation
# --------------------------------------------------------------------------- #

def _aggregate(records: Sequence[dict], agent: str, field: str):
    """median / 10th / 90th percentile across seeds, at every logged t."""
    buckets = defaultdict(list)
    for r in records:
        if r["agent"] != agent or _isnan(r.get(field)):
            continue
        buckets[r["t"]].append(r[field])
    if not buckets:
        return None
    ts = sorted(buckets)
    med = np.array([np.median(buckets[t]) for t in ts])
    lo = np.array([np.percentile(buckets[t], 10) for t in ts])
    hi = np.array([np.percentile(buckets[t], 90) for t in ts])
    return np.array(ts, dtype=float), med, lo, hi


def plot_ablation(
    records: Sequence[dict],
    agents: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
):
    """Left: recovered orthogonal offset Sigma^perp(t) vs. the true signal size.
    Right: predicted-mean error eps^base(t) per agent, overlaid with beta_N(t)
    for agents that report a confidence width."""
    import matplotlib.pyplot as plt

    agents = list(agents) if agents else sorted({r["agent"] for r in records})
    fig, (ax_perp, ax_eps) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ag in agents:
        agg = _aggregate(records, ag, "sigma_perp")
        if agg is None:
            continue
        ts, med, lo, hi = agg
        line, = ax_perp.plot(ts, med, label=rf"$\Sigma^\perp$ ({ag})")
        ax_perp.fill_between(ts, lo, hi, alpha=0.2, color=line.get_color())

    for ag in agents:
        sig = _aggregate(records, ag, "perp_signal")
        if sig is None:
            continue
        _, med, lo, hi = sig
        ax_perp.axhline(float(np.median(med)), ls="--", color="k", lw=1,
                         label=r"$|\mu^{\star\perp}|$ (signal)")
        ax_perp.axhspan(float(np.median(lo)), float(np.median(hi)), color="k", alpha=0.05)
        break

    ax_perp.set_xscale("log")
    ax_perp.set_yscale("log")
    ax_perp.set_xlabel("$t$")
    ax_perp.set_ylabel("magnitude")
    ax_perp.set_title("orthogonal offset: estimate vs. true signal")
    ax_perp.legend(fontsize=8)

    for ag in agents:
        agg = _aggregate(records, ag, "eps_base")
        if agg is None:
            continue
        ts, med, lo, hi = agg
        line, = ax_eps.plot(ts, med, label=rf"$\varepsilon^{{base}}$ ({ag})")
        ax_eps.fill_between(ts, lo, hi, alpha=0.2, color=line.get_color())

        bagg = _aggregate(records, ag, "beta")
        if bagg is not None:
            bts, bmed, _, _ = bagg
            ax_eps.plot(bts, bmed, ls="--", color=line.get_color(), label=rf"$\beta_N$ ({ag})")

    ax_eps.set_xscale("log")
    ax_eps.set_yscale("log")
    ax_eps.set_xlabel("$t$")
    ax_eps.set_ylabel("magnitude")
    ax_eps.set_title("predicted-mean error vs. confidence width")
    ax_eps.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    import os
    import tempfile

    rng = np.random.default_rng(0)
    scratch = tempfile.mkdtemp(prefix="colf_plots_selftest_")

    # --- benchmark: gopt fails often on S2/S3, colf never does ---
    bench = []
    for scen, agent_fail_p in [("S1", {"gopt": 0.03, "colf": 0.03}),
                                ("S2", {"gopt": 0.55, "colf": 0.04}),
                                ("S3", {"gopt": 0.50, "colf": 0.05})]:
        for agent, p_fail in agent_fail_p.items():
            for seed in range(30):
                correct = rng.random() > p_fail
                reversal = scen != "S1"
                bench.append({
                    "scenario": scen, "agent": agent, "seed": seed,
                    "T_delta": int(rng.integers(500, 5000)),
                    "psi": 0, "a_star": 0, "correct": bool(correct),
                    "censored": bool(scen == "S3" and agent == "gopt" and rng.random() < 0.3),
                    "reversal": reversal, "obs_correct_prob": 0.5, "Delta_2": 0.2,
                })

    summary = summarize_benchmark(bench)
    for scen in ("S1", "S2", "S3"):
        assert summary[(scen, "colf")]["error_rate"] < summary[(scen, "gopt")]["error_rate"] or scen == "S1"
    assert summary[("S3", "gopt")]["censor_rate"] > 0.0

    fig1 = plot_benchmark(bench, delta=0.05, out_path=os.path.join(scratch, "benchmark_selftest.png"))
    assert fig1.axes and len(fig1.axes) == 6
    assert os.path.getsize(os.path.join(scratch, "benchmark_selftest.png")) > 0

    # --- ablation: colf's offset error and eps decay, gopt's eps plateaus ---
    ts = np.unique(np.geomspace(20, 20_000, 40).astype(int))
    abl = []
    for seed in range(10):
        signal = 0.1 + 0.02 * rng.standard_normal()
        bias = 0.08 + 0.01 * rng.standard_normal()
        for t in ts:
            noise = 0.05 * rng.standard_normal()
            abl.append({
                "agent": "colf", "seed": seed, "t": int(t),
                "sigma_perp": abs(signal * np.exp(-t / 3000) + 0.01 * noise),
                "perp_signal": abs(signal),
                "eps_base": abs(0.3 / np.sqrt(t) + 0.02 * noise),
                "beta": 0.6 / np.sqrt(t),
            })
            abl.append({
                "agent": "gopt", "seed": seed, "t": int(t),
                "sigma_perp": None,
                "perp_signal": None,
                "eps_base": abs(bias + 0.1 / np.sqrt(t) + 0.02 * noise),
                "beta": None,
            })

    agg = _aggregate(abl, "colf", "sigma_perp")
    assert agg is not None and agg[1][-1] < agg[1][0], "sigma_perp should decay"
    agg_base = _aggregate(abl, "gopt", "eps_base")
    agg_colf = _aggregate(abl, "colf", "eps_base")
    assert agg_base[1][-1] > agg_colf[1][-1], "baseline should not converge to zero"

    fig2 = plot_ablation(abl, out_path=os.path.join(scratch, "ablation_selftest.png"))
    assert len(fig2.axes) == 2
    assert os.path.getsize(os.path.join(scratch, "ablation_selftest.png")) > 0

    print("plots.py self-test passed")


if __name__ == "__main__":
    _selftest()
