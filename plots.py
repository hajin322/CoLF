"""Figure 1 (benchmark) and Figure 2 (ablation).

Both functions take plain records, so expt/benchmark.py and expt/ablation.py
stay the only place that touches the environment or the agents.

Benchmark record (one row per scenario x agent x instance), expt/benchmark.py:
    scenario, agent, seed, T_delta, correct, censored, Delta_2

Ablation record (one row per agent x instance x checkpoint t), expt/ablation.py:
    agent, seed, t, eps, width
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _clean_log(ax, which: str = "both") -> None:
    """Log axes here span two or three decades, where matplotlib's minor ticks
    are eight unlabelled marks per decade and read as clutter. Major only."""
    from matplotlib.ticker import NullLocator
    if which in ("x", "both"):
        ax.xaxis.set_minor_locator(NullLocator())
    if which in ("y", "both"):
        ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(which="major", ls=":", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.set_axisbelow(True)


def _rate_pair(rows) -> tuple:
    """(error rate, censor rate) over a group of runs."""
    if not rows:
        return float("nan"), float("nan")
    return (float(np.mean([not r["correct"] for r in rows])),
            float(np.mean([r["censored"] for r in rows])))


# --------------------------------------------------------------------------- #
# Figure 1: benchmark
# --------------------------------------------------------------------------- #

# Colour is the method family, dash is the variant within it: CoLF blue,
# G-allocation red, the variant of each dashed.
AGENT_COLOR = {"colf": "#1f77b4", "colf_noaug": "#1f77b4",
               "gopt": "#d62728", "oracle": "#d62728"}
AGENT_DASH = {"colf_noaug": (4, 2), "oracle": (4, 2)}
AGENT_LABEL = {"colf": "CoLF",
               "colf_noaug": "CoLF without augmentation",
               "gopt": "G-allocation",
               "oracle": "G-allocation with true feature"}


AGENT_ORDER = ["colf", "colf_noaug", "gopt", "oracle"]


def _order(names) -> list:
    """Method first, baseline next, reference last -- not alphabetical."""
    return sorted(names, key=lambda a: (AGENT_ORDER.index(a)
                                        if a in AGENT_ORDER else len(AGENT_ORDER), a))


def _agent_style(agent: str, i: int):
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    return AGENT_COLOR.get(agent, palette[i % len(palette)]), AGENT_LABEL.get(agent, agent)


def _dash(agent: str) -> dict:
    d = AGENT_DASH.get(agent)
    return {"dashes": d} if d else {}


def plot_benchmark(
    records: Sequence[dict],
    scenarios: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[str]] = None,
    delta: float = 0.05,
    out_path: Optional[str] = None,
    titles: Optional[dict] = None,
):
    """One panel per scenario: how often each agent named a suboptimal arm, or
    never stopped.

    What is on trial here is soundness, so there is no stopping-time panel: one
    beside it invites the reading that the two trade off -- they do not, since
    a censored run has no stopping time to put on the axis and a fast wrong
    answer is not a cheap one."""
    import matplotlib.pyplot as plt

    scenarios = list(scenarios) if scenarios else sorted({r["scenario"] for r in records})
    agents = list(agents) if agents else _order({r["agent"] for r in records})
    styles = {ag: _agent_style(ag, i) for i, ag in enumerate(agents)}

    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(3.6 * len(scenarios), 3.2), squeeze=False,
        sharey=True,
    )

    rates = [
        r for scen in scenarios for ag in agents
        for r in _rate_pair([x for x in records if x["scenario"] == scen and x["agent"] == ag])
    ]
    ymax = max([delta] + [v for v in rates if not _isnan(v)]) * 1.35

    for col, scen in enumerate(scenarios):
        rows = [r for r in records if r["scenario"] == scen]
        ax_e = axes[0, col]
        ax_e.set_title(titles.get(scen, scen) if titles else scen, fontsize=11)

        err, cens = zip(*[_rate_pair([r for r in rows if r["agent"] == ag])
                          for ag in agents])
        x, w = np.arange(len(agents)), 0.34
        b1 = ax_e.bar(x - w / 2, err, width=w,
                      color=[styles[ag][0] for ag in agents])
        ax_e.bar(x + w / 2, cens, width=w,
                 color=[styles[ag][0] for ag in agents], alpha=0.35, hatch="//")
        # the delta line is back: the bars now straddle it, so it carries the
        # claim rather than sitting far below every bar
        ax_e.axhline(delta, ls="--", color="k", lw=1.1, zorder=3)
        if col == len(scenarios) - 1:
            ax_e.text(len(agents) - 0.55, delta, rf"$\delta$={delta}",
                      va="bottom", ha="right", fontsize=8)
        for rect, v in zip(b1, err):
            ax_e.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v),
                          textcoords="offset points", xytext=(0, 2),
                          ha="center", fontsize=8)
        ax_e.set_xticks(x)
        ax_e.set_xticklabels([styles[ag][1].split(" (")[0] for ag in agents])
        ax_e.set_ylim(0, ymax)
        ax_e.grid(axis="y", ls=":", alpha=0.5)
        ax_e.set_axisbelow(True)
        if col == 0:
            ax_e.set_ylabel("rate")
            # the bars are coloured by agent, so the legend may only speak
            # about solid-vs-hatched; a coloured key would say the wrong thing
            from matplotlib.patches import Patch
            ax_e.legend(
                handles=[Patch(facecolor="0.45", label="recommend suboptimal arm"),
                         Patch(facecolor="0.45", alpha=0.35, hatch="//",
                               label="hit maximum complexity")],
                fontsize=8, loc="upper left", framealpha=0.9,
            )

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Figure 2: ablation
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
    """One question per panel, over an ablation ladder whose adjacent rungs
    differ by exactly one thing (see expt/ablation.py).

    1. Does the predicted mean converge?  max_a |Yhat_a - z_a^T theta_star|.
    2. Is the confidence width honest?  that same error over beta_N, against
       the line at one.  Above it the agent is confident and wrong, so lower
       is better on both panels.
    """
    import matplotlib.pyplot as plt

    agents = list(agents) if agents else _order({r["agent"] for r in records})
    styles = {ag: _agent_style(ag, i) for i, ag in enumerate(agents)}
    fig, (ax_eps, ax_cov) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # --- 2. the predicted-mean error ------------------------------------- #
    for ag in agents:
        agg = _aggregate(records, ag, "eps")
        if agg is None:
            continue
        ts, med, lo, hi = agg
        color = styles[ag][0]
        ax_eps.plot(ts, med, color=color, label=styles[ag][1], **_dash(ag))
        ax_eps.fill_between(ts, lo, hi, alpha=0.2, color=color)

    ax_eps.set_ylabel(r"$\max_a\,|\hat Y_a - \mathbf{z}_a^\top\boldsymbol{\theta}_\star|$")

    # --- 3. does the width cover the error? ------------------------------ #
    # log scale first: the shaded band below y = 1 is placed from the data
    # limits, and those are meaningless while the axis is still linear.
    ax_cov.set_xscale("log")
    ax_cov.set_yscale("log")

    curves, seen = {}, []
    for ag in agents:
        e = _aggregate(records, ag, "eps")
        b = _aggregate(records, ag, "width")
        if e is None or b is None:
            continue
        e_at = dict(zip(e[0], e[1]))
        ts = np.array([t for t in b[0] if t in e_at])
        # error over width: below one the width covers the error, and the
        # error is the numerator on both panels, so down is good on both.
        curves[ag] = (ts, np.array([e_at[t] / b[1][list(b[0]).index(t)] for t in ts]))
        seen.append(curves[ag][1])

    for ag, (ts, ratio) in curves.items():
        ax_cov.plot(ts, ratio, color=styles[ag][0], label=styles[ag][1], **_dash(ag))

    if seen:
        flat = np.concatenate(seen)
        lo, hi = min(float(flat.min()) / 1.6, 0.5), float(flat.max()) * 1.6
        ax_cov.set_ylim(lo, hi)
        ax_cov.axhspan(1.0, hi, color="#d62728", alpha=0.08, zorder=0)
        # just above the line it labels, at the right edge: the rising curves
        # cross y = 1 on the left and occupy the top right, so this corner is
        # the one region of the band no curve passes through
        ax_cov.text(0.97, 1.12, "error zone",
                    transform=ax_cov.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=9, color="#d62728")
    ax_cov.axhline(1.0, ls="--", color="k", lw=1.1, zorder=1)

    ax_cov.set_ylabel(r"$\max_a\,|\hat Y_a - \mathbf{z}_a^\top"
                      r"\boldsymbol{\theta}_\star|\ /\ \beta_N$")

    for ax in (ax_eps, ax_cov):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$t$")
        _clean_log(ax)

    # one legend for both panels, below them, so it covers no data
    handles, labels = ax_eps.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    import os
    import tempfile

    rng = np.random.default_rng(0)
    scratch = tempfile.mkdtemp(prefix="colf_plots_selftest_")

    bench = []
    for scen, fail in (("S1", {"gopt": 0.03, "colf": 0.0}),
                       ("S2", {"gopt": 0.30, "colf": 0.0}),
                       ("S3", {"gopt": 0.35, "colf": 0.0})):
        for agent, p_fail in fail.items():
            for seed in range(30):
                bench.append({
                    "scenario": scen, "agent": agent, "seed": seed,
                    "T_delta": int(rng.integers(500, 5000)),
                    "correct": bool(rng.random() > p_fail),
                    "censored": bool(scen == "S3" and agent == "gopt"
                                     and rng.random() < 0.1),
                    "Delta_2": 0.2,
                })
    path = os.path.join(scratch, "benchmark.png")
    fig1 = plot_benchmark(bench, delta=0.05, out_path=path)
    assert len(fig1.axes) == 3 and os.path.getsize(path) > 0, "one panel per scenario"

    abl = []
    for seed in range(10):
        for t in np.unique(np.geomspace(20, 20_000, 40).astype(int)):
            noise = 0.05 * rng.standard_normal()
            abl.append({"agent": "colf", "seed": seed, "t": int(t),
                        "eps": abs(0.3 / np.sqrt(t) + 0.02 * noise),
                        "width": 0.6 / np.sqrt(t)})
            abl.append({"agent": "gopt", "seed": seed, "t": int(t),
                        "eps": abs(0.5 + 0.1 / np.sqrt(t) + 0.02 * noise),
                        "width": 0.2 / np.sqrt(t)})
    ts, colf_eps, _, _ = _aggregate(abl, "colf", "eps")
    assert colf_eps[-1] < colf_eps[0], "the median error should decay"
    path = os.path.join(scratch, "ablation.png")
    fig2 = plot_ablation(abl, out_path=path)
    assert len(fig2.axes) == 2 and os.path.getsize(path) > 0

    print("plots.py self-test passed")


if __name__ == "__main__":
    _selftest()
