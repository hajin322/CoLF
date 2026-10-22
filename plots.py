"""Plotting for experiment 1 (benchmark) and experiment 2 (ablation).

Neither function talks to the environment, agents, or expt/*; both take plain
records so benchmark.py / ablation.py stay the only place that touches those.

Benchmark record (one row per scenario x agent x seed), matches expt/benchmark.py:
    scenario, agent, seed, T_delta, psi, a_star, correct, censored,
    reversal, proj_reversal, shift_spread, Delta_2

Ablation record (one row per agent x seed x checkpoint t), matches expt/ablation.py:
    agent, seed, t, sigma_perp, perp_signal, eps_base, beta, correct
    sigma_perp / beta are None for agents without an augmented block / width.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def binom_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound on a rate, exact by bisection.

    A measured 0/200 is not evidence that the true error rate is zero -- it is
    evidence that it is below 1.5%. Plotting the point estimate alone would
    claim more than the runs support, and on a log axis a zero cannot be drawn
    at all.
    """
    from math import comb
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0

    def cdf(pp):                       # P(X <= k) for X ~ Binomial(n, pp)
        return sum(comb(n, i) * pp ** i * (1 - pp) ** (n - i) for i in range(k + 1))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _decade_ticks(ax, which: str = "y") -> None:
    """Label 1-2-3-5 within each decade instead of only the decade itself.

    These axes span barely one decade, where powers of ten alone leave two
    labels and no way to read a value off the plot."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


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
        prev = np.array([r.get("proj_reversal", False) for r in rows], dtype=bool)
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
            "error_rate_given_proj_reversal": (
                float(np.mean(~correct[prev])) if prev.any() else float("nan")
            ),
            "proj_reversal_rate": float(np.mean(prev)),
        }
    return out


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
    """Top row: sample complexity per agent, shared y so the panels compare.
    Bottom row: how often each agent named a suboptimal arm, or never stopped.

    Neither outliers nor the mean are drawn.  Censored runs sit at T_max, so
    they would stretch a log axis over four decades and would drag the mean
    above the box while not being a mean at all -- a censored run's true
    stopping time is unknown.  The bottom row counts them instead."""
    import matplotlib.pyplot as plt

    scenarios = list(scenarios) if scenarios else sorted({r["scenario"] for r in records})
    agents = list(agents) if agents else _order({r["agent"] for r in records})
    styles = {ag: _agent_style(ag, i) for i, ag in enumerate(agents)}

    fig, axes = plt.subplots(
        2, len(scenarios), figsize=(3.6 * len(scenarios), 6.0), squeeze=False,
        sharey="row", gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    rates = [
        r for scen in scenarios for ag in agents
        for r in _rate_pair([x for x in records if x["scenario"] == scen and x["agent"] == ag])
    ]
    ymax = max([delta] + [v for v in rates if not _isnan(v)]) * 1.35

    for col, scen in enumerate(scenarios):
        rows = [r for r in records if r["scenario"] == scen]
        ax_t, ax_e = axes[0, col], axes[1, col]

        box = ax_t.boxplot(
            [[r["T_delta"] for r in rows if r["agent"] == ag] for ag in agents],
            tick_labels=[styles[ag][1].split(" (")[0] for ag in agents],
            showmeans=False, showfliers=False, patch_artist=True, widths=0.55,
            medianprops=dict(color="black", lw=1.4),
        )
        for patch, ag in zip(box["boxes"], agents):
            patch.set_facecolor(styles[ag][0])
            patch.set_alpha(0.35)
        ax_t.set_yscale("log")
        ax_t.set_title(titles.get(scen, scen) if titles else scen, fontsize=11)
        _clean_log(ax_t, "y")
        if col == 0:
            ax_t.set_ylabel("sample complexity")

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


def summarize_ablation(records: Sequence[dict], n_rows: int = 6) -> list[dict]:
    """A few log-spaced checkpoints: per agent, the median predicted-mean error,
    the median width, and their ratio (a valid width keeps the ratio above one
    and roughly flat in t)."""
    ts = sorted({r["t"] for r in records})
    picks = [ts[i] for i in np.unique(np.linspace(0, len(ts) - 1, n_rows).astype(int))]
    agents = sorted({r["agent"] for r in records})

    out = []
    for t in picks:
        for ag in agents:
            rows = [r for r in records if r["agent"] == ag and r["t"] == t]
            eps = [r["eps_base"] for r in rows if not _isnan(r.get("eps_base"))]
            bet = [r["beta"] for r in rows if not _isnan(r.get("beta"))]
            perp = [r["sigma_perp"] for r in rows if not _isnan(r.get("sigma_perp"))]
            ok = [r["correct"] for r in rows if not _isnan(r.get("correct"))]
            if not eps:
                continue
            e = float(np.median(eps))
            b = float(np.median(bet)) if bet else float("nan")
            out.append({
                "t": t, "agent": ag, "n": len(eps), "eps_base": e, "beta": b,
                "ratio": b / e if bet else float("nan"),
                "error": 1.0 - float(np.mean([float(v) for v in ok])) if ok else float("nan"),
                "sigma_perp": float(np.median(perp)) if perp else float("nan"),
            })
    return out


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
        agg = _aggregate(records, ag, "eps_base")
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
        e = _aggregate(records, ag, "eps_base")
        b = _aggregate(records, ag, "beta")
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
        ax_cov.text(0.97, 0.93, "error zone", transform=ax_cov.transAxes,
                    ha="right", va="top", fontsize=9, color="#d62728")
    ax_cov.axhline(1.0, ls="--", color="k", lw=1.1, zorder=1)

    ax_cov.set_ylabel(r"$\max_a\,|\hat Y_a - \mathbf{z}_a^\top"
                      r"\boldsymbol{\theta}_\star|\ /\ \beta_N$")

    for ax, loc in ((ax_eps, "lower left"), (ax_cov, "upper left")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$t$")
        _clean_log(ax)
        ax.legend(fontsize=8, loc=loc)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# experiment 3: one-factor sweeps
# --------------------------------------------------------------------------- #

FACTOR_LABEL = {
    "du": ("$d_u$", "how much of the reward is hidden"),
    "delta": (r"$\delta$", r"is the realised error under $\delta$?"),
    "min_gap": (r"$\Delta_2$ floor", r"the $1/\Delta^2$ term and its floor"),
    "K": ("$K$", "Theorem 1's leading term"),
}


def plot_sweep(
    summary: Sequence[dict],
    out_path: Optional[str] = None,
    delta: float = 0.05,
    K: Optional[int] = None,
):
    """One column per factor. Top: error rate against the delta target.
    Bottom: median T_delta with the inter-quartile band.

    `summary` is the output of expt/sweep.py::summarize_sweep, possibly
    concatenated over several factors.
    """
    import matplotlib.pyplot as plt

    factors = []
    for row in summary:
        if row["factor"] not in factors:
            factors.append(row["factor"])
    agents = sorted({row["agent"] for row in summary})
    styles = {ag: _agent_style(ag, i) for i, ag in enumerate(agents)}

    fig, axes = plt.subplots(2, len(factors), figsize=(3.7 * len(factors), 6.2),
                             squeeze=False)

    for col, factor in enumerate(factors):
        ax_e, ax_t = axes[0, col], axes[1, col]
        xlabel = FACTOR_LABEL.get(factor, (factor, factor))[0]
        log_x = factor in ("delta", "K")

        for ag in agents:
            rows = sorted([r for r in summary
                           if r["factor"] == factor and r["agent"] == ag],
                          key=lambda r: r["value"])
            if not rows:
                continue
            x = np.array([r["value"] for r in rows], dtype=float)
            color, label = styles[ag]
            e = np.array([r["error_rate"] for r in rows])
            # a measured zero is only evidence of a small rate, so carry the
            # one-sided 95% upper bound; without it CoLF's curve is a flat
            # line on zero that says more than the runs do
            up = np.array([binom_upper(int(round(r["error_rate"] * r["n"])), r["n"])
                           for r in rows])
            ax_e.fill_between(x, e, up, color=color, alpha=0.18)
            ax_e.plot(x, e, "o-", color=color, label=label, ms=4)
            ax_t.plot(x, [r["T_median"] for r in rows], "o-", color=color,
                      label=label, ms=4)
            ax_t.fill_between(x, [r["T_q25"] for r in rows],
                              [r["T_q75"] for r in rows], color=color, alpha=0.18)

        if factor == "delta":
            lo = min(r["value"] for r in summary if r["factor"] == factor)
            hi = max(r["value"] for r in summary if r["factor"] == factor)
            ax_e.plot([lo, hi], [lo, hi], "k--", lw=1.1, label=r"$\delta$ target")
        else:
            ax_e.axhline(delta, ls="--", color="k", lw=1.1, label=rf"$\delta$={delta}")

        ax_e.set_ylabel("error rate" if col == 0 else None)
        ax_e.set_ylim(0, None)
        ax_t.set_ylabel("sample complexity(median)" if col == 0 else None)
        ax_t.set_yscale("log")
        # the levels span well under a decade; letting the axis start at zero
        # squashes every curve into the top third of the panel
        tv = [v for r in summary if r["factor"] == factor
              for v in (r["T_q25"], r["T_q75"])]
        ax_t.set_ylim(min(tv) / 1.25, max(tv) * 1.25)
        _decade_ticks(ax_t, "y")
        xs = sorted({r["value"] for r in summary if r["factor"] == factor})
        for ax in (ax_e, ax_t):
            ax.set_xlabel(xlabel)
            if log_x:
                ax.set_xscale("log")
            _clean_log(ax, "both" if log_x else "y")
            # a log axis over 10..50 or 0.01..0.2 labels one decade and hides
            # the rest, so tick the levels that were actually run
            if log_x:
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{v:g}" for v in xs])
            ax.legend(fontsize=8)

    fig.tight_layout()
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
                    "reversal": reversal, "proj_reversal": reversal,
                    "shift_spread": 0.2, "Delta_2": 0.2,
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
                "correct": int(t > 100),
            })
            abl.append({
                "agent": "gopt", "seed": seed, "t": int(t),
                "sigma_perp": None,
                "perp_signal": None,
                "eps_base": abs(bias + 0.1 / np.sqrt(t) + 0.02 * noise),
                "beta": None,
                "correct": int(rng.random() > min(0.6, t / 20_000)),
            })

    summ = summarize_ablation(abl, n_rows=4)
    assert {r["agent"] for r in summ} == {"colf", "gopt"}
    colf_rows = [r for r in summ if r["agent"] == "colf"]
    assert all(r["ratio"] > 1.0 for r in colf_rows), "width should cover the error"
    assert colf_rows[-1]["eps_base"] < colf_rows[0]["eps_base"], "eps should decay"
    assert np.isnan([r for r in summ if r["agent"] == "gopt"][0]["beta"]), \
        "an agent with no width should report beta as nan"

    agg = _aggregate(abl, "colf", "sigma_perp")
    assert agg is not None and agg[1][-1] < agg[1][0], "sigma_perp should decay"
    agg_base = _aggregate(abl, "gopt", "eps_base")
    agg_colf = _aggregate(abl, "colf", "eps_base")
    assert agg_base[1][-1] > agg_colf[1][-1], "baseline should not converge to zero"

    # --- sweeps: baseline error grows with du, CoLF's does not ---
    sw = []
    for du in (0, 1, 2, 4, 8):
        for ag, e, T in (("colf", 0.01, 300 + 20 * du), ("gopt", 0.02 + 0.06 * du, 260 + 15 * du)):
            sw.append({"factor": "du", "value": du, "agent": ag, "n": 50,
                       "error_rate": e, "censor_rate": 0.0, "T_median": T,
                       "T_q25": T * 0.7, "T_q75": T * 1.4, "Delta_2": 0.25,
                       "proj_reversal_rate": min(0.5, 0.1 * du)})
    for d_ in (0.2, 0.1, 0.05, 0.01):
        for ag, e in (("colf", d_ * 0.3), ("gopt", 0.12)):
            sw.append({"factor": "delta", "value": d_, "agent": ag, "n": 50,
                       "error_rate": e, "censor_rate": 0.0,
                       "T_median": 300 / d_ ** 0.2, "T_q25": 200, "T_q75": 500,
                       "Delta_2": 0.25, "proj_reversal_rate": 0.5})
    assert binom_upper(0, 200) < 0.02 and binom_upper(0, 20) > 0.1, "CP upper bound"
    assert 0.05 < binom_upper(5, 100) < 0.12, f"CP upper bound: {binom_upper(5, 100)}"
    fig3 = plot_sweep(sw, delta=0.05, K=20,
                      out_path=os.path.join(scratch, "sweep_selftest.png"))
    assert len(fig3.axes) == 4, "one column per factor, two rows"
    assert os.path.getsize(os.path.join(scratch, "sweep_selftest.png")) > 0

    fig2 = plot_ablation(abl, out_path=os.path.join(scratch, "ablation_selftest.png"))
    assert len(fig2.axes) == 2
    assert os.path.getsize(os.path.join(scratch, "ablation_selftest.png")) > 0

    print("plots.py self-test passed")


if __name__ == "__main__":
    _selftest()
