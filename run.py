"""CLI entry point: config -> instances -> agents -> experiment -> plots.

    python run.py benchmark   experiment 1: sample complexity + error rate
    python run.py ablation    experiment 2: estimation-error diagnostics
    python run.py sweep       experiment 3: one factor at a time
    python run.py             self-test (small end-to-end run, ~5s)

Agent imports are deferred into the command functions so that `-h` and the
self-test's argument checks stay cheap. The seams:

    agents.gopt.GOptAgent(X, delta, sigma, rng=None)
    agents.colf.CoLFAgent(X, delta, sigma, rng=None, **algorithm_knobs)
    expt.benchmark.run_benchmark(instance_cfgs, agent_factories, run_cfg)
        -> list[dict]  (schema: plots.summarize_benchmark / plots.plot_benchmark)
    expt.ablation.run_ablation(instance_cfg, agent_factories, run_cfg)
        -> list[dict]  (schema: plots.plot_ablation)
    expt.sweep.run_sweep(factor, values, base_cfg, agent_factories, run_cfg)
        -> list[dict]  (schema: expt.sweep.summarize_sweep -> plots.plot_sweep)

`agent_factories` is {name: Callable[[X, delta, sigma, rng], Agent]}.

Every default is read from InstanceConfig / RunConfig rather than written out
here, so the CLI cannot drift from the dataclasses; _selftest pins that.
Algorithm knobs belong to the agents, so they are wired in _agent_factories.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from config import DEFAULT_SCENARIOS, InstanceConfig, RunConfig, Scenario
import plots

SCENARIO_BY_NAME = {s.value: s for s in Scenario}


def _agent_factories(args: argparse.Namespace) -> dict:
    """Algorithm knobs live on the agents, so they are wired in here rather
    than carried around inside RunConfig."""
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent

    f = {
        "gopt": lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng),
        "colf": lambda X, delta, sigma, rng: CoLFAgent(
            X, delta, sigma, rng, beta_scale=args.beta_scale, ipw=args.ipw,
            p=args.p),
    }
    return f


def _run_config(args: argparse.Namespace) -> RunConfig:
    kwargs = dict(delta=args.delta, n_reps=args.n_reps, T_max=args.t_max,
                  seed0=args.seed0)
    if hasattr(args, "ablation_T"):
        kwargs["ablation_T"] = args.ablation_T
        kwargs["ablation_points"] = args.ablation_points
    return RunConfig(**kwargs)


def _instance_configs(args: argparse.Namespace) -> Sequence[InstanceConfig]:
    names = args.scenarios or [s.value for s in DEFAULT_SCENARIOS]
    base = dict(K=args.K, d=args.d, sigma=args.sigma, min_gap=args.min_gap)
    return [InstanceConfig(scenario=SCENARIO_BY_NAME[name], **base) for name in names]


def _write_csv(records: Sequence[dict], path: Path) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _print_summary(summary: dict) -> None:
    print(f"{'scenario':<10}{'agent':<8}{'n':>5}{'T_median':>10}"
          f"{'error':>8}{'censor':>8}{'err|rev':>9}{'err|proj-rev':>14}")
    for (scen, agent), s in sorted(summary.items()):
        print(f"{scen:<10}{agent:<8}{s['n']:>5}{s['T_median']:>10.1f}"
              f"{s['error_rate']:>8.3f}{s['censor_rate']:>8.3f}"
              f"{s['error_rate_given_reversal']:>9.3f}"
              f"{s['error_rate_given_proj_reversal']:>14.3f}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from expt.benchmark import run_benchmark

    run_cfg = _run_config(args)
    instance_cfgs = _instance_configs(args)
    records = run_benchmark(instance_cfgs, _agent_factories(args), run_cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, out_dir / "benchmark.csv")
    _print_summary(plots.summarize_benchmark(records))
    plots.plot_benchmark(
        records, delta=run_cfg.delta,
        titles={s.value: s.label for s in Scenario},
        out_path=out_dir / "benchmark.png",
    )


def cmd_ablation(args: argparse.Namespace) -> None:
    from expt.ablation import run_ablation

    run_cfg = _run_config(args)
    scenario = SCENARIO_BY_NAME[args.ablation_scenario]
    instance_cfg = InstanceConfig(
        K=args.K, d=args.d, scenario=scenario, sigma=args.sigma,
        min_gap=args.min_gap,
    )
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent
    factories = _agent_factories(args)
    # the ablation ladder: remove one thing at a time, and a ceiling
    factories["colf_noaug"] = lambda X, delta, sigma, rng: CoLFAgent(
        X, delta, sigma, rng, beta_scale=args.beta_scale, ipw=args.ipw,
        p=args.p, augment=False)
    factories["oracle"] = lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng)
    records = run_ablation(instance_cfg, factories, run_cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, out_dir / "ablation.csv")

    print(f"{'t':>8}  {'agent':<6}{'n':>5}{'eps_base':>11}{'beta':>11}"
          f"{'beta/eps':>10}{'error':>8}{'sigma_perp':>12}")
    for r in plots.summarize_ablation(records):
        print(f"{r['t']:>8}  {r['agent']:<6}{r['n']:>5}{r['eps_base']:>11.5f}"
              f"{r['beta']:>11.5f}{r['ratio']:>10.2f}{r['error']:>8.3f}"
              f"{r['sigma_perp']:>12.5f}")

    plots.plot_ablation(records, out_path=out_dir / "ablation.png")


# factor -> the levels swept when `run.py sweep` is called without --values.
# Chosen so each of the draft's claims gets its own curve; see expt/sweep.py.
SWEEP_LEVELS = {
    "du": [0, 1, 2, 4, 8, 16],
    "delta": [0.2, 0.1, 0.05, 0.01],
    "min_gap": [0.10, 0.20, 0.30, 0.40, 0.50],
    "K": [10, 20, 30, 40, 50],
}


def cmd_sweep(args: argparse.Namespace) -> None:
    from expt.sweep import run_sweep, summarize_sweep

    run_cfg = _run_config(args)
    base = InstanceConfig(K=args.K, d=args.d, scenario=SCENARIO_BY_NAME[args.scenario],
                          sigma=args.sigma, min_gap=args.min_gap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, records = [], []
    for factor in args.factors:
        values = args.values if args.values else SWEEP_LEVELS[factor]
        values = [type(SWEEP_LEVELS[factor][0])(v) for v in values]
        rows = run_sweep(factor, values, base, _agent_factories(args), run_cfg)
        records.extend(rows)
        summary.extend(summarize_sweep(rows))

    _write_csv(records, out_dir / "sweep.csv")
    print(f"{'factor':<9}{'value':>8}{'agent':>7}{'n':>5}{'error':>8}{'censor':>8}"
          f"{'T_med':>9}{'Delta_2':>9}{'proj-rev':>10}")
    for r in summary:
        print(f"{r['factor']:<9}{r['value']:>8}{r['agent']:>7}{r['n']:>5}"
              f"{r['error_rate']:>8.3f}{r['censor_rate']:>8.3f}{r['T_median']:>9.0f}"
              f"{r['Delta_2']:>9.4f}{r['proj_reversal_rate']:>10.2f}")

    plots.plot_sweep(summary, delta=run_cfg.delta, K=args.K,
                     out_path=out_dir / "sweep.png")


def _add_common_args(sp: argparse.ArgumentParser) -> None:
    # Defaults are read from the dataclasses, never retyped: a hardcoded CLI
    # default once shadowed a config field and silently reverted every run.
    ic, rc = InstanceConfig(), RunConfig()
    # CoLF's constructor defaults, read the same way rather than retyped
    import inspect
    from agents.colf import CoLFAgent
    cd = {k: v.default for k, v in inspect.signature(CoLFAgent).parameters.items()}
    sp.add_argument("--K", type=int, default=ic.K)
    sp.add_argument("--d", type=int, default=ic.d)
    sp.add_argument("--sigma", type=float, default=ic.sigma)
    sp.add_argument("--min-gap", type=float, default=ic.min_gap)
    sp.add_argument("--delta", type=float, default=rc.delta)
    sp.add_argument("--t-max", type=int, default=rc.T_max)
    sp.add_argument("--seed0", type=int, default=rc.seed0)
    # CoLF's own knobs, not run settings.
    sp.add_argument("--beta-scale", type=float, default=cd["beta_scale"])
    # w = 1/p is the draft's Eq. (12)-(13); w = K is the value at which the
    # imputation drops out of mu_hat entirely. See agents/colf.py.
    sp.add_argument("--ipw", choices=("paper", "round_robin"), default=cd["ipw"])
    sp.add_argument("--p", type=float, default=cd["p"],
                    help="pseudo-action coupling probability, Eq. (12)")
    sp.add_argument("--out-dir", default="results")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CoLF experiments (BAI under partial observability)")
    sub = p.add_subparsers(dest="cmd", required=True)

    rc = RunConfig()

    b = sub.add_parser("benchmark", help="experiment 1: sample complexity + error rate")
    b.add_argument("--scenarios", nargs="+", choices=list(SCENARIO_BY_NAME), default=None)
    b.add_argument("--n-reps", type=int, default=rc.n_reps)
    _add_common_args(b)
    b.set_defaults(func=cmd_benchmark)

    a = sub.add_parser("ablation", help="experiment 2: estimation-error diagnostics")
    a.add_argument("--ablation-scenario", choices=list(SCENARIO_BY_NAME), default=Scenario.PARTIAL.value)
    a.add_argument("--n-reps", type=int, default=rc.n_reps)
    a.add_argument("--ablation-T", type=int, default=rc.ablation_T, dest="ablation_T")
    a.add_argument("--ablation-points", type=int, default=rc.ablation_points)
    _add_common_args(a)
    a.set_defaults(func=cmd_ablation)

    w = sub.add_parser("sweep", help="experiment 3: one factor at a time")
    w.add_argument("--factors", nargs="+", choices=list(SWEEP_LEVELS),
                   default=list(SWEEP_LEVELS))
    w.add_argument("--values", nargs="+", default=None,
                   help="override the levels; only meaningful with one factor")
    w.add_argument("--scenario", choices=list(SCENARIO_BY_NAME),
                   default=Scenario.PARTIAL.value)
    w.add_argument("--n-reps", type=int, default=rc.n_reps)
    _add_common_args(w)
    w.set_defaults(func=cmd_sweep)
    return p


def _parse(argv: Sequence[str], cmd: str) -> argparse.Namespace:
    return _build_parser().parse_args([cmd, *argv])


def main(argv: Sequence[str] = None) -> None:
    args = _build_parser().parse_args(argv)
    args.func(args)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    import shutil
    import tempfile

    for argv in (["-h"], ["benchmark", "-h"], ["ablation", "-h"]):
        try:
            main(argv)
        except SystemExit as e:
            assert e.code == 0, f"{argv} exited with code {e.code}"
        else:
            raise AssertionError(f"{argv} should have exited")

    # parsing with no flags must reproduce the dataclass defaults exactly
    ic, rc = InstanceConfig(), RunConfig()
    # CoLF's constructor defaults, read the same way rather than retyped
    import inspect
    from agents.colf import CoLFAgent
    cd = {k: v.default for k, v in inspect.signature(CoLFAgent).parameters.items()}
    for cmd in ("benchmark", "ablation"):
        args = _parse([], cmd)
        for field in ("delta", "T_max", "seed0", "n_reps"):
            got = getattr(_run_config(args), field)
            assert got == getattr(rc, field), (
                f"{cmd}: CLI default for {field} ({got!r}) has drifted from "
                f"RunConfig ({getattr(rc, field)!r})"
            )
        for cli, field in (("K", "K"), ("d", "d"), ("sigma", "sigma"),
                           ("min_gap", "min_gap")):
            got = getattr(args, cli)
            assert got == getattr(ic, field), (
                f"{cmd}: CLI default for --{cli} ({got!r}) has drifted from "
                f"InstanceConfig ({getattr(ic, field)!r})"
            )

    out_dir = tempfile.mkdtemp(prefix="colf_run_selftest_")
    try:
        main([
            "benchmark", "--K", "10", "--d", "4", "--n-reps", "4", "--t-max", "10000",
            "--min-gap", "0.15", "--sigma", "0.1", "--out-dir", out_dir,
        ])
        assert (Path(out_dir) / "benchmark.csv").exists()
        assert (Path(out_dir) / "benchmark.png").exists()

        main([
            "ablation", "--K", "10", "--d", "4", "--ablation-scenario", "S2",
            "--min-gap", "0.10", "--sigma", "0.1", "--out-dir", out_dir,
            "--n-reps", "4", "--ablation-T", "4000", "--ablation-points", "15",
        ])
        assert (Path(out_dir) / "ablation.csv").exists()
        assert (Path(out_dir) / "ablation.png").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    print("run.py self-test passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
