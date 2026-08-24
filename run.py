"""CLI entry point: config -> instances -> agents -> experiment -> plots.

    python run.py benchmark   experiment 1: sample complexity + error rate
    python run.py ablation    experiment 2: estimation-error diagnostics
    python run.py             self-test (small end-to-end run, ~5s)

Agent imports are deferred into the command functions so that `-h` and the
self-test's argument checks stay cheap. The seams:

    agents.gopt.GOptAgent(X, delta, sigma, rng=None)
    agents.colf.CoLFAgent(X, delta, sigma, rng=None, run_cfg=RunConfig())
    expt.benchmark.run_benchmark(instance_cfgs, agent_factories, run_cfg, dcfg=None)
        -> list[dict]  (schema: plots.summarize_benchmark / plots.plot_benchmark)
    expt.ablation.run_ablation(instance_cfg, agent_factories, run_cfg, dcfg=None)
        -> list[dict]  (schema: plots.plot_ablation)

`agent_factories` is {name: Callable[[X, delta, sigma, rng], Agent]}.

Every run-level default is read from RunConfig rather than written out here,
so the CLI cannot drift from the dataclass; _selftest pins that.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from config import DEFAULT_SCENARIOS, DesignConfig, InstanceConfig, RunConfig, Scenario
import plots

SCENARIO_BY_NAME = {s.value: s for s in Scenario}


def _agent_factories(run_cfg: RunConfig) -> dict:
    from agents.colf import CoLFAgent
    from agents.gopt import GOptAgent

    return {
        "gopt": lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng),
        "colf": lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng, run_cfg),
    }


def _run_config(args: argparse.Namespace) -> RunConfig:
    kwargs = dict(delta=args.delta, n_reps=args.n_reps, T_max=args.t_max,
                  seed0=args.seed0, beta_scale=args.beta_scale)
    if hasattr(args, "ablation_T"):
        kwargs["ablation_T"] = args.ablation_T
        kwargs["ablation_points"] = args.ablation_points
    return RunConfig(**kwargs)


def _instance_configs(args: argparse.Namespace) -> Sequence[InstanceConfig]:
    names = args.scenarios or [s.value for s in DEFAULT_SCENARIOS]
    base = dict(K=args.K, d=args.d, sigma=args.sigma, latent_scale=args.latent_scale,
                min_gap=args.min_gap)
    return [InstanceConfig(scenario=SCENARIO_BY_NAME[name], **base) for name in names]


def _write_csv(records: Sequence[dict], path: Path) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _print_summary(summary: dict) -> None:
    header = (f"{'scenario':<10}{'agent':<8}{'n':>5}{'T_median':>12}"
              f"{'error_rate':>12}{'censor_rate':>13}{'err|reversal':>14}")
    print(header)
    for (scen, agent), s in sorted(summary.items()):
        print(f"{scen:<10}{agent:<8}{s['n']:>5}{s['T_median']:>12.1f}"
              f"{s['error_rate']:>12.3f}{s['censor_rate']:>13.3f}"
              f"{s['error_rate_given_reversal']:>14.3f}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from expt.benchmark import run_benchmark

    run_cfg = _run_config(args)
    dcfg = DesignConfig()
    instance_cfgs = _instance_configs(args)
    factories = _agent_factories(run_cfg)

    records = run_benchmark(instance_cfgs, factories, run_cfg, dcfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, out_dir / "benchmark.csv")
    _print_summary(plots.summarize_benchmark(records))
    plots.plot_benchmark(records, delta=run_cfg.delta, out_path=out_dir / "benchmark.png")


def cmd_ablation(args: argparse.Namespace) -> None:
    from expt.ablation import run_ablation

    run_cfg = _run_config(args)
    dcfg = DesignConfig()
    scenario = SCENARIO_BY_NAME[args.ablation_scenario]
    instance_cfg = InstanceConfig(
        K=args.K, d=args.d, scenario=scenario, sigma=args.sigma,
        latent_scale=args.latent_scale, min_gap=args.min_gap,
    )
    factories = _agent_factories(run_cfg)

    records = run_ablation(instance_cfg, factories, run_cfg, dcfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, out_dir / f"ablation_{scenario.value}.csv")
    plots.plot_ablation(records, out_path=out_dir / f"ablation_{scenario.value}.png")


def _add_common_args(sp: argparse.ArgumentParser) -> None:
    # Run-level defaults come from RunConfig so the CLI cannot silently drift
    # from the dataclass (a hardcoded CLI default once did exactly that,
    # overriding the config and quietly reverting beta to an uncalibrated width).
    rc = RunConfig()
    sp.add_argument("--K", type=int, default=20)
    sp.add_argument("--d", type=int, default=8)
    sp.add_argument("--sigma", type=float, default=0.1)
    sp.add_argument("--latent-scale", type=float, default=1.0)
    sp.add_argument("--min-gap", type=float, default=0.10)
    sp.add_argument("--delta", type=float, default=rc.delta)
    sp.add_argument("--t-max", type=int, default=rc.T_max)
    sp.add_argument("--seed0", type=int, default=rc.seed0)
    sp.add_argument("--beta-scale", type=float, default=rc.beta_scale)
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
    a.add_argument("--ablation-scenario", choices=list(SCENARIO_BY_NAME), default=Scenario.HALF.value)
    a.add_argument("--n-reps", type=int, default=rc.n_reps)
    a.add_argument("--ablation-T", type=int, default=rc.ablation_T, dest="ablation_T")
    a.add_argument("--ablation-points", type=int, default=rc.ablation_points)
    _add_common_args(a)
    a.set_defaults(func=cmd_ablation)
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

    # parsing with no flags must reproduce RunConfig's own defaults exactly.
    # A hardcoded CLI default once shadowed beta_scale and silently reverted
    # every run to the uncalibrated width; this pins the two together.
    rc = RunConfig()
    for cmd in ("benchmark", "ablation"):
        parsed = _run_config(_parse(["--K", "10", "--d", "4"], cmd))
        for field in ("delta", "T_max", "seed0", "beta_scale", "n_reps"):
            assert getattr(parsed, field) == getattr(rc, field), (
                f"{cmd}: CLI default for {field} ({getattr(parsed, field)!r}) "
                f"has drifted from RunConfig ({getattr(rc, field)!r})"
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
        assert (Path(out_dir) / "ablation_S2.csv").exists()
        assert (Path(out_dir) / "ablation_S2.png").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    print("run.py self-test passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
