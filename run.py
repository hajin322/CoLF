"""CLI entry point: every figure and table of the paper, and nothing else.

    python run.py benchmark   Figure 1, Table 1     -> benchmark.png, table1_benchmark.csv
    python run.py ablation    Figure 2, ablation    -> ablation.png, table_ablation.csv
    python run.py sweep       Tables 2 and 3        -> table2_du.csv, table3_delta.csv
    python run.py             self-test (small end-to-end run)

With no flags each command reproduces the paper's setting (PAPER below);
instance and run defaults are read from InstanceConfig / RunConfig and CoLF's
from its constructor, never retyped, so the CLI cannot drift from them.
Agent imports are deferred into the commands so that `-h` stays cheap.

`agent_factories` is {name: Callable[[X, delta, sigma, rng], Agent]}.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from config import DEFAULT_SCENARIOS, InstanceConfig, RunConfig, Scenario
import plots

# The paper's instance counts, scenarios and levels, per figure and table.
PAPER = {
    "benchmark_reps": 200,
    "figure2_reps": 100, "figure2_scenario": Scenario.PARTIAL,
    "ablation_reps": 150, "ablation_scenario": Scenario.REVERSED,
    "du_reps": 150, "du_scenario": Scenario.PARTIAL, "du_levels": [0, 1, 2, 4, 8, 16],
    "delta_reps": 200, "delta_scenario": Scenario.REVERSED,
    "delta_levels": [0.2, 0.1, 0.05, 0.01, 0.005, 0.001],
}


def _gopt():
    from agents.gopt import GOptAgent
    return lambda X, delta, sigma, rng: GOptAgent(X, delta, sigma, rng)


def _colf(args, **kw):
    from agents.colf import CoLFAgent
    return lambda X, delta, sigma, rng: CoLFAgent(X, delta, sigma, rng, p=args.p, **kw)


def _instance_config(args, scenario: Scenario) -> InstanceConfig:
    return InstanceConfig(K=args.K, d=args.d, scenario=scenario, sigma=args.sigma,
                          min_gap=args.min_gap)


def _run_config(args, n_reps: int, **kw) -> RunConfig:
    return RunConfig(delta=args.delta, n_reps=n_reps, T_max=args.t_max,
                     seed0=args.seed0, **kw)


def _write_table(rows: Sequence[dict], path: Path) -> None:
    """Write the rows to CSV and echo them, so a run shows what it wrote."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{path}")
    print("  ".join(f"{k:>14}" for k in rows[0]))
    for r in rows:
        print("  ".join(f"{v:>14.4g}" if isinstance(v, float) else f"{v!s:>14}"
                        for v in r.values()))


def _out_dir(args) -> Path:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_benchmark(args) -> None:
    from expt.benchmark import run_benchmark, table

    out = _out_dir(args)
    records = run_benchmark(
        [_instance_config(args, s) for s in DEFAULT_SCENARIOS],
        {"gopt": _gopt(), "colf": _colf(args)},
        _run_config(args, args.n_reps),
    )
    plots.plot_benchmark(records, delta=args.delta,
                         titles={s.value: s.label for s in Scenario},
                         out_path=out / "benchmark.png")
    _write_table(table(records), out / "table1_benchmark.csv")


def cmd_ablation(args) -> None:
    from expt.ablation import run_ablation, run_stopping_ablation, table

    def ladder(eliminate):
        return {"gopt": _gopt(),
                "colf": _colf(args, eliminate=eliminate),
                "colf_noaug": _colf(args, eliminate=eliminate, augment=False),
                "oracle": _gopt()}

    out = _out_dir(args)
    # Figure 2 compares estimators on one round-robin schedule, so CoLF's
    # phase-1 estimator is kept running with no elimination; the table runs
    # every agent to its own stopping rule.
    records = run_ablation(
        _instance_config(args, PAPER["figure2_scenario"]), ladder(eliminate=False),
        _run_config(args, args.figure_reps, ablation_T=args.ablation_T,
                    ablation_points=args.ablation_points),
    )
    plots.plot_ablation(records, out_path=out / "ablation.png")

    rows = run_stopping_ablation(
        _instance_config(args, PAPER["ablation_scenario"]), ladder(eliminate=True),
        _run_config(args, args.table_reps),
    )
    _write_table(table(rows), out / "table_ablation.csv")


def cmd_sweep(args) -> None:
    from expt.sweep import run_sweep, table_delta, table_du

    out = _out_dir(args)
    factories = {"gopt": _gopt(), "colf": _colf(args)}
    rows = run_sweep("du", args.du_levels,
                     _instance_config(args, PAPER["du_scenario"]), factories,
                     _run_config(args, args.du_reps))
    _write_table(table_du(rows), out / "table2_du.csv")
    rows = run_sweep("delta", args.delta_levels,
                     _instance_config(args, PAPER["delta_scenario"]), factories,
                     _run_config(args, args.delta_reps))
    _write_table(table_delta(rows), out / "table3_delta.csv")


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def _add_common_args(sp: argparse.ArgumentParser) -> None:
    import inspect
    from agents.colf import CoLFAgent

    ic, rc = InstanceConfig(), RunConfig()
    p_default = inspect.signature(CoLFAgent).parameters["p"].default
    sp.add_argument("--K", type=int, default=ic.K)
    sp.add_argument("--d", type=int, default=ic.d)
    sp.add_argument("--sigma", type=float, default=ic.sigma)
    sp.add_argument("--min-gap", type=float, default=ic.min_gap)
    sp.add_argument("--delta", type=float, default=rc.delta)
    sp.add_argument("--t-max", type=int, default=rc.T_max)
    sp.add_argument("--seed0", type=int, default=rc.seed0)
    sp.add_argument("--p", type=float, default=p_default,
                    help="CoLF's pseudo-action matching probability, Eq. (7)-(10)")
    sp.add_argument("--out-dir", default="results")


def _build_parser() -> argparse.ArgumentParser:
    rc = RunConfig()
    parser = argparse.ArgumentParser(description="CoLF: the paper's figures and tables")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="Figure 1 and Table 1")
    b.add_argument("--n-reps", type=int, default=PAPER["benchmark_reps"])
    _add_common_args(b)
    b.set_defaults(func=cmd_benchmark)

    a = sub.add_parser("ablation", help="Figure 2 and the ablation table")
    a.add_argument("--figure-reps", type=int, default=PAPER["figure2_reps"])
    a.add_argument("--table-reps", type=int, default=PAPER["ablation_reps"])
    a.add_argument("--ablation-T", type=int, default=rc.ablation_T, dest="ablation_T")
    a.add_argument("--ablation-points", type=int, default=rc.ablation_points)
    _add_common_args(a)
    a.set_defaults(func=cmd_ablation)

    w = sub.add_parser("sweep", help="Tables 2 and 3")
    w.add_argument("--du-reps", type=int, default=PAPER["du_reps"])
    w.add_argument("--du-levels", type=int, nargs="+", default=PAPER["du_levels"])
    w.add_argument("--delta-reps", type=int, default=PAPER["delta_reps"])
    w.add_argument("--delta-levels", type=float, nargs="+",
                   default=PAPER["delta_levels"])
    _add_common_args(w)
    w.set_defaults(func=cmd_sweep)
    return parser


def main(argv: Sequence[str] = None) -> None:
    args = _build_parser().parse_args(argv)
    args.func(args)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    import shutil
    import tempfile

    for argv in (["-h"], ["benchmark", "-h"], ["ablation", "-h"], ["sweep", "-h"]):
        try:
            main(argv)
        except SystemExit as e:
            assert e.code == 0, f"{argv} exited with code {e.code}"
        else:
            raise AssertionError(f"{argv} should have exited")

    # parsing with no flags must reproduce the dataclass defaults exactly
    ic, rc = InstanceConfig(), RunConfig()
    for cmd in ("benchmark", "ablation", "sweep"):
        args = _build_parser().parse_args([cmd])
        for cli, ref, field in (("K", ic, "K"), ("d", ic, "d"), ("sigma", ic, "sigma"),
                                ("min_gap", ic, "min_gap"), ("delta", rc, "delta"),
                                ("t_max", rc, "T_max"), ("seed0", rc, "seed0")):
            assert getattr(args, cli) == getattr(ref, field), (
                f"{cmd}: CLI default for --{cli} has drifted from the config"
            )

    small = ["--K", "10", "--d", "4", "--t-max", "20000", "--min-gap", "0.15"]
    out = tempfile.mkdtemp(prefix="colf_run_selftest_")
    try:
        main(["benchmark", "--n-reps", "3", "--out-dir", out, *small])
        main(["ablation", "--figure-reps", "2", "--table-reps", "2", "--ablation-T", "2000",
              "--ablation-points", "10", "--out-dir", out, *small])
        main(["sweep", "--du-reps", "2", "--du-levels", "0", "4", "--delta-reps", "2",
              "--delta-levels", "0.1", "0.01", "--out-dir", out, *small])
        written = sorted(p.name for p in Path(out).iterdir())
        assert written == ["ablation.png", "benchmark.png", "table1_benchmark.csv",
                           "table2_du.csv", "table3_delta.csv", "table_ablation.csv"], written
    finally:
        shutil.rmtree(out, ignore_errors=True)

    print("run.py self-test passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
