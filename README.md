# CoLF — Confident identification with Latent Features

Reference implementation of **CoLF**, a fixed-confidence best-arm-identification
(BAI) algorithm for linear bandits whose reward-generating feature vectors are
only *partially observable*, together with the observed-features-only
`G-allocation` baseline it is measured against and the experiment harness that
produces every number and figure in the accompanying paper.

The setting: each arm `a` carries a feature vector `z_a = [x_a; u_a]`, the
learner is shown only the observed half `x_a`, and the reward is linear in the
whole of it. The unrecorded half contributes an arm-specific shift
`u_a^T theta_u` that can reverse the ranking of the arms, so a stopping rule
built on `x_a` alone certifies a suboptimal arm at high nominal confidence. CoLF
removes that failure without assuming anything about the latent features beyond
the linearity of the reward.

## How it works

1. **Mixed G-optimal design.** A G-optimal design over the observed features is
   mixed with the uniform distribution, which guarantees every arm a positive
   sampling weight, and the features and rewards are rescaled by it.
2. **Orthogonal augmentation.** The design-scaled features are augmented with an
   orthonormal basis of the orthogonal complement of their row space. The basis
   is computable from the observed features alone, so the latent contribution is
   absorbed into two low-dimensional coefficients and the problem becomes an
   ordinary linear bandit. No tuning parameter governs the augmentation and the
   latent dimension is never estimated.
3. **Doubly robust estimation.** A DR pseudo-reward lets every arm contribute to
   the information matrix in every round, so one confidence width serves all
   arms and the imputation error cancels over a round-robin sweep.
4. **Two-phase elimination.** Arms are eliminated against that width until at
   most `d` survive, after which CoLF switches to successive elimination on
   per-arm empirical means, which are unbiased regardless of the latent features.

## Requirements

Python 3.11 or newer, with:

```
numpy
matplotlib
```

No other third-party dependency; `matplotlib` is needed only for the figures.

## Usage

```bash
python run.py                # self-test, small end-to-end run (~5 s)
python run.py benchmark      # experiment 1: error rate and sample complexity
python run.py ablation       # experiment 2: estimation-error diagnostics
python run.py sweep          # experiment 3: one factor at a time
```

Every run writes into `results/`:

| command | output |
| --- | --- |
| `benchmark` | `benchmark.csv`, `benchmark.png` |
| `ablation` | `ablation.csv`, `ablation.png` |
| `sweep` | `sweep.csv`, `sweep_{du,delta,min_gap,K}.csv`, `sweep_summary.csv`, `sweep.png` |

Common options (`python run.py <command> -h` for the full list):

```
--K --d --sigma --min-gap     the instance
--delta --t-max --seed0       the run
--p --ipw --beta-scale        CoLF's own knobs
--n-reps --out-dir
```

`sweep` varies one factor at a time over `du`, `delta`, `min_gap` and `K`,
holding everything else fixed and reusing the same seeds at every level.

Every module is also runnable on its own and then executes its own regression
test, for example `python agents/colf.py` or `python env.py`.

## Layout

```
run.py            CLI: config -> instances -> agents -> experiment -> plots
config.py         InstanceConfig / RunConfig and the three scenarios
env.py            the reward-generating environment; agents only ever see X
plots.py          figures for the benchmark, the ablation and the sweeps
agents/
  base.py         the interface every agent implements
  colf.py         CoLF: mixed design, orthogonal augmentation, DR estimator
  gopt.py         observed-features-only G-allocation elimination baseline
expt/
  runner.py       one (instance, agent) execution loop
  benchmark.py    experiment 1
  ablation.py     experiment 2
  sweep.py        experiment 3
results/          csv and png written by the commands above
```

The information barrier of the setting is enforced by the code rather than by
convention: an agent is constructed from `instance.observed()` and nothing else,
so the latent features and the true parameter are unreachable from inside an
agent.

## Scenarios

| name | `du` | drawn |
| --- | --- | --- |
| `S1` fully observable | 0 | nothing is hidden; the control |
| `S2` partially observable | `d` | without further conditioning |
| `S3` partially observable (reversed) | `d` | conditional on the observed features alone favouring a suboptimal arm |

## Reproducibility

Instances are drawn from `seed0` and each arm carries its own noise stream, so
the n-th pull of an arm is the same variate whatever the other arms did in
between. The two algorithms therefore run on the same instances with the same
noise, and a difference between them is never a difference in the draw.

## Citation

Sangjin Kim and Wonyoung Kim. *Best Arm Identification in Linear Bandits with
Partially Observable Features.* Second Workshop on ML×OR, NeurIPS 2026.
