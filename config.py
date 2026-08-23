"""Configuration objects for the BAI-with-partially-observable-features experiments.

Three scenarios, following the discussion in the draft:

    S1 (FULL)     du = 0.  Both algorithms are well specified.  Measures the
                  price CoLF pays for robustness it does not need here.
    S2 (HALF)     du = d.  The main claim: an observed-features-only stopping
                  rule certifies the wrong arm at high nominal confidence.
    S3 (UNIDENT)  du = d, and one pair of arms shares an observed feature
                  vector while differing in the latent one.  This is the
                  experimental counterpart of the `+K` term in Theorem 1:
                  the latent offset of an arm is not identified by samples of
                  the other arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Scenario(str, Enum):
    FULL = "S1"      # fully observable
    HALF = "S2"      # half observed, half latent
    UNIDENT = "S3"   # half latent, with an observationally identical arm pair

    @property
    def label(self) -> str:
        return {
            Scenario.FULL: "S1: fully observable",
            Scenario.HALF: "S2: half latent",
            Scenario.UNIDENT: "S3: unidentifiable pair",
        }[self]


# --------------------------------------------------------------------------- #
# Instance
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InstanceConfig:
    """Everything that defines the reward-generating environment.

    The learner is handed `X` and `d` only; every other field below is oracle
    information used for evaluation.
    """

    K: int = 20                     # number of arms
    d: int = 8                      # observed feature dimension (must be < K)
    scenario: Scenario = Scenario.HALF

    latent_scale: float = 1.0       # kappa: L2 norm of each latent column u_a
    sigma: float = 0.1              # sub-Gaussian noise scale
    reward_bound: float = 1.0       # rescale theta so max_a |z_a^T theta| <= this

    min_gap: float = 0.10           # reject instances with Delta_2 below this
    max_gap: float = float("inf")   # optional upper cap on Delta_2
    max_resample: int = 2000

    arm_noise: bool = False         # False: eta_t shared across arms (draft Eq. 2)

    def __post_init__(self) -> None:
        if self.d >= self.K:
            raise ValueError(f"the setting requires d < K, got d={self.d}, K={self.K}")
        if self.min_gap <= 0:
            raise ValueError("min_gap must be positive")

    @property
    def du(self) -> int:
        return 0 if self.scenario is Scenario.FULL else self.d

    @property
    def dz(self) -> int:
        return self.d + self.du

    @property
    def tag(self) -> str:
        return f"{self.scenario.value}_K{self.K}_d{self.d}"


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DesignConfig:
    """Parameters of the G-optimal design solver and the augmentation."""

    gamma_mix: float = 0.5          # pi = (1 - g) pi^G + g * uniform;  draft uses 1/2
    fw_max_iter: int = 20_000       # Frank-Wolfe / Fedorov-Wynn iterations
    fw_tol: float = 1e-8            # stop when max_a ||x_a||^2_{V^-1} / d - 1 <= tol
    prune: float = 1e-10            # zero out negligible pi^G weights, then renormalize
    rank_rtol: float = 1e-10        # relative tolerance for the rank of Xbar
    check: bool = True              # run the block-structure assertions on build

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma_mix <= 1.0:
            raise ValueError("gamma_mix must lie in (0, 1]")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunConfig:
    """Parameters of a single benchmark / ablation sweep."""

    delta: float = 0.05             # target confidence level
    n_reps: int = 50                # independent instances per scenario
    T_max: int = 200_000            # censoring horizon
    seed0: int = 20260823           # base seed; instance r uses seed0 + r

    coupling_p: float = 0.6         # pseudo-action coupling probability, in (1/2, 1)
    lam_impute: float = 1e-2        # ridge penalty of the imputation estimator
    lam_dr: float = 1e-2            # ridge penalty of the DR estimator

    beta_mode: str = "sharp"        # {"theory", "sharp", "calibrated"}
    beta_scale: float = 1.0         # multiplier used when beta_mode == "calibrated"

    check_every_sweeps: int = 1     # evaluate the stopping rule every m round-robin sweeps
    ablation_T: int = 20_000        # horizon of the estimation-error experiment
    ablation_points: int = 40       # number of log-spaced checkpoints

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if not 0.5 < self.coupling_p < 1.0:
            raise ValueError("coupling_p must lie in (1/2, 1)")
        if self.beta_mode not in {"theory", "sharp", "calibrated"}:
            raise ValueError(f"unknown beta_mode: {self.beta_mode}")


DEFAULT_SCENARIOS = (Scenario.FULL, Scenario.HALF, Scenario.UNIDENT)