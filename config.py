"""Configuration objects for the BAI-with-partially-observable-features experiments.

Three scenarios, following the discussion in the draft:

    S1 (FULL)      du = 0.  Everything that drives the reward is recorded, so
                   both algorithms are well specified.  Measures the price CoLF
                   pays for robustness it does not need here.
    S2 (PARTIAL)   du = d.  Half the covariates go unrecorded.  Drawn without
                   further conditioning, so the latent shifts reverse the
                   ordering on some instances and not others.
    S3 (REVERSED)  du = d, drawn conditional on the latent shifts actually
                   reversing the ordering: the observed features alone rank a
                   suboptimal arm strictly first.  This is the abstract's claim
                   in isolation -- an observed-only rule cannot identify the
                   best arm here however long it samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Scenario(str, Enum):
    FULL = "S1"      # fully observable
    PARTIAL = "S2"   # half observed, half latent
    REVERSED = "S3"  # half latent, conditioned on the ordering being reversed

    @property
    def label(self) -> str:
        """Two lines, because this is only ever used as a figure panel title."""
        return {
            Scenario.FULL: "fully observable",
            Scenario.PARTIAL: "partially observable",
            Scenario.REVERSED: "partially observable(reversed)",
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
    scenario: Scenario = Scenario.PARTIAL

    sigma: float = 0.1              # sub-Gaussian noise scale
    reward_bound: float = 1.0       # rescale theta so max_a |z_a^T theta| <= this

    # How much of the reward is hidden. None lets the scenario decide (0 for
    # S1, d otherwise); set it to sweep du as a factor in its own right, which
    # is the one knob the draft's claim is actually about.
    latent_dim: Optional[int] = None

    # Delta_2, the gap between the best and second-best arm, drives the sample
    # complexity as 1/Delta_2^2. Without a floor, one nearly-tied draw dominates
    # the mean T_delta and the comparison measures luck rather than the
    # algorithms, so instances outside the band are redrawn.
    min_gap: float = 0.10
    max_gap: float = float("inf")
    max_instances: int = 2000       # give up after this many redraws

    def __post_init__(self) -> None:
        if self.d >= self.K:
            raise ValueError(f"the setting requires d < K, got d={self.d}, K={self.K}")
        if self.min_gap <= 0:
            raise ValueError("min_gap must be positive")
        if self.latent_dim is not None and self.latent_dim < 0:
            raise ValueError("latent_dim must be non-negative")
        if self.scenario is Scenario.REVERSED and self.du == 0:
            raise ValueError(
                "S3 conditions on the latent shifts reversing the ordering, "
                "which is impossible with du = 0"
            )

    @property
    def du(self) -> int:
        if self.latent_dim is not None:
            return self.latent_dim
        return 0 if self.scenario is Scenario.FULL else self.d

    @property
    def dz(self) -> int:
        return self.d + self.du

    @property
    def tag(self) -> str:
        return f"{self.scenario.value}_K{self.K}_d{self.d}"


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunConfig:
    """How to run a sweep.  Nothing here belongs to an algorithm: the
    ridge penalties, the coupling probability, the confidence-width scale and
    the stopping cadence are all constructor arguments of the agents that use
    them."""

    delta: float = 0.05             # target confidence level
    n_reps: int = 50                # independent instances per scenario
    T_max: int = 200_000            # censoring horizon
    seed0: int = 20260823           # base seed; instance r uses seed0 + r

    ablation_T: int = 20_000        # horizon of the estimation-error experiment
    ablation_points: int = 40       # number of log-spaced checkpoints

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.n_reps <= 0 or self.T_max <= 0:
            raise ValueError("n_reps and T_max must be positive")


DEFAULT_SCENARIOS = (Scenario.FULL, Scenario.PARTIAL, Scenario.REVERSED)