"""The reward-generating environment.

`Instance` holds the full problem (X, U, theta) and hands out rewards.  Agents
receive `instance.observed()` and nothing else, so the information barrier of
the setting is enforced by the code rather than by convention.

Noise.  Each arm draws from its own independent stream, matching eta_{a,t} of
the draft's Eq. (7) and notation table.  (Eq. (2)/(3) write eta_t instead, but
only one arm is pulled per round, so a single run's reward law is the same
either way -- the indexing only decides whether two agents that pull different
arms would have shared a noise value.)  The n-th pull of arm a is reproducible
whatever the other arms did, so runs remain comparable across agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import DEFAULT_SCENARIOS, InstanceConfig, Scenario


# --------------------------------------------------------------------------- #
# Instance
# --------------------------------------------------------------------------- #

@dataclass
class Instance:
    """A single problem instance.  Only `X` and `cfg.d` are visible to a learner."""

    cfg: InstanceConfig
    X: np.ndarray                 # (d, K)   observed features
    U: np.ndarray                 # (du, K)  latent features; (0, K) when du = 0
    theta_o: np.ndarray           # (d,)
    theta_u: np.ndarray           # (du,)
    means: np.ndarray             # (K,)     m_a = z_a^T theta_star
    seed: int

    _noise_rng: list = field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------------- basics --

    def __post_init__(self) -> None:
        if self._noise_rng is None:
            self.reset_noise(self.seed)

    @property
    def K(self) -> int:
        return self.cfg.K

    @property
    def d(self) -> int:
        return self.cfg.d

    @property
    def a_star(self) -> int:  # prohibited to agents
        return int(np.argmax(self.means))

    @property
    def gaps(self) -> np.ndarray:  # prohibited to agents
        """Delta_a = m_1 - m_a, zero at the optimal arm."""
        return self.means.max() - self.means

    @property
    def gap_min(self) -> float:  # prohibited to agents
        """Delta_2, the gap to the second best arm."""
        g = np.sort(self.gaps)
        return float(g[1])

    def observed(self) -> np.ndarray:
        """The only thing a learner is allowed to see (besides d, K)."""
        return self.X.copy()

    # ----------------------------------------------------------------- pulls --

    def reset_noise(self, seed: Optional[int] = None) -> None:
        """Restart every arm's noise stream from `seed`."""
        s = self.seed if seed is None else seed
        self._noise_rng = [
            np.random.default_rng(child)
            for child in np.random.SeedSequence(s).spawn(self.K)
        ]

    def pull(self, a: int) -> float:
        """Draw y_{a,t} = z_a^T theta_star + eta_{a,t}.

        Arm a has its own stream, so its n-th pull is the same variate however
        many times the other arms were pulled in between.
        """
        eta = float(self._noise_rng[a].standard_normal()) * self.cfg.sigma
        return float(self.means[a]) + eta


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def make_instance(cfg: InstanceConfig, seed: int) -> Instance:
    """Draw one instance of the requested scenario.

    Common to all scenarios:
      * columns of X (and of U) are drawn iid standard normal and normalised to
        unit L2 norm;
      * theta is rescaled so that max_a |z_a^T theta| = reward_bound, matching
        the bounded-mean-reward convention of the draft;
      * instances whose gap to the second best arm falls below `min_gap` are
        rejected, since they only inflate the stopping time without changing
        what the experiment shows.

    Scenario S3 is drawn conditional on the latent shifts actually reversing the
    ordering: the observed-only limit must rank some suboptimal arm strictly
    above a_star.  Rescaling theta by a positive scalar changes no ranking, so
    the conditioning may be checked before the rescaling step.
    """
    rng = np.random.default_rng(seed)
    K, d, du = cfg.K, cfg.d, cfg.du

    for _ in range(cfg.max_instances):
        X = rng.standard_normal((d, K))
        X /= np.linalg.norm(X, axis=0)
        if np.linalg.matrix_rank(X) < d:
            continue

        if du:
            U = rng.standard_normal((du, K))
            U /= np.linalg.norm(U, axis=0)
        else:
            U = np.zeros((0, K))

        theta_o = rng.standard_normal(d)
        theta_u = rng.standard_normal(du) if du else np.zeros(0)

        means = X.T @ theta_o + (U.T @ theta_u if du else 0.0)

        if cfg.scenario is Scenario.REVERSED:
            # require the shift to actually reverse the ordering: the recorded
            # covariates on their own must favour a suboptimal arm. This is the
            # abstract's claim isolated into a scenario.
            if int(np.argmax(X.T @ theta_o)) == int(np.argmax(means)):
                continue

        # rescale so that max_a |m_a| = reward_bound
        scale = cfg.reward_bound / max(float(np.abs(means).max()), 1e-12)
        theta_o = theta_o * scale
        theta_u = theta_u * scale
        means = means * scale

        srt = np.sort(means)[::-1]
        gap2 = float(srt[0] - srt[1])
        if not (cfg.min_gap <= gap2 <= cfg.max_gap):
            continue

        return Instance(
            cfg=cfg,
            X=X,
            U=U,
            theta_o=theta_o,
            theta_u=theta_u,
            means=means,
            seed=seed,
        )

    raise RuntimeError(
        f"failed to draw a {cfg.scenario.value} instance with gap in "
        f"[{cfg.min_gap}, {cfg.max_gap}] after {cfg.max_instances} attempts"
    )


def make_batch(cfg: InstanceConfig, n_reps: int, seed0: int) -> list[Instance]:
    """`n_reps` instances with seeds seed0, seed0 + 1, ...  Paired across agents."""
    return [make_instance(cfg, seed0 + r) for r in range(n_reps)]


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    for scenario in DEFAULT_SCENARIOS:
        cfg = InstanceConfig(K=20, d=8, scenario=scenario, sigma=0.1, min_gap=0.10)
        insts = make_batch(cfg, n_reps=50, seed0=20260823)

        rev, gaps = [], []
        for ins in insts:
            # the recorded covariates' own favourite, read off theta_o
            reversal = int(np.argmax(ins.X.T @ ins.theta_o)) != ins.a_star
            rev.append(reversal)
            gaps.append(ins.gap_min)
            if scenario is Scenario.FULL:
                assert not reversal, (
                    "S1: nothing is hidden, so the observed ranking must be right"
                )
            if scenario is Scenario.REVERSED:
                assert reversal, "S3 conditioning failed: the ordering is not reversed"

        print(f"{scenario.value} | reversal {np.mean(rev):5.1%} | "
              f"median Delta_2 {np.median(gaps):.4f}")

    # each arm has its own stream: the n-th pull of a given arm is reproducible,
    # and different arms get independent draws
    ins = make_instance(InstanceConfig(scenario=Scenario.PARTIAL), 7)
    ins.reset_noise(123)
    a1 = [ins.pull(t % ins.K) - ins.means[t % ins.K] for t in range(50)]
    ins.reset_noise(123)
    a2 = [ins.pull((t * 3) % ins.K) - ins.means[(t * 3) % ins.K] for t in range(50)]
    assert not np.allclose(a1, a2), "per-arm noise should depend on which arm is pulled"

    ins.reset_noise(123)                       # same arm, same order -> identical
    a3 = [ins.pull(t % ins.K) - ins.means[t % ins.K] for t in range(50)]
    assert np.allclose(a1, a3), "per-arm streams are not reproducible"

    ins.reset_noise(123)                       # arm 0's n-th pull is independent
    first = [ins.pull(0) - ins.means[0] for _ in range(5)]   # of the other arms
    ins.reset_noise(123)
    for t in range(20):
        ins.pull(t % ins.K)
    ins.reset_noise(123)
    interleaved = []
    for t in range(5):
        for b in range(ins.K):
            v = ins.pull(b) - ins.means[b]
            if b == 0:
                interleaved.append(v)
    assert np.allclose(first, interleaved), (
        "arm 0's stream should not depend on what the other arms did"
    )

    sd = float(np.std([ins.pull(3) - ins.means[3] for _ in range(20000)]))
    assert abs(sd - ins.cfg.sigma) < 0.01 * ins.cfg.sigma, f"noise scale is {sd}"

    print("env.py self-test passed")


if __name__ == "__main__":
    _selftest()
