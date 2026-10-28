"""The interface every agent implements.

The constructor takes the observed features, the confidence level, the noise
scale and a random stream, and nothing else.  `U`, `theta_star` and the true
means are never passed to an agent, so the information barrier of the setting
is enforced by the signature rather than by discipline.

Runner contract
---------------
    while True:
        a = agent.sample()
        if a is None:            # the stopping rule fired
            break
        agent.update(a, instance.pull(a))
        if agent.t >= T_max:     # censoring is the runner's business
            break
    psi = agent.recommend()

`sample` returns the arm to pull, or None once the agent has stopped.  It must
not have side effects beyond bookkeeping: an agent that is polled twice without
an intervening `update` should return the same arm.

Evaluation hooks
----------------
    predicted_means()     estimate of E[y_a] for every arm, on the original
                          reward scale, comparable across agents and against
                          `instance.means`.  Figure 2 plots its error.
    confidence_width()    the current half-width the elimination rule uses, so
                          Figure 2 can divide the realised error by it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np


class StopReason(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"      # the stopping rule fired
    CENSORED = "censored"    # the runner hit T_max first


class Agent(ABC):
    """Base class for a fixed-confidence best-arm-identification agent."""

    name: str = "agent"

    def __init__(
        self,
        X: np.ndarray,
        delta: float,
        sigma: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be (d, K), got shape {X.shape}")
        self.X = np.array(X, dtype=float, copy=True)
        self.d, self.K = self.X.shape
        if self.d >= self.K:
            raise ValueError(f"the setting requires d < K, got d={self.d}, K={self.K}")
        self.delta = float(delta)
        self.sigma = float(sigma)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.t: int = 0                                  # rounds elapsed
        self.n_pulls = np.zeros(self.K, dtype=np.int64)  # pulls per arm
        self.reason: StopReason = StopReason.RUNNING

    # ------------------------------------------------------------ interface --

    @abstractmethod
    def sample(self) -> Optional[int]:
        """Arm to pull next, or None once the stopping rule has fired."""

    @abstractmethod
    def update(self, a: int, reward: float) -> None:
        """Fold the observed reward of arm `a` into the agent's state."""

    @abstractmethod
    def recommend(self) -> int:
        """The arm the agent certifies.  Well defined at any point, so that a
        censored run still yields an answer to score."""

    @abstractmethod
    def predicted_means(self) -> np.ndarray:
        """Estimate of E[y_a] for every arm, on the original reward scale.

        Agents that work in design-scaled coordinates must undo the scaling
        here, so that the returned vector is comparable across agents and
        against `instance.means`.
        """

    # ------------------------------------------------------------- optional --

    def confidence_width(self) -> Optional[float]:
        """Current half-width used by the elimination rule, if the agent has one."""
        return None

    # ------------------------------------------------------------ mechanics --

    @property
    def stopped(self) -> bool:
        return self.reason is not StopReason.RUNNING

    def _stop(self, reason: StopReason = StopReason.STOPPED) -> None:
        self.reason = reason

    def _record_pull(self, a: int) -> None:
        """Bookkeeping every `update` implementation should call first."""
        self.t += 1
        self.n_pulls[a] += 1

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}(name={self.name!r}, K={self.K}, d={self.d}, "
            f"t={self.t}, reason={self.reason.value})"
        )


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def argmax_random_tie(v: np.ndarray, rng: np.random.Generator, tol: float = 1e-12) -> int:
    """Argmax with ties broken uniformly at random.

    When two arms score exactly equal, deterministic tie-breaking would let an
    agent look either lucky or hopeless purely through arm labelling.  Breaking
    at random makes the coin flip explicit.
    """
    v = np.asarray(v, dtype=float)
    hit = np.flatnonzero(v >= v.max() - tol * max(1.0, float(np.abs(v).max())))
    return int(hit[0]) if hit.size == 1 else int(rng.choice(hit))


def round_robin(t: int, K: int) -> int:
    """Arm pulled at round t (zero-indexed) under a round-robin schedule.

    After every sweep of K rounds the empirical Gram matrix of the augmented
    features equals sum_a xhat_a xhat_a^T exactly rather than in expectation,
    which is what removes the matrix-concentration step for the design
    component of the analysis.
    """
    return t % K


def confidence_radius(sigma: float, K: int, delta: float, scale: float = 1.0) -> float:
    """sigma * sqrt(2 log(2K/delta)), the one constant both agents' widths are
    built from, so a faster stopping time can't come from a looser budget."""
    return scale * float(sigma) * float(np.sqrt(2.0 * np.log(2.0 * K / delta)))


class _NullAgent(Agent):
    """Minimal concrete subclass, used only to exercise the base class."""

    name = "null"

    def __init__(self, X, delta, sigma, rng=None, horizon: int = 10):
        super().__init__(X, delta, sigma, rng)
        self.horizon = horizon
        self.total = np.zeros(self.K)

    def sample(self):
        if self.stopped:
            return None
        if self.t >= self.horizon:
            self._stop()
            return None
        return round_robin(self.t, self.K)

    def update(self, a, reward):
        self._record_pull(a)
        self.total[a] += reward

    def predicted_means(self):
        n = np.maximum(self.n_pulls, 1)
        return self.total / n

    def recommend(self):
        return argmax_random_tie(self.predicted_means(), self.rng)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 8))
    ag = _NullAgent(X, delta=0.05, sigma=0.1, rng=rng, horizon=16)

    # polling twice without an update must not advance the schedule
    assert ag.sample() == ag.sample() == 0

    seen = []
    while (a := ag.sample()) is not None:
        seen.append(a)
        ag.update(a, float(a))
    assert seen == list(range(8)) * 2, "round-robin schedule is wrong"
    assert ag.t == 16 and ag.stopped and ag.reason is StopReason.STOPPED
    assert np.allclose(ag.predicted_means(), np.arange(8.0))
    assert ag.recommend() == 7

    # ties are broken uniformly, not by index order
    counts = np.zeros(3)
    for _ in range(4000):
        counts[argmax_random_tie(np.array([1.0, 1.0, 0.0]), rng)] += 1
    assert counts[2] == 0 and abs(counts[0] / 4000 - 0.5) < 0.05, counts

    try:
        _NullAgent(rng.standard_normal((8, 3)), 0.05, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("d < K is not being enforced")

    r1 = confidence_radius(sigma=0.1, K=20, delta=0.05)
    assert r1 > 0.0
    assert confidence_radius(0.1, 20, 0.005) > r1, "smaller delta must widen the radius"
    assert confidence_radius(0.1, 40, 0.05) > r1, "more arms must widen the radius"
    assert np.isclose(confidence_radius(0.1, 20, 0.05, scale=2.0), 2.0 * r1)

    print("agents/base.py self-test passed")


if __name__ == "__main__":
    _selftest()
