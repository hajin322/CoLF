"""Observed-features-only G-allocation elimination baseline.

Static G-optimal rounding (the allocation is never recomputed over the
surviving arms), ridge estimation in the raw d-dim observed space, and
elimination against the current leader at width beta * sqrt(d / t), with
beta = confidence_radius(sigma, K, delta) = sigma * sqrt(2 log(2K/delta))
from agents/base.py. Under the G-optimal design max_a ||x_a||^2_{V^-1} = d,
so sqrt(d / t) is the right per-arm prediction scale.

NOTE on fairness of the T_delta comparison: this agent spends the whole delta
budget on its width, while CoLF splits delta in half (the other half pays for
the pseudo-action cap of Eq. 15). CoLF is therefore held to the stricter
budget of the two, so the comparison does not flatter it.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import Agent, StopReason, confidence_radius  # noqa: E402
from config import DesignConfig  # noqa: E402
from design import gopt_design  # noqa: E402


class GOptAgent(Agent):
    name = "gopt"

    def __init__(
        self,
        X: np.ndarray,
        delta: float,
        sigma: float,
        rng: Optional[np.random.Generator] = None,
        dcfg: Optional[DesignConfig] = None,
        lam: float = 1e-2,
    ) -> None:
        super().__init__(X, delta, sigma, rng)
        self.pi_G, _ = gopt_design(self.X, dcfg or DesignConfig())
        self.lam = float(lam)
        self.beta_const = confidence_radius(self.sigma, self.K, self.delta)

        self.active = np.ones(self.K, dtype=bool)
        self._A_inv = np.eye(self.d) / self.lam
        self._c = np.zeros(self.d)
        self._psi: Optional[int] = None

    # ------------------------------------------------------------- internals --

    def _theta_hat(self) -> np.ndarray:
        return self._A_inv @ self._c

    def _width(self) -> float:
        return self.beta_const * float(np.sqrt(self.d / max(self.t, 1)))

    # --------------------------------------------------------------- interface --

    def sample(self) -> Optional[int]:
        # deterministic argmax: sample() must be idempotent between updates,
        # and random tie-breaking would violate that when scores tie exactly
        # (e.g. every active arm at t = 0).
        if self.stopped:
            return None
        target = self.pi_G * self.t - self.n_pulls
        target = np.where(self.active, target, -np.inf)
        return int(np.argmax(target))

    def update(self, a: int, reward: float) -> None:
        self._record_pull(a)
        x = self.X[:, a]

        Av = self._A_inv @ x
        denom = 1.0 + float(x @ Av)
        self._A_inv -= np.outer(Av, Av) / denom
        self._c += x * reward

        scores = self.X.T @ self._theta_hat()
        leader = int(np.argmax(np.where(self.active, scores, -np.inf)))
        eliminated = self.active & (scores[leader] - scores > 4.0 * self._width())
        self.active[eliminated] = False

        if self.active.sum() == 1:
            self._psi = int(np.flatnonzero(self.active)[0])
            self._stop(StopReason.STOPPED)

    def recommend(self) -> int:
        if self._psi is not None:
            return self._psi
        means = np.where(self.active, self.predicted_means(), -np.inf)
        return int(np.argmax(means))

    def predicted_means(self) -> np.ndarray:
        return self.X.T @ self._theta_hat()

    def components(self) -> dict:
        return {"par": self._theta_hat(), "perp": None}

    def confidence_width(self) -> Optional[float]:
        return self._width()


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    from config import InstanceConfig, RunConfig, Scenario
    from env import make_instance
    from expt.runner import run_single

    cfg = InstanceConfig(K=10, d=4, scenario=Scenario.FULL, sigma=0.1, min_gap=0.15)
    run_cfg = RunConfig(T_max=20_000, delta=0.05)

    correct, censored = [], []
    for seed in range(15):
        inst = make_instance(cfg, seed=seed)
        rng = np.random.default_rng(seed)
        agent = GOptAgent(inst.observed(), delta=run_cfg.delta, sigma=cfg.sigma, rng=rng)

        # polling twice without an update must not advance the schedule
        assert agent.sample() == agent.sample()

        res = run_single(inst, agent, run_cfg, noise_seed=seed)
        correct.append(res.correct)
        censored.append(res.censored)
        assert agent.active.sum() >= 1
        assert np.isfinite(agent.predicted_means()).all()
        assert agent.components()["perp"] is None

    acc = float(np.mean(correct))
    cens = float(np.mean(censored))
    print(f"S1 (fully observable) | accuracy {acc:.2%} | censored {cens:.2%}")
    assert acc >= 0.8, "baseline should be reliably correct when nothing is hidden"
    assert cens <= 0.2

    # reproducibility: same seeds, independent agent instances -> identical outcome
    inst = make_instance(cfg, seed=0)
    a1 = GOptAgent(inst.observed(), delta=0.05, sigma=cfg.sigma, rng=np.random.default_rng(0))
    r1 = run_single(inst, a1, run_cfg, noise_seed=0)
    a2 = GOptAgent(inst.observed(), delta=0.05, sigma=cfg.sigma, rng=np.random.default_rng(0))
    r2 = run_single(inst, a2, run_cfg, noise_seed=0)
    assert r1.T_delta == r2.T_delta and r1.psi == r2.psi

    print("agents/gopt.py self-test passed")


if __name__ == "__main__":
    _selftest()
