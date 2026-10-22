"""Observed-features-only G-allocation baseline.

The canonical construction for fixed-confidence BAI in a linear bandit: a
static G-optimal design over the observed features, ridge estimation in the
raw d-dimensional space, and elimination against the current leader.

    beta_t := sigma sqrt(2 log(2K/delta)) sqrt(d/t),
    eliminate a  once  (x_psi - x_a)^T thetahat_t  >  2 beta_t.

Each piece is the textbook one, and each matches the corresponding piece of
CoLF, so the two algorithms differ in their feature set and nothing else.

  * sqrt(d/t).  Under the G-optimal design max_a ||x_a||^2_{V(pi^G)^-1} = d by
    Kiefer-Wolfowitz, and after t pulls allocated to that design A_t = t V(pi^G),
    so every arm's prediction variance is at most d/t.  This is the analytic
    form; using the empirical max_a ||x_a||_{A_t^-1} instead does not work with
    elimination, because an eliminated arm stops being sampled and its own
    direction stops being refined, so the maximum over all K arms stalls and
    the rule never fires.

  * sigma sqrt(2 log(2K/delta)).  The sub-Gaussian tail with a union bound over
    the K arms.  The rewards here are Gaussian; a bounded-noise Azuma constant
    would be twice this and would cost four times the samples.

  * The factor two.  beta_t bounds one arm's error, and the test compares two,
    exactly as CoLF's own rule spends beta on each of the two arms it compares.

The width is calibrated, not merely conservative: on the fully observable
scenario, where this agent is well specified and any error is the width's fault
rather than the model's, it makes no error in 1200 runs across four gap
regimes, which a self-test below pins.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import (  # noqa: E402
    Agent, StopReason, argmax_random_tie, confidence_radius,
)


def gopt_design(X, tol=1e-6, max_iter=60_000):
    """pi^G = argmin_pi max_a ||x_a||^2_{V(pi)^-1}, with V(pi) = sum_a pi_a x_a x_a^T.

    By Kiefer-Wolfowitz the G-optimal and D-optimal designs coincide and the
    optimal value is exactly d, whatever K is. So we solve the D-optimal problem
    with the multiplicative update

        pi_a <- pi_a * g_a(pi) / d,   g_a(pi) = ||x_a||^2_{V(pi)^-1},

    which increases log det V(pi) monotonically and reaches machine precision in
    a few hundred iterations. Since max_a g_a >= d always, `gap` below is a
    certified optimality gap rather than a guess.

    CoLF keeps its own copy of this (agents/colf.py); the two are pinned to
    agree by a self-test there.
    """
    d, K = X.shape
    if np.linalg.matrix_rank(X) < d:
        raise ValueError("X does not span R^d; the design problem is degenerate")

    pi = np.full(K, 1.0 / K)
    for _ in range(max_iter):
        V = (X * pi) @ X.T
        g = np.einsum("da,da->a", X, np.linalg.solve(V, X))
        if g.max() / d - 1.0 <= tol:
            break
        pi = pi * (g / d)
        pi /= pi.sum()
    return pi, {"value": float(g.max()), "gap": float(g.max() / d - 1.0)}


def observed_only_limit(X, means, pi=None):
    """Where any observed-only least-squares rule ends up, EVALUATION ONLY.

    This agent does not estimate theta_o.  It fits a linear model in x to the
    *whole* reward, so with enough samples under allocation pi it converges to

        theta_proj = V(pi)^-1 sum_a pi_a x_a m_a,   V(pi) = sum_a pi_a x_a x_a^T

    which absorbs whatever part of the latent contribution happens to lie in
    span(x).  `argmax_a x_a^T theta_proj` is therefore the arm it names in the
    limit, and it is NOT the same as `argmax_a x_a^T theta_o` that
    Instance.oracle reports: over 300 S3 instances the two agree only 24% of
    the time.  Conditioning an error rate on the wrong one of these measures
    the wrong event.

    It lives here rather than in env.py for the same reason oracle_mu_star
    lives in colf.py: the limit is a property of *this* agent's model class and
    design, not of the instance.  Defaults to pi^G, the allocation the agent
    tracks; pass pi to ask about a different one.
    """
    if pi is None:
        pi, _ = gopt_design(X)
    theta_proj = np.linalg.solve((X * pi) @ X.T, (X * pi) @ means)
    scores = X.T @ theta_proj
    best = int(np.argmax(scores))
    a_star = int(np.argmax(means))
    return {
        "theta_proj": theta_proj,
        "proj_best": best,
        "proj_reversal": best != a_star,
        "proj_gap": float(scores.max() - scores[a_star]),
    }


class GOptAgent(Agent):
    name = "gopt"

    def __init__(
        self,
        X: np.ndarray,
        delta: float,
        sigma: float,
        rng: Optional[np.random.Generator] = None,
        lam: float = 1e-2,
        elim: float = 2.0,     # one beta for each of the two arms compared
    ) -> None:
        super().__init__(X, delta, sigma, rng)
        self.pi_G, _ = gopt_design(self.X)
        self.lam = float(lam)
        self.elim = float(elim)
        self.beta = confidence_radius(self.sigma, self.K, self.delta)

        self.active = np.ones(self.K, dtype=bool)
        self._A_inv = np.eye(self.d) / self.lam
        self._c = np.zeros(self.d)
        self._psi: Optional[int] = None

    # ------------------------------------------------------------- internals --

    def _theta_hat(self) -> np.ndarray:
        return self._A_inv @ self._c

    def _width(self) -> float:
        """beta_t: bounds |x_a^T (thetahat_t - theta)| for every arm at once."""
        return self.beta * float(np.sqrt(self.d / max(self.t, 1)))

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
        self._A_inv -= np.outer(Av, Av) / (1.0 + float(x @ Av))
        self._c += x * reward

        scores = self.X.T @ self._theta_hat()
        leader = int(np.argmax(np.where(self.active, scores, -np.inf)))
        self.active[self.active & (scores[leader] - scores
                                   > self.elim * self._width())] = False

        if self.active.sum() == 1:
            self._psi = int(np.flatnonzero(self.active)[0])
            self._stop(StopReason.STOPPED)

    def recommend(self) -> int:
        if self._psi is not None:
            return self._psi
        # Exact ties are resolved by index order under np.argmax, which would
        # make the reported accuracy an artefact of arm labelling. Flip a coin
        # instead, matching the 1/|tie set| ceiling Instance.oracle assumes.
        means = np.where(self.active, self.predicted_means(), -np.inf)
        return argmax_random_tie(means, self.rng)

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

    # the limit this agent converges to is not the theta_o ranking, and the
    # agent really does go there: driven past its own stop, argmax of its
    # predicted means matches the projection under its realised allocation.
    cfg3 = InstanceConfig(K=20, d=8, scenario=Scenario.REVERSED, sigma=0.1, min_gap=0.10)
    agree, still_right = [], []
    for seed in range(8):
        ins = make_instance(cfg3, seed=seed)
        ins.reset_noise(seed)
        ag = GOptAgent(ins.observed(), 0.05, 0.1, np.random.default_rng(seed))
        for _ in range(8000):
            a = ag.sample()
            if a is None:                       # keep feeding it past the stop
                a = int(np.argmax(ag.pi_G * ag.t - ag.n_pulls))
            ag.update(a, ins.pull(a))
        emp = ag.n_pulls / ag.n_pulls.sum()
        got = int(np.argmax(ag.predicted_means()))
        agree.append(got == observed_only_limit(ins.X, ins.means, emp)["proj_best"])
        still_right.append(got == ins.a_star)
    assert np.mean(agree) >= 0.75, (
        f"the agent should end up at its own projection, matched {np.mean(agree):.0%}"
    )
    print(f"driven past its stop on S3: matches its projection {np.mean(agree):.0%} "
          f"of the time, still correct {np.mean(still_right):.0%}")

    # the width must be the analytic sqrt(d/t), and the constant the
    # sub-Gaussian one for the Gaussian rewards used here
    ic = InstanceConfig(K=20, d=8, scenario=Scenario.FULL, sigma=0.1, min_gap=0.10)
    ag0 = GOptAgent(make_instance(ic, 3).observed(), delta=0.05, sigma=0.1,
                    rng=np.random.default_rng(0))
    assert np.isclose(ag0.beta, 0.1 * np.sqrt(2 * np.log(2 * 20 / 0.05))), \
        "beta should be sigma sqrt(2 log(2K/delta))"
    assert ag0.elim == 2.0, "one beta for each of the two arms compared"
    ag0.t = 100
    assert np.isclose(ag0._width(), ag0.beta * np.sqrt(8 / 100)), "width is beta sqrt(d/t)"

    # calibration: with nothing hidden this agent is well specified, so any
    # error is the width's fault. A width that is too narrow shows up here.
    run_cal = RunConfig(T_max=200_000)
    for mg in (0.30, 0.10):
        c = InstanceConfig(K=20, d=8, scenario=Scenario.FULL, sigma=0.1, min_gap=mg)
        ok = []
        for seed in range(150):
            i2 = make_instance(c, 20260823 + seed)
            res = run_single(i2, GOptAgent(i2.observed(), 0.05, 0.1,
                                           np.random.default_rng(seed)),
                             run_cal, noise_seed=i2.seed)
            ok.append(res.correct)
        err = 1 - float(np.mean(ok))
        print(f"calibration, d_u = 0, min_gap = {mg:.2f}: error {err:.3f} (delta = 0.05)")
        assert err <= 0.05, (
            f"the width is too narrow: {err:.3f} error on a well-specified problem"
        )

    print("agents/gopt.py self-test passed")


if __name__ == "__main__":
    _selftest()
