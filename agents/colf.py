"""CoLF: G-optimal design + doubly robust ridge on the augmented features.

WHAT THE DR STEP IS FOR
-----------------------
Only one arm is pulled per round, so only one reward is observed. The DR
construction turns that single reward into a pseudo-reward for *every* arm:
impute all K with the current ridge fit, then correct the one arm you actually
saw. Two things follow, and they are the whole reason the DR step exists:

  * the information matrix is N * G at *every* round (Eq. 16), exactly and with
    G = blockdiag(V(pi), I) full rank -- so the stopping rule can be evaluated
    every round, not only at sweep boundaries where a pulled-arm Gram would
    happen to equal a multiple of G;
  * no arm is ever poorly covered, so one confidence width serves all arms.

THE IPW WEIGHT
--------------
The correction must be up-weighted by the inverse probability that this arm is
the one being corrected this round. Write the per-round DR sum as

    sum_a xhat_a Yhat_{a,tau} = G mu_check + w * xhat_t xhat_t^T (mu* - mu_check)

with w the IPW weight on the single pulled arm. Over one round-robin sweep
(each arm exactly once, mu_check ~frozen) the numerator is
K G mu_check + w G (mu* - mu_check) against a denominator of K G, so

    mu_hat = mu_check + (w / K) (mu* - mu_check).

The correction moves mu_hat only w/K of the way from the imputation to the
truth. Under round-robin each arm is the pulled arm in exactly 1/K of rounds,
so the propensity is 1/K and **w = K**, which lands on mu* exactly.

The draft's Eq. (12)-(13) instead use w = 1/phi = 1/p. That is the propensity
of the *pseudo-action* matching, not of the arm being pulled, and it gives
w/K = 1/(pK) ~ 0.08 at K=20: mu_hat barely leaves mu_check, and inherits its
lag instead of correcting it. Measured, w=1/p is ~2x less accurate than w=K at
every horizon. `ipw` selects between them so the draft's variant stays runnable.

The pseudo-action redraw of Eq. (15) is kept -- it is what makes the round a
random event with a known probability rather than a deterministic one -- but
note that redrawing *until match* succeeds on all but a vanishing fraction of
rounds, so it does not by itself supply the randomness the correction is
weighted against. This is the open item for Section C.

TWO DELIBERATE DEPARTURES FROM THE ALGORITHM 1 BOX
--------------------------------------------------
1. `mu_check` excludes the current round's own sample. Algorithm 1 updates it
   (line 14) before forming Yhat (line 15), but the draft's own zero-mean claim
   needs mu_check to be F_{t-1}-measurable; folding the sample in first shrinks
   the residual by its own ridge leverage (measured: correction collapses to 17%
   of its true size).
2. Selection and stopping divide out sqrt(pi_a). Algorithm 1 line 19 returns
   argmax_a xhat_a^T mu_hat, but xhat_a carries a sqrt(pi_a) factor, so that
   compares design-weighted rewards; Section 2 defines the best arm by
   unweighted reward.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import Agent, StopReason, round_robin  # noqa: E402
from config import DesignConfig, RunConfig  # noqa: E402
from design import build_design  # noqa: E402


class CoLFAgent(Agent):
    name = "colf"

    def __init__(self, X, delta, sigma, rng=None, run_cfg=None, dcfg=None,
                 ipw="round_robin", use_pseudo_action=True):
        super().__init__(X, delta, sigma, rng)
        cfg = run_cfg or RunConfig()
        self.cfg = cfg
        self.D = build_design(self.X, dcfg or DesignConfig())

        self.p = cfg.coupling_p
        self.use_pseudo_action = use_pseudo_action
        # delta is split evenly between the two ways the run can fail: the
        # redraw cap missing on some round (Eq. 15), and the confidence width
        # being violated (Eq. 17). The draft writes delta' for both but never
        # fixes the split.
        self.delta_prime = self.delta / 2.0
        self.ipw = ipw
        self.w = float(self.K) if ipw == "round_robin" else 1.0 / self.p

        # imputation ridge, held as its inverse Gram so updates are rank-1
        self.A_inv = np.eye(self.K) / cfg.lam_impute
        self.Xy = np.zeros(self.K)

        # DR accumulators over matched rounds
        self.n_matched = 0
        self.sum_imp = np.zeros(self.K)          # sum of mu_check_tau
        self.sum_corr = np.zeros(self.K)         # sum of w * residual * xhat
        self.sum_var = np.zeros((self.K, self.K))  # sum of pi_a xhat xhat^T

        self.psi = None

    # ------------------------------------------------------------ estimates --

    def imputation(self):
        """mu_check: plain ridge on the arms actually pulled so far."""
        return self.A_inv @ self.Xy

    def estimate(self):
        """mu_hat: the DR estimator of Eq. (14)."""
        if self.n_matched == 0:
            return np.zeros(self.K)
        V = self.n_matched * self.D.G + self.cfg.lam_dr * np.eye(self.K)
        return np.linalg.solve(V, self.D.G @ self.sum_imp + self.sum_corr)

    def predicted_means(self):
        """Estimated E[y_a], on the original reward scale."""
        return (self.D.Xhat.T @ self.estimate()) / np.sqrt(self.D.pi)

    def components(self):
        mu = self.estimate()
        return {"par": mu[: self.d], "perp": mu[self.d:]}

    # --------------------------------------------------------- uncertainty --

    def confidence_width(self):
        """beta_N: bounds max_a |xhat_a^T (mu_hat - mu*)|.

        Noise reaches mu_hat through the correction term as
        w * sqrt(pi_tau) * eta_tau * xhat_tau, so the exact variance factor is
        xhat_a^T V^-1 (sum_tau pi_tau xhat xhat^T) V^-1 xhat_a, and the
        sub-Gaussian tail over K arms contributes sqrt(2 log(2K/delta')).

        This is the noise path only. It does NOT cover the imputation error
        (mu_check - mu*) leaking through (G - w xhat xhat^T), which vanishes
        only as mu_check converges. Section C owes that term; `beta_scale`
        stands in for it meanwhile.
        """
        N = self.n_matched
        if N == 0:
            return float("inf")
        V = N * self.D.G + self.cfg.lam_dr * np.eye(self.K)
        W = np.linalg.solve(V, self.D.Xhat)
        var = np.einsum("ia,ia->a", W, self.sum_var @ W).max()
        return self.cfg.beta_scale * self.w * self.sigma * np.sqrt(
            2.0 * var * np.log(2.0 * self.K / self.delta_prime)
        )

    # ------------------------------------------------------------- stopping --

    def _redraw_matches(self, t):
        """Eq. (12)+(15): redraw the pseudo-action until it agrees with the
        pulled arm, capped at rho_t tries. Each try matches w.p. p, and only
        the match/no-match bit is ever used downstream."""
        if not self.use_pseudo_action:
            return True
        rho = int(np.ceil(
            np.log((t + 1) ** 2 / self.delta_prime) / np.log(1.0 / (1.0 - self.p))
        ))
        return any(self.rng.random() < self.p for _ in range(rho))

    def _check_stopping(self):
        """Eq. (17): stop once the leader beats every other arm by more than
        the two arms' combined error budget."""
        means = self.predicted_means()
        psi = int(np.argmax(means))

        # beta lives in xhat-coordinates; dividing by sqrt(pi_a) converts it to
        # a reward-scale budget, which differs per arm because pi does.
        budget = self.confidence_width() / np.sqrt(self.D.pi)
        gap = means[psi] - means
        others = np.arange(self.K) != psi

        if np.all(gap[others] > budget[psi] + budget[others]):
            self.psi = psi
            self._stop(StopReason.STOPPED)

    # ------------------------------------------------------------ interface --

    def sample(self):
        return None if self.stopped else round_robin(self.t, self.K)

    def update(self, a, reward):
        xhat = self.D.Xhat[:, a]
        ybar = np.sqrt(self.D.pi[a]) * reward

        # read the imputation BEFORE folding in this round's sample, so it
        # depends only on rounds < t (see departure 1 in the module docstring)
        mu_check = self.imputation()

        self._record_pull(a)
        if self._redraw_matches(self.t):
            self.sum_corr += self.w * (ybar - xhat @ mu_check) * xhat
            self.sum_imp += mu_check
            self.sum_var += self.D.pi[a] * np.outer(xhat, xhat)
            self.n_matched += 1

        Av = self.A_inv @ xhat
        self.A_inv -= np.outer(Av, Av) / (1.0 + xhat @ Av)
        self.Xy += xhat * ybar

        # every round, not just at sweep boundaries: V_N = N*G holds exactly at
        # each matched round, which is what the DR step buys us
        if self.t % self.cfg.check_every_rounds == 0:
            self._check_stopping()

    def recommend(self):
        return self.psi if self.psi is not None else int(np.argmax(self.predicted_means()))


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest():
    from config import InstanceConfig, Scenario
    from env import make_instance
    from expt.runner import run_single

    def drive(agent, inst, n, noise_seed=0):
        inst.reset_noise(noise_seed)
        for _ in range(n):
            a = agent.sample()
            agent.update(a, inst.pull(a))
        return agent

    cfg = InstanceConfig(K=20, d=8, scenario=Scenario.HALF, sigma=0.1, min_gap=0.10)
    inst = make_instance(cfg, seed=0)
    D = inst.design()
    orc = inst.oracle(D)
    mu_star = np.concatenate([orc["mu_par"], orc["mu_perp"]])
    never_stop = RunConfig(delta=0.05, check_every_rounds=10 ** 9)

    def err(agent):
        return float(np.max(np.abs(D.Xhat.T @ (agent.estimate() - mu_star))))

    # 0. the imputation used by the correction must exclude the current sample
    ag = CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(5),
                   never_stop, use_pseudo_action=False)
    inst.reset_noise(77)
    A, Xy, corr = never_stop.lam_impute * np.eye(cfg.K), np.zeros(cfg.K), np.zeros(cfg.K)
    for _ in range(60):
        a = ag.sample()
        y = inst.pull(a)
        xhat, ybar = D.Xhat[:, a], np.sqrt(D.pi[a]) * y
        corr += ag.w * (ybar - xhat @ np.linalg.solve(A, Xy)) * xhat
        A += np.outer(xhat, xhat)
        Xy += xhat * ybar
        ag.update(a, y)
    assert np.allclose(ag.sum_corr, corr, atol=1e-9), \
        "correction does not match an explicit pre-round imputation fit"
    print("imputation excludes the current round's own sample")

    # 1. w = K must beat the draft's w = 1/p, and match the plain ridge
    e_rr = err(drive(CoLFAgent(inst.observed(), 0.05, cfg.sigma,
                               np.random.default_rng(0), never_stop), inst, 2000))
    ag_paper = drive(CoLFAgent(inst.observed(), 0.05, cfg.sigma,
                               np.random.default_rng(0), never_stop, ipw="paper"),
                     inst, 2000)
    e_paper = err(ag_paper)
    e_ridge = float(np.max(np.abs(D.Xhat.T @ (ag_paper.imputation() - mu_star))))
    print(f"after 2000 rounds: w=K {e_rr:.5f} | w=1/p {e_paper:.5f} | ridge {e_ridge:.5f}")
    assert e_rr < e_paper, "w=K should beat the draft's w=1/p"
    assert e_rr < 2 * e_ridge, "w=K should be comparable to the plain ridge"

    # 2. consistency, and the augmented block recovers the latent offset
    ag = drive(CoLFAgent(inst.observed(), 0.05, cfg.sigma,
                         np.random.default_rng(0), never_stop), inst, 20_000)
    assert ag.n_matched >= 0.99 * 20_000, "redraw-until-match should use nearly every round"
    assert np.max(np.abs(ag.predicted_means() - inst.means)) < 0.02
    perp = D.B @ ag.components()["perp"]
    assert np.max(np.abs(perp - orc["perp_per_arm"])) < 0.02, \
        "augmented block should recover the latent offset"
    print(f"consistency at t=20000: means {np.max(np.abs(ag.predicted_means()-inst.means)):.4f}"
          f" | offset {np.max(np.abs(perp-orc['perp_per_arm'])):.4f}"
          f" (signal {orc['perp_signal']:.4f})")

    # 3. sample() has no side effects
    ag = CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(0))
    assert ag.sample() == ag.sample() == 0

    # 4. end to end
    run_cfg = RunConfig(T_max=50_000, delta=0.05)
    for scen in (Scenario.FULL, Scenario.HALF, Scenario.UNIDENT):
        icfg = InstanceConfig(K=20, d=8, scenario=scen, sigma=0.1, min_gap=0.10)
        ok, T = [], []
        for seed in range(20):
            ins = make_instance(icfg, seed=seed)
            agent = CoLFAgent(ins.observed(), 0.05, icfg.sigma,
                              np.random.default_rng(seed), run_cfg)
            res = run_single(ins, agent, run_cfg, noise_seed=seed)
            ok.append(res.correct)
            T.append(res.T_delta)
        acc = float(np.mean(ok))
        print(f"{scen.value} | accuracy {acc:.0%} | median T {np.median(T):.0f}")
        assert acc >= 0.9, f"CoLF should stay accurate in {scen.value}"

    print("agents/colf.py self-test passed")


if __name__ == "__main__":
    _selftest()
