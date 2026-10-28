"""CoLF, Algorithm 1: G-optimal design, orthogonal augmentation, a doubly
robust ridge estimator, and two-phase elimination.

PHASE 1 (lines 5-20), while |A_t| > d
    * a_t cycles over all of [K], eliminated arms included (line 6);
    * the pseudo-action is redrawn up to rho_t times, Eq. (10); on no match the
      round pulls nothing and mu_hat is carried forward (lines 7-12);
    * on a match: pull, refit the ridge imputation mu_check with penalty p/K,
      Eq. (7), and the DR estimator mu_hat, Eq. (9), with weight
      1/phi_{a_t,t} = 1/p on the pulled arm (lines 14-16);
    * eliminate a once max_{a' in A} xhat_{a'}^T mu_hat / sqrt(pi_{a'})
      - xhat_a^T mu_hat / sqrt(pi_a) > 2 beta_N(delta), with beta_N the
      Theorem 3 width at N matched rounds (line 18).

PHASE 2 (lines 21-25), while |A_t| >= 2
    successive elimination on per-arm empirical means of the raw rewards,
    every sample since round 1 reused, radius sigma sqrt(2 log(t^2/delta)/N_a).

WHICH mu_check THE PSEUDO-REWARDS USE
-------------------------------------
Eq. (8) writes Yhat_{a,tau} with mu_check_tau, but the proofs of Theorems 2
and 3 factor out mu_check_t - mu_star for every tau <= t, i.e. all past
pseudo-rewards are re-imputed with the current fit. That is the reading used
here.

The DR estimator then collapses onto the imputation. The ridge
normal equations give sum_tau xhat_tau (ybar_tau - xhat_tau^T mu_check)
= (p/K) mu_check, so the numerator of Eq. (9) is
N G mu_check + (1/p)(p/K) mu_check = Vhat mu_check and mu_hat = mu_check
exactly -- which is why Eq. (7) carries p/K as its penalty. p therefore
reaches the estimate only through that penalty; where it matters is the
6 sigma / p in beta_N. The self-test pins the collapse.

THE LEVERAGE IS 1, NOT d/K
--------------------------
Xhat is square and invertible, so Xhat^T G^-1 Xhat = I and
||xhat_a||^2_{Vhat^-1} / pi_a ~ 1 / (pi_a N) <= 2K / N for every arm. The
predicted means therefore converge at sigma sqrt(K / N), and Lemma 6's
(d + 1) / t does not hold; beta_N is still wide enough in every experiment
because its constants (the 3/2 above all) dominate at these K. The self-test
pins the identity.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import Agent, StopReason, round_robin  # noqa: E402


# --------------------------------------------------------------------------- #
# the design and the augmentation, both computed from X alone before round 1
# --------------------------------------------------------------------------- #

class AugmentedDesign:
    """pi, B and the augmented features, all deterministic given X.

    Shapes (rows are dimensions, columns are arms -- getting this backwards is
    the one mistake that stays silent):
        Xbar  (d, K)      xbar_a = sqrt(pi_a) x_a
        B     (K, K-d)    orthonormal basis of R(Xbar)^perp
        Xhat  (K, K)      xhat_a = [xbar_a ; B^T e_a]
        G     (K, K)      Xhat Xhat^T = blockdiag(V(pi), I), a CONSTANT
    """

    def __init__(self, X, gamma=0.5, check=True, augment=True):
        d, K = X.shape
        self.gamma = gamma
        self.augment = augment
        self.pi, self.info = mixed_design(X, gamma)

        self.Xbar = X * np.sqrt(self.pi)
        _, s, Vt = np.linalg.svd(self.Xbar, full_matrices=True)
        if int(np.sum(s > s[0] * 1e-10)) != d:
            raise ValueError(f"Xbar is rank deficient, expected rank d = {d}")
        self.V = self.Xbar @ self.Xbar.T
        if augment:
            self.B = np.ascontiguousarray(Vt[d:, :].T)
            self.Xhat = np.vstack([self.Xbar, self.B.T])
        else:
            # `augment=False` is the ablation: everything else about CoLF is
            # kept -- round-robin, the design, the DR estimator, the stopping
            # rule -- and only the K - d appended directions are removed. The
            # feature matrix is then d x K rather than square, so
            # Xhat^T G^-1 Xhat is the projection P_Xbar instead of I and the
            # estimator can no longer reproduce a reward outside R(Xbar).
            self.B = np.zeros((K, 0))
            self.Xhat = self.Xbar
        self.G = self.Xhat @ self.Xhat.T
        self.K, self.d = K, d
        self.dim = self.Xhat.shape[0]
        if check:
            self.verify()

    def verify(self, atol=1e-9):
        """The identities the analysis rests on (draft Eq. 9, 10)."""
        K, d = self.K, self.d
        if self.augment:
            assert np.allclose(self.B.T @ self.B, np.eye(K - d), atol=atol), "B not orthonormal"
            assert np.allclose(self.Xbar @ self.B, 0.0, atol=atol), "Xbar B != 0"
            ref = np.zeros((K, K))
            ref[:d, :d] = self.V
            ref[d:, d:] = np.eye(K - d)
            assert np.allclose(self.G, ref, atol=atol), "Xhat Xhat^T is not blockdiag(V, I)"
            P = self.Xbar.T @ np.linalg.solve(self.V, self.Xbar)
            assert np.allclose(self.B @ self.B.T, np.eye(K) - P, atol=atol), "BB^T != I - P"
        else:
            assert np.allclose(self.G, self.V, atol=atol), "without B, G must be V(pi)"
        # `gap` is a certificate on max_a g_a / d - 1, and beta scales as
        # sqrt(max_a g_a), so 1e-3 here is a 5e-4 relative effect on the width.
        # The multiplicative update slows down with K: at K = 100 it needs
        # ~60k iterations to reach 1e-6, so a tighter gate would only reject
        # designs that are already good enough.
        assert self.info["gap"] <= 1e-3, f"G-optimal solver did not converge: {self.info}"
        # V(pi) >= (1 - gamma) V(pi^G), so mixing inflates the worst-case
        # prediction variance by at most 1 / (1 - gamma) -- the bound behind
        # Eq. (5). At gamma = 1/2 that is 2d.
        if self.gamma < 1.0:
            # the same 1e-3 slack the solver's own certified gap allows
            assert self.info["mixed_value"] <= d / (1.0 - self.gamma) * (1.0 + 1e-3), (
                f"mixing broke the {d}/(1-gamma) variance bound: {self.info}"
            )


def oracle_mu_star(X, U, theta_o, theta_u, D):
    """The true value of the parameter CoLF estimates, draft Eq. (8).

        mu_par  = theta_o + V(pi)^-1 Xbar Ubar^T theta_u      in R^d
        mu_perp = B^T Ubar^T theta_u                          in R^{K-d}

    EVALUATION ONLY -- it reads U and theta, which no agent may see. It lives
    here rather than in env.py because mu_star is defined by *this* file's
    augmentation: change B and mu_perp changes with it. Feeding it a different
    design than the agent used makes the two incomparable.
    """
    Xbar = X * np.sqrt(D.pi)
    w = (U * np.sqrt(D.pi)).T @ theta_u if U.size else np.zeros(X.shape[1])
    mu_par = theta_o + np.linalg.solve(Xbar @ Xbar.T, Xbar @ w)
    mu_perp = D.B.T @ w
    perp_per_arm = D.B @ mu_perp               # entry a is (B^T e_a)^T mu_perp
    return {
        "mu_par": mu_par,
        "mu_perp": mu_perp,
        "mu_star": np.concatenate([mu_par, mu_perp]),
        "perp_per_arm": perp_per_arm,
        "perp_signal": float(np.abs(perp_per_arm).max()),
    }


def mixed_design(X, gamma=0.5, tol=1e-6, max_iter=60_000):
    """The design CoLF samples from, draft Eq. (5):

        pi   = (1 - gamma) pi^G + gamma / K
        pi^G = argmin_pi max_a ||x_a||^2_{V(pi)^-1},   V(pi) = sum_a pi_a x_a x_a^T

    CoLF never wants pi^G on its own -- an arm with pi_a = 0 carries a latent
    offset that no other arm's samples identify, so its augmented feature would
    never enter the information matrix -- hence one function returning the
    mixture.  The baseline in agents/gopt.py *does* want pi^G unmixed and keeps
    its own copy; a self-test below pins the two solvers to agree.

    HOW THE SOLVE WORKS.  Write g_a(pi) = ||x_a||^2_{V(pi)^-1}, the prediction
    variance at arm a under design pi.  Kiefer-Wolfowitz says three things at
    once: the G-optimal design (minimise max_a g_a) and the D-optimal design
    (maximise log det V) coincide, the optimal value of max_a g_a is exactly d
    whatever K is, and pi is optimal iff g_a = d at every arm with pi_a > 0.

    That last line is the whole algorithm.  sum_a pi_a g_a = tr(V^-1 V) = d
    holds for *every* pi, so g_a / d is a set of multipliers averaging to one
    under pi.  Multiply pi_a by it: arms whose variance is above average gain
    weight, arms below average lose it, and the fixed points are exactly the
    designs with g_a constant on the support.  Since max_a g_a >= d always,
    `gap = max_a g_a / d - 1` is a *certificate* -- it bounds how far the
    returned design is from optimal without needing to know the optimum.
    """
    d, K = X.shape
    if np.linalg.matrix_rank(X) < d:
        raise ValueError("X does not span R^d; the design problem is degenerate")

    piG = np.full(K, 1.0 / K)
    for _ in range(max_iter):
        V = (X * piG) @ X.T
        g = np.einsum("da,da->a", X, np.linalg.solve(V, X))
        if g.max() / d - 1.0 <= tol:
            break
        piG = piG * (g / d)
        piG /= piG.sum()

    pi = (1.0 - gamma) * piG + gamma / K
    pi /= pi.sum()

    Vm = (X * pi) @ X.T
    mixed_value = float(np.einsum("da,da->a", X, np.linalg.solve(Vm, X)).max())
    return pi, {
        "pi_G": piG,
        "value": float(g.max()),          # max_a g_a under pi^G; equals d
        "gap": float(g.max() / d - 1.0),  # certified distance from G-optimal
        "mixed_value": mixed_value,       # max_a g_a under the mixture
    }


class CoLFAgent(Agent):
    name = "colf"

    def __init__(self, X, delta, sigma, rng=None, *,
                 p=0.6,                 # Eq. (7)-(10); the theorems ask p in (1/2, 1)
                 gamma=0.5,             # pi = (1-gamma) pi^G + gamma/K, Eq. (4)
                 eliminate=True,        # False keeps phase 1's estimator running
                                        # with no elimination, for the ablation
                 augment=True):         # False ablates the augmentation only
        super().__init__(X, delta, sigma, rng)
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie in (0, 1)")
        self.p = float(p)
        self.eliminate = eliminate
        self.D = AugmentedDesign(self.X, gamma, augment=augment)
        self.dim = self.D.dim               # K when augmented, d when not
        self.sqrt_pi = np.sqrt(self.D.pi)

        # Vhat = N G + I/K, inverted through G's eigenbasis
        self.g_eig, self.g_vec = np.linalg.eigh(self.D.G)

        # imputation ridge, Eq. (7), held as its inverse Gram so updates are rank-1
        self.A_inv = np.eye(self.dim) / (self.p / self.K)
        self.Sxx = np.zeros((self.dim, self.dim))
        self.Sxy = np.zeros(self.dim)
        self.mu_check = np.zeros(self.dim)
        self.mu_hat = np.zeros(self.dim)
        self.N = 0                              # matched rounds, N(M_<=t)

        self.active = np.ones(self.K, dtype=bool)
        self.phase = 1
        self.round = 0                          # rounds elapsed, pulls or not
        self.sum_y = np.zeros(self.K)
        self.p2_ptr = 0
        self.T_phase1 = None                    # pulls when phase 2 began
        self.psi = None
        self._pending = None

    # ------------------------------------------------------------ phase 1 --

    def _rho(self, t):
        """Eq. (10), t one-indexed."""
        return int(np.ceil(np.log(2.0 * (t + 1) ** 2 / self.delta)
                           / np.log(1.0 / (1.0 - self.p))))

    def _matched(self, t):
        """Lines 7-10. Each draw agrees w.p. phi_{a_t,t} = p, and only the
        match bit is used later, so P(some of rho draws match) stands in for
        the loop."""
        return bool(self.rng.random() < 1.0 - (1.0 - self.p) ** self._rho(t))

    def beta(self, N=None):
        """Theorem 3 width at N matched rounds."""
        N = self.N if N is None else N
        if N == 0:
            return np.inf
        return float(np.sqrt(2.0 * (self.d + 1) / N)
                     * (6.0 * self.sigma / self.p * np.sqrt(np.log(N ** 2 / self.delta))
                        + 1.5))

    def _dr_estimate(self):
        """Eq. (9)."""
        # sum_tau xhat_tau (ybar_tau - xhat_tau^T mu_check_t), computed from
        # Sxx directly so the collapse onto mu_check is checked by the
        # self-test rather than assumed
        resid = self.Sxy - self.Sxx @ self.mu_check
        num = self.N * self.D.G @ self.mu_check + resid / self.p
        lam = self.N * self.g_eig + 1.0 / self.K
        return self.g_vec @ ((self.g_vec.T @ num) / lam)

    def _eliminate_phase1(self):
        """Line 18, then the switch of line 5."""
        m = self.D.Xhat.T @ self.mu_hat / self.sqrt_pi
        lead = m[self.active].max()
        self.active &= ~(lead - m > 2.0 * self.beta())
        if self.active.sum() <= self.d:
            self.phase = 2
            self.T_phase1 = self.t
            self._maybe_finish()

    # ------------------------------------------------------------ phase 2 --

    def _radius(self, n):
        return self.sigma * np.sqrt(2.0 * np.log(self.round ** 2 / self.delta) / n)

    def _eliminate_phase2(self):
        """Lines 23-24, t the round index."""
        idx = np.flatnonzero(self.active)
        n = self.n_pulls[idx]
        mean = self.sum_y[idx] / n
        rad = self._radius(n)
        j = int(np.argmax(mean))
        self.active[idx[mean[j] - mean > rad + rad[j]]] = False
        self._maybe_finish()

    def _maybe_finish(self):
        if self.active.sum() == 1:
            self.psi = int(np.flatnonzero(self.active)[0])
            self._stop(StopReason.STOPPED)

    # ------------------------------------------------------------ interface --

    def sample(self):
        if self.stopped:
            return None
        if self._pending is not None:
            return self._pending
        if self.phase == 1:
            while True:
                a = round_robin(self.round, self.K)         # line 6
                if self._matched(self.round + 1):
                    break
                self.round += 1                             # lines 11-12: no pull,
                                                            # mu_hat carried forward
        else:
            idx = np.flatnonzero(self.active)               # line 22
            a = int(idx[self.p2_ptr % idx.size])
            self.p2_ptr += 1
        self._pending = a
        return a

    def update(self, a, reward):
        self._pending = None
        self._record_pull(a)
        self.round += 1
        self.sum_y[a] += reward
        if self.phase == 2:
            self._eliminate_phase2()
            return

        xhat = self.D.Xhat[:, a]
        ybar = self.sqrt_pi[a] * reward
        Av = self.A_inv @ xhat
        self.A_inv -= np.outer(Av, Av) / (1.0 + xhat @ Av)
        self.Sxx += np.outer(xhat, xhat)
        self.Sxy += xhat * ybar
        self.mu_check = self.A_inv @ self.Sxy               # line 15
        self.N += 1
        self.mu_hat = self._dr_estimate()                   # line 16
        if self.eliminate:
            self._eliminate_phase1()

    def predicted_means(self):
        """Estimated E[y_a] on the original reward scale: the DR prediction,
        and in phase 2 the empirical means that phase decides on."""
        m = self.D.Xhat.T @ self.mu_hat / self.sqrt_pi
        if self.phase == 2:
            m = m.copy()
            m[self.active] = self.sum_y[self.active] / self.n_pulls[self.active]
        return m

    def confidence_width(self):
        """The half-width the current phase eliminates against, reward scale."""
        if self.phase == 1:
            return self.beta()
        return float(self._radius(self.n_pulls[self.active]).max())

    def recommend(self):
        if self.psi is not None:
            return self.psi
        return int(np.argmax(np.where(self.active, self.predicted_means(), -np.inf)))


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest():
    from config import InstanceConfig, RunConfig, Scenario
    from env import make_instance
    from expt.runner import run_single
    from agents.gopt import gopt_design as gopt_design_baseline

    # gopt.py carries its own copy of the solver. Pin the two together, so a
    # fix that misses one fails here rather than silently putting the two
    # agents on different designs. Also check Kiefer-Wolfowitz itself: the
    # optimal value is d, and mixing costs at most a factor 1/(1-gamma).
    rng0 = np.random.default_rng(0)
    for d, K in ((3, 12), (8, 20), (5, 9)):
        Z = rng0.standard_normal((d, K))
        Z /= np.linalg.norm(Z, axis=0)
        pi, info = mixed_design(Z)
        assert np.allclose(info["pi_G"], gopt_design_baseline(Z)[0], atol=1e-10), (
            f"colf.py and gopt.py disagree on pi^G at d={d}, K={K}"
        )
        assert 0.0 <= info["gap"] < 1e-4, (
            f"Kiefer-Wolfowitz value should be d={d}, got {info['value']}"
        )
        assert d <= info["mixed_value"] <= 2 * d, (
            f"mixed design value {info['mixed_value']} outside [d, 2d] at d={d}"
        )
        assert pi.min() >= 0.5 / K - 1e-12, "mixing must floor every pi_a at gamma/K"

        # the equilibrium the update is solving for: g_a = d on the support,
        # g_a <= d off it. This is what makes max_a g_a equal to d.
        piG = info["pi_G"]
        Vg = (Z * piG) @ Z.T
        gg = np.einsum("ia,ia->a", Z, np.linalg.solve(Vg, Z))
        sup = piG > 1e-4
        assert np.allclose(gg[sup], d, rtol=1e-3), \
            f"g_a is not flat on the support: {gg[sup].min()} .. {gg[sup].max()}"
        assert gg.max() <= d * (1.0 + 1e-3), "some arm exceeds the optimal value d"
    print("pi^G hits the Kiefer-Wolfowitz value d; mixing stays inside [d, 2d]")

    # the multiplicative update is a monotone ascent on log det V, which is
    # what makes it converge at all -- there is no line search to fall back on.
    Z = rng0.standard_normal((6, 15))
    Z /= np.linalg.norm(Z, axis=0)
    piG, ld = np.full(15, 1 / 15), []
    for _ in range(500):
        Vg = (Z * piG) @ Z.T
        ld.append(np.linalg.slogdet(Vg)[1])
        gg = np.einsum("ia,ia->a", Z, np.linalg.solve(Vg, Z))
        piG = piG * (gg / 6)
        piG /= piG.sum()
    assert np.min(np.diff(ld)) > -1e-12, "the update must not decrease log det V"
    print(f"log det V ascends monotonically: {ld[0]:.5f} -> {ld[-1]:.5f}")

    # draft Eq. (7)-(8): the augmentation reproduces every arm's reward exactly,
    # xhat_a^T mu_star = sqrt(pi_a) * m_a. This is the claim that the
    # augmentation removes the misspecification rather than merely bounding it.
    for scen in (Scenario.FULL, Scenario.PARTIAL, Scenario.REVERSED):
        icfg = InstanceConfig(K=20, d=8, scenario=scen, sigma=0.1, min_gap=0.10)
        for seed in range(5):
            ins = make_instance(icfg, seed=seed)
            Dv = AugmentedDesign(ins.observed())
            truth = oracle_mu_star(ins.X, ins.U, ins.theta_o, ins.theta_u, Dv)
            assert np.allclose(Dv.Xhat.T @ truth["mu_star"],
                               np.sqrt(Dv.pi) * ins.means, atol=1e-9), (
                f"augmentation does not reproduce the rewards in {scen.value}"
            )
    print("augmentation reproduces the rewards exactly (Eq. 7-8)")

    def drive(agent, inst, n, noise_seed=0):
        """Run for exactly n rounds of round-robin, ignoring the stopping rule."""
        inst.reset_noise(noise_seed)
        for t in range(n):
            a = round_robin(t, agent.K)
            agent.update(a, inst.pull(a))
        return agent

    cfg = InstanceConfig(K=20, d=8, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.10)
    inst = make_instance(cfg, seed=0)
    D = AugmentedDesign(inst.observed())
    orc = oracle_mu_star(inst.X, inst.U, inst.theta_o, inst.theta_u, D)

    # 0. Xhat is square and invertible, so every arm has leverage exactly one
    #    in G^-1 -- the identity behind the sqrt(K / N) rate.
    assert np.abs(D.Xhat.T @ np.linalg.solve(D.G, D.Xhat) - np.eye(cfg.K)).max() < 1e-9, \
        "Xhat^T G^-1 Xhat != I"
    print(f"Xhat^T G^-1 Xhat = I; max_a 1/pi_a = {1 / D.pi.min():.1f} "
          f"against Lemma 6's d + 1 = {cfg.d + 1}")

    # 1. the DR estimator is the ridge imputation, at every p
    for pp in (0.5, 0.6, 0.9):
        ag = CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(0),
                       p=pp, eliminate=False)
        inst.reset_noise(3)
        for t in range(3 * cfg.K + 7):
            a = round_robin(t, cfg.K)
            ag.update(a, inst.pull(a))
            assert np.allclose(ag.mu_hat, ag.mu_check, atol=1e-8), "mu_hat != mu_check"
    print("mu_hat = mu_check at every round, p = 0.5, 0.6, 0.9")

    # 2. consistency, and the augmented block recovers the latent offset
    ag = drive(CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(0),
                         eliminate=False), inst, 20_000)
    e_means = float(np.max(np.abs(ag.predicted_means() - inst.means)))
    perp = D.B @ ag.mu_hat[cfg.d:]
    e_perp = float(np.max(np.abs(perp - orc["perp_per_arm"])))
    print(f"t=20000: means {e_means:.4f} | offset {e_perp:.4f} "
          f"(signal {orc['perp_signal']:.4f}) | beta {ag.beta():.4f}")
    assert e_means < 0.02 and e_perp < 0.02
    assert e_means < ag.beta(), "the Theorem 3 width should cover the error"

    # 3. sample() has no side effects
    ag = CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(0))
    assert ag.sample() == ag.sample() == 0

    # 4. end to end: both phases run, and the answer is right
    run_cfg = RunConfig(T_max=200_000, delta=0.05)
    for scen in (Scenario.FULL, Scenario.PARTIAL, Scenario.REVERSED):
        icfg = InstanceConfig(K=20, d=8, scenario=scen, sigma=0.1, min_gap=0.10)
        ok, T, T1 = [], [], []
        for seed in range(10):
            ins = make_instance(icfg, seed=seed)
            agent = CoLFAgent(ins.observed(), 0.05, icfg.sigma, np.random.default_rng(seed))
            res = run_single(ins, agent, run_cfg, noise_seed=seed)
            assert res.reason == "stopped" and agent.T_phase1 is not None
            ok.append(res.correct)
            T.append(res.T_delta)
            T1.append(agent.T_phase1)
        print(f"{scen.value} | accuracy {np.mean(ok):.0%} | median T {np.median(T):.0f}"
              f" | median phase-1 pulls {np.median(T1):.0f}")
        assert np.mean(ok) >= 0.9, f"CoLF should stay accurate in {scen.value}"

    print("agents/colf.py self-test passed")


if __name__ == "__main__":
    _selftest()
