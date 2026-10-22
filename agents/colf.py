"""CoLF: G-optimal design + doubly robust ridge on the augmented features.

WHAT THE DR STEP IS FOR
-----------------------
Only one arm is pulled per round, so only one reward is observed. The DR
construction turns that single reward into a pseudo-reward for *every* arm:
impute all K with the current ridge fit, then correct the one arm you actually
saw. Two things follow, and they are the whole reason the DR step exists:

  * the information matrix is N * G at *every* round (Eq. 16), exactly and with
    G = blockdiag(V(pi), I) full rank, so one width serves every arm with no
    arm-dependent leverage to track;
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

The principle w = 1/phi is right; what phi equals depends on two conventions,
and 1/p is only ever half of it. phi is the probability that *this arm's*
correction fires this round, which factors as

    phi = P(this arm is the pulled arm) * P(the coupling matched) = (1/K) * p

Whether the p survives depends on how N is counted. Measured on a noiseless
instance with mu_check frozen at 0, relative error ||mu_hat - mu*|| / ||mu*||:

    w          round-robin              iid a ~ pi
               matched  all rounds      matched  all rounds
    1/p         0.917     0.950          0.903     0.942
    K           0.003     0.399          0.296     0.334
    K/p         0.667     0.004          1.029     0.297
    1/pi_a      0.241     0.447          0.012     0.401
    1/(p pi_a)  0.719     0.241          0.665     0.012

Each column has exactly one right answer, and 1/p is never it -- at 1/p the
estimator stays within 9% of the imputation it started from. But K/p *is*
right when unmatched rounds are counted, and there the draft's 1/p is literally
present; the missing factor is the 1/K, not the 1/p. This file drops unmatched
rounds from N, which cancels the p and leaves w = K.

Eq. (6) is what pins the column: sqrt(pi_a) is folded into the features so that
"V(pi) is reproduced exactly rather than in expectation", and that holds only
when each arm appears exactly once per sweep. So P(this arm is pulled) = 1/K.

EQ. (15) DECIDES THE OTHER HALF, AND IT CANCELS THE p
-----------------------------------------------------
The redraw of Eq. (15) draws the pseudo-action up to
rho_t = ceil(log((t+1)^2/delta') / log(1/(1-p))) times and takes any match, so
the probability that a round matches is not p but 1 - (1-p)^rho_t:

    t          1        10       100     10000
    rho_t      6        10        15        25
    P(match)   0.9959   0.99990   0.999999  1.000000

Eq. (15) exists precisely to drive that to one. But 1/p in Eq. (12)-(13) is
compensating for the match *failing*, which Eq. (15) has just removed. The two
equations undo each other; you want one or the other, not both.

Residual ||M_s|| left by one closed sweep, over 4000 sweeps:

    pseudo-action        w      counts    ||E[M_s]||   E||M_s||
    single draw         1/p     matched     11.024      11.975
    single draw         K/p     all          0.291      19.397
    single draw         K       matched      0.162      12.068
    rho_t (Eq. 15)      1/p     matched     18.333      18.333
    rho_t (Eq. 15)      K/p     all         13.333      13.333
    rho_t (Eq. 15)      K       matched      0.000       0.000

Under a single draw both K/p and K are unbiased but only in expectation, with
large per-sweep noise. Under Eq. (15) the match is deterministic in effect, and
w = K is exactly zero every sweep -- not zero on average, zero. That is the
combination this file uses. `ipw` keeps the other two runnable.

The pseudo-action redraw of Eq. (15) is kept -- it is what makes the round a
random event with a known probability rather than a deterministic one -- but
note that redrawing *until match* succeeds on all but a vanishing fraction of
rounds, so it does not by itself supply the randomness the correction is
weighted against. This is the open item for Section C.

WHERE THE ERROR COMES FROM, AND WHY mu_check IS FROZEN
-----------------------------------------------------
Writing V_N = N G + lambda I, the estimator's error splits exactly into three
pieces (verified numerically to 6e-16 in the self-test):

    mu_hat - mu* = V_N^-1 sum_tau (G - w xhat xhat^T)(mu_check_tau - mu*)   (i)
                 + w V_N^-1 sum_tau sqrt(pi_tau) eta_tau xhat_tau           (ii)
                 - lambda V_N^-1 mu*                                       (iii)

Only (ii) carries sigma. (i) and (iii) are of order ||mu*|| however quiet the
rewards are, so a width built from (ii) alone shrinks with sigma while the
error does not -- at sigma = 0.03 the stopping rule fired within a few rounds
on an estimate built from almost nothing.

(i) is the one that matters, and freezing mu_check over a whole round-robin
sweep removes it outright. With mu_check constant across the sweep the sum
factors as M (mu_frozen - mu*) with M = sum (G - w xhat xhat^T), and over a
complete sweep M = K G - w G = 0 exactly at w = K. Measured at sigma = 0.03,
t = 400: total error 0.0127 -> 0.0030, and what remains is (ii) to four
decimals. So beta being proportional to sigma is now a statement about the
estimator, not an omission in the width.

Inside an unfinished sweep M is not yet zero, and the same factoring bounds
what it leaks; `confidence_width` carries that term explicitly. The bound is
crude there -- it has to fall back on ||mu*|| <= S_mu -- so in practice the
stopping rule fires at sweep boundaries, where the term is exactly zero. That
is the honest reading of the guarantee: it is a sweep-level one. It also keeps
the rule quiet before the first sweep closes.

ONE MORE DEPARTURE FROM THE ALGORITHM 1 BOX
-------------------------------------------
Selection and stopping divide out sqrt(pi_a). Algorithm 1 line 19 returns
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


# w, and whether a round with no pseudo-action match still counts towards N.
# phi = P(this arm's correction fires) = (1/K) * p, and w = 1/phi, so the two
# conventions differ only in whether the p is carried in w or absorbed by not
# counting the rounds it drops.
_IPW = {
    "paper":       lambda K, p: (1.0 / p, False),   # as drafted; the 1/K is missing
    "coupled":     lambda K, p: (K / p, True),      # the draft's 1/p with 1/K restored
    "round_robin": lambda K, p: (float(K), False),  # p absorbed by dropping the rounds
}


class CoLFAgent(Agent):
    name = "colf"

    def __init__(self, X, delta, sigma, rng=None, *,
                 gamma=0.5,             # pi = (1-gamma) pi^G + gamma/K, draft Eq. (5)
                 p=None,                # coupling probability of Eq. (12); None
                                        # gives 1/K, at which Eq. (12) is the
                                        # uniform draw -- see below
                 lam_impute=1e-2,       # ridge penalty of the imputation fit
                 lam_dr=0.0,            # ridge penalty of the DR fit; G is full
                                        # rank, so no penalty is needed at all
                 beta_scale=1.0,        # inflation of the confidence width
                 reward_bound=1.0,      # known S with max_a |m_a| <= S, Section 2
                 ipw="paper",           # w = 1/phi, which is K at p = 1/K
                 use_pseudo_action=True,
                 augment=True):         # False ablates the augmentation only
        super().__init__(X, delta, sigma, rng)
        p = 1.0 / self.K if p is None else float(p)
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie in (0, 1)")
        # At p = 1/K, Eq. (12) is the uniform draw: phi_a = p = 1/K on the
        # pulled arm and (1-p)/(K-1) = 1/K on every other, so atilde is drawn
        # uniformly over the arms -- the propensity a round-robin visit has.
        # Then 1/phi_{a,tau} = K and Eq. (13) is correct exactly as written,
        # the correction is unbiased round by round, and one sweep contributes
        # sum_a (G - w xhat_a xhat_a^T) = (K - w) G = 0 identically.
        # Away from 1/K the draft's p in (1/2, 1) leaves (K - 1/p) G behind on
        # every sweep -- 18.33 G at p = 0.6 -- and a width that covers that
        # residual overshoots the realised error by 40x to 233x.
        if beta_scale <= 0.0:
            raise ValueError("beta_scale must be positive")
        self.D = AugmentedDesign(self.X, gamma, augment=augment)
        self.dim = self.D.dim               # K when augmented, d when not

        self.p = p
        self.lam_impute = lam_impute
        self.lam_dr = lam_dr
        self.beta_scale = beta_scale
        self.use_pseudo_action = use_pseudo_action
        # delta is split three ways, over the three things that must all hold:
        # the redraw cap never misses (Eq. 15), the imputation stays inside its
        # own ridge bound, and the DR width is never violated (Eq. 17). The
        # draft writes delta' throughout and never fixes the split.
        self.delta_prime = self.delta / 3.0
        if ipw not in _IPW:
            raise ValueError(f"ipw must be one of {tuple(_IPW)}, got {ipw!r}")
        self.ipw = ipw
        self.w, self.count_all = _IPW[ipw](self.K, self.p)

        # ||mu*|| <= S / sqrt(lambda_min(G)).  Xhat^T mu* = sqrt(pi) * m, and
        # ||sqrt(pi) * m|| <= S ||sqrt(pi)|| = S since pi sums to one; invert
        # through sigma_min(Xhat)^2 = lambda_min(G).  Known from X and S alone.
        self.g_eig = np.linalg.eigvalsh(self.D.G)
        self.S_mu = float(reward_bound) / np.sqrt(self.g_eig.min())

        # imputation ridge, held as its inverse Gram so updates are rank-1
        self.A_inv = np.eye(self.dim) / lam_impute
        self.Xy = np.zeros(self.dim)

        # DR accumulators, split between closed sweeps and the sweep in progress
        self.n_closed = 0
        self.c_imp = np.zeros(self.dim)
        self.c_corr = np.zeros(self.dim)
        self.c_var = np.zeros((self.dim, self.dim))

        self.mu_frozen = np.zeros(self.dim)      # mu_check held over the sweep
        self.sum_ME = 0.0                        # sum of ||M_s|| E_s over closed sweeps
        self._new_sweep()

        self.psi = None

    def _new_sweep(self):
        """Freeze the imputation, record how far it can be from mu*, and clear
        the sweep buffer."""
        self.mu_frozen = self.imputation()
        self.E_frozen = self._imputation_error_bound()
        self.b_n = 0
        self.b_imp = np.zeros(self.dim)
        self.b_corr = np.zeros(self.dim)
        self.b_var = np.zeros((self.dim, self.dim))
        self.b_M = np.zeros((self.dim, self.dim))  # sum of (G - w xhat xhat^T)

    def _close_sweep(self):
        """Commit the sweep if it is complete, discard it otherwise.

        A complete sweep has b_M = K G - w G = 0 exactly at w = K, which is what
        makes the imputation term of the error vanish (see the module docstring).
        A sweep with a round missing does not telescope, so it is dropped rather
        than left to contaminate a width that assumes it did.  The pseudo-action
        misses a round with probability at most delta'/(t+1)^2, so this discards
        under 0.05 sweeps in a whole run.
        """
        if self.b_n == self.K:
            self.n_closed += self.K
            self.c_imp += self.b_imp
            self.c_corr += self.b_corr
            self.c_var += self.b_var
            # ||M_s|| ||mu_check_s - mu*|| is all the width needs from a closed
            # sweep, and it is a scalar, so the matrices need not be kept.
            # M_s = 0 exactly at w = K, which contributes nothing at all.
            self.sum_ME += float(np.linalg.norm(self.b_M, 2)) * self.E_frozen
        self._new_sweep()

    @property
    def n_matched(self) -> int:
        return self.n_closed + self.b_n

    # ------------------------------------------------------------ estimates --

    def imputation(self):
        """mu_check: plain ridge on the arms actually pulled so far."""
        return self.A_inv @ self.Xy

    def _imputation_error_bound(self):
        """E: a high-probability bound on ||mu_check - mu*||, needed because
        the imputation does not cancel out of mu_hat unless w = K.

        Standard self-normalised ridge bound (Abbasi-Yadkori et al.), which is
        already uniform over t so no union over sweeps is required:

            ||mu_check - mu*||_A <= sigma sqrt(2 log(1/delta') + log det A/det lam I)
                                    + sqrt(lam) ||mu*||

        and ||.||_2 <= ||.||_A / sqrt(lambda_min(A)).  Under round-robin the
        Gram is exactly A = s G + lam I after s whole sweeps of pulls, so both
        det A and lambda_min(A) come straight off G's eigenvalues -- this is
        called only at sweep boundaries.
        """
        crude = float(np.linalg.norm(self.mu_frozen) + self.S_mu)
        s = self.t // self.K
        if s == 0:
            return crude
        eig = s * self.g_eig + self.lam_impute
        logdet = float(np.sum(np.log(eig / self.lam_impute)))
        num = self.sigma * np.sqrt(2.0 * np.log(1.0 / self.delta_prime) + logdet)
        num += np.sqrt(self.lam_impute) * self.S_mu
        # the triangle-inequality bound is better than the self-normalised one
        # early on, when A is still close to lam I
        return min(float(num / np.sqrt(eig.min())), crude)

    def estimate(self):
        """mu_hat: the DR estimator of Eq. (14), over closed sweeps plus the
        sweep in progress."""
        N = self.n_matched
        if N == 0:
            return np.zeros(self.dim)
        V = N * self.D.G + self.lam_dr * np.eye(self.dim)
        return np.linalg.solve(
            V, self.D.G @ (self.c_imp + self.b_imp) + self.c_corr + self.b_corr
        )

    def predicted_means(self):
        """Estimated E[y_a], on the original reward scale."""
        return (self.D.Xhat.T @ self.estimate()) / np.sqrt(self.D.pi)

    def components(self):
        mu = self.estimate()
        return {"par": mu[: self.d],
                "perp": mu[self.d:] if self.D.augment else None}

    # --------------------------------------------------------- uncertainty --

    def reward_budget(self):
        """Per-arm confidence width on the REWARD scale: bounds |mhat_a - m_a|,
        all three terms of the error decomposition at the top of this file.

            noise   w sigma sqrt(2 var_a log(2K/delta')),
                    var_a = xhat_a^T V^-1 (sum pi_tau xhat xhat^T) V^-1 xhat_a
            ridge   lambda ||V^-1 xhat_a|| S_mu
            sweep   ||M^T V^-1 xhat_a|| (||mu_frozen|| + S_mu)

        Only the first is proportional to sigma.  The imputation term is of
        order ||mu*|| however quiet the rewards are, which is why a width built
        from the noise alone fired on almost no data at small sigma.

        Freezing mu_check over a sweep is what makes the imputation term
        bounded at all: it factors as V^-1 M (mu_frozen - mu*) with M the sum
        of (G - w xhat xhat^T) over the sweep.  A closed sweep gives the same
        M_s = (K - w) G every time, so closed sweeps need only the running sum
        of ||mu_check_s - mu*|| bounds; the sweep in progress carries its own
        partial M.  At w = K, (K - w) G = 0 and the whole closed-sweep term
        disappears -- that is the one value of w that removes the imputation
        from the error rather than merely bounding it.
        """
        N = self.n_matched
        if N == 0:
            return np.full(self.K, np.inf)
        V = N * self.D.G + self.lam_dr * np.eye(self.dim)
        W = np.linalg.solve(V, self.D.Xhat)            # columns V^-1 xhat_a

        # a quadratic form in a PSD matrix, so non-negative in exact arithmetic;
        # clipped because it lands at -5e-19 on the first round
        var = np.einsum("ia,ia->a", W, (self.c_var + self.b_var) @ W).clip(0.0)
        noise = self.w * self.sigma * np.sqrt(
            2.0 * var * np.log(2.0 * self.K / self.delta_prime)
        )
        Wn = np.linalg.norm(W, axis=0)
        ridge = self.lam_dr * Wn * self.S_mu
        leak = Wn * (self.sum_ME + np.linalg.norm(self.b_M, 2) * self.E_frozen)

        # Divide by sqrt(pi_a) per arm, and only then take a maximum. Doing it
        # the other way round -- one scalar width in xhat-coordinates, spent as
        # beta / sqrt(pi_a) -- inflates the budget of every arm by
        # sqrt(pi_max / pi_a), which is 1.3x at K = 10 and 2.1x at K = 50, and
        # costs the square of that in samples. The inflation is pure slack:
        # Xhat is square and invertible so Xhat^T G^-1 Xhat = I exactly, hence
        # Var(mhat_a) = sigma^2 / (number of sweeps) for EVERY arm and the
        # correct per-arm budget is the same number for all of them.
        return self.beta_scale * (noise + ridge + leak) / np.sqrt(self.D.pi)

    def confidence_width(self):
        """The widest per-arm reward-scale budget, for reporting."""
        b = self.reward_budget()
        return float(b if np.isscalar(b) else np.max(b))

    # ------------------------------------------------------------- stopping --

    def _redraw_matches(self, t):
        """Eq. (12)+(15): draw the pseudo-action up to rho_t times; the round
        counts if any draw agrees with the pulled arm, and is skipped entirely
        if none does (Algorithm 1 lines 7-12). Each draw agrees w.p. p, so only
        the match/no-match bit matters -- the pseudo-action's identity is never
        used by Eq. (13)-(14)."""
        if not self.use_pseudo_action:
            return True
        rho = int(np.ceil(
            np.log((t + 1) ** 2 / self.delta_prime) / np.log(1.0 / (1.0 - self.p))
        ))
        # only the match/no-match bit is ever used, and P(at least one of rho
        # draws matches) = 1 - (1-p)^rho, so one uniform stands in for rho of
        # them. At p = 1/K that is ~470 draws a round saved by t = 20000.
        return bool(self.rng.random() < 1.0 - (1.0 - self.p) ** rho)

    def _check_stopping(self):
        """Eq. (17): stop once the leader beats every other arm by more than
        the two arms' combined error budget.

        Never before every arm has been pulled at least once. Theorem 1 makes
        that a necessary condition -- an arm's latent offset is not identified
        by samples of the other arms, so its reward is undetermined however
        large its gap. The width's sweep term now enforces the same thing on
        its own, since M is far from zero until the first sweep closes; the
        guard stays because it is the statement of the theorem rather than a
        consequence of how the width happens to be built.
        """
        if self.n_pulls.min() == 0:
            return
        means = self.predicted_means()
        psi = int(np.argmax(means))

        budget = self.reward_budget()
        gap = means[psi] - means
        others = np.arange(self.K) != psi

        if np.all(gap[others] > budget[psi] + budget[others]):
            self.psi = psi
            self._stop(StopReason.STOPPED)

    # ------------------------------------------------------------ interface --

    def sample(self):
        return None if self.stopped else round_robin(self.t, self.K)

    def update(self, a, reward):
        if self.t % self.K == 0 and self.t > 0:
            self._close_sweep()

        xhat = self.D.Xhat[:, a]
        ybar = np.sqrt(self.D.pi[a]) * reward
        mu_check = self.mu_frozen

        self._record_pull(a)
        matched = self._redraw_matches(self.t)
        if matched or self.count_all:
            # the round enters N and contributes its imputation to the numerator
            self.b_imp += mu_check
            self.b_M += self.D.G
            self.b_n += 1
        if matched:
            self.b_corr += self.w * (ybar - xhat @ mu_check) * xhat
            self.b_var += self.D.pi[a] * np.outer(xhat, xhat)
            self.b_M -= self.w * np.outer(xhat, xhat)

        Av = self.A_inv @ xhat
        self.A_inv -= np.outer(Av, Av) / (1.0 + xhat @ Av)
        self.Xy += xhat * ybar

        # every round: V_N = N*G holds exactly at each matched round, which is
        # what the DR step buys us
        self._check_stopping()

    def recommend(self):
        return self.psi if self.psi is not None else int(np.argmax(self.predicted_means()))


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
        """Run for exactly n rounds, ignoring the stopping rule -- these tests
        are about the estimator, so they drive round_robin directly."""
        inst.reset_noise(noise_seed)
        for t in range(n):
            a = round_robin(t, agent.K)
            agent.update(a, inst.pull(a))
        return agent

    cfg = InstanceConfig(K=20, d=8, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.10)
    inst = make_instance(cfg, seed=0)
    D = AugmentedDesign(inst.observed())
    orc = oracle_mu_star(inst.X, inst.U, inst.theta_o, inst.theta_u, D)
    mu_star = orc["mu_star"]

    def err(agent):
        return float(np.max(np.abs(D.Xhat.T @ (agent.estimate() - mu_star))))

    # 0. the correction must use the imputation frozen at the start of the
    #    sweep, the three-term decomposition must be exact, and a closed sweep
    #    must leave exactly M = (K - w) G behind -- zero only at w = K.
    #    p is inert: the two w = K rows below differ only in p and agree to
    #    machine precision, while the w = 1/p row does not.
    for label, ipw, pp in (("w=1/p  p=0.6", "paper", 0.6),
                           ("w=K    p=1/K", "paper", 1.0 / cfg.K),
                           ("w=K    p=0.6", "round_robin", 0.6)):
        ag = CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(5),
                       ipw=ipw, p=pp, use_pseudo_action=False)
        inst.reset_noise(77)
        A, Xy, corr = ag.lam_impute * np.eye(cfg.K), np.zeros(cfg.K), np.zeros(cfg.K)
        M = np.zeros((cfg.K, cfg.K))
        imp, noise, frozen = np.zeros(cfg.K), np.zeros(cfg.K), None
        for t in range(3 * cfg.K):
            a = round_robin(t, cfg.K)
            if t % cfg.K == 0:
                frozen = np.linalg.solve(A, Xy)
            y = inst.pull(a)
            xhat, ybar = D.Xhat[:, a], np.sqrt(D.pi[a]) * y
            R = D.G - ag.w * np.outer(xhat, xhat)
            corr += ag.w * (ybar - xhat @ frozen) * xhat
            M += R
            imp += R @ (frozen - mu_star)
            noise += ag.w * np.sqrt(D.pi[a]) * (y - inst.means[a]) * xhat
            A += np.outer(xhat, xhat)
            Xy += xhat * ybar
            ag.update(a, y)

        assert np.allclose(ag.c_corr + ag.b_corr, corr, atol=1e-9), \
            "correction does not use the imputation frozen at the sweep boundary"
        assert np.allclose(M, 3 * (cfg.K - ag.w) * D.G, atol=1e-9), \
            "a closed sweep must telescope to (K - w) G"

        V = ag.n_matched * D.G + ag.lam_dr * np.eye(cfg.K)
        parts = [np.linalg.solve(V, imp), np.linalg.solve(V, noise),
                 -ag.lam_dr * np.linalg.solve(V, mu_star)]
        assert np.abs(ag.estimate() - mu_star - sum(parts)).max() < 1e-12, \
            "the three-term error decomposition is not exact"
        leak = np.abs(parts[0]).max()
        if np.isclose(ag.w, cfg.K):
            assert leak < 1e-12, "at w = K the imputation term must be exactly zero"
        else:
            assert leak > 1e-3, "away from w = K the imputation term should survive"
        print(f"  {label:12s} w={ag.w:6.3f}  imputation term after 3 sweeps = {leak:.5f}")
    print("error decomposition exact; imputation term vanishes exactly at w = K")

    # 0a. Eq. (15) drives the match probability to one, which is what leaves
    #     1/K as the only propensity and cancels the p out of w.
    ag = CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(0))
    for t in (1, 100, 10_000):
        rho = int(np.ceil(np.log((t + 1) ** 2 / ag.delta_prime)
                          / np.log(1.0 / (1.0 - ag.p))))
        assert 1.0 - (1.0 - ag.p) ** rho > 0.995, \
            "Eq. (15) is supposed to make a match near certain"
    hits = sum(ag._redraw_matches(t) for t in range(1, 2001))
    assert hits >= 1990, f"redraw should match nearly every round, got {hits}/2000"
    print(f"Eq. (15) redraw matches {hits}/2000 rounds: the p in 1/p has nothing left to fix")

    # 0aa. Xhat is square and invertible, so Xhat^T G^-1 Xhat = I exactly and
    #      Var(mhat_a) is the same for every arm. The reward-scale budget must
    #      therefore be flat across arms; if it is not, a max has been taken in
    #      the wrong coordinates and every arm pays sqrt(pi_max / pi_a) for it.
    for Kc in (10, 20, 40):
        ic = InstanceConfig(K=Kc, d=8, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.10)
        ins = make_instance(ic, seed=0)
        ag = CoLFAgent(ins.observed(), 0.05, ic.sigma, np.random.default_rng(0))
        assert np.abs(ag.D.Xhat.T @ np.linalg.solve(ag.D.G, ag.D.Xhat)
                      - np.eye(Kc)).max() < 1e-9, "Xhat^T G^-1 Xhat != I"
        ins.reset_noise(1)
        for t in range(5 * Kc):
            a = round_robin(t, Kc)
            ag.update(a, ins.pull(a))
        b = ag.reward_budget()
        assert b.max() / b.min() < 1.001, (
            f"reward budget is not flat across arms at K={Kc}: "
            f"spread {b.max() / b.min():.3f}"
        )
    print("reward-scale budget is flat across arms, as Xhat^T G^-1 Xhat = I requires")

    # 0b. Eq. (10)'s factor 1 - 1(atilde = a)/phi_a is mean zero under the raw
    #     multinomial of Eq. (9) but not after resampling, where its mean is
    #     exactly 1 - 1/(K p). So the draft's w = 1/p is right at p = 1/K, and
    #     there it is the same number as w = K.
    def factor_mean(pp, matched, reps=40_000, seed=0):
        rng = np.random.default_rng(seed)
        acc = np.zeros(cfg.K)
        others = [[k for k in range(cfg.K) if k != j] for j in range(cfg.K)]
        for _ in range(reps):
            a_t = int(rng.integers(cfg.K))
            at = a_t if (matched or rng.random() < pp) else int(rng.choice(others[a_t]))
            f = np.ones(cfg.K)
            f[at] -= 1.0 / (pp if at == a_t else (1 - pp) / (cfg.K - 1))
            acc += f
        return float(np.abs(acc / reps).max())

    for pp in (0.6, 1.0 / cfg.K):
        assert factor_mean(pp, matched=False) < 0.1, "Eq. (9) should make the factor mean zero"
        assert abs(factor_mean(pp, matched=True) - abs(1 - 1 / (cfg.K * pp))) < 0.05, \
            "conditioning on the match should leave a mean of 1 - 1/(K p)"
    assert np.isclose(CoLFAgent(inst.observed(), 0.05, cfg.sigma, p=1.0 / cfg.K).w, cfg.K), \
        "at p = 1/K the draft's w = 1/p should coincide with w = K"
    print(f"Eq. (10) factor after resampling: mean 1 - 1/(K p), zero only at p = 1/K")

    # 0b. the width must not vanish with sigma: only its noise part scales
    def width_at(sig):
        a = drive(CoLFAgent(inst.observed(), 0.05, sig, np.random.default_rng(1),
                            ), inst, 5 * cfg.K)
        return a.confidence_width()
    w_hi, w_lo = width_at(0.10), width_at(0.01)
    assert w_lo > 0.1 * w_hi, "width still collapses with sigma"
    print(f"width at sigma 0.10 / 0.01: {w_hi:.5f} / {w_lo:.5f}")

    # 1. w = K must beat w = 1/p at the draft's p = 0.6, and match the plain ridge
    e_rr = err(drive(CoLFAgent(inst.observed(), 0.05, cfg.sigma,
                               np.random.default_rng(0),
                               ipw="round_robin"), inst, 2000))
    ag_paper = drive(CoLFAgent(inst.observed(), 0.05, cfg.sigma, np.random.default_rng(0),
                               ipw="paper", p=0.6), inst, 2000)
    e_paper = err(ag_paper)
    e_ridge = float(np.max(np.abs(D.Xhat.T @ (ag_paper.imputation() - mu_star))))
    print(f"after 2000 rounds: w=K {e_rr:.5f} | w=1/p {e_paper:.5f} | ridge {e_ridge:.5f}")
    assert e_rr < e_paper, "w=K should beat w=1/p at the draft's p=0.6"
    assert e_rr < 2 * e_ridge, "w=K should be comparable to the plain ridge"

    # 2. consistency, and the augmented block recovers the latent offset
    ag = drive(CoLFAgent(inst.observed(), 0.05, cfg.sigma,
                         np.random.default_rng(0), ipw="round_robin"), inst, 20_000)
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

    # 4. end to end, at w = K
    run_cfg = RunConfig(T_max=20_000, delta=0.05)
    for scen in (Scenario.FULL, Scenario.PARTIAL, Scenario.REVERSED):
        icfg = InstanceConfig(K=20, d=8, scenario=scen, sigma=0.1, min_gap=0.10)
        ok, T = [], []
        for seed in range(20):
            ins = make_instance(icfg, seed=seed)
            agent = CoLFAgent(ins.observed(), 0.05, icfg.sigma,
                              np.random.default_rng(seed), ipw="round_robin")
            res = run_single(ins, agent, run_cfg, noise_seed=seed)
            ok.append(res.correct)
            T.append(res.T_delta)
        acc = float(np.mean(ok))
        print(f"{scen.value} | accuracy {acc:.0%} | median T {np.median(T):.0f}")
        assert acc >= 0.9, f"CoLF should stay accurate in {scen.value}"

    # 5. the same run at the draft's w = 1/p does not terminate. Recorded as a
    #    test because it is the experimental content of the IPW discussion: the
    #    imputation survives in mu_hat, and a width that bounds it honestly
    #    never clears the gap.
    icfg = InstanceConfig(K=20, d=8, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.10)
    cen = []
    for seed in range(5):
        ins = make_instance(icfg, seed=seed)
        agent = CoLFAgent(ins.observed(), 0.05, icfg.sigma,
                          np.random.default_rng(seed), ipw="paper", p=0.6)
        cen.append(run_single(ins, agent, RunConfig(T_max=5_000), noise_seed=seed).censored)
    assert all(cen), "w = 1/p at p = 0.6 is expected to censor; if it stops, revisit the width"
    print(f"w = 1/p at p = 0.6 censors {sum(cen)}/{len(cen)} runs at T_max = 5000")

    print("agents/colf.py self-test passed")


if __name__ == "__main__":
    _selftest()
