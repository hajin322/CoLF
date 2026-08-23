"""G-optimal design, orthogonal augmentation, and the oracle reparameterisation.

Everything here is a pure function of arrays.  Nothing in this module touches
the environment, so it can be imported by both the agents (which may use only
`X`) and the evaluation code (which may use `U` and `theta`).

Shape convention, matching the draft
------------------------------------
    X      (d, K)      column a is the observed feature x_a
    U      (du, K)     column a is the latent feature u_a
    Xbar   (d, K)      Xbar = X D,  D = diag(sqrt(pi_1), ..., sqrt(pi_K))
    B      (K, K - d)  orthonormal basis of R(Xbar)^perp  (a subspace of R^K)
    Xhat   (K, K)      Xhat = [Xbar ; B^T];  column a is xhat_a = [xbar_a ; B^T e_a]

`Xhat[:, a]` is the augmented feature of arm a.  Rows index the augmented
dimension, columns index arms.  Getting this backwards is the single most
likely source of a silent bug downstream.

Key identities verified by `verify_design`
------------------------------------------
    V(pi)      = Xbar Xbar^T = sum_a pi_a x_a x_a^T          (d, d)
    Xbar B     = 0                                            (Eq. 9)
    Xhat Xhat^T = blockdiag(V(pi), I_{K-d})                   (Eq. 10)
    xhat_a^T mu_star = sqrt(pi_a) * z_a^T theta_star          (Eq. 7-8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config import DesignConfig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def design_scale(X: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Xbar = X D with D = diag(sqrt(pi)).  Scales column a by sqrt(pi_a)."""
    return X * np.sqrt(pi)[None, :]


def prediction_variances(X: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """g_a = ||x_a||^2_{V(pi)^-1} for every arm, where V(pi) = sum_a pi_a x_a x_a^T."""
    V = (X * pi[None, :]) @ X.T
    Z = np.linalg.solve(V, X)          # (d, K)
    return np.einsum("da,da->a", X, Z)


# --------------------------------------------------------------------------- #
# 1. G-optimal design
# --------------------------------------------------------------------------- #

def gopt_design(
    X: np.ndarray,
    cfg: Optional[DesignConfig] = None,
) -> Tuple[np.ndarray, dict]:
    """Solve  pi^G = argmin_pi max_a ||x_a||^2_{V(pi)^-1}.

    By the Kiefer-Wolfowitz equivalence theorem the G-optimal and D-optimal
    designs coincide and the optimal value is exactly d, independent of K.  We
    therefore solve the D-optimal problem with the multiplicative (Titterington)
    update

        pi_a  <-  pi_a * g_a(pi) / d,        g_a(pi) = ||x_a||^2_{V(pi)^-1},

    which monotonically increases log det V(pi) and reaches machine precision in
    a few hundred iterations, where Frank-Wolfe with the Fedorov-Wynn step
    stalls at an O(1/k) gap.  Since max_a g_a >= d always holds, the returned
    `info["gap"] = max_a g_a / d - 1 >= 0` is a certified optimality gap.

    Returns
    -------
    pi_G : (K,) design weights summing to one
    info : {"value", "gap", "n_iter", "support"}
    """
    cfg = cfg or DesignConfig()
    d, K = X.shape

    if np.linalg.matrix_rank(X) < d:
        raise ValueError("X does not span R^d; the design problem is degenerate")

    pi = np.full(K, 1.0 / K)
    n_iter, gap = 0, np.inf
    for n_iter in range(1, cfg.fw_max_iter + 1):
        g = prediction_variances(X, pi)
        gap = float(g.max() / d - 1.0)
        if gap <= cfg.fw_tol:
            break
        pi = pi * (g / d)
        pi /= pi.sum()

    # Drop negligible weights, but only if doing so does not degrade the design;
    # the mixed design adds gamma / K to every arm afterwards regardless.
    if cfg.prune > 0:
        pruned = np.where(pi < cfg.prune, 0.0, pi)
        if pruned.sum() > 0 and np.linalg.matrix_rank(X[:, pruned > 0]) == d:
            pruned = pruned / pruned.sum()
            if float(prediction_variances(X, pruned).max()) / d - 1.0 <= max(gap, cfg.fw_tol):
                pi = pruned

    g = prediction_variances(X, pi)
    info = {
        "value": float(g.max()),
        "gap": float(g.max() / d - 1.0),
        "n_iter": n_iter,
        "support": int(np.count_nonzero(pi)),
    }
    return pi, info


# --------------------------------------------------------------------------- #
# 2. mixed design
# --------------------------------------------------------------------------- #

def mixed_design(pi_G: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """pi = (1 - gamma) pi^G + gamma * (1/K) 1_K.

    The draft takes gamma = 1/2.  Mixing is not cosmetic: an arm with pi_a = 0
    carries a latent offset that the samples of the other arms cannot identify,
    so its augmented feature would never enter the information matrix.
    """
    K = pi_G.shape[0]
    pi = (1.0 - gamma) * pi_G + gamma / K
    pi /= pi.sum()
    if not np.all(pi > 0):
        raise ValueError("mixed design produced a zero weight")
    return pi


# --------------------------------------------------------------------------- #
# 3. orthonormal basis of R(Xbar)^perp
# --------------------------------------------------------------------------- #

def aug_basis(
    X: np.ndarray,
    pi: np.ndarray,
    cfg: Optional[DesignConfig] = None,
) -> np.ndarray:
    """Orthonormal basis B of R(Xbar)^perp = null(Xbar), as a (K, K-d) matrix.

    Computed from the observed features and the design alone: no reward, no
    tuning parameter, and no estimate of du is formed at any point.  Any basis
    of the same subspace gives the same algorithm; we take the trailing right
    singular vectors of Xbar, which is deterministic given Xbar.
    """
    cfg = cfg or DesignConfig()
    d, K = X.shape
    Xbar = design_scale(X, pi)

    _, s, Vt = np.linalg.svd(Xbar, full_matrices=True)   # Vt is (K, K)
    rank = int(np.sum(s > s[0] * max(cfg.rank_rtol, np.finfo(float).eps * K)))
    if rank != d:
        raise ValueError(f"Xbar has numerical rank {rank}, expected d = {d}")

    B = Vt[d:, :].T                                       # (K, K - d)
    return np.ascontiguousarray(B)


# --------------------------------------------------------------------------- #
# 4. augmented features
# --------------------------------------------------------------------------- #

def aug_feat(X: np.ndarray, pi: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Xhat = [Xbar ; B^T] in R^{K x K}; column a is the augmented feature of arm a."""
    Xbar = design_scale(X, pi)
    return np.vstack([Xbar, B.T])


# --------------------------------------------------------------------------- #
# 5. oracle reparameterisation
# --------------------------------------------------------------------------- #

def reparam_mu(
    X: np.ndarray,
    U: np.ndarray,
    theta_o: np.ndarray,
    theta_u: np.ndarray,
    pi: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oracle values of (mu_par, mu_perp) defined in Eq. (8).

        mu_par  = theta_o + V(pi)^-1 Xbar Ubar^T theta_u          in R^d
        mu_perp = B^T (I - P_Xbar) Ubar^T theta_u = B^T Ubar^T theta_u   in R^{K-d}

    The second equality holds because the columns of B already lie in
    R(Xbar)^perp, so B^T P_Xbar = 0.  Evaluation only; agents never call this.

    Note that mu_par is also the population least-squares limit of an
    observed-features-only estimator under the design pi, so
    `argmax_a x_a^T mu_par` is the arm such an algorithm converges to.
    """
    Xbar = design_scale(X, pi)
    w = (U * np.sqrt(pi)[None, :]).T @ theta_u if U.size else np.zeros(X.shape[1])
    V = Xbar @ Xbar.T
    mu_par = theta_o + np.linalg.solve(V, Xbar @ w)
    mu_perp = B.T @ w
    return mu_par, mu_perp


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Design:
    """Everything derived from the observed features and computed once, before
    the first pull.  All fields are F_0-measurable and deterministic."""

    pi_G: np.ndarray        # (K,)   G-optimal design
    pi: np.ndarray          # (K,)   mixed design actually used
    Xbar: np.ndarray        # (d, K)
    B: np.ndarray           # (K, K-d)
    Xhat: np.ndarray        # (K, K) column a = xhat_a
    V: np.ndarray           # (d, d) = Xbar Xbar^T = V(pi)
    G: np.ndarray           # (K, K) = Xhat Xhat^T = blockdiag(V, I_{K-d})
    gopt_value: float       # max_a ||x_a||^2_{V(pi^G)^-1}, should equal d
    mixed_value: float      # max_a ||x_a||^2_{V(pi)^-1},  bounded by d / (1-gamma)
    info: dict

    @property
    def K(self) -> int:
        return self.Xhat.shape[1]

    @property
    def d(self) -> int:
        return self.Xbar.shape[0]

    def xhat(self, a: int) -> np.ndarray:
        return self.Xhat[:, a]

    def perp_block(self, a: int) -> np.ndarray:
        """B^T e_a, the augmented half of xhat_a."""
        return self.B[a, :]


def build_design(
    X: np.ndarray,
    cfg: Optional[DesignConfig] = None,
) -> Design:
    """Run the whole pre-round-one pipeline and, optionally, verify it."""
    cfg = cfg or DesignConfig()
    pi_G, info = gopt_design(X, cfg)
    pi = mixed_design(pi_G, cfg.gamma_mix)
    B = aug_basis(X, pi, cfg)
    Xhat = aug_feat(X, pi, B)
    Xbar = design_scale(X, pi)
    V = Xbar @ Xbar.T
    G = Xhat @ Xhat.T
    info = dict(info, gamma_mix=cfg.gamma_mix)

    design = Design(
        pi_G=pi_G,
        pi=pi,
        Xbar=Xbar,
        B=B,
        Xhat=Xhat,
        V=V,
        G=G,
        gopt_value=info["value"],
        mixed_value=float(prediction_variances(X, pi).max()),
        info=info,
    )
    if cfg.check:
        verify_design(X, design)
    return design


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify_design(X: np.ndarray, D: Design, atol: float = 1e-9) -> None:
    """Assert the algebraic identities the analysis relies on."""
    d, K = X.shape

    assert D.B.shape == (K, K - d), f"B has shape {D.B.shape}, expected {(K, K - d)}"
    assert np.allclose(D.B.T @ D.B, np.eye(K - d), atol=atol), "B is not orthonormal"
    assert np.allclose(D.Xbar @ D.B, 0.0, atol=atol), "Xbar B != 0"

    # Eq. (10): the Gram matrix over the whole action space is block diagonal,
    # the observed block attains the design information matrix and the
    # augmented block is the identity.
    G_ref = np.zeros((K, K))
    G_ref[:d, :d] = D.V
    G_ref[d:, d:] = np.eye(K - d)
    assert np.allclose(D.G, G_ref, atol=atol), "Xhat Xhat^T is not blockdiag(V, I)"

    # BB^T is the projection onto R(Xbar)^perp
    P = D.Xbar.T @ np.linalg.solve(D.V, D.Xbar)
    assert np.allclose(D.B @ D.B.T, np.eye(K) - P, atol=atol), "BB^T != I - P_Xbar"

    # Kiefer-Wolfowitz: the optimal value is exactly d.  Any feasible design has
    # max_a g_a >= d, so the solver approaches d from above.
    assert D.gopt_value >= d * (1 - 1e-8), f"G-optimal value {D.gopt_value} below d = {d}"
    assert D.gopt_value <= d * (1 + 1e-4), (
        f"G-optimal solver did not converge: value {D.gopt_value}, d = {d}"
    )

    # V(pi) >= (1 - gamma) V(pi^G) in the psd order, so mixing inflates the
    # worst-case prediction variance by at most 1 / (1 - gamma).
    gamma = D.info["gamma_mix"]
    assert D.mixed_value <= d / (1.0 - gamma) * (1 + 1e-4), (
        f"mixed design value {D.mixed_value} exceeds d / (1 - gamma) = {d / (1 - gamma)}"
    )


def verify_reparam(
    X: np.ndarray,
    U: np.ndarray,
    theta_o: np.ndarray,
    theta_u: np.ndarray,
    D: Design,
    atol: float = 1e-9,
) -> None:
    """Check xhat_a^T mu_star = sqrt(pi_a) * z_a^T theta_star for every arm."""
    mu_par, mu_perp = reparam_mu(X, U, theta_o, theta_u, D.pi, D.B)
    mu = np.concatenate([mu_par, mu_perp])
    means = X.T @ theta_o + (U.T @ theta_u if U.size else 0.0)
    lhs = D.Xhat.T @ mu
    rhs = np.sqrt(D.pi) * means
    assert np.allclose(lhs, rhs, atol=atol), "augmentation does not reproduce the rewards"


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for K, d, du in [(20, 8, 8), (12, 3, 5), (30, 10, 0), (20, 8, 8)]:
        X = rng.standard_normal((d, K))
        X /= np.linalg.norm(X, axis=0, keepdims=True)
        U = rng.standard_normal((du, K)) if du else np.zeros((0, K))
        if du:
            U /= np.linalg.norm(U, axis=0, keepdims=True)
        theta_o = rng.standard_normal(d)
        theta_u = rng.standard_normal(du)

        D = build_design(X)
        verify_reparam(X, U, theta_o, theta_u, D)
        print(
            f"K={K:3d} d={d:2d} du={du:2d} | "
            f"g-opt value {D.gopt_value:.6f} (=d {d}) support {D.info['support']:2d} "
            f"iters {D.info['n_iter']:5d} | mixed value {D.mixed_value:.4f} "
            f"| min pi {D.pi.min():.4f}"
        )

    # duplicate-column instance (scenario S3): design must still be well posed
    X = rng.standard_normal((8, 20))
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    X[:, 5] = X[:, 3]
    D = build_design(X)
    sep = np.linalg.norm(D.B[3] - D.B[5])
    print(f"duplicate pair (3,5): ||B^T e_3 - B^T e_5|| = {sep:.4f}  (must be > 0)")
    assert sep > 1e-6
    print("design.py self-test passed")


if __name__ == "__main__":
    _selftest()