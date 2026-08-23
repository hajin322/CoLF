"""The reward-generating environment.

`Instance` holds the full problem (X, U, theta) and hands out rewards.  Agents
receive `instance.observed()` and nothing else, so the information barrier of
the setting is enforced by the code rather than by convention.

Pairing.  The draft's Eq. (2) attaches a single noise variable eta_t to round t
rather than one per arm, so an instance draws exactly one normal per pull.  Two
agents run on the same instance with the same noise seed therefore see the
identical noise sequence (eta_1, eta_2, ...) as a function of the round index,
which removes most of the between-agent variance in the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

import design as dg
from config import DesignConfig, InstanceConfig, Scenario


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
    twin: Optional[Tuple[int, int]] = None   # (a_star, a_prime) in scenario S3

    _noise_rng: np.random.Generator = field(default=None, repr=False, compare=False)
    _n_pulls: np.ndarray = field(default=None, repr=False, compare=False)
    _design: Optional[Tuple[DesignConfig, "dg.Design"]] = field(
        default=None, repr=False, compare=False
    )

    # ---------------------------------------------------------------- basics --

    def __post_init__(self) -> None:
        if self._noise_rng is None:
            self.reset_noise(self.seed)
        if self._n_pulls is None:
            self._n_pulls = np.zeros(self.K, dtype=np.int64)

    @property
    def K(self) -> int:
        return self.cfg.K

    @property
    def d(self) -> int:
        return self.cfg.d

    @property
    def a_star(self) -> int:
        return int(np.argmax(self.means))

    @property
    def gaps(self) -> np.ndarray:
        """Delta_a = m_1 - m_a, zero at the optimal arm."""
        return self.means.max() - self.means

    @property
    def gap_min(self) -> float:
        """Delta_2, the gap to the second best arm."""
        g = np.sort(self.gaps)
        return float(g[1])

    def observed(self) -> np.ndarray:
        """The only thing a learner is allowed to see (besides d, K)."""
        return self.X.copy()

    def design(self, dcfg: Optional[DesignConfig] = None) -> "dg.Design":
        """The design and augmentation derived from X alone, built once.

        Both agents and the evaluation code share this object.  It is a
        deterministic function of X, so caching changes nothing except runtime.
        """
        dcfg = dcfg or DesignConfig()
        if self._design is None or self._design[0] != dcfg:
            self._design = (dcfg, dg.build_design(self.X, dcfg))
        return self._design[1]

    # ----------------------------------------------------------------- pulls --

    def reset_noise(self, seed: Optional[int] = None) -> None:
        self._noise_rng = np.random.default_rng(self.seed if seed is None else seed)
        self._n_pulls = np.zeros(self.K, dtype=np.int64)

    def pull(self, a: int) -> float:
        """Draw y_{a,t} = z_a^T theta_star + eta_t.

        Exactly one normal variate is consumed per call, which is what makes the
        noise stream comparable across agents.
        """
        eta = float(self._noise_rng.standard_normal()) * self.cfg.sigma
        self._n_pulls[a] += 1
        return float(self.means[a]) + eta

    @property
    def n_pulls(self) -> np.ndarray:
        return self._n_pulls.copy()

    # --------------------------------------------------------------- oracles --

    def oracle(self, D: "dg.Design") -> dict:
        """Oracle quantities that depend on the design, for evaluation only.

        `mu_par` doubles as the population least-squares limit of an
        observed-features-only estimator under the design pi, so `obs_best` is
        the arm such an algorithm converges to and `reversal` flags the
        instances where partial observability actually flips the ranking.
        """
        mu_par, mu_perp = dg.reparam_mu(
            self.X, self.U, self.theta_o, self.theta_u, D.pi, D.B
        )
        obs_scores = self.X.T @ mu_par
        perp_per_arm = D.B @ mu_perp          # entry a is (B^T e_a)^T mu_perp

        # An observed-features-only rule cannot separate arms whose observed
        # scores coincide, which is exactly what scenario S3 constructs.  The
        # best it can do is break the tie at random, so record the tie set and
        # the resulting ceiling on its probability of returning a_star.
        tol = 1e-9 * max(1.0, float(np.abs(obs_scores).max()))
        tie_set = np.flatnonzero(obs_scores >= obs_scores.max() - tol)
        obs_correct_prob = (
            1.0 / tie_set.size if self.a_star in tie_set else 0.0
        )
        return {
            "mu_par": mu_par,
            "mu_perp": mu_perp,
            "mu_star": np.concatenate([mu_par, mu_perp]),
            "perp_per_arm": perp_per_arm,
            "perp_signal": float(np.abs(perp_per_arm).max()),
            "obs_best": int(np.argmax(obs_scores)),
            "obs_tie_size": int(tie_set.size),
            "obs_correct_prob": float(obs_correct_prob),
            "reversal": bool(obs_correct_prob < 1.0),
            "obs_gap": float(obs_scores.max() - obs_scores[self.a_star]),
        }

    def summary(self) -> dict:
        return {
            "scenario": self.cfg.scenario.value,
            "seed": self.seed,
            "K": self.K,
            "d": self.d,
            "du": self.cfg.du,
            "a_star": self.a_star,
            "gap_min": self.gap_min,
            "gap_max": float(self.gaps.max()),
            "mean_max": float(self.means.max()),
            "twin": self.twin,
        }


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def _unit_columns(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=0, keepdims=True)
    n[n == 0.0] = 1.0
    return A / n


def make_instance(
    cfg: InstanceConfig,
    seed: int,
    dcfg: Optional[DesignConfig] = None,
) -> Instance:
    """Draw one instance of the requested scenario.

    Common to all scenarios:
      * columns of X (and of U) are drawn iid standard normal and normalised to
        unit L2 norm; U is then scaled by `latent_scale`;
      * theta is rescaled so that max_a |z_a^T theta| = reward_bound, matching
        the bounded-mean-reward convention of the draft;
      * instances whose gap to the second best arm falls below `min_gap` are
        rejected, since they only inflate the stopping time without changing
        what the experiment shows.

    Scenario S3 additionally performs the twin surgery, and is drawn
    conditional on the twin pair attaining the maximum observed score; see
    `_twin_surgery`.  Rescaling theta by a positive scalar changes neither the
    ranking nor the tie structure, so the conditioning may be checked before
    the rescaling step.
    """
    dcfg = dcfg or DesignConfig()
    rng = np.random.default_rng(seed)
    K, d, du = cfg.K, cfg.d, cfg.du

    for _ in range(cfg.max_resample):
        X = _unit_columns(rng.standard_normal((d, K)))
        if np.linalg.matrix_rank(X) < d:
            continue

        if du:
            U = _unit_columns(rng.standard_normal((du, K))) * cfg.latent_scale
        else:
            U = np.zeros((0, K))

        theta_o = rng.standard_normal(d)
        theta_u = rng.standard_normal(du) if du else np.zeros(0)

        twin = None
        if cfg.scenario is Scenario.UNIDENT:
            out = _twin_surgery(X, U, theta_o, theta_u, dcfg)
            if out is None:
                continue
            X, U, twin = out

        means = X.T @ theta_o + (U.T @ theta_u if du else 0.0)

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
            twin=twin,
        )

    raise RuntimeError(
        f"failed to draw a {cfg.scenario.value} instance with gap in "
        f"[{cfg.min_gap}, {cfg.max_gap}] after {cfg.max_resample} attempts"
    )


def _observed_scores(
    X: np.ndarray,
    U: np.ndarray,
    theta_o: np.ndarray,
    theta_u: np.ndarray,
    dcfg: DesignConfig,
) -> np.ndarray:
    """x_a^T mu_par: the limit an observed-features-only estimator converges to."""
    D = dg.build_design(X, DesignConfig(**{**dcfg.__dict__, "check": False}))
    mu_par, _ = dg.reparam_mu(X, U, theta_o, theta_u, D.pi, D.B)
    return X.T @ mu_par


def _twin_surgery(
    X: np.ndarray,
    U: np.ndarray,
    theta_o: np.ndarray,
    theta_u: np.ndarray,
    dcfg: DesignConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
    """Make one pair of arms observationally identical but latently opposite.

    Let a_star be the optimal arm and a_prime the highest-scoring arm under the
    observed-only limit mu_par (excluding a_star).  Copy x_{a_star} onto
    a_prime and set u_{a'} = -u_{a*}, so that

        m_{a*} = x_{a*}^T theta_o + u_{a*}^T theta_u,
        m_{a'} = x_{a*}^T theta_o - u_{a*}^T theta_u.

    Since m_{a*} is untouched and was the global maximum, the optimal arm stays
    inside the pair by construction.  Overwriting the observed-best arm rather
    than an arbitrary one is what makes the pair likely to top the observed
    ranking afterwards, which is the condition we then enforce by rejection:
    the point of S3 is a pair the learner ranks first but cannot split, not an
    ordinary ranking reversal, which S2 already covers.

    Any policy mapping observed features to selection probabilities assigns the
    same probability to both arms (Definition 1 of Kim et al. 2025), so it can
    never separate them.  The augmentation can: the design treats the duplicated
    columns symmetrically, so pi_{a*} = pi_{a'} and Xbar (e_{a*} - e_{a'}) = 0,
    which places the entire difference direction inside R(Xbar)^perp.

    Returns None if the drawn instance fails the conditioning.
    """
    means = X.T @ theta_o + (U.T @ theta_u if U.size else 0.0)
    a_star = int(np.argmax(means))

    s = _observed_scores(X, U, theta_o, theta_u, dcfg)
    order = np.argsort(-s)
    a_prime = int(order[0]) if order[0] != a_star else int(order[1])

    X = X.copy()
    U = U.copy()
    X[:, a_prime] = X[:, a_star]
    U[:, a_prime] = -U[:, a_star]
    if np.linalg.matrix_rank(X) < X.shape[0]:
        return None

    # condition: the pair, and only the pair, attains the maximum observed score
    s2 = _observed_scores(X, U, theta_o, theta_u, dcfg)
    tol = 1e-9 * max(1.0, float(np.abs(s2).max()))
    tie = set(np.flatnonzero(s2 >= s2.max() - tol).tolist())
    if tie != {a_star, a_prime}:
        return None

    # The sign flip u_{a'} = -u_{a*} makes a_prime the better of the two whenever
    # u_{a*}^T theta_u < 0.  The optimum is still inside the pair either way; we
    # just order the tuple so that twin[0] is the optimal arm.
    means = X.T @ theta_o + U.T @ theta_u
    pair = (a_star, a_prime) if means[a_star] >= means[a_prime] else (a_prime, a_star)
    return X, U, pair


def make_batch(
    cfg: InstanceConfig,
    n_reps: int,
    seed0: int,
    dcfg: Optional[DesignConfig] = None,
) -> list[Instance]:
    """`n_reps` instances with seeds seed0, seed0 + 1, ...  Paired across agents."""
    return [make_instance(cfg, seed0 + r, dcfg) for r in range(n_reps)]


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    dcfg = DesignConfig()
    for scenario in (Scenario.FULL, Scenario.HALF, Scenario.UNIDENT):
        cfg = InstanceConfig(K=20, d=8, scenario=scenario, sigma=0.1, min_gap=0.10)
        insts = make_batch(cfg, n_reps=50, seed0=20260823)

        rev, sig, gaps, obsgap, ceil, ties = [], [], [], [], [], []
        for ins in insts:
            D = ins.design(dcfg)
            dg.verify_reparam(ins.X, ins.U, ins.theta_o, ins.theta_u, D)
            o = ins.oracle(D)
            rev.append(o["reversal"])
            sig.append(o["perp_signal"])
            gaps.append(ins.gap_min)
            obsgap.append(o["obs_gap"])
            ceil.append(o["obs_correct_prob"])
            ties.append(o["obs_tie_size"])
            if scenario is Scenario.UNIDENT:
                i, j = ins.twin
                assert np.allclose(ins.X[:, i], ins.X[:, j]), "twin arms differ in X"
                assert ins.a_star == i, "twin surgery lost the optimal arm"
                assert o["obs_tie_size"] == 2 and o["obs_correct_prob"] == 0.5, (
                    "S3 conditioning failed: the twins do not top the observed ranking"
                )
                assert abs(D.pi[i] - D.pi[j]) < 1e-9, "design is asymmetric in the twins"
                # the difference direction sits entirely in the augmented block
                diff = D.Xhat[:, i] - D.Xhat[:, j]
                assert np.allclose(diff[: ins.d], 0.0, atol=1e-9), (
                    "twin arms are separable in the observed block"
                )
                assert np.linalg.norm(diff) > 1e-6

        print(
            f"{scenario.value} | reversal {np.mean(rev):5.1%} | "
            f"obs-only accuracy ceiling {np.mean(ceil):5.1%} | "
            f"tie size {np.mean(ties):.2f} | perp signal {np.mean(sig):.4f} | "
            f"Delta_2 {np.mean(gaps):.4f} | obs-gap {np.mean(obsgap):.4f}"
        )

    # noise pairing: two runs with the same seed see the same eta sequence
    ins = make_instance(InstanceConfig(scenario=Scenario.HALF), 7)
    ins.reset_noise(123)
    r1 = [ins.pull(t % ins.K) - ins.means[t % ins.K] for t in range(50)]
    ins.reset_noise(123)
    r2 = [ins.pull((t * 3) % ins.K) - ins.means[(t * 3) % ins.K] for t in range(50)]
    assert np.allclose(r1, r2), "noise stream is not reproducible across pull orders"
    print("env.py self-test passed")


if __name__ == "__main__":
    _selftest()