"""Single (instance, agent) execution loop, shared by benchmark.py and ablation.py.

Runner contract (agents/base.py):
    while True:
        a = agent.sample()
        if a is None: break
        agent.update(a, instance.pull(a))
        if agent.t >= T_max: agent._stop(CENSORED); break
    psi = agent.recommend()

`on_step(agent, instance)` fires once per pull; ablation.py uses it to record
checkpoints, benchmark.py can leave it unset.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import Agent, StopReason  # noqa: E402
from config import RunConfig  # noqa: E402
from env import Instance  # noqa: E402


@dataclass
class RunResult:
    psi: int
    a_star: int
    correct: bool
    T_delta: int
    censored: bool
    reason: str
    n_pulls: np.ndarray
    predicted_means: np.ndarray
    components: dict
    trace: dict


def run_single(
    instance: Instance,
    agent: Agent,
    run_cfg: RunConfig,
    noise_seed: Optional[int] = None,
    on_step: Optional[Callable[[Agent, Instance], None]] = None,
) -> RunResult:
    """Run `agent` to completion on `instance` and score the recommendation.

    Resets the instance's noise stream first, so that two agents given the
    same `noise_seed` see the identical (eta_1, eta_2, ...) sequence.
    """
    instance.reset_noise(instance.seed if noise_seed is None else noise_seed)

    while True:
        a = agent.sample()
        if a is None:
            break
        agent.update(a, instance.pull(a))
        if on_step is not None:
            on_step(agent, instance)
        if not agent.stopped and agent.t >= run_cfg.T_max:
            agent._stop(StopReason.CENSORED)
            break

    psi = agent.recommend()
    return RunResult(
        psi=psi,
        a_star=instance.a_star,
        correct=psi == instance.a_star,
        T_delta=agent.t,
        censored=agent.reason is StopReason.CENSORED,
        reason=agent.reason.value,
        n_pulls=agent.n_pulls,
        predicted_means=agent.predicted_means(),
        components=agent.components(),
        trace=agent.trace(),
    )


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    from config import InstanceConfig, Scenario
    from env import make_instance

    class _RoundRobinAgent(Agent):
        """Deterministic round-robin agent, used only to exercise the loop."""

        name = "rr"

        def __init__(self, X, delta, sigma, rng=None, horizon=None):
            super().__init__(X, delta, sigma, rng)
            self.horizon = horizon
            self.total = np.zeros(self.K)

        def sample(self):
            if self.stopped:
                return None
            if self.horizon is not None and self.t >= self.horizon:
                self._stop()
                return None
            return self.t % self.K

        def update(self, a, reward):
            self._record_pull(a)
            self.total[a] += reward

        def predicted_means(self):
            return self.total / np.maximum(self.n_pulls, 1)

        def recommend(self):
            return int(np.argmax(self.predicted_means()))

    cfg = InstanceConfig(K=10, d=4, scenario=Scenario.PARTIAL, sigma=0.1, min_gap=0.10)
    inst = make_instance(cfg, seed=1)
    rng = np.random.default_rng(0)

    # stops on its own before T_max: reason is STOPPED, T_delta == horizon
    run_cfg = RunConfig(T_max=1000)
    agent = _RoundRobinAgent(inst.observed(), delta=0.05, sigma=0.1, rng=rng, horizon=30)
    res = run_single(inst, agent, run_cfg, noise_seed=42)
    assert res.T_delta == 30 and res.reason == "stopped" and not res.censored
    assert res.n_pulls.sum() == 30
    assert res.a_star == inst.a_star

    # never stops on its own: runner must censor at T_max
    agent2 = _RoundRobinAgent(inst.observed(), delta=0.05, sigma=0.1, rng=rng, horizon=None)
    run_cfg_small = RunConfig(T_max=25)
    res2 = run_single(inst, agent2, run_cfg_small, noise_seed=42)
    assert res2.T_delta == 25 and res2.censored and res2.reason == "censored"

    # noise pairing: same seed, same deterministic policy -> bit-identical outcome
    agent3 = _RoundRobinAgent(inst.observed(), delta=0.05, sigma=0.1, rng=rng, horizon=30)
    res3 = run_single(inst, agent3, run_cfg, noise_seed=42)
    assert np.allclose(res.predicted_means, res3.predicted_means)
    assert np.array_equal(res.n_pulls, res3.n_pulls)

    # on_step fires exactly once per pull
    counter = {"n": 0}
    agent4 = _RoundRobinAgent(inst.observed(), delta=0.05, sigma=0.1, rng=rng, horizon=17)
    res4 = run_single(inst, agent4, run_cfg, noise_seed=7, on_step=lambda ag, ins: counter.__setitem__("n", counter["n"] + 1))
    assert counter["n"] == res4.T_delta == 17

    print("expt/runner.py self-test passed")


if __name__ == "__main__":
    _selftest()
