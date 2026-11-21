"""Tabular TD control agents (Q-learning, SARSA, Expected SARSA) and a random baseline.

Equal budgets: every run gets exactly `train_steps` environment steps. Episodes are capped at
`episode_cap` steps; hitting the cap is TRUNCATION, so the TD target still bootstraps from the
next state (only reaching the goal is terminal). Behaviour policy: epsilon-greedy, fixed epsilon,
uniform tie-breaking. Evaluation: every `eval_every` steps, greedy (epsilon = 0) on a separate
environment instance with its own random stream, plus the exact expected capped return of the
greedy policy computed from the MDP tables.
"""
from __future__ import annotations

import random
import sys
from functools import lru_cache

import numpy as np

from cliff_mdp import (GOAL, N_ACTIONS, N_STATES, START, SlipCliffWalkingEnv, build_tables,
                       policy_value_capped, references)

ALGOS = ("qlearning", "sarsa", "expected_sarsa")
ALL_ALGOS = ALGOS + ("random",)
_ALGO_ID = {a: i for i, a in enumerate(ALL_ALGOS)}


def td_target(algo, q_next, a_next, r, terminated, gamma, eps):
    """TD target. `terminated` is True only at the goal; truncation must NOT be passed here."""
    if terminated:
        return r
    if algo == "qlearning":
        return r + gamma * max(q_next)
    if algo == "sarsa":
        return r + gamma * q_next[a_next]
    if algo == "expected_sarsa":  # expectation under the epsilon-greedy policy (ties irrelevant)
        return r + gamma * ((1 - eps) * max(q_next) + eps * sum(q_next) / len(q_next))
    raise ValueError(algo)


class _Uniform:
    """Buffered U[0,1) stream from a NumPy Generator (fast scalar draws, reproducible)."""

    def __init__(self, rng):
        self.rng, self.buf, self.i = rng, [], 0

    def __call__(self):
        if self.i >= len(self.buf):
            self.buf, self.i = self.rng.random(4096).tolist(), 0
        u = self.buf[self.i]
        self.i += 1
        return u


def _greedy(q):
    return max(range(len(q)), key=q.__getitem__)  # first argmax


@lru_cache(maxsize=None)
def _tables(slip):
    return build_tables(slip)


@lru_cache(maxsize=None)
def _refs(slip, eps, gamma):
    return references(slip, eps, gamma)


def _seed(seed, algo, slip, role):
    ss = np.random.SeedSequence([int(seed), _ALGO_ID[algo], int(round(slip * 1000)), role])
    return int(ss.generate_state(1)[0])


def _seed_global_generators(seed_int):
    """Seed Python's `random`, NumPy's legacy global generator and PyTorch (only if it is imported).

    The agents and environments draw from their own seeded Generators, so these calls do not change
    any result; they make the global generators reproducible too, as the brief requires.
    """
    random.seed(seed_int)
    np.random.seed(seed_int % (2 ** 32))
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed_int)


def _exact_greedy_value(tb, Q, horizon, gamma):
    g = np.array([_greedy(q) for q in Q])
    pi = np.zeros((N_STATES, N_ACTIONS)); pi[np.arange(N_STATES), g] = 1.0
    return float(policy_value_capped(tb, pi, horizon, gamma)[START])


def run_agent(algo, slip, alpha, seed, cfg):
    """One training run -> dict of raw arrays and per-seed scores."""
    eps, gamma, T = cfg["epsilon"], cfg["gamma"], cfg["train_steps"]
    _seed_global_generators(_seed(seed, algo, slip, 0))
    cap, every, n_ev = cfg["episode_cap"], cfg["eval_every"], cfg["eval_episodes"]
    tb = _tables(slip)
    env = SlipCliffWalkingEnv(slip, cap, tables=tb)
    eval_env = SlipCliffWalkingEnv(slip, cap, tables=tb)          # separate instance
    ref = _refs(slip, eps, gamma)
    u = _Uniform(np.random.default_rng(_seed(seed, algo, slip, 2)))
    q0 = float(cfg.get("q_init", 0.0))
    Q = [[q0] * N_ACTIONS for _ in range(N_STATES)]
    random_exact = None
    if algo == "random":
        pi = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
        random_exact = float(policy_value_capped(tb, pi, cap, gamma)[START])

    def act(q):
        if algo == "random" or u() < eps:
            return int(u() * N_ACTIONS)
        m = max(q)
        best = [i for i in range(N_ACTIONS) if q[i] == m]
        return best[0] if len(best) == 1 else best[int(u() * len(best))]

    s, _ = env.reset(seed=_seed(seed, algo, slip, 1))
    eval_env.reset(seed=_seed(seed, algo, slip, 3))
    a = act(Q[s])
    ep_ret = ep_len = ep_falls = total_falls = 0
    ep_returns, ep_lens, ep_fall_counts, ep_end = [], [], [], []
    ev_steps, ev_exact, ev_sim = [], [], []
    for t in range(1, T + 1):
        s2, r, term, trunc, info = env.step(a)
        a2 = None
        if algo != "random":
            if algo == "sarsa" and not term:
                a2 = act(Q[s2])
            target = td_target(algo, Q[s2], a2, r, term, gamma, eps)
            Q[s][a] += alpha * (target - Q[s][a])
        ep_ret += r; ep_len += 1
        if info["fell"]:
            ep_falls += 1; total_falls += 1
        if term or trunc:
            ep_returns.append(ep_ret); ep_lens.append(ep_len); ep_fall_counts.append(ep_falls); ep_end.append(t)
            ep_ret = ep_len = ep_falls = 0
            s, _ = env.reset()
            a = act(Q[s])
        else:
            s = s2
            a = a2 if (algo == "sarsa") else act(Q[s2])
        if t % every == 0:
            ev_steps.append(t)
            if algo == "random":
                ev_exact.append(random_exact); ev_sim.append(np.nan)
            else:
                ev_exact.append(_exact_greedy_value(tb, Q, cap, gamma))
                tot = 0.0
                for _ in range(n_ev):
                    es, _ = eval_env.reset(); done = False
                    while not done:
                        es, er, et, etr, _ = eval_env.step(_greedy(Q[es]))
                        tot += er; done = et or etr
                ev_sim.append(tot / n_ev)

    ev_exact_a = np.array(ev_exact)
    v_star = ref["v_star"]
    thr = v_star - cfg["threshold_fraction"] * abs(v_star)          # "within 10% of optimal"
    hit = np.flatnonzero(ev_exact_a >= thr)
    scores = {
        "online_return": float(np.mean(ep_returns)) if ep_returns else np.nan,
        "fall_rate_per_1k": 1000.0 * total_falls / T,
        "final_eval_exact": float(ev_exact_a[-1]),
        "final_eval_sim": float(ev_sim[-1]),
        "gap_vs_vstar": float(v_star - ev_exact_a[-1]),
        "auc_eval": float(ev_exact_a.mean()),
        "steps_to_threshold": float(ev_steps[hit[0]]) if len(hit) else float("nan"),
        "censored": bool(len(hit) == 0),
        "n_episodes": len(ep_returns),
        "total_falls": total_falls,
    }
    return {
        "algo": algo, "slip": slip, "alpha": alpha, "seed": seed, "scores": scores,
        "ep_returns": np.array(ep_returns, dtype=np.float32), "ep_lens": np.array(ep_lens, dtype=np.int32),
        "ep_falls": np.array(ep_fall_counts, dtype=np.int32), "ep_end_step": np.array(ep_end, dtype=np.int32),
        "eval_steps": np.array(ev_steps), "eval_exact": ev_exact_a, "eval_sim": np.array(ev_sim),
        "q_final": np.array(Q, dtype=np.float32),
    }
