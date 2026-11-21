"""Slippery CliffWalking: explicit MDP tables, DP solvers and a Gymnasium environment.

Layout (4 x 12, state = row * 12 + col), same as Gymnasium's CliffWalking-v1 (ENV_ID below):
  start = 36 (bottom-left), goal = 47 (bottom-right), cliff = 37..46.
Actions follow Gymnasium: 0 up, 1 right, 2 down, 3 left.

Slip model (our own wrapper; Gymnasium's built-in slip is fixed at 1/3 per direction):
  intended move with probability 1 - slip, each perpendicular move with probability slip/2.
  slip = 2/3 reproduces Gymnasium's is_slippery=True; slip = 0 is deterministic.
Rewards: -1 per step, -100 for stepping into the cliff (agent is returned to the start and
the episode CONTINUES), episode TERMINATES only on reaching the goal. Hitting a step cap is
TRUNCATION, not termination (the target must still bootstrap).
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_ID = "CliffWalking-v1"  # Gymnasium environment whose 4 x 12 layout, rewards and cliff rules are reproduced here
ROWS, COLS = 4, 12
N_STATES, N_ACTIONS = ROWS * COLS, 4
START, GOAL = 36, 47
CLIFF = frozenset(range(37, 47))
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left


def _move(s: int, a: int):
    """Deterministic move -> (next_state, reward, terminated, fell)."""
    if s == GOAL:
        return GOAL, 0.0, True, False
    r, c = divmod(s, COLS)
    dr, dc = MOVES[a]
    r2 = min(max(r + dr, 0), ROWS - 1)
    c2 = min(max(c + dc, 0), COLS - 1)
    s2 = r2 * COLS + c2
    if s2 in CLIFF:
        return START, -100.0, False, True
    return s2, -1.0, s2 == GOAL, False


@dataclass(frozen=True, eq=False)
class Tables:
    slip: float
    p: np.ndarray      # (S, A, 3) outcome probabilities: [intended, left-of, right-of]
    ns: np.ndarray     # (S, A, 3) next state
    r: np.ndarray      # (S, A, 3) reward
    term: np.ndarray   # (S, A, 3) terminated flag
    fell: np.ndarray   # (S, A, 3) cliff-fall flag
    T: np.ndarray      # (S, A, S) non-terminal transition probabilities (dense)
    R: np.ndarray      # (S, A) expected immediate reward


def build_tables(slip: float) -> Tables:
    if not 0.0 <= slip <= 1.0:
        raise ValueError("slip must be in [0, 1]")
    S, A = N_STATES, N_ACTIONS
    p = np.zeros((S, A, 3)); ns = np.zeros((S, A, 3), dtype=np.int64)
    r = np.zeros((S, A, 3)); term = np.zeros((S, A, 3), dtype=bool)
    fell = np.zeros((S, A, 3), dtype=bool)
    probs = (1.0 - slip, slip / 2.0, slip / 2.0)
    for s in range(S):
        for a in range(A):
            for k, act in enumerate((a, (a - 1) % 4, (a + 1) % 4)):
                n, rew, t, f = _move(s, act)
                p[s, a, k], ns[s, a, k], r[s, a, k], term[s, a, k], fell[s, a, k] = probs[k], n, rew, t, f
    T = np.zeros((S, A, S)); R = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            for k in range(3):
                R[s, a] += p[s, a, k] * r[s, a, k]
                if not term[s, a, k]:
                    T[s, a, ns[s, a, k]] += p[s, a, k]
    return Tables(slip, p, ns, r, term, fell, T, R)


# ----------------------------------------------------------------- dynamic programming
def value_iteration(tb: Tables, gamma: float = 1.0, tol: float = 1e-10, max_iter: int = 200_000):
    """Optimal values V*, Q* (undiscounted by default; terminal value 0)."""
    V = np.zeros(N_STATES)
    for _ in range(max_iter):
        Q = tb.R + gamma * (tb.T @ V)
        Vn = Q.max(1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    else:
        raise RuntimeError("value iteration did not converge")
    return V, tb.R + gamma * (tb.T @ V)


def eps_soft_optimal(tb: Tables, eps: float, gamma: float = 1.0, tol: float = 1e-10, max_iter: int = 200_000):
    """Best achievable value among eps-soft policies (every action has probability >= eps/|A|)."""
    V = np.zeros(N_STATES)
    for _ in range(max_iter):
        Q = tb.R + gamma * (tb.T @ V)
        Vn = (1 - eps) * Q.max(1) + eps * Q.mean(1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    else:
        raise RuntimeError("eps-soft value iteration did not converge")
    return V


def eps_greedy_policy(Q: np.ndarray, eps: float) -> np.ndarray:
    """(S, A) action probabilities of the eps-greedy policy, ties split uniformly."""
    best = (Q >= Q.max(1, keepdims=True) - 1e-12).astype(float)
    best /= best.sum(1, keepdims=True)
    return eps / Q.shape[1] + (1 - eps) * best


def policy_value_exact(tb: Tables, pi: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Exact value of a stochastic policy (linear solve). Returns NaNs if the policy is improper."""
    P = np.einsum("sa,sat->st", pi, tb.T)
    r = (pi * tb.R).sum(1)
    try:
        return np.linalg.solve(np.eye(N_STATES) - gamma * P, r)
    except np.linalg.LinAlgError:
        return np.full(N_STATES, np.nan)


def policy_value_capped(tb: Tables, pi: np.ndarray, horizon: int, gamma: float = 1.0) -> np.ndarray:
    """Expected return over at most `horizon` steps (matches an episode cap with truncation)."""
    P = np.einsum("sa,sat->st", pi, tb.T)
    r = (pi * tb.R).sum(1)
    V = np.zeros(N_STATES)
    for _ in range(horizon):
        V = r + gamma * (P @ V)
    return V


def references(slip: float, eps: float, gamma: float = 1.0) -> dict:
    """DP reference values at the start state, plus the van Seijen Type 1 check."""
    tb = build_tables(slip)
    V, Q = value_iteration(tb, gamma)
    Ve = eps_soft_optimal(tb, eps, gamma)
    Vg = policy_value_exact(tb, eps_greedy_policy(Q, eps), gamma)
    return {"slip": slip, "v_star": float(V[START]), "v_eps_soft_opt": float(Ve[START]),
            "v_eps_greedy_of_qstar": float(Vg[START]),
            "type1_margin": float(Ve[START] - Vg[START])}


# ------------------------------------------------------------------- gymnasium env
class SlipCliffWalkingEnv(gym.Env):
    """Gymnasium-compatible slippery CliffWalking (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, slip: float = 0.0, max_steps: int = 500, tables: Tables | None = None):
        self.tables = tables if tables is not None else build_tables(slip)
        self.slip, self.max_steps = self.tables.slip, max_steps
        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)
        t = self.tables
        self._ns, self._r = t.ns.tolist(), t.r.tolist()
        self._term, self._fell = t.term.tolist(), t.fell.tolist()
        self._c1, self._c2 = 1.0 - self.slip, 1.0 - self.slip / 2.0
        self._buf, self._i = [], 0
        self.s, self.t, self.last_fell = START, 0, False

    def _u(self) -> float:
        if self._i >= len(self._buf):
            self._buf, self._i = self.np_random.random(4096).tolist(), 0
        u = self._buf[self._i]
        self._i += 1
        return u

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._buf, self._i = [], 0
        self.s, self.t, self.last_fell = START, 0, False
        return START, {}

    def step(self, action):
        u = self._u()
        k = 0 if u < self._c1 else (1 if u < self._c2 else 2)
        s, a = self.s, int(action)
        s2, r, term, fell = self._ns[s][a][k], self._r[s][a][k], self._term[s][a][k], self._fell[s][a][k]
        self.s, self.last_fell = s2, fell
        self.t += 1
        truncated = (not term) and self.t >= self.max_steps
        return s2, r, term, truncated, {"fell": fell}
