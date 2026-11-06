#!/usr/bin/env python3
"""Train the cart-pole swing-up policy shown in the "Swing up" demo, using numpy only.

Method: OpenAI-style evolution strategies (Salimans et al., 2017) with antithetic
sampling, centred-rank fitness shaping and Adam, on a small tanh MLP.
The dynamics, integrator, observation scaling and wall model below are mirrored
exactly in assets/js/demos/cartpole.js; change both or neither.

Outputs assets/data/cartpole-policy.json: weights, training metadata, the measured
swing-up success rate from the down position over N seeded trials, a push-recovery
rate, and the same numbers for the classical (energy swing-up + LQR) controller.

    python3 tools/train_pendulum.py              # train + evaluate + write JSON (~6 min, 1 core)
    python3 tools/train_pendulum.py --eval-only  # re-evaluate the shipped JSON
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---- plant (SI units). theta = 0 is upright; positive theta leans the pole toward +x. ----
G = 9.81            # m/s^2
M_CART = 1.0        # kg
M_POLE = 0.3        # kg
L_HALF = 0.5        # m, pivot to pole centre of mass (pole is 1 m long)
F_MAX = 20.0        # N, actuator limit
X_MAX = 2.0         # m, track half-length; the cart stops dead at a wall
DT = 0.02           # s, control period (50 Hz), RK4 with zero-order-hold force
EPISODE_S = 8.0
OBS_SCALE = np.array([1 / X_MAX, 1 / 3.0, 1.0, 1.0, 1 / 8.0])

# ---- success criterion ----
UPRIGHT_RAD = np.deg2rad(12.0)
HOLD_S = 3.0          # must stay upright (and off the walls) for the last HOLD_S of the trial
EVAL_S = 10.0
SWINGUP_LIMIT_S = EVAL_S - HOLD_S

HIDDEN = (16, 16)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "data" / "cartpole-policy.json"


def deriv(s: np.ndarray, f: np.ndarray) -> np.ndarray:
    """State derivative for s[..., (x, xd, th, thd)] under cart force f[...] (N)."""
    xd, th, thd = s[..., 1], s[..., 2], s[..., 3]
    sin, cos = np.sin(th), np.cos(th)
    mt = M_CART + M_POLE
    tmp = (f + M_POLE * L_HALF * thd * thd * sin) / mt
    thdd = (G * sin - cos * tmp) / (L_HALF * (4.0 / 3.0 - M_POLE * cos * cos / mt))
    xdd = tmp - M_POLE * L_HALF * thdd * cos / mt
    return np.stack([xd, xdd, thd, thdd], axis=-1)


def step(s: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 step; returns (new state, hit_wall mask). Walls are inelastic stops."""
    k1 = deriv(s, f)
    k2 = deriv(s + 0.5 * DT * k1, f)
    k3 = deriv(s + 0.5 * DT * k2, f)
    k4 = deriv(s + DT * k3, f)
    n = s + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    hit = np.abs(n[..., 0]) > X_MAX
    n[..., 0] = np.clip(n[..., 0], -X_MAX, X_MAX)
    n[..., 1] = np.where(hit, 0.0, n[..., 1])
    return n, hit


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def observe(s: np.ndarray) -> np.ndarray:
    o = np.stack([s[..., 0], s[..., 1], np.cos(s[..., 2]), np.sin(s[..., 2]), s[..., 3]], axis=-1)
    return o * OBS_SCALE


# ---- policy: obs(5) -> tanh(16) -> tanh(16) -> tanh(1) * F_MAX ----
def shapes() -> list[tuple[int, ...]]:
    dims = (5, *HIDDEN, 1)
    out = []
    for a, b in zip(dims[:-1], dims[1:]):
        out += [(a, b), (b,)]
    return out


N_PARAMS = int(sum(np.prod(s) for s in shapes()))


def unflatten(theta: np.ndarray) -> list[np.ndarray]:
    """theta[..., N_PARAMS] -> list of arrays with leading batch dims."""
    out, i = [], 0
    lead = theta.shape[:-1]
    for s in shapes():
        n = int(np.prod(s))
        out.append(theta[..., i:i + n].reshape(*lead, *s))
        i += n
    return out


def policy(params: list[np.ndarray], obs: np.ndarray) -> np.ndarray:
    """params with leading dim P, obs [P, E, 5] -> force [P, E]."""
    h = obs
    for li in range(0, len(params), 2):
        w, b = params[li], params[li + 1]
        h = np.tanh(np.einsum("pei,pij->pej", h, w) + b[:, None, :])
    return h[..., 0] * F_MAX


def init_states(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mostly hanging-down starts, some anywhere (teaches balancing and recovery)."""
    s = np.zeros((n, 4))
    down = rng.random(n) < 0.7
    s[:, 0] = rng.uniform(-0.6, 0.6, n)
    s[:, 1] = rng.normal(0, 0.2, n)
    s[:, 2] = np.where(down, np.pi + rng.normal(0, 0.25, n), rng.uniform(-np.pi, np.pi, n))
    s[:, 3] = rng.normal(0, 0.5, n)
    return s


def rollout(theta: np.ndarray, starts: np.ndarray, kicks: np.ndarray) -> np.ndarray:
    """Return total reward per member. theta [P, N], starts [E, 4], kicks [T, E] (rad/s)."""
    p = theta.shape[0]
    params = unflatten(theta)
    s = np.broadcast_to(starts, (p, *starts.shape)).copy()
    total = np.zeros(p)
    for t in range(kicks.shape[0]):
        s[..., 3] += kicks[t]
        f = policy(params, observe(s))
        s, hit = step(s, f)
        cos = np.cos(s[..., 2])
        r = cos - 0.1 * s[..., 0] ** 2 - 0.002 * s[..., 3] ** 2 - 0.02 * (f / F_MAX) ** 2 - 2.0 * hit
        r += 0.5 * (cos > np.cos(UPRIGHT_RAD))  # bonus for being inside the success cone
        total += r.mean(axis=1)
    return total


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def train(seed: int, gens: int, pairs: int, envs: int, sigma: float, lr: float, log) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.3, N_PARAMS) * 0.5
    m = np.zeros(N_PARAMS)
    v = np.zeros(N_PARAMS)
    steps = int(EPISODE_S / DT)
    best = (-np.inf, theta.copy(), 0)
    t0 = time.time()
    for g in range(1, gens + 1):
        eps = rng.normal(0, 1, (pairs, N_PARAMS))
        cand = np.concatenate([theta + sigma * eps, theta - sigma * eps])
        starts = init_states(rng, envs)  # common random numbers across the population
        kicks = np.where(rng.random((steps, envs)) < 0.004, rng.normal(0, 2.5, (steps, envs)), 0.0)
        fit = rollout(cand, starts, kicks)
        w = centred_ranks(fit)
        grad = (w[:pairs] - w[pairs:]) @ eps / (2 * pairs * sigma) - 0.005 * theta
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta = theta + lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
        if g % 25 == 0 or g == gens:
            sr, _, _ = evaluate_policy(theta, n=20, seed=10_000 + g)  # probe seeds, disjoint from the final eval
            if sr >= best[0]:
                best = (sr, theta.copy(), g)
            log(f"gen {g:4d}/{gens}  mean fit {fit.mean():8.1f}  max {fit.max():8.1f}  "
                f"swing-up {sr * 100:5.1f}% (20 probe seeds)  {time.time() - t0:6.1f} s")
    meta = {"generations": gens, "best_generation": best[2], "population": 2 * pairs,
            "envs_per_member": envs, "episode_s": EPISODE_S, "sigma": sigma, "lr": lr,
            "episodes": gens * 2 * pairs * envs, "wall_s": round(time.time() - t0, 1)}
    return best[1], meta


# ---- evaluation (shared criterion for learned and classical controllers) ----
def run_trials(ctrl, starts: np.ndarray, seconds: float, kick_at_s: float | None = None,
               kicks: np.ndarray | None = None) -> np.ndarray:
    """Simulate trials; return per-step 'upright and not at a wall' flags [E, T]."""
    s = starts.copy()
    steps = int(round(seconds / DT))
    up = np.zeros((len(s), steps), bool)
    wall = np.zeros((len(s), steps), bool)
    for t in range(steps):
        if kick_at_s is not None and t == int(round(kick_at_s / DT)):
            s[:, 3] += kicks
        f = np.clip(ctrl(s), -F_MAX, F_MAX)
        s, hit = step(s, f)
        up[:, t] = np.abs(wrap(s[:, 2])) < UPRIGHT_RAD
        wall[:, t] = hit
    return up & ~wall


def summarise(ok: np.ndarray, hold_steps: int, limit_steps: int) -> tuple[float, list[float | None]]:
    """Success = upright for the final hold window; time = start of the final upright run."""
    times = []
    for row in ok:
        if not row[-hold_steps:].all():
            times.append(None)
            continue
        idx = np.where(~row)[0]
        first = 0 if len(idx) == 0 else idx[-1] + 1
        times.append(first * DT if first <= limit_steps else None)
    return sum(t is not None for t in times) / len(times), times


def down_starts(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.zeros((n, 4))
    s[:, 2] = np.pi + rng.uniform(-0.05, 0.05, n)
    return s


def learned_ctrl(theta: np.ndarray):
    params = unflatten(theta[None])
    return lambda s: policy(params, observe(s)[None])[0]


def evaluate_policy(theta: np.ndarray, n: int, seed: int):
    ok = run_trials(learned_ctrl(theta), down_starts(n, seed), EVAL_S)
    sr, times = summarise(ok, int(HOLD_S / DT), int(SWINGUP_LIMIT_S / DT))
    return sr, times, ok


def push_recovery(ctrl, n: int, seed: int, kick_radps: float) -> float:
    """Start balanced, kick the pole at 1 s by +/- kick_radps, require upright for the last 3 s of 7 s."""
    rng = np.random.default_rng(seed)
    s = np.zeros((n, 4))
    s[:, 2] = rng.uniform(-0.05, 0.05, n)
    kicks = kick_radps * np.where(rng.random(n) < 0.5, -1.0, 1.0)
    ok = run_trials(ctrl, s, 7.0, kick_at_s=1.0, kicks=kicks)
    return float(ok[:, -int(HOLD_S / DT):].all(axis=1).mean())


# ---- classical baseline: energy-pumping swing-up + discrete LQR near the top ----
def lqr_gain() -> np.ndarray:
    """Discrete LQR on a finite-difference linearisation of step() about upright."""
    eps = 1e-5
    x0 = np.zeros(4)
    a = np.zeros((4, 4))
    b = np.zeros((4, 1))
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        a[:, i] = (step(x0 + d, np.array(0.0))[0] - step(x0 - d, np.array(0.0))[0]) / (2 * eps)
    b[:, 0] = (step(x0, np.array(eps))[0] - step(x0, np.array(-eps))[0]) / (2 * eps)
    q = np.diag([1.0, 0.5, 40.0, 2.0])
    r = np.array([[0.05]])
    p = q.copy()
    for _ in range(5000):
        k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        p_new = q + a.T @ p @ (a - b @ k)
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new
            break
        p = p_new
    else:
        raise RuntimeError("Riccati iteration did not converge")
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)[0]


# Gains picked by a small sweep (k_energy in {1.2, 3, 6, 10}); 6.0 keeps the cart centred.
CLASSIC = {"k_energy": 6.0, "k_x": 1.0, "k_v": 1.5, "switch_rad": 0.4, "switch_radps": 4.0}


def classical_ctrl(k_lqr: np.ndarray):
    j = 4.0 / 3.0 * M_POLE * L_HALF ** 2  # pole inertia about the pivot

    def ctrl(s: np.ndarray) -> np.ndarray:
        x, xd, th, thd = s[:, 0], s[:, 1], wrap(s[:, 2]), s[:, 3]
        energy = 0.5 * j * thd ** 2 + M_POLE * G * L_HALF * (np.cos(th) - 1.0)  # 0 at rest upright
        # pump energy toward 0: cart acceleration opposite to the pole's horizontal swing
        a = CLASSIC["k_energy"] * energy * thd * np.cos(th) - CLASSIC["k_x"] * x - CLASSIC["k_v"] * xd
        f_swing = (M_CART + M_POLE) * a * 4.0
        f_lqr = -(np.stack([x, xd, th, thd], axis=1) @ k_lqr)
        near = (np.abs(th) < CLASSIC["switch_rad"]) & (np.abs(thd) < CLASSIC["switch_radps"])
        return np.where(near, f_lqr, f_swing)
    return ctrl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gens", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.eval_only:
        data = json.loads(OUT.read_text())
        theta = np.concatenate([np.asarray(a, float).ravel() for a in data["weights"]])
        meta = data["training"]
    else:
        log(f"ES on a {N_PARAMS}-parameter MLP {(5, *HIDDEN, 1)}; seed {args.seed}")
        theta, meta = train(args.seed, args.gens, args.pairs, args.envs, args.sigma, args.lr, log)
        meta["seed"] = args.seed
        theta = np.round(theta, 5)  # evaluate exactly what ships

    eval_seed = 2026
    sr, times, _ = evaluate_policy(theta, args.trials, eval_seed)
    ok_t = [t for t in times if t is not None]
    push = push_recovery(learned_ctrl(theta), args.trials, eval_seed + 1, 3.0)
    k = lqr_gain()
    c_ok = run_trials(classical_ctrl(k), down_starts(args.trials, eval_seed), EVAL_S)
    _, c_times = summarise(c_ok, int(HOLD_S / DT), int(SWINGUP_LIMIT_S / DT))
    c_push = push_recovery(classical_ctrl(k), args.trials, eval_seed + 1, 3.0)
    c_ok_t = [t for t in c_times if t is not None]
    nan = float("nan")
    log(f"learned:   swing-up {sr * 100:.0f}% ({len(ok_t)}/{args.trials}), "
        f"time median {np.median(ok_t) if ok_t else nan:.2f} s max {max(ok_t) if ok_t else nan:.2f} s, "
        f"push recovery (3 rad/s) {push * 100:.0f}%")
    log(f"classical: swing-up {len(c_ok_t)}/{args.trials}, "
        f"time median {np.median(c_ok_t) if c_ok_t else nan:.2f} s, push recovery {c_push * 100:.0f}%")

    def res(t: list[float], pr: float) -> dict:
        return {"success": len(t), "median_s": round(float(np.median(t)), 2) if t else None,
                "max_s": round(float(max(t)), 2) if t else None, "push_recovery": round(pr, 3)}

    out = {
        "about": "Cart-pole swing-up policy for assets/js/demos/cartpole.js. Generated by tools/train_pendulum.py.",
        "plant": {"g": G, "m_cart": M_CART, "m_pole": M_POLE, "l_half": L_HALF, "f_max": F_MAX,
                  "x_max": X_MAX, "dt": DT, "obs_scale": OBS_SCALE.tolist()},
        "arch": [5, *HIDDEN, 1],
        "params": N_PARAMS,
        "method": "evolution strategies (antithetic, rank-shaped, Adam), numpy",
        "training": meta,
        "eval": {
            "criterion": f"start hanging down (|offset| <= 0.05 rad, cart at 0); upright = |theta| < "
                         f"{np.rad2deg(UPRIGHT_RAD):.0f} deg off the walls; success = upright for the last "
                         f"{HOLD_S:.0f} s of a {EVAL_S:.0f} s trial; time = start of that final upright run",
            "trials": args.trials, "seed": eval_seed, "push_kick_radps": 3.0,
            "learned": res(ok_t, push), "classical": res(c_ok_t, c_push),
        },
        "classical": {**CLASSIC, "k_lqr": [round(float(v), 5) for v in k]},
        "weights": [a.ravel().tolist() for a in unflatten(theta)],
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    log(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
