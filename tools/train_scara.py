#!/usr/bin/env python3
"""Train the obstacle-aware SCARA reaching policy shown in the "Reach around it" demo, numpy only.

Method: OpenAI-style evolution strategies (Salimans et al., 2017) with antithetic sampling,
centred-rank fitness shaping and Adam, on a tanh MLP. Each generation evaluates every
population member on the same batch of tasks (common random numbers). Task curriculum:
  * a share of tasks are "elbow-swap" tasks: the target pose is near a q1 limit and the arm
    starts on the other elbow side (a greedy reacher runs into the joint limit there);
  * a share are replayed from tasks the policy failed at the most recent probe;
  * obstacles are randomised (position, radius, or none) and phased in: the obstacle share
    ramps from 0 to --p-obstacle over the first --obstacle-ramp of the generations;
  * the ES noise and the learning rate anneal linearly to --sigma-final and --lr-final;
  * the shipped weights are the best checkpoint on a fixed --probe-tasks probe set.

Plant (top view, no gravity): two-link arm with the link lengths and joint limits of
assets/js/demos/scara.js. Each joint is a velocity servo: the policy commands a joint
velocity in [-V_MAX, V_MAX]; the joint velocity follows it with time constant SERVO_TAU_S and
acceleration clipped to A_MAX (the same limits the IK planner obeys). Joint limits are hard
stops (angle clamped, velocity zeroed). 50 Hz. Touching the obstacle freezes the arm in
training (one-off --contact-penalty, no reach bonus afterwards), and counts as a failure in evaluation.

Observation (OBS_NAMES, 28 values); every position is in metres before scaling, and the
"forearm frame" is rotated by -(q1 + q2), the "upper-arm frame" by -q1:
   0-3   cos q1, sin q1, cos q2, sin q2
   4-5   q1 / pi, q2 / pi
   6-7   q1d / V_MAX, q2d / V_MAX
   8-9   target - tip, forearm frame, / 0.15
  10-11  same, / 0.02 clipped to [-1, 1]
  12-15  joint-limit margins: (lim - q1), (q1 + lim), (lim - q2), (q2 + lim), each / 0.5 rad, clipped to [0, 1]
  16-17  target - base, base frame, / REACH
  18     obstacle present (0 / 1); when absent, 19-27 are 0 except the clearances (1)
  19-20  obstacle centre - base, base frame, / REACH
  21     obstacle radius / 0.1
  22-23  obstacle centre - tip, forearm frame, / 0.3
  24-25  obstacle centre - elbow, upper-arm frame, / 0.3
  26-27  clearance of link 1 and link 2 to the obstacle surface (m, minus LINK_R) / 0.1, clipped to [-1, 1]
With --ik-obs 1 (obs_version 3) six more values follow: for each elbow e = +1, -1 the analytic IK
solution for the target, as (feasible, (q1_goal - q1) / pi, (q2_goal - q2) / pi), where feasible
means within the joint limits and clear of the obstacle (0 and zeros otherwise). The joints do
not wrap, so the sign of q1_goal - q1 is the only legal direction of rotation.
The plant, observation and policy are mirrored exactly in scara.js; change both or neither.

Outputs assets/data/scara-policy.json (or --out): weights, training metadata and results on
held-out seeded task sets (no obstacle; random obstacles; the demo's default obstacle),
with success per target bearing sector and obstacle-touch rates, plus the analytic IK +
minimum-jerk baseline on the obstacle-free set.

    python3 tools/train_scara.py --workers 24          # full run
    python3 tools/train_scara.py --eval-only           # re-evaluate the shipped JSON
Results depend only on --seed and the hyper-parameters, not on --workers.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per process; parallelism comes from --workers

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# ---- arm (SI units, angles in rad). Mirrors ARM in scara.js. ----
L1 = 0.32                                   # m
L2 = 0.26                                   # m
REACH = L1 + L2                             # m
Q1_LIM = np.deg2rad(160.0)
Q2_LIM = np.deg2rad(150.0)
V_MAX = 2.4                                 # rad/s, per joint
A_MAX = 7.0                                 # rad/s^2, per joint
SERVO_TAU_S = 0.05                          # s
LINK_R = 0.022                              # m, link half-width for collision checks
DT = 0.02                                   # s, 50 Hz
QLIM = np.array([Q1_LIM, Q2_LIM])
# Default obstacle of the demo relative to the arm base: (0.98, 0.86) - (0.62, 0.60), r = 0.07 m.
DEMO_OBSTACLE = (0.36, 0.26, 0.07)
OBS_R_RANGE = (0.04, 0.09)                  # m, random obstacle radius in training/eval
OBS_DIST_RANGE = (0.12, 0.56)               # m, obstacle centre distance from the base

# ---- success criterion ----
TOL_M = 0.01
STOP_MPS = 0.02
REACH_LIMIT_S = 5.0
TRIAL_S = 6.0

OBS_COARSE_M = 0.15
OBS_FINE_M = 0.02
LIMIT_MARGIN_RAD = 0.5
OBST_REL_M = 0.3
CLEAR_SCALE_M = 0.1
OBS_NAMES = ["cq1", "sq1", "cq2", "sq2", "q1/pi", "q2/pi", "q1d", "q2d", "tgt_fx_c", "tgt_fy_c", "tgt_fx_f",
             "tgt_fy_f", "m1hi", "m1lo", "m2hi", "m2lo", "tgt_bx", "tgt_by", "obs_on", "obs_bx", "obs_by", "obs_r",
             "obs_fx", "obs_fy", "obs_ux", "obs_uy", "clear1", "clear2"]
IK_OBS_NAMES = ["ikp_ok", "ikp_dq1", "ikp_dq2", "ikn_ok", "ikn_dq1", "ikn_dq2"]
# Run configuration shared by every function, including worker processes (set in main before the
# pool forks). ik_obs selects obs_version 3 (with the IK-goal block) over 2.
CFG = {"ik_obs": True}


def obs_names() -> list[str]:
    return OBS_NAMES + (IK_OBS_NAMES if CFG["ik_obs"] else [])


def n_obs() -> int:
    return len(obs_names())
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "data" / "scara-policy.json"
SECTORS = ((-180, -90, "down-left"), (-90, 0, "down-right"), (0, 90, "up-right"), (90, 180, "up-left"))


# ---------------------------------------------------------------- kinematics / plant
def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q[..., 2] -> (elbow[..., 2], tip[..., 2]) relative to the base."""
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    elbow = np.stack([L1 * np.cos(q1), L1 * np.sin(q1)], axis=-1)
    tip = elbow + np.stack([L2 * np.cos(q12), L2 * np.sin(q12)], axis=-1)
    return elbow, tip


def tip_velocity(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    w1, w12 = qd[..., 0], qd[..., 0] + qd[..., 1]
    vx = -L1 * np.sin(q1) * w1 - L2 * np.sin(q12) * w12
    vy = L1 * np.cos(q1) * w1 + L2 * np.cos(q12) * w12
    return np.stack([vx, vy], axis=-1)


def step(q: np.ndarray, qd: np.ndarray, v_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 50 Hz step of the velocity-servo plant. Returns (q, qd, hit_limit[...])."""
    acc = np.clip((v_cmd - qd) / SERVO_TAU_S, -A_MAX, A_MAX)
    qd = qd + acc * DT
    q = q + qd * DT
    over = np.abs(q) > QLIM
    q = np.clip(q, -QLIM, QLIM)
    qd = np.where(over, 0.0, qd)
    return q, qd, over.any(axis=-1)


def seg_point_dist(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v = b - a
    t = np.clip(np.sum((c - a) * v, axis=-1) / np.maximum(np.sum(v * v, axis=-1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[..., None] * v - c, axis=-1)


def clearances(q: np.ndarray, obst: np.ndarray) -> np.ndarray:
    """Surface clearance (m) of link 1 and link 2 to the obstacle; obst[..., 4] = (on, x, y, r).
    Absent obstacles give +inf. Negative = touching."""
    elbow, tip = fk(q)
    c = obst[..., 1:3]
    base = np.zeros_like(elbow)
    d1 = seg_point_dist(base, elbow, c) - obst[..., 3] - LINK_R
    d2 = seg_point_dist(elbow, tip, c) - obst[..., 3] - LINK_R
    on = obst[..., 0] > 0.5
    return np.stack([np.where(on, d1, np.inf), np.where(on, d2, np.inf)], axis=-1)


def rot(x: np.ndarray, y: np.ndarray, ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express world vector (x, y) in a frame rotated by ang."""
    c, s = np.cos(ang), np.sin(ang)
    return c * x + s * y, -s * x + c * y


def ik_batch(target: np.ndarray, elbow: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised IK; returns (q[..., 2], within_limits[...]). Mirrors ik() and ikPose() in scara.js."""
    x, y = target[..., 0], target[..., 1]
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    ok = np.abs(c2) <= 1
    q2 = elbow * np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = np.arctan2(np.sin(q1), np.cos(q1))
    q = np.stack([q1, q2], axis=-1)
    return q, ok & (np.abs(q1) <= Q1_LIM) & (np.abs(q2) <= Q2_LIM)


def ik_goal_features(q: np.ndarray, target: np.ndarray, obst: np.ndarray) -> list[np.ndarray]:
    out = []
    for e in (1.0, -1.0):
        g, ok = ik_batch(target, e)
        ok = ok & (clearances(g, obst).min(axis=-1) > 0)
        f = ok.astype(float)
        out += [f, f * (g[..., 0] - q[..., 0]) / np.pi, f * (g[..., 1] - q[..., 1]) / np.pi]
    return out


def observe(q: np.ndarray, qd: np.ndarray, target: np.ndarray, obst: np.ndarray) -> np.ndarray:
    elbow, tip = fk(q)
    q1, q2 = q[..., 0], q[..., 1]
    phi = q1 + q2
    lx, ly = rot(target[..., 0] - tip[..., 0], target[..., 1] - tip[..., 1], phi)
    on = obst[..., 0]
    ox, oy = obst[..., 1], obst[..., 2]
    fx, fy = rot(ox - tip[..., 0], oy - tip[..., 1], phi)
    ux, uy = rot(ox - elbow[..., 0], oy - elbow[..., 1], q1)
    cl = np.clip(clearances(q, obst) / CLEAR_SCALE_M, -1, 1)
    return np.stack([
        np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2),
        q1 / np.pi, q2 / np.pi, qd[..., 0] / V_MAX, qd[..., 1] / V_MAX,
        lx / OBS_COARSE_M, ly / OBS_COARSE_M,
        np.clip(lx / OBS_FINE_M, -1, 1), np.clip(ly / OBS_FINE_M, -1, 1),
        np.clip((Q1_LIM - q1) / LIMIT_MARGIN_RAD, 0, 1), np.clip((q1 + Q1_LIM) / LIMIT_MARGIN_RAD, 0, 1),
        np.clip((Q2_LIM - q2) / LIMIT_MARGIN_RAD, 0, 1), np.clip((q2 + Q2_LIM) / LIMIT_MARGIN_RAD, 0, 1),
        target[..., 0] / REACH, target[..., 1] / REACH,
        on, on * ox / REACH, on * oy / REACH, on * obst[..., 3] / 0.1,
        on * fx / OBST_REL_M, on * fy / OBST_REL_M, on * ux / OBST_REL_M, on * uy / OBST_REL_M,
        cl[..., 0], cl[..., 1],
        *(ik_goal_features(q, target, obst) if CFG["ik_obs"] else []),
    ], axis=-1)


# ---------------------------------------------------------------- policy
def shapes(hidden: tuple[int, ...]) -> list[tuple[int, ...]]:
    dims = (n_obs(), *hidden, 2)
    out = []
    for a, b in zip(dims[:-1], dims[1:]):
        out += [(a, b), (b,)]
    return out


def n_params(hidden: tuple[int, ...]) -> int:
    return int(sum(np.prod(s) for s in shapes(hidden)))


def unflatten(theta: np.ndarray, hidden: tuple[int, ...]) -> list[np.ndarray]:
    out, i = [], 0
    lead = theta.shape[:-1]
    for s in shapes(hidden):
        n = int(np.prod(s))
        out.append(theta[..., i:i + n].reshape(*lead, *s))
        i += n
    return out


def policy(params: list[np.ndarray], obs: np.ndarray) -> np.ndarray:
    """params with leading dim P, obs [P, E, n_obs()] -> joint velocity command [P, E, 2] (rad/s)."""
    h = obs
    for li in range(0, len(params), 2):
        h = np.tanh(h @ params[li] + params[li + 1][:, None, :])
    return h * V_MAX


# ---------------------------------------------------------------- tasks
def ik(target: np.ndarray, elbow: float) -> np.ndarray | None:
    x, y = target
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return None
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = np.arctan2(np.sin(q1), np.cos(q1))
    q = np.array([q1, q2])
    return q if np.all(np.abs(q) <= QLIM) else None


def clear_goals(tg: np.ndarray, obst: np.ndarray) -> list[np.ndarray]:
    """IK poses (both elbows) for target tg that are within limits and clear of the obstacle."""
    out = []
    for e in (1.0, -1.0):
        g = ik(tg, e)
        if g is not None and clearances(g, obst).min() > 0:
            out.append(g)
    return out


def random_obstacle(rng: np.random.Generator) -> np.ndarray:
    d = rng.uniform(*OBS_DIST_RANGE)
    a = rng.uniform(-np.pi, np.pi)
    return np.array([1.0, d * np.cos(a), d * np.sin(a), rng.uniform(*OBS_R_RANGE)])


NO_OBSTACLE = np.array([0.0, 0.0, 0.0, 0.0])


def sample_task(rng: np.random.Generator, kind: str, p_obstacle: float, min_move_m: float,
                obstacle: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kind 'uniform': start and target pose uniform over (5 deg inside) the joint limits.
    kind 'swap': target pose with |q1| in 100..155 deg, start pose on the other elbow side.
    The start pose and at least one IK pose for the target are clear of the obstacle."""
    lim = QLIM - np.deg2rad(5.0)
    while True:
        if obstacle is not None:
            obst = obstacle.copy()
        else:
            obst = random_obstacle(rng) if rng.random() < p_obstacle else NO_OBSTACLE.copy()
        if kind == "swap":
            qt = np.array([rng.choice([-1, 1]) * np.deg2rad(rng.uniform(100, 155)), rng.uniform(-lim[1], lim[1])])
            q0 = np.array([rng.uniform(-lim[0], lim[0]), -np.sign(qt[1]) * rng.uniform(0.2, lim[1])])
        else:
            qt = rng.uniform(-lim, lim)
            q0 = rng.uniform(-lim, lim)
        tg = fk(qt)[1]
        if np.linalg.norm(tg - fk(q0)[1]) < min_move_m:
            continue
        if clearances(q0, obst).min() <= 0.005 or not clear_goals(tg, obst):
            continue
        return q0, tg, obst


def sample_batch(rng, n, swap_frac, p_obstacle, min_move_m=0.1, obstacle=None):
    rows = [sample_task(rng, "swap" if rng.random() < swap_frac else "uniform", p_obstacle, min_move_m, obstacle)
            for _ in range(n)]
    return tuple(np.array(c) for c in zip(*rows))


# ---------------------------------------------------------------- C-space connectivity (eval filter)
CS_GRID = 96


def path_exists(q0: np.ndarray, goals: list[np.ndarray], obst: np.ndarray) -> bool:
    """True if a collision-free joint-space path from q0 to one of goals exists on a CS_GRID^2
    C-space grid (8-connected, cell centres checked exactly). Joint limits do not wrap."""
    if obst[0] < 0.5:
        return bool(goals)
    a = np.linspace(-Q1_LIM, Q1_LIM, CS_GRID, endpoint=False) + Q1_LIM / CS_GRID
    b = np.linspace(-Q2_LIM, Q2_LIM, CS_GRID, endpoint=False) + Q2_LIM / CS_GRID
    qq = np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1)
    free = clearances(qq, np.broadcast_to(obst, (CS_GRID, CS_GRID, 4))).min(axis=-1) > 0
    cell = lambda q: (min(CS_GRID - 1, int((q[0] + Q1_LIM) / (2 * Q1_LIM) * CS_GRID)),  # noqa: E731
                      min(CS_GRID - 1, int((q[1] + Q2_LIM) / (2 * Q2_LIM) * CS_GRID)))
    reach = np.zeros_like(free)
    reach[cell(q0)] = True
    free[cell(q0)] = True
    targets = [cell(g) for g in goals]
    for g in targets:
        free[g] = True
    while True:
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        g2 = grown.copy()
        g2[:, 1:] |= grown[:, :-1]
        g2[:, :-1] |= grown[:, 1:]
        g2 &= free
        if any(g2[t] for t in targets):
            return True
        if (g2 == reach).all():
            return False
        reach = g2


# ---------------------------------------------------------------- training
def reward_rollout(theta: np.ndarray, hidden, q0, tg, obst, steps, contact_penalty, clear_m) -> np.ndarray:
    """Mean per-step reward for each member. theta [P, N]; q0/tg [E, 2]; obst [E, 4].
    float32 policy for speed (batched small matmuls dominate); evaluation is float64."""
    p = theta.shape[0]
    params = unflatten(theta.astype(np.float32), hidden)
    q = np.broadcast_to(q0, (p, *q0.shape)).copy()
    qd = np.zeros_like(q)
    target = np.broadcast_to(tg, q.shape)
    ob = np.broadcast_to(obst, (p, *obst.shape))
    dead = np.zeros(q.shape[:2], bool)
    total = np.zeros(p)
    for _ in range(steps):
        v = policy(params, observe(q, qd, target, ob).astype(np.float32)).astype(np.float64)
        q_new, qd_new, hit = step(q, qd, v)
        q = np.where(dead[..., None], q, q_new)
        qd = np.where(dead[..., None], 0.0, qd_new)
        _, tip = fk(q)
        d = np.linalg.norm(target - tip, axis=-1)
        speed = np.linalg.norm(tip_velocity(q, qd), axis=-1)
        clear = clearances(q, ob).min(axis=-1)
        touch = clear <= 0
        r = (-d / 0.3 + 0.5 * np.exp(-d / 0.01)
             + 1.0 * ((d < TOL_M) & (speed < STOP_MPS))
             - 0.3 * (d < 0.05) * speed
             - 0.01 * np.sum((v / V_MAX) ** 2, axis=-1)
             - 0.5 * hit
             - 0.2 * np.clip(1.0 - clear / clear_m, 0.0, 1.0))  # clearance shaping inside clear_m
        new_touch = touch & ~dead
        dead |= touch
        # contact freezes the arm: a one-off penalty, then only the distance term (no reach bonus).
        # Per-step rewards stay in the same range as normal play, so contact is never a shortcut
        # and does not dominate the return (the v2 -3/step rule made the policy timid).
        r = np.where(dead, -d / 0.3, r) - contact_penalty * new_touch
        total += r.mean(axis=1)
    return total / steps


def _worker(args):
    cfg, theta, hidden, q0, tg, obst, steps, cp, cm = args
    CFG.update(cfg)
    return reward_rollout(theta, hidden, q0, tg, obst, steps, cp, cm)


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def train(a, pool, log) -> tuple[np.ndarray, dict]:
    hidden = tuple(a.hidden)
    npar = n_params(hidden)
    rng = np.random.default_rng(a.seed)
    theta = rng.normal(0, 1, npar) * np.concatenate(
        [np.full(int(np.prod(s)), 1 / np.sqrt(s[0]) if len(s) == 2 else 0.0) for s in shapes(hidden)])
    m = np.zeros(npar)
    v = np.zeros(npar)
    steps = int(round(a.episode_s / DT))
    best = (-np.inf, theta.copy(), 0)
    hard = []  # replay pool of failed probe tasks: (q0, tg, obst)
    # fixed probe set (own seed stream, disjoint from training and eval) for checkpoint selection
    prng = np.random.default_rng(10_000_000 + a.seed)
    probe = sample_batch(prng, a.probe_tasks, 0.25, a.p_obstacle)
    t0 = time.time()
    for g in range(1, a.gens + 1):
        frac = (g - 1) / max(1, a.gens - 1)
        sigma = a.sigma + (a.sigma_final - a.sigma) * frac
        lr = a.lr + (a.lr_final - a.lr) * frac
        p_obs = a.p_obstacle * min(1.0, frac / a.obstacle_ramp) if a.obstacle_ramp > 0 else a.p_obstacle
        eps = rng.normal(0, 1, (a.pairs, npar))
        cand = np.concatenate([theta + sigma * eps, theta - sigma * eps])
        q0, tg, ob = sample_batch(rng, a.envs, a.swap_frac, p_obs)
        n_rep = min(len(hard), int(round(a.replay_frac * a.envs)))
        if n_rep:
            for i, j in enumerate(rng.choice(len(hard), n_rep, replace=False)):
                q0[i], tg[i], ob[i] = hard[j]
        chunks = np.array_split(cand, a.workers)
        args = (dict(CFG), hidden, q0, tg, ob, steps, a.contact_penalty, a.clear_m)
        if pool is None:
            fit = _worker((args[0], cand, *args[1:]))
        else:
            fit = np.concatenate(pool.map(_worker, [(args[0], c, *args[1:]) for c in chunks]))
        w = centred_ranks(fit)
        grad = (w[:a.pairs] - w[a.pairs:]) @ eps / (2 * a.pairs * sigma) - a.wd * theta
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta = theta + lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
        if g % a.probe_every == 0 or g == a.gens:
            res = evaluate(theta, hidden, *probe, pool, a.workers)
            sr = float(res["success"].mean())
            # replay pool: the probe tasks the current policy fails (newest 400 kept)
            fail = np.where(~res["success"])[0]
            hard = (hard + [(probe[0][i], probe[1][i], probe[2][i]) for i in fail])[-400:]
            # checkpoints are only eligible once the obstacle share has reached its final value
            if sr >= best[0] and (a.obstacle_ramp <= 0 or frac >= a.obstacle_ramp):
                best = (sr, theta.copy(), g)
            el = time.time() - t0
            no = probe[2][:, 0] < 0.5
            log(f"gen {g:5d}/{a.gens}  fit mean {fit.mean():7.3f} max {fit.max():7.3f}  probe success "
                f"{sr * 100:5.1f}% (free {res['success'][no].mean() * 100:5.1f}%, obstacle "
                f"{res['success'][~no].mean() * 100:5.1f}%)  touch {res['touch'].mean() * 100:4.1f}%  "
                f"p_obs {p_obs:.2f} sigma {sigma:.3f}  {el:7.1f} s  ETA {el / g * (a.gens - g) / 60:5.1f} min")
    if best[2] == 0:  # no eligible checkpoint (e.g. a very short run): ship the last weights
        best = (None, theta.copy(), a.gens)
    meta = {"generations": a.gens, "best_generation": best[2], "population": 2 * a.pairs,
            "envs_per_member": a.envs, "episode_s": a.episode_s, "sigma": [a.sigma, a.sigma_final],
            "lr": [a.lr, a.lr_final], "swap_frac": a.swap_frac, "hard_task_replay": a.replay_frac,
            "p_obstacle": a.p_obstacle, "obstacle_ramp": a.obstacle_ramp, "contact_penalty": a.contact_penalty,
            "clear_shaping_m": a.clear_m, "probe_tasks": a.probe_tasks,
            "episodes": a.gens * 2 * a.pairs * a.envs, "workers": a.workers,
            "wall_s": round(time.time() - t0, 1)}
    return best[1], meta


# ---------------------------------------------------------------- evaluation
def _eval_chunk(args):
    cfg, theta, hidden, q0, tg, ob = args
    CFG.update(cfg)
    params = unflatten(theta[None], hidden)
    q, qd = q0.copy(), np.zeros_like(q0)
    n = len(q0)
    reach_t = np.full(n, np.nan)
    touch = clearances(q, ob).min(axis=-1) <= 0
    for k in range(int(round(TRIAL_S / DT))):
        v = policy(params, observe(q, qd, tg, ob)[None])[0]
        q, qd, _ = step(q, qd, v)
        tip = fk(q)[1]
        d = np.linalg.norm(tg - tip, axis=-1)
        ok = (d < TOL_M) & (np.linalg.norm(tip_velocity(q, qd), axis=-1) < STOP_MPS)
        reach_t = np.where(np.isnan(reach_t) & ok, (k + 1) * DT, reach_t)
        touch |= clearances(q, ob).min(axis=-1) <= 0
    final_err = np.linalg.norm(tg - fk(q)[1], axis=-1)
    return reach_t, final_err, touch


def evaluate(theta, hidden, q0, tg, ob, pool=None, workers=1) -> dict:
    """Success = reached within REACH_LIMIT_S, still within TOL_M at TRIAL_S, and never touched the obstacle."""
    idx = np.array_split(np.arange(len(q0)), max(1, workers))
    jobs = [(dict(CFG), theta, hidden, q0[i], tg[i], ob[i]) for i in idx if len(i)]
    parts = pool.map(_eval_chunk, jobs) if pool is not None else [_eval_chunk(j) for j in jobs]
    reach_t, final_err, touch = (np.concatenate(x) for x in zip(*parts))
    reached = (reach_t <= REACH_LIMIT_S) & (final_err < TOL_M)
    return {"success": reached & ~touch, "reached": reached, "time": reach_t, "final_err": final_err, "touch": touch}


def min_jerk_time(d: float) -> float:
    """Mirror of segmentTime() in scara.js: d = largest joint change (rad)."""
    return max(0.15, 1.875 * d / V_MAX, np.sqrt(5.7735 * d / A_MAX))


def ik_baseline(q0: np.ndarray, tg: np.ndarray) -> dict:
    """Obstacle-free set only: analytic IK (elbow +1 preferred, as in the demo) + one straight
    min-jerk joint move; time = first moment the 1 cm / 2 cm/s criterion is met."""
    times = []
    for qs, t in zip(q0, tg):
        qg = ik(t, 1.0)
        qg = ik(t, -1.0) if qg is None else qg
        T = min_jerk_time(float(np.max(np.abs(qg - qs))))
        ts = np.arange(1, int(np.ceil(T / DT)) + 1) * DT
        s = np.clip(ts / T, 0, 1)
        sh = s ** 3 * (10 - 15 * s + 6 * s * s)
        sd = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
        qq = qs + sh[:, None] * (qg - qs)
        d = np.linalg.norm(t - fk(qq)[1], axis=-1)
        vv = np.linalg.norm(tip_velocity(qq, sd[:, None] * (qg - qs)), axis=-1)
        hit = np.where((d < TOL_M) & (vv < STOP_MPS))[0]
        times.append(ts[hit[0]] if len(hit) else T)
    return {"success": len(times), "median_s": round(float(np.median(times)), 2),
            "p90_s": round(float(np.percentile(times, 90)), 2)}


def _filter_chunk(args):
    q0, tg, ob = args
    return [path_exists(a, clear_goals(b, c), c) for a, b, c in zip(q0, tg, ob)]


def eval_set(n: int, seed: int, mode: str, pool=None, workers=1):
    """Held-out tasks: target >= 20 cm from the start tip, uniform start/target poses.
    mode 'free': no obstacle. 'random': a random obstacle. 'demo': the demo's default obstacle.
    With an obstacle, tasks are kept only if a collision-free joint-space path exists (path_exists)."""
    rng = np.random.default_rng(seed)
    obstacle = {"free": NO_OBSTACLE, "random": None, "demo": np.array([1.0, *DEMO_OBSTACLE])}[mode]
    p_obs = 0.0 if mode == "free" else 1.0
    keep_q0, keep_tg, keep_ob = [], [], []
    while len(keep_q0) < n:
        q0, tg, ob = sample_batch(rng, 2 * (n - len(keep_q0)), 0.0, p_obs, 0.2, obstacle)
        if mode == "free":
            ok = [True] * len(q0)
        else:
            idx = np.array_split(np.arange(len(q0)), max(1, workers))
            jobs = [(q0[i], tg[i], ob[i]) for i in idx if len(i)]
            ok = sum(pool.map(_filter_chunk, jobs) if pool is not None else [_filter_chunk(j) for j in jobs], [])
        for a, b, c, k in zip(q0, tg, ob, ok):
            if k and len(keep_q0) < n:
                keep_q0.append(a)
                keep_tg.append(b)
                keep_ob.append(c)
    return np.array(keep_q0), np.array(keep_tg), np.array(keep_ob)


def summarise(r: dict, tg: np.ndarray) -> dict:
    t = r["time"][r["success"]]
    bearing = np.degrees(np.arctan2(tg[:, 1], tg[:, 0]))
    sectors = []
    for lo, hi, name in SECTORS:
        msk = (bearing >= lo) & (bearing < hi)
        sectors.append({"name": name, "from_deg": lo, "to_deg": hi, "success": int(r["success"][msk].sum()),
                        "trials": int(msk.sum())})
    return {"trials": len(tg), "success": int(r["success"].sum()), "touch": int(r["touch"].sum()),
            "median_s": round(float(np.median(t)), 2) if len(t) else None,
            "p90_s": round(float(np.percentile(t, 90)), 2) if len(t) else None,
            "median_final_err_mm": round(float(np.median(r["final_err"]) * 1000), 1),
            "by_bearing": sectors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=1, help="processes; results do not depend on it")
    ap.add_argument("--gens", type=int, default=4000)
    ap.add_argument("--pairs", type=int, default=256, help="population = 2 * pairs")
    ap.add_argument("--envs", type=int, default=64, help="tasks per member per generation")
    ap.add_argument("--ik-obs", type=int, default=1, choices=(0, 1),
                    help="1: observe the analytic IK goals (obs_version 3); 0: obs_version 2")
    ap.add_argument("--episode-s", type=float, default=6.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--sigma-final", type=float, default=0.015)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-final", type=float, default=0.003)
    ap.add_argument("--p-obstacle", type=float, default=0.6, help="final share of tasks with an obstacle")
    ap.add_argument("--obstacle-ramp", type=float, default=0.25, help="share of the run over which obstacles phase in")
    ap.add_argument("--contact-penalty", type=float, default=300.0, help="one-off, in per-step reward units")
    ap.add_argument("--clear-m", type=float, default=0.02, help="clearance shaping distance (m)")
    ap.add_argument("--wd", type=float, default=0.001, help="L2 pull on the parameters, per generation")
    ap.add_argument("--swap-frac", type=float, default=0.3)
    ap.add_argument("--replay-frac", type=float, default=0.2)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--probe-tasks", type=int, default=600)
    ap.add_argument("--trials", type=int, default=1000, help="tasks per held-out set")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dump-tasks", type=Path, default=None,
                    help="also write the held-out task sets (base-relative) as JSON, for tools/eval_scara_ik.mjs")
    a = ap.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    if a.eval_only:
        CFG["ik_obs"] = json.loads(a.out.read_text()).get("obs_version") == 3
    else:
        CFG["ik_obs"] = bool(a.ik_obs)
    pool = mp.get_context("fork").Pool(a.workers) if a.workers > 1 else None
    try:
        if a.eval_only:
            data = json.loads(a.out.read_text())
            hidden = tuple(data["arch"][1:-1])
            theta = np.concatenate([np.asarray(x, float).ravel() for x in data["weights"]])
            meta = data["training"]
        else:
            hidden = tuple(a.hidden)
            log(f"ES on a {n_params(hidden)}-parameter MLP {(n_obs(), *hidden, 2)}; obs_version "
                f"{3 if CFG['ik_obs'] else 2}; seed {a.seed}; "
                f"population {2 * a.pairs} x {a.envs} tasks x {a.episode_s} s; {a.workers} workers")
            theta, meta = train(a, pool, log)
            meta["seed"] = a.seed
            theta = np.round(theta, 4)  # evaluate exactly what ships

        eval_seed = 2026  # disjoint from the training (seed), probe (10_000_000 + g) streams
        t0 = time.time()
        sets = {}
        dumped = {}
        for k, (mode, sd) in enumerate((("free", eval_seed), ("random", eval_seed + 1), ("demo", eval_seed + 2))):
            q0, tg, ob = eval_set(a.trials, sd, mode, pool, a.workers)
            r = evaluate(theta, hidden, q0, tg, ob, pool, a.workers)
            dumped[mode] = {"q0": q0.tolist(), "tg": tg.tolist(), "ob": ob.tolist(),
                            "success": r["success"].tolist(), "touch": r["touch"].tolist()}
            sets[mode] = summarise(r, tg)
            sets[mode]["seed"] = sd
            s = sets[mode]
            log(f"[{mode:6s}] success {s['success']}/{s['trials']}  touch {s['touch']}  median {s['median_s']} s  "
                f"p90 {s['p90_s']} s  median final err {s['median_final_err_mm']} mm  | "
                + "  ".join(f"{x['name']} {x['success']}/{x['trials']}" for x in s["by_bearing"]))
            if mode == "free":
                sets["ik_min_jerk_free"] = ik_baseline(q0, tg)
                log(f"[free  ] IK + min-jerk: {sets['ik_min_jerk_free']}")
        log(f"evaluation {time.time() - t0:.0f} s")
        if a.dump_tasks:
            a.dump_tasks.write_text(json.dumps({"note": "held-out SCARA task sets, positions relative to the arm base; "
                                                "ob = (present, x, y, r)", **dumped}))
            log(f"wrote {a.dump_tasks}")

        out = {
            "about": "SCARA reaching policy for assets/js/demos/scara.js. Generated by tools/train_scara.py.",
            "obs_version": 3 if CFG["ik_obs"] else 2,
            "obs_names": obs_names(),
            "plant": {"l1": L1, "l2": L2, "q_lim": [round(float(x), 6) for x in QLIM], "v_max": V_MAX,
                      "a_max": A_MAX, "servo_tau_s": SERVO_TAU_S, "dt": DT, "link_r": LINK_R,
                      "obs_coarse_m": OBS_COARSE_M, "obs_fine_m": OBS_FINE_M, "limit_margin_rad": LIMIT_MARGIN_RAD,
                      "obst_rel_m": OBST_REL_M, "clear_scale_m": CLEAR_SCALE_M},
            "arch": [n_obs(), *hidden, 2],
            "params": n_params(hidden),
            "method": "evolution strategies (antithetic, rank-shaped, Adam, annealed noise) with an obstacle "
                      "phase-in and elbow-swap curriculum and failed-task replay, numpy",
            "training": meta,
            "eval": {
                "criterion": f"random start pose at rest, reachable target >= 20 cm away; success = tip within "
                             f"{TOL_M * 100:.0f} cm and slower than {STOP_MPS * 100:.0f} cm/s within "
                             f"{REACH_LIMIT_S:.0f} s, still within {TOL_M * 100:.0f} cm at {TRIAL_S:.0f} s, and no "
                             f"obstacle contact; obstacle sets keep only tasks with a collision-free joint-space path",
                **sets,
            },
            "weights": [x.ravel().tolist() for x in unflatten(theta, hidden)],
        }
        a.out.write_text(json.dumps(out, separators=(",", ":")))
        log(f"wrote {a.out} ({a.out.stat().st_size} bytes)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
