#!/usr/bin/env python3
"""Train the SCARA reaching policy shown in the "Reach around it" demo, using numpy only.

Method: OpenAI-style evolution strategies (Salimans et al., 2017) with antithetic
sampling, centred-rank fitness shaping and Adam, on a small tanh MLP (same recipe as
tools/train_pendulum.py).

Plant (top view, so no gravity): a two-link arm with the link lengths and joint limits of
assets/js/demos/scara.js. Each joint is a velocity servo: the policy commands a joint
velocity in [-V_MAX, V_MAX]; the joint velocity follows it with time constant SERVO_TAU_S and
acceleration clipped to A_MAX (the same speed/acceleration limits the IK planner obeys).
Joint limits are hard stops (angle clamped, velocity zeroed). Control at 50 Hz.
The plant, observation and policy below are mirrored exactly in scara.js; change both or neither.

Observation (12): cos/sin of both joints, both joints scaled by 1/pi (so the limits are
visible), both joint velocities / V_MAX, and the target position relative to the end
effector expressed in the forearm frame (rotated by -(q1 + q2)), twice: /0.15 m (coarse) and
/0.02 m clipped to [-1, 1] (fine, for the last centimetre).
The policy does not observe the obstacle. Its obstacle-contact rate against the demo's
default obstacle is measured and reported instead.

Outputs assets/data/scara-policy.json: weights, training metadata and results on a held-out
seeded set of (start pose, target) pairs for the policy and for the analytic IK +
minimum-jerk baseline.

    python3 tools/train_scara.py              # train + evaluate + write JSON
    python3 tools/train_scara.py --eval-only  # re-evaluate the shipped JSON
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---- arm (SI units, angles in rad). Mirrors ARM in scara.js. ----
L1 = 0.32                                   # m
L2 = 0.26                                   # m
Q1_LIM = np.deg2rad(160.0)                  # |q1| <= 160 deg
Q2_LIM = np.deg2rad(150.0)                  # |q2| <= 150 deg
V_MAX = 2.4                                 # rad/s, per joint
A_MAX = 7.0                                 # rad/s^2, per joint
SERVO_TAU_S = 0.05                          # s, joint velocity servo time constant
LINK_R = 0.022                              # m, link half-width for collision checks
DT = 0.02                                   # s, 50 Hz
# Default obstacle of the demo relative to the arm base: (0.98, 0.86) - (0.62, 0.60), r = 0.07 m.
OBSTACLE = (0.36, 0.26, 0.07)

# ---- success criterion ----
TOL_M = 0.01            # end effector within 1 cm ...
STOP_MPS = 0.02         # ... and moving slower than 2 cm/s
REACH_LIMIT_S = 5.0     # must happen within 5 s ...
TRIAL_S = 6.0           # ... and the tip must still be within 1 cm at the end of a 6 s trial
EPISODE_S = 4.0         # training episode length

OBS_COARSE_M = 0.15
OBS_FINE_M = 0.02
HIDDEN = (32, 32)
N_OBS = 12
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "data" / "scara-policy.json"
QLIM = np.array([Q1_LIM, Q2_LIM])


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


def observe(q: np.ndarray, qd: np.ndarray, target: np.ndarray) -> np.ndarray:
    _, tip = fk(q)
    d = target - tip
    phi = q[..., 0] + q[..., 1]
    c, s = np.cos(phi), np.sin(phi)
    lx = c * d[..., 0] + s * d[..., 1]
    ly = -s * d[..., 0] + c * d[..., 1]
    return np.stack([
        np.cos(q[..., 0]), np.sin(q[..., 0]), np.cos(q[..., 1]), np.sin(q[..., 1]),
        q[..., 0] / np.pi, q[..., 1] / np.pi, qd[..., 0] / V_MAX, qd[..., 1] / V_MAX,
        lx / OBS_COARSE_M, ly / OBS_COARSE_M,
        np.clip(lx / OBS_FINE_M, -1, 1), np.clip(ly / OBS_FINE_M, -1, 1),
    ], axis=-1)


# ---- policy: obs(12) -> tanh(32) -> tanh(32) -> tanh(2) * V_MAX ----
def shapes() -> list[tuple[int, ...]]:
    dims = (N_OBS, *HIDDEN, 2)
    out = []
    for a, b in zip(dims[:-1], dims[1:]):
        out += [(a, b), (b,)]
    return out


N_PARAMS = int(sum(np.prod(s) for s in shapes()))


def unflatten(theta: np.ndarray) -> list[np.ndarray]:
    out, i = [], 0
    lead = theta.shape[:-1]
    for s in shapes():
        n = int(np.prod(s))
        out.append(theta[..., i:i + n].reshape(*lead, *s))
        i += n
    return out


def policy(params: list[np.ndarray], obs: np.ndarray) -> np.ndarray:
    """params with leading dim P, obs [P, E, 12] -> joint velocity command [P, E, 2] (rad/s)."""
    h = obs
    for li in range(0, len(params), 2):
        h = np.tanh(h @ params[li] + params[li + 1][:, None, :])
    return h * V_MAX


# ---- task sampling ----
def ik(target: np.ndarray, elbow: float) -> np.ndarray | None:
    """Analytic IK for one target (relative to base). elbow = +1 (q2 > 0) or -1. None if unreachable."""
    x, y = target
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return None
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = np.arctan2(np.sin(q1), np.cos(q1))
    q = np.array([q1, q2])
    return q if np.all(np.abs(q) <= QLIM) else None


def sample_tasks(rng: np.random.Generator, n: int, min_move_m: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Random start pose (5 deg inside the limits, at rest) and a reachable target >= min_move_m away."""
    q0 = np.zeros((n, 2))
    tg = np.zeros((n, 2))
    lim = QLIM - np.deg2rad(5.0)
    i = 0
    while i < n:
        q = rng.uniform(-lim, lim)
        # target = tip of a random in-limit pose, i.e. uniform over joint space, which covers the whole
        # reachable set including the wedge behind the base
        t = fk(rng.uniform(-lim, lim))[1]
        if np.linalg.norm(t - fk(q)[1]) < min_move_m:
            continue
        q0[i], tg[i] = q, t
        i += 1
    return q0, tg


def rollout(theta: np.ndarray, q0: np.ndarray, tg: np.ndarray, steps: int) -> np.ndarray:
    """Return mean per-step reward for each member. theta [P, N], q0/tg [E, 2].
    The policy runs in float32 here for speed (batched small matmuls dominate); evaluation is float64."""
    p = theta.shape[0]
    params = unflatten(theta.astype(np.float32))
    q = np.broadcast_to(q0, (p, *q0.shape)).copy()
    qd = np.zeros_like(q)
    target = np.broadcast_to(tg, q.shape)
    total = np.zeros(p)
    for _ in range(steps):
        v = policy(params, observe(q, qd, target).astype(np.float32)).astype(np.float64)
        q, qd, hit = step(q, qd, v)
        _, tip = fk(q)
        d = np.linalg.norm(target - tip, axis=-1)
        speed = np.linalg.norm(tip_velocity(q, qd), axis=-1)
        r = (-d / 0.3 + 0.5 * np.exp(-d / 0.01)
             + 1.0 * ((d < TOL_M) & (speed < STOP_MPS))
             - 0.3 * (d < 0.05) * speed
             - 0.01 * np.sum((v / V_MAX) ** 2, axis=-1)
             - 0.5 * hit)
        total += r.mean(axis=1)
    return total / steps


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def train(seed: int, gens: int, pairs: int, envs: int, sigma: float, lr: float, replay_frac: float,
          log) -> tuple[np.ndarray, dict]:
    """ES with hard-task replay: replay_frac of each generation's tasks are drawn from the probe tasks
    the current policy failed most recently (greedy reaching gets stuck at the q1 limit unless it
    learns to swap elbows, and those tasks are rare under uniform sampling)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.15, N_PARAMS)
    m = np.zeros(N_PARAMS)
    v = np.zeros(N_PARAMS)
    steps = int(EPISODE_S / DT)
    best = (-np.inf, theta.copy(), 0)
    hard_q0, hard_tg = np.zeros((0, 2)), np.zeros((0, 2))
    t0 = time.time()
    for g in range(1, gens + 1):
        eps = rng.normal(0, 1, (pairs, N_PARAMS))
        cand = np.concatenate([theta + sigma * eps, theta - sigma * eps])
        q0, tg = sample_tasks(rng, envs)  # common random numbers across the population
        n_rep = min(len(hard_q0), int(round(replay_frac * envs)))
        if n_rep:
            pick = rng.choice(len(hard_q0), n_rep, replace=False)
            q0[:n_rep], tg[:n_rep] = hard_q0[pick], hard_tg[pick]
        fit = rollout(cand, q0, tg, steps)
        w = centred_ranks(fit)
        grad = (w[:pairs] - w[pairs:]) @ eps / (2 * pairs * sigma) - 0.002 * theta
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta = theta + lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
        if g % 25 == 0 or g == gens:
            probe_q0, probe_tg = sample_tasks(np.random.default_rng(10_000 + g), 100)
            res = evaluate(learned_ctrl(theta), probe_q0, probe_tg)
            sr = float(res["reached"].mean())
            fail = ~res["reached"]
            hard_q0 = np.concatenate([hard_q0, probe_q0[fail]])[-200:]
            hard_tg = np.concatenate([hard_tg, probe_tg[fail]])[-200:]
            if sr >= best[0]:
                best = (sr, theta.copy(), g)
            log(f"gen {g:4d}/{gens}  mean fit {fit.mean():7.3f}  max {fit.max():7.3f}  "
                f"success {sr * 100:5.1f}% (100 probe tasks)  median final err "
                f"{np.median(res['final_err']) * 1000:6.1f} mm  {time.time() - t0:6.1f} s")
    meta = {"generations": gens, "best_generation": best[2], "population": 2 * pairs,
            "envs_per_member": envs, "hard_task_replay": replay_frac, "episode_s": EPISODE_S, "sigma": sigma, "lr": lr,
            "episodes": gens * 2 * pairs * envs, "wall_s": round(time.time() - t0, 1)}
    return best[1], meta


# ---- evaluation (same criterion for the policy and the IK baseline) ----
def seg_point_dist(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v = b - a
    t = np.clip(np.sum((c - a) * v, axis=-1) / np.maximum(np.sum(v * v, axis=-1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[..., None] * v - c, axis=-1)


def touches_obstacle(q: np.ndarray) -> np.ndarray:
    ox, oy, r = OBSTACLE
    c = np.array([ox, oy])
    elbow, tip = fk(q)
    base = np.zeros_like(elbow)
    return np.minimum(seg_point_dist(base, elbow, c), seg_point_dist(elbow, tip, c)) < r + LINK_R


def learned_ctrl(theta: np.ndarray):
    params = unflatten(theta[None])
    return lambda q, qd, tg: policy(params, observe(q, qd, tg)[None])[0]


def evaluate(ctrl, q0: np.ndarray, tg: np.ndarray) -> dict:
    """Run every task for TRIAL_S with the policy. reached = the success criterion in the docstring."""
    q, qd = q0.copy(), np.zeros_like(q0)
    steps = int(round(TRIAL_S / DT))
    n = len(q0)
    reach_t = np.full(n, np.nan)
    contact = touches_obstacle(q)
    for k in range(steps):
        q, qd, _ = step(q, qd, ctrl(q, qd, tg))
        _, tip = fk(q)
        d = np.linalg.norm(tg - tip, axis=-1)
        ok = (d < TOL_M) & (np.linalg.norm(tip_velocity(q, qd), axis=-1) < STOP_MPS)
        reach_t = np.where(np.isnan(reach_t) & ok, (k + 1) * DT, reach_t)
        contact |= touches_obstacle(q)
    final_err = np.linalg.norm(tg - fk(q)[1], axis=-1)
    reached = (reach_t <= REACH_LIMIT_S) & (final_err < TOL_M)
    return {"reached": reached, "time": reach_t, "final_err": final_err, "contact": contact}


def min_jerk_time(d: float) -> float:
    """Mirror of segmentTime() in scara.js: d = largest joint change (rad)."""
    return max(0.15, 1.875 * d / V_MAX, np.sqrt(5.7735 * d / A_MAX))


def ik_baseline(q0: np.ndarray, tg: np.ndarray) -> dict:
    """Analytic IK (elbow +1 preferred, as in the demo) + one straight min-jerk joint move, no obstacle.
    The time is when the same 1 cm / 2 cm/s criterion is first met along that move."""
    times, ok = [], []
    for qs, t in zip(q0, tg):
        qg = ik(t, 1.0)
        if qg is None:
            qg = ik(t, -1.0)
        if qg is None:
            ok.append(False)
            times.append(np.nan)
            continue
        T = min_jerk_time(float(np.max(np.abs(qg - qs))))
        ts = np.arange(1, int(np.ceil(T / DT)) + 1) * DT
        s = np.clip(ts / T, 0, 1)
        sd = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
        sh = s ** 3 * (10 - 15 * s + 6 * s * s)
        qq = qs + sh[:, None] * (qg - qs)
        qv = sd[:, None] * (qg - qs)
        d = np.linalg.norm(t - fk(qq)[1], axis=-1)
        v = np.linalg.norm(tip_velocity(qq, qv), axis=-1)
        hit = np.where((d < TOL_M) & (v < STOP_MPS))[0]
        times.append(ts[hit[0]] if len(hit) else T)
        ok.append(True)
    return {"reached": np.array(ok), "time": np.array(times)}


def eval_set(n: int, seed: int, avoid_obstacle: bool) -> tuple[np.ndarray, np.ndarray]:
    """Held-out tasks. With avoid_obstacle, start pose and target pose both clear of the default obstacle."""
    rng = np.random.default_rng(seed)
    q0s, tgs = [], []
    while len(q0s) < n:
        q0, tg = sample_tasks(rng, 1, min_move_m=0.2)
        if avoid_obstacle:
            goal = ik(tg[0], 1.0)
            if goal is None:
                goal = ik(tg[0], -1.0)
            if touches_obstacle(q0)[0] or goal is None or touches_obstacle(goal[None])[0]:
                continue
        q0s.append(q0[0])
        tgs.append(tg[0])
    return np.array(q0s), np.array(tgs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--gens", type=int, default=700)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--replay-frac", type=float, default=0.25)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.eval_only:
        data = json.loads(args.out.read_text())
        theta = np.concatenate([np.asarray(a, float).ravel() for a in data["weights"]])
        meta = data["training"]
    else:
        log(f"ES on a {N_PARAMS}-parameter MLP {(N_OBS, *HIDDEN, 2)}; seed {args.seed}")
        theta, meta = train(args.seed, args.gens, args.pairs, args.envs, args.sigma, args.lr, args.replay_frac, log)
        meta["seed"] = args.seed
        theta = np.round(theta, 4)  # evaluate exactly what ships

    eval_seed = 2026  # disjoint from the training (seed) and probe (10_000 + g) streams
    q0, tg = eval_set(args.trials, eval_seed, avoid_obstacle=False)
    res = evaluate(learned_ctrl(theta), q0, tg)
    base = ik_baseline(q0, tg)
    oq0, otg = eval_set(args.trials, eval_seed + 1, avoid_obstacle=True)
    ores = evaluate(learned_ctrl(theta), oq0, otg)

    def summary(r: dict) -> dict:
        t = r["time"][r["reached"]]
        return {"success": int(r["reached"].sum()),
                "median_s": round(float(np.median(t)), 2) if len(t) else None,
                "p90_s": round(float(np.percentile(t, 90)), 2) if len(t) else None}

    learned = summary(res)
    learned["median_final_err_mm"] = round(float(np.median(res["final_err"]) * 1000), 1)
    # success by target bearing from the base (deg, x axis = 0): shows where the policy is weak
    bearing = np.degrees(np.arctan2(tg[:, 1], tg[:, 0]))
    learned["by_bearing"] = []
    for lo in (-180, -90, 0, 90):
        m = (bearing >= lo) & (bearing < lo + 90)
        learned["by_bearing"].append({"from_deg": lo, "to_deg": lo + 90, "success": int(res["reached"][m].sum()),
                                      "trials": int(m.sum())})
        log(f"  bearing {lo:4d}..{lo + 90:4d} deg: {int(res['reached'][m].sum())}/{int(m.sum())}")
    baseline = summary(base)
    contact = round(float(ores["contact"].mean()), 3)
    log(f"learned: {learned['success']}/{args.trials} reached, median {learned['median_s']} s, "
        f"p90 {learned['p90_s']} s, median final error {learned['median_final_err_mm']} mm")
    log(f"IK + min-jerk: {baseline['success']}/{args.trials}, median {baseline['median_s']} s, p90 {baseline['p90_s']} s")
    log(f"learned policy touched the default obstacle in {contact * 100:.1f}% of {args.trials} obstacle-clear tasks")

    out = {
        "about": "SCARA reaching policy for assets/js/demos/scara.js. Generated by tools/train_scara.py.",
        "plant": {"l1": L1, "l2": L2, "q_lim": [round(float(v), 6) for v in QLIM], "v_max": V_MAX, "a_max": A_MAX,
                  "servo_tau_s": SERVO_TAU_S, "dt": DT, "obs_coarse_m": OBS_COARSE_M, "obs_fine_m": OBS_FINE_M},
        "arch": [N_OBS, *HIDDEN, 2],
        "params": N_PARAMS,
        "method": "evolution strategies (antithetic, rank-shaped, Adam), numpy",
        "training": meta,
        "eval": {
            "criterion": f"random start pose at rest and a reachable target at least 20 cm away; success = tip "
                         f"within {TOL_M * 100:.0f} cm and slower than {STOP_MPS * 100:.0f} cm/s within "
                         f"{REACH_LIMIT_S:.0f} s, and still within {TOL_M * 100:.0f} cm at {TRIAL_S:.0f} s; "
                         f"time = first moment the criterion is met",
            "trials": args.trials, "seed": eval_seed,
            "learned": learned, "ik_min_jerk": baseline,
            "obstacle": {"trials": args.trials, "seed": eval_seed + 1, "learned_contact_rate": contact,
                         "note": "the policy does not observe the obstacle; tasks whose start or IK goal pose "
                                 "touches it are excluded"},
        },
        "weights": [a.ravel().tolist() for a in unflatten(theta)],
    }
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    log(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
