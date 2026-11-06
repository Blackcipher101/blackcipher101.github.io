// MPPI (Model Predictive Path Integral) planner playground.
// Each control tick: sample K noisy control sequences around the nominal one, roll
// out a unicycle model, score every rollout, and update the nominal sequence with
// the exponentially weighted average of the noise (Williams et al., 2017).
import { View, Loop, COLORS, gauss, clamp } from './common.js?v=20261106j';

const W = 16, H = 10;               // world, metres
const K = 256;                      // samples
const T = 32;                       // horizon steps
const DT = 0.08;                    // rollout step, s (horizon 2.56 s)
const CONTROL_DT = 0.05;            // controller period, s (20 Hz)
const SIGMA_V = 0.45, SIGMA_W = 1.1;
const V_MIN = -0.2, V_MAX = 1.6;    // m/s
const W_MAX = 2.2;                  // rad/s
const LAMBDA = 0.6;                 // temperature
const GAMMA = 0.06;                 // control-cost weight (fraction-of-lambda style)
const ROBOT_R = 0.3;                // m
const GOAL_TOL_M = 0.25;
const CLEARANCE_M = 0.5;

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, W, H);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);

  const U = new Float32Array(T * 2);              // nominal [v, w] sequence
  const eps = new Float32Array(K * T * 2);        // sampled perturbations
  const cost = new Float64Array(K);
  const traj = new Float32Array(K * (T + 1) * 2); // rollout positions for drawing
  const best = new Float32Array((T + 1) * 2);
  const order = new Uint16Array(K);
  const wts = new Float64Array(K);
  let robot, goal, obstacles, acc = 0, arrived = false, solveMs = 0, minCost = 0;
  let minClearance = Infinity; // closest robot-to-obstacle surface distance since the last goal, m

  function defaultScene() {
    robot = [1.6, 5, 0];
    goal = [14.2, 5];
    obstacles = [
      { x: 6, y: 5.3, r: 1.1 }, { x: 9.5, y: 3.2, r: 0.9 }, { x: 10, y: 7.2, r: 1.0 }, { x: 12.3, y: 5.2, r: 0.6 },
    ];
    U.fill(0);
    arrived = false;
    minClearance = Infinity;
  }

  function stepCost(x, y) {
    let c = 0;
    if (x < ROBOT_R || y < ROBOT_R || x > W - ROBOT_R || y > H - ROBOT_R) c += 200;
    for (let o = 0; o < obstacles.length; o++) {
      const ob = obstacles[o];
      const d = Math.hypot(x - ob.x, y - ob.y) - ob.r - ROBOT_R;
      if (d < 0) c += 200;
      else if (d < CLEARANCE_M) c += 12 * (CLEARANCE_M - d) * (CLEARANCE_M - d);
    }
    return c;
  }

  function iterate() {
    const t0 = performance.now();
    let cmin = Infinity;
    for (let k = 0; k < K; k++) {
      let x = robot[0], y = robot[1], th = robot[2];
      let c = 0;
      const tb = k * (T + 1) * 2;
      traj[tb] = x; traj[tb + 1] = y;
      for (let t = 0; t < T; t++) {
        const e = (k * T + t) * 2;
        // the first sample is the unperturbed nominal (helps when all others are bad)
        const ev = k === 0 ? 0 : SIGMA_V * gauss();
        const ew = k === 0 ? 0 : SIGMA_W * gauss();
        const v = clamp(U[2 * t] + ev, V_MIN, V_MAX);
        const w = clamp(U[2 * t + 1] + ew, -W_MAX, W_MAX);
        // store the effective (post-clamp) perturbation so the update stays feasible
        eps[e] = v - U[2 * t];
        eps[e + 1] = w - U[2 * t + 1];
        th += w * DT;
        x += v * Math.cos(th) * DT;
        y += v * Math.sin(th) * DT;
        traj[tb + 2 * (t + 1)] = x; traj[tb + 2 * (t + 1) + 1] = y;
        const dg = Math.hypot(x - goal[0], y - goal[1]);
        c += (dg + stepCost(x, y) + (v < 0 ? 0.6 : 0)) * DT; // goal distance, obstacles, discourage reversing
        // control cost  gamma * u^T Sigma^-1 eps
        c += GAMMA * (U[2 * t] * eps[e] / (SIGMA_V * SIGMA_V) + U[2 * t + 1] * eps[e + 1] / (SIGMA_W * SIGMA_W)) * DT;
      }
      c += 4 * Math.hypot(x - goal[0], y - goal[1]);
      cost[k] = c;
      if (c < cmin) cmin = c;
    }
    // weights and update
    let wsum = 0;
    for (let k = 0; k < K; k++) { wts[k] = Math.exp(-(cost[k] - cmin) / LAMBDA); wsum += wts[k]; }
    for (let t = 0; t < T; t++) {
      let dv = 0, dw = 0;
      for (let k = 0; k < K; k++) {
        const e = (k * T + t) * 2;
        dv += wts[k] * eps[e];
        dw += wts[k] * eps[e + 1];
      }
      U[2 * t] = clamp(U[2 * t] + dv / wsum, V_MIN, V_MAX);
      U[2 * t + 1] = clamp(U[2 * t + 1] + dw / wsum, -W_MAX, W_MAX);
    }
    // nominal rollout for display
    let x = robot[0], y = robot[1], th = robot[2];
    best[0] = x; best[1] = y;
    for (let t = 0; t < T; t++) {
      th += U[2 * t + 1] * DT;
      x += U[2 * t] * Math.cos(th) * DT;
      y += U[2 * t] * Math.sin(th) * DT;
      best[2 * (t + 1)] = x; best[2 * (t + 1) + 1] = y;
    }
    for (let k = 0; k < K; k++) order[k] = k;
    order.sort((a, b) => cost[b] - cost[a]); // worst first, best drawn on top
    minCost = cmin;
    solveMs = 0.9 * solveMs + 0.1 * (performance.now() - t0);
  }

  function control() {
    const dGoal = Math.hypot(robot[0] - goal[0], robot[1] - goal[1]);
    if (dGoal < GOAL_TOL_M) {
      if (!arrived) { arrived = true; U.fill(0); }
      return;
    }
    arrived = false;
    iterate();
    const v = U[0], w = U[1];
    robot[2] += w * CONTROL_DT;
    robot[0] += v * Math.cos(robot[2]) * CONTROL_DT;
    robot[1] += v * Math.sin(robot[2]) * CONTROL_DT;
    for (const o of obstacles) {
      minClearance = Math.min(minClearance, Math.hypot(robot[0] - o.x, robot[1] - o.y) - o.r - ROBOT_R);
    }
    // receding horizon: shift by the elapsed fraction of a rollout step
    const shift = CONTROL_DT / DT; // 0.625 steps; interpolate
    for (let t = 0; t < T; t++) {
      const f = t + shift;
      const i0 = Math.min(T - 1, Math.floor(f)), i1 = Math.min(T - 1, i0 + 1), a = f - Math.floor(f);
      U[2 * t] = (1 - a) * U[2 * i0] + a * U[2 * i1];
      U[2 * t + 1] = (1 - a) * U[2 * i0 + 1] + a * U[2 * i1 + 1];
    }
  }

  const BUCKETS = [
    { a: 0.0, col: 'rgba(125,112,255,0.10)' },
    { a: 0.5, col: 'rgba(125,112,255,0.22)' },
    { a: 0.8, col: 'rgba(255,180,138,0.35)' },
    { a: 0.95, col: 'rgba(217,244,91,0.75)' },
  ];

  function draw() {
    const { ctx } = view;
    const d = view.dpr;
    view.clear();
    view.grid(1);
    // obstacles
    for (const o of obstacles) {
      ctx.beginPath();
      ctx.arc(view.sx(o.x), view.sy(o.y), o.r * view.scale, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(244,239,228,0.9)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(view.sx(o.x), view.sy(o.y), (o.r + ROBOT_R) * view.scale, 0, 2 * Math.PI);
      ctx.setLineDash([3 * d, 4 * d]);
      ctx.strokeStyle = 'rgba(244,239,228,0.25)';
      ctx.lineWidth = 1 * d;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // rollouts, bucketed by cost rank so we stroke 4 paths instead of 256
    if (!arrived) {
      ctx.lineWidth = 1.2 * d;
      for (let b = 0; b < BUCKETS.length; b++) {
        const lo = Math.floor(BUCKETS[b].a * K);
        const hi = b + 1 < BUCKETS.length ? Math.floor(BUCKETS[b + 1].a * K) : K;
        ctx.beginPath();
        for (let r = lo; r < hi; r++) {
          const tb = order[r] * (T + 1) * 2;
          ctx.moveTo(view.sx(traj[tb]), view.sy(traj[tb + 1]));
          for (let t = 1; t <= T; t++) ctx.lineTo(view.sx(traj[tb + 2 * t]), view.sy(traj[tb + 2 * t + 1]));
        }
        ctx.strokeStyle = BUCKETS[b].col;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(view.sx(best[0]), view.sy(best[1]));
      for (let t = 1; t <= T; t++) ctx.lineTo(view.sx(best[2 * t]), view.sy(best[2 * t + 1]));
      ctx.strokeStyle = COLORS.coral;
      ctx.lineWidth = 3.5 * d;
      ctx.lineCap = 'round';
      ctx.stroke();
    }
    // goal
    const gx = view.sx(goal[0]), gy = view.sy(goal[1]);
    ctx.beginPath();
    ctx.arc(gx, gy, 9 * d, 0, 2 * Math.PI);
    ctx.strokeStyle = COLORS.lime;
    ctx.lineWidth = 2.5 * d;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(gx, gy, 3 * d, 0, 2 * Math.PI);
    ctx.fillStyle = COLORS.lime;
    ctx.fill();
    // robot footprint + heading
    ctx.beginPath();
    ctx.arc(view.sx(robot[0]), view.sy(robot[1]), ROBOT_R * view.scale, 0, 2 * Math.PI);
    ctx.fillStyle = COLORS.violet;
    ctx.fill();
    view.robot(robot[0], robot[1], robot[2], 0.22, COLORS.cream, null);
  }

  function updateReadout() {
    out('ms').textContent = `${solveMs.toFixed(1)} ms`;
    out('cost').textContent = arrived ? '–' : minCost.toFixed(1);
    out('clear').textContent = Number.isFinite(minClearance) ? `${minClearance.toFixed(2)} m` : '–';
    out('v').textContent = `${(arrived ? 0 : U[0]).toFixed(2)} m/s`;
    out('msg').textContent = arrived ? 'Arrived. Tap the map to set a new goal.' : 'Tap to set the goal. Drag on empty space to draw an obstacle, drag one to move it.';
  }

  // pointer interaction
  let drag = null;
  canvas.addEventListener('pointerdown', (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    const p = view.toWorld(ev);
    const hit = obstacles.find((o) => Math.hypot(o.x - p[0], o.y - p[1]) < o.r);
    drag = { start: p, sx: ev.clientX, sy: ev.clientY, moved: false, obstacle: hit || null, offset: hit ? [hit.x - p[0], hit.y - p[1]] : null, created: null };
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const p = view.toWorld(ev);
    if (!drag.moved && Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy) > 8) {
      drag.moved = true;
      if (!drag.obstacle && obstacles.length < 14) {
        drag.created = { x: drag.start[0], y: drag.start[1], r: 0.3 };
        obstacles.push(drag.created);
      }
    }
    if (!drag.moved) return;
    if (drag.obstacle) {
      drag.obstacle.x = clamp(p[0] + drag.offset[0], 0, W);
      drag.obstacle.y = clamp(p[1] + drag.offset[1], 0, H);
    } else if (drag.created) {
      drag.created.r = clamp(Math.hypot(p[0] - drag.start[0], p[1] - drag.start[1]), 0.3, 2.2);
    }
    view.dirty = true;
  });
  const endDrag = (ev) => {
    if (!drag) return;
    if (!drag.moved && ev.type === 'pointerup') {
      const p = view.toWorld(ev);
      goal = [clamp(p[0], 0.4, W - 0.4), clamp(p[1], 0.4, H - 0.4)];
      arrived = false;
      minClearance = Infinity;
    }
    drag = null;
    view.dirty = true;
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  // keyboard-reachable equivalents of the pointer actions
  btn('goal').addEventListener('click', () => {
    for (let i = 0; i < 50; i++) {
      const g = [1 + Math.random() * (W - 2), 1 + Math.random() * (H - 2)];
      if (stepCost(g[0], g[1]) === 0 && Math.hypot(g[0] - robot[0], g[1] - robot[1]) > 4) { goal = g; break; }
    }
    arrived = false;
    minClearance = Infinity;
  });
  btn('scatter').addEventListener('click', () => {
    obstacles = [];
    for (let i = 0; i < 7; i++) {
      const o = { x: 2 + Math.random() * (W - 4), y: 1 + Math.random() * (H - 2), r: 0.4 + Math.random() * 0.8 };
      if (Math.hypot(o.x - robot[0], o.y - robot[1]) > o.r + 1 && Math.hypot(o.x - goal[0], o.y - goal[1]) > o.r + 1) obstacles.push(o);
    }
    arrived = false;
    minClearance = Infinity;
  });
  btn('reset').addEventListener('click', defaultScene);

  let uiTimer = 0;
  const loop = new Loop((dt) => {
    acc += dt;
    let steps = 0;
    while (acc >= CONTROL_DT && steps < 3) { control(); acc -= CONTROL_DT; steps++; }
    if (steps === 3) acc = 0; // fell behind: drop time rather than spiral
    draw();
    uiTimer += dt;
    if (uiTimer > 0.2) { updateReadout(); uiTimer = 0; }
  });

  defaultScene();
  iterate();
  draw();
  updateReadout();
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
