// 2-link SCARA arm, seen from above: a learned reaching policy and a classical planner,
// overlaid on the same base and sent to the same targets.
// Classical (peach): analytic inverse kinematics (both elbow configurations) with joint
// limits, a configuration-space (C-space) collision map for one circular obstacle, A* in
// C-space when the straight joint-space move would hit it, and minimum-jerk time scaling
// of every segment under joint speed and acceleration limits.
// Learned (lime): a 12-32-32-2 tanh MLP trained offline with evolution strategies in numpy
// (tools/train_scara.py), commanding joint velocities to a velocity-servo plant with the
// same speed/acceleration limits. Its weights and held-out results ship in
// assets/data/scara-policy.json; the plant and observation here mirror that script exactly.
// The policy does not observe the obstacle: its contacts are counted, not prevented.
import { View, Loop, COLORS, clamp } from './common.js?v=20261106i';

export const ARM = {
  base: [0.62, 0.6],               // m, world position of joint 1
  l1: 0.32, l2: 0.26,              // m
  q1: [-160, 160].map((d) => d * Math.PI / 180),
  q2: [-150, 150].map((d) => d * Math.PI / 180),
  vmax: 2.4,                       // rad/s, per joint
  amax: 7.0,                       // rad/s^2, per joint
  linkR: 0.022,                    // m, link half-width used for collision
};
const W = 2.0, H = 1.2;            // world, metres
const GRID = 72;                   // C-space cells per joint axis
const GRID_MARGIN_M = 0.02;        // obstacle inflation for the A* grid, so paths between free cell centres stay clear
const REACH_TOL_M = 0.01;          // same "reached" criterion as tools/train_scara.py:
const REACH_STOP_MPS = 0.02;       // tip within 1 cm and slower than 2 cm/s
const REACH_LIMIT_S = 5;           // ... within 5 s of the target being set
const CS = { x0: 1.32, y0: 0.2, w: 0.62, h: 0.8 }; // C-space inset, world coordinates
const DEFAULT_OBS = { x: 0.98, y: 0.86, r: 0.07 };
const HOME = [0.6, 1.2];

const POLICY_URL = new URL('../../data/scara-policy.json', import.meta.url);
POLICY_URL.search = new URL(import.meta.url).search;

export function fk(q1, q2) {
  const [bx, by] = ARM.base;
  const ex = bx + ARM.l1 * Math.cos(q1), ey = by + ARM.l1 * Math.sin(q1);
  return { elbow: [ex, ey], ee: [ex + ARM.l2 * Math.cos(q1 + q2), ey + ARM.l2 * Math.sin(q1 + q2)] };
}

const inLimits = (q1, q2) => q1 >= ARM.q1[0] && q1 <= ARM.q1[1] && q2 >= ARM.q2[0] && q2 <= ARM.q2[1];

/** IK for the world point (x, y). elbow = +1 (left, q2 > 0) or -1 (right). null if no valid solution. */
export function ik(x, y, elbow) {
  const dx = x - ARM.base[0], dy = y - ARM.base[1];
  const c2 = (dx * dx + dy * dy - ARM.l1 * ARM.l1 - ARM.l2 * ARM.l2) / (2 * ARM.l1 * ARM.l2);
  if (c2 > 1 || c2 < -1) return null;
  const q2 = elbow * Math.acos(c2);
  let q1 = Math.atan2(dy, dx) - Math.atan2(ARM.l2 * Math.sin(q2), ARM.l1 + ARM.l2 * Math.cos(q2));
  q1 = Math.atan2(Math.sin(q1), Math.cos(q1));
  return inLimits(q1, q2) ? [q1, q2] : null;
}

function segCircleDist(ax, ay, bx, by, cx, cy) {
  const vx = bx - ax, vy = by - ay;
  const t = clamp(((cx - ax) * vx + (cy - ay) * vy) / (vx * vx + vy * vy), 0, 1);
  return Math.hypot(ax + t * vx - cx, ay + t * vy - cy);
}

export function collides(q1, q2, obs, margin = 0) {
  if (!obs) return false;
  const { elbow, ee } = fk(q1, q2);
  const r = obs.r + ARM.linkR + margin;
  return segCircleDist(ARM.base[0], ARM.base[1], elbow[0], elbow[1], obs.x, obs.y) < r ||
    segCircleDist(elbow[0], elbow[1], ee[0], ee[1], obs.x, obs.y) < r;
}

const cellOf = (q, lim) => clamp(Math.floor((q - lim[0]) / (lim[1] - lim[0]) * GRID), 0, GRID - 1);
const centreOf = (i, lim) => lim[0] + (i + 0.5) / GRID * (lim[1] - lim[0]);

/**
 * C-space occupancy (Uint8Array GRID*GRID, index i1*GRID+i2): 2 = the cell centre collides,
 * 1 = it is clear but within GRID_MARGIN_M of the obstacle, 0 = free.
 */
export function cspace(obs) {
  const occ = new Uint8Array(GRID * GRID);
  if (!obs) return occ;
  for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
    const a = centreOf(i, ARM.q1), b = centreOf(j, ARM.q2);
    occ[i * GRID + j] = collides(a, b, obs) ? 2 : collides(a, b, obs, GRID_MARGIN_M) ? 1 : 0;
  }
  return occ;
}

/** Exact check of the straight joint-space segment a->b, sampled every 0.5 deg. */
function segmentFree(a, b, obs) {
  const n = Math.max(2, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * Math.PI / 180)));
  for (let k = 0; k <= n; k++) {
    const t = k / n;
    if (collides(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), obs)) return false;
  }
  return true;
}

/** 8-connected A* on the C-space grid, then shortcut smoothing with exact segment checks. */
function astar(q0, qg, obs, occ) {
  const s = cellOf(q0[0], ARM.q1) * GRID + cellOf(q0[1], ARM.q2);
  const g = cellOf(qg[0], ARM.q1) * GRID + cellOf(qg[1], ARM.q2);
  const N = GRID * GRID, cost = new Float64Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1);
  const closed = new Uint8Array(N), open = [s];
  const si = Math.floor(s / GRID), sj = s % GRID, gi = Math.floor(g / GRID), gj = g % GRID;
  const h = (c) => Math.hypot(Math.floor(c / GRID) - gi, (c % GRID) - gj);
  cost[s] = 0;
  const f = new Float64Array(N).fill(Infinity);
  f[s] = h(s);
  while (open.length) {
    let bi = 0;
    for (let i = 1; i < open.length; i++) if (f[open[i]] < f[open[bi]]) bi = i;
    const c = open[bi];
    open[bi] = open[open.length - 1];
    open.pop();
    if (c === g) break;
    if (closed[c]) continue;
    closed[c] = 1;
    const ci = Math.floor(c / GRID), cj = c % GRID;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      if (!di && !dj) continue;
      const ni = ci + di, nj = cj + dj;
      if (ni < 0 || nj < 0 || ni >= GRID || nj >= GRID) continue;
      const n = ni * GRID + nj;
      // margin cells are allowed only next to the start and goal, which may sit close to the obstacle
      const nearEnds = Math.max(Math.abs(ni - si), Math.abs(nj - sj)) <= 2 || Math.max(Math.abs(ni - gi), Math.abs(nj - gj)) <= 2;
      if (closed[n] || (n !== g && (occ[n] === 2 || (occ[n] === 1 && !nearEnds)))) continue;
      const nc = cost[c] + Math.hypot(di, dj);
      if (nc < cost[n]) { cost[n] = nc; prev[n] = c; f[n] = nc + h(n); open.push(n); }
    }
  }
  if (prev[g] === -1 && g !== s) return null;
  const cells = [];
  for (let c = g; c !== -1 && c !== s; c = prev[c]) cells.push(c);
  const via = cells.reverse().slice(0, -1).map((c) => [centreOf(Math.floor(c / GRID), ARM.q1), centreOf(c % GRID, ARM.q2)]);
  const pts = [q0, ...via, qg];
  // shortcut: from each kept point jump to the farthest point reachable in a straight free line
  const out = [pts[0]];
  let i = 0;
  while (i < pts.length - 1) {
    let j = pts.length - 1;
    while (j > i + 1 && !segmentFree(pts[i], pts[j], obs)) j--;
    if (j === i + 1 && !segmentFree(pts[i], pts[j], obs)) return null; // grid cell centre path clipped the obstacle
    out.push(pts[j]);
    i = j;
  }
  return out;
}

/**
 * Plan from joint config q0 to the world point (x, y). Tries the preferred elbow first.
 * Returns { path: [[q1,q2],...], elbow, method } or { error }.
 */
export function plan(q0, x, y, elbowPref, obs, occ = cspace(obs)) {
  const goals = [elbowPref, -elbowPref].map((e) => ({ e, q: ik(x, y, e) })).filter((g) => g.q && !collides(g.q[0], g.q[1], obs));
  if (!goals.length) {
    return { error: ik(x, y, 1) || ik(x, y, -1) ? 'Every arm pose that reaches that point hits the obstacle.' : 'Out of reach.' };
  }
  for (const g of goals) if (segmentFree(q0, g.q, obs)) return { path: [q0, g.q], elbow: g.e, method: 'direct' };
  for (const g of goals) {
    const path = astar(q0, g.q, obs, occ);
    if (path) return { path, elbow: g.e, method: 'astar' };
  }
  return { error: 'No collision-free path in C-space.' };
}

/** Minimum-jerk duration for a joint-space segment under per-joint speed/accel limits. */
export function segmentTime(a, b) {
  const d = Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]));
  // min-jerk peak speed = 1.875 d/T, peak acceleration = 5.7735 d/T^2
  return Math.max(0.15, 1.875 * d / ARM.vmax, Math.sqrt(5.7735 * d / ARM.amax));
}
const minJerk = (t) => t * t * t * (10 - 15 * t + 6 * t * t);

/** Nearest reachable world point to (x, y) (coarse polar search around the base), or null. */
function nearestReachable(x, y) {
  const ang = Math.atan2(y - ARM.base[1], x - ARM.base[0]);
  let best = null;
  for (let k = 0; k <= 400; k++) {
    const rr = (ARM.l1 + ARM.l2) * k / 400;
    for (let m = -30; m <= 30; m++) {
      const a = ang + m * Math.PI / 60;
      const px = ARM.base[0] + rr * 0.999 * Math.cos(a), py = ARM.base[1] + rr * 0.999 * Math.sin(a);
      if (!ik(px, py, 1) && !ik(px, py, -1)) continue;
      const dd = Math.hypot(px - x, py - y);
      if (!best || dd < best.d) best = { x: px, y: py, d: dd };
    }
  }
  return best;
}

export async function loadPolicy() {
  const r = await fetch(POLICY_URL);
  if (!r.ok) throw new Error(`policy fetch failed: HTTP ${r.status}`);
  return r.json();
}

/**
 * Learned arm: velocity-servo plant + MLP policy, mirroring step(), observe() and policy() in
 * tools/train_scara.py. Buffers are preallocated so the 50 Hz step does not allocate.
 */
export class LearnedArm {
  constructor(data) {
    this.data = data;
    this.p = data.plant;
    this.q = new Float64Array(2);
    this.qd = new Float64Array(2);
    this.v = new Float64Array(2);
    this.bufs = data.arch.map((n) => new Float64Array(n));
  }
  reset(q) { this.q.set(q); this.qd.fill(0); this.v.fill(0); }
  /** Target (dx, dy) relative to the base, metres. Runs the policy and one plant step of p.dt. */
  step(tx, ty) {
    const { p, q, qd, v, bufs } = this;
    const { ee } = fk(q[0], q[1]);
    const dx = tx - (ee[0] - ARM.base[0]), dy = ty - (ee[1] - ARM.base[1]);
    const phi = q[0] + q[1], c = Math.cos(phi), s = Math.sin(phi);
    const lx = c * dx + s * dy, ly = -s * dx + c * dy;
    const o = bufs[0];
    o[0] = Math.cos(q[0]); o[1] = Math.sin(q[0]); o[2] = Math.cos(q[1]); o[3] = Math.sin(q[1]);
    o[4] = q[0] / Math.PI; o[5] = q[1] / Math.PI; o[6] = qd[0] / p.v_max; o[7] = qd[1] / p.v_max;
    o[8] = lx / p.obs_coarse_m; o[9] = ly / p.obs_coarse_m;
    o[10] = clamp(lx / p.obs_fine_m, -1, 1); o[11] = clamp(ly / p.obs_fine_m, -1, 1);
    const dims = this.data.arch, wts = this.data.weights;
    for (let l = 0; l < dims.length - 1; l++) {
      const w = wts[2 * l], b = wts[2 * l + 1], hin = bufs[l], hout = bufs[l + 1], nIn = dims[l], nOut = dims[l + 1];
      for (let j = 0; j < nOut; j++) {
        let a = b[j];
        for (let i = 0; i < nIn; i++) a += hin[i] * w[i * nOut + j];
        hout[j] = Math.tanh(a);
      }
    }
    const outp = bufs[dims.length - 1];
    for (let k = 0; k < 2; k++) {
      v[k] = outp[k] * p.v_max;
      const acc = clamp((v[k] - qd[k]) / p.servo_tau_s, -p.a_max, p.a_max);
      qd[k] += acc * p.dt;
      q[k] += qd[k] * p.dt;
      if (Math.abs(q[k]) > p.q_lim[k]) { q[k] = clamp(q[k], -p.q_lim[k], p.q_lim[k]); qd[k] = 0; }
    }
  }
  tipSpeed() {
    const { q, qd } = this, a = q[0], b = q[0] + q[1], w1 = qd[0], w12 = qd[0] + qd[1];
    return Math.hypot(-ARM.l1 * Math.sin(a) * w1 - ARM.l2 * Math.sin(b) * w12, ARM.l1 * Math.cos(a) * w1 + ARM.l2 * Math.cos(b) * w12);
  }
}

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, W, H);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);

  let q = HOME.slice(), elbowPref = 1, target = null, traj = null, msg = '', planInfo = '–';
  let obs = null, occ = null, occImg = null, reachImg = null, drag = null;
  let learned = null, acc = 0;
  // per-arm metrics since the target was set; a = learned policy, b = IK planner
  const mk = () => ({ t: 0, reach: null, path: 0, contacts: 0, touching: false, err: null, lastEE: null, trail: [] });
  const met = { a: mk(), b: mk() };

  function setObstacle(o) {
    obs = o;
    occ = cspace(obs);
    occImg = null;
  }
  function clearMetrics(keepContacts) {
    for (const k of ['a', 'b']) {
      const m = met[k];
      Object.assign(m, { t: 0, reach: null, path: 0, err: null });
      if (!keepContacts) { m.contacts = 0; m.touching = false; m.trail.length = 0; }
    }
  }
  function reset() {
    q = HOME.slice();
    learned?.reset(HOME);
    elbowPref = 1;
    btn('elbow').setAttribute('aria-pressed', 'false');
    btn('elbow').textContent = 'IK elbow: left';
    btn('obstacle').setAttribute('aria-pressed', 'true');
    setObstacle({ ...DEFAULT_OBS });
    target = null; traj = null; planInfo = '–';
    clearMetrics(false);
    msg = 'Tap inside the shaded workspace to set a target for both arms. Drag the obstacle to move it.';
  }

  function goTo(x, y, replan = false) {
    let tx = x, ty = y, note = '';
    if (!ik(x, y, 1) && !ik(x, y, -1)) {
      const n = nearestReachable(x, y);
      if (!n) { msg = 'Out of reach.'; return; }
      tx = n.x; ty = n.y; note = 'Out of reach: aiming at the nearest reachable point. ';
    }
    target = [tx, ty];
    if (!replan) clearMetrics(true);
    const r = plan(q, tx, ty, elbowPref, obs, occ);
    if (r.error) { traj = null; planInfo = 'blocked'; msg = note + 'IK planner: ' + r.error; return; }
    const segs = [];
    let t0 = 0;
    for (let i = 0; i + 1 < r.path.length; i++) {
      const T = segmentTime(r.path[i], r.path[i + 1]);
      segs.push({ a: r.path[i], b: r.path[i + 1], t0, T });
      t0 += T;
    }
    traj = { segs, t: 0, total: t0, path: r.path };
    planInfo = r.method === 'direct' ? 'direct' : `A* · ${r.path.length - 2} via`;
    const sw = r.elbow !== elbowPref ? `Switched to elbow ${r.elbow > 0 ? 'left' : 'right'}: the other pose is outside the joint limits or blocked. ` : '';
    msg = note + 'IK planner: ' + sw + (r.method === 'direct' ? 'straight joint-space move.' : 'the straight move would hit the obstacle, so A* routed around it in C-space.');
  }

  /** Update reach/path/contact metrics of one arm from its current joint pose. */
  function track(m, q1, q2, speed, dt) {
    const { ee } = fk(q1, q2);
    if (m.lastEE) m.path += Math.hypot(ee[0] - m.lastEE[0], ee[1] - m.lastEE[1]);
    if (m.lastEE && speed === null) speed = Math.hypot(ee[0] - m.lastEE[0], ee[1] - m.lastEE[1]) / Math.max(dt, 1e-3);
    m.lastEE = ee;
    m.trail.push(ee[0], ee[1]);
    if (m.trail.length > 240) m.trail.splice(0, 2);
    const touching = !!obs && collides(q1, q2, obs);
    if (touching && !m.touching) m.contacts++;
    m.touching = touching;
    if (!target) return;
    m.t += dt;
    m.err = Math.hypot(ee[0] - target[0], ee[1] - target[1]);
    if (m.reach === null && m.err < REACH_TOL_M && speed !== null && speed < REACH_STOP_MPS) m.reach = m.t;
  }

  function updateIK(dt) {
    if (traj && !(drag && drag.obs && drag.moved)) { // hold still while the obstacle is being moved
      traj.t = Math.min(traj.total, traj.t + dt);
      const sg = traj.segs.find((g) => traj.t <= g.t0 + g.T) || traj.segs[traj.segs.length - 1];
      const s = minJerk(clamp((traj.t - sg.t0) / sg.T, 0, 1));
      q = [sg.a[0] + s * (sg.b[0] - sg.a[0]), sg.a[1] + s * (sg.b[1] - sg.a[1])];
      if (traj.t >= traj.total) traj = null;
    }
    track(met.b, q[0], q[1], null, dt);
  }

  function updateLearned(dt) {
    if (!learned) return;
    const pdt = learned.p.dt;
    acc += dt;
    let n = 0;
    while (acc >= pdt && n < 4) {
      if (target) learned.step(target[0] - ARM.base[0], target[1] - ARM.base[1]);
      track(met.a, learned.q[0], learned.q[1], learned.tipSpeed(), pdt);
      acc -= pdt; n++;
    }
    if (n === 4) acc = 0; // fell behind: drop time rather than spiral
  }

  // reachability shading: per-cell IK test (exact at the cell centre), cached offscreen
  function buildReach() {
    const cw = Math.round(W * 120), chh = Math.round(H * 120);
    const c = document.createElement('canvas');
    c.width = cw; c.height = chh;
    const g = c.getContext('2d'), img = g.createImageData(cw, chh);
    for (let j = 0; j < chh; j++) for (let i = 0; i < cw; i++) {
      const x = (i + 0.5) / 120, y = H - (j + 0.5) / 120;
      if (x > CS.x0 - 0.06) continue;
      const a = ik(x, y, 1), b = ik(x, y, -1), o = 4 * (j * cw + i);
      if (a || b) { img.data[o] = 125; img.data[o + 1] = 112; img.data[o + 2] = 255; img.data[o + 3] = a && b ? 46 : 26; }
    }
    g.putImageData(img, 0, 0);
    return c;
  }
  function buildOcc() {
    const c = document.createElement('canvas');
    c.width = GRID; c.height = GRID;
    const g = c.getContext('2d'), img = g.createImageData(GRID, GRID);
    for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
      const o = 4 * ((GRID - 1 - j) * GRID + i); // x = q1, y = q2 (up)
      const blocked = occ[i * GRID + j] === 2;
      img.data[o] = blocked ? 255 : 40; img.data[o + 1] = blocked ? 107 : 37; img.data[o + 2] = blocked ? 87 : 52; img.data[o + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return c;
  }

  const csx = (q1) => view.sx(CS.x0 + (q1 - ARM.q1[0]) / (ARM.q1[1] - ARM.q1[0]) * CS.w);
  const csy = (q2) => view.sy(CS.y0 + (q2 - ARM.q2[0]) / (ARM.q2[1] - ARM.q2[0]) * CS.h);

  function drawTrail(tr, color) {
    if (tr.length <= 4) return;
    const { ctx } = view;
    ctx.beginPath();
    ctx.moveTo(view.sx(tr[0]), view.sy(tr[1]));
    for (let i = 2; i < tr.length; i += 2) ctx.lineTo(view.sx(tr[i]), view.sy(tr[i + 1]));
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 * view.dpr; ctx.stroke();
  }
  function drawArm(q1, q2, upper, fore, width, hit) {
    const { ctx } = view;
    const { elbow, ee } = fk(q1, q2);
    const b = ARM.base;
    ctx.lineCap = 'round';
    ctx.strokeStyle = hit ? COLORS.coral : upper; ctx.lineWidth = 2 * width * ARM.linkR * view.scale;
    ctx.beginPath(); ctx.moveTo(view.sx(b[0]), view.sy(b[1])); ctx.lineTo(view.sx(elbow[0]), view.sy(elbow[1])); ctx.stroke();
    ctx.strokeStyle = hit ? COLORS.coral : fore; ctx.lineWidth = 1.6 * width * ARM.linkR * view.scale;
    ctx.beginPath(); ctx.moveTo(view.sx(elbow[0]), view.sy(elbow[1])); ctx.lineTo(view.sx(ee[0]), view.sy(ee[1])); ctx.stroke();
    for (const [p, r] of [[elbow, 0.026 * width], [ee, 0.016]]) {
      ctx.beginPath(); ctx.arc(view.sx(p[0]), view.sy(p[1]), r * view.scale, 0, 2 * Math.PI); ctx.fillStyle = COLORS.cream; ctx.fill();
    }
  }

  function draw() {
    const { ctx } = view, d = view.dpr;
    view.clear();
    view.grid(0.1);
    if (!reachImg) reachImg = buildReach();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(reachImg, view.sx(0), view.sy(H), W * view.scale, H * view.scale);
    if (obs) {
      ctx.beginPath();
      ctx.arc(view.sx(obs.x), view.sy(obs.y), obs.r * view.scale, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,107,87,0.85)';
      ctx.fill();
    }
    drawTrail(met.b.trail, 'rgba(255,180,138,0.5)');
    drawTrail(met.a.trail, 'rgba(217,244,91,0.5)');
    if (target) {
      const tx = view.sx(target[0]), ty = view.sy(target[1]);
      ctx.strokeStyle = COLORS.cream; ctx.lineWidth = 2 * d;
      ctx.beginPath(); ctx.arc(tx, ty, 8 * d, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx - 12 * d, ty); ctx.lineTo(tx + 12 * d, ty); ctx.moveTo(tx, ty - 12 * d); ctx.lineTo(tx, ty + 12 * d); ctx.stroke();
    }
    // arms: IK planner (peach, wider) under the learned policy (lime, narrower) so both stay visible
    drawArm(q[0], q[1], 'rgba(255,180,138,0.75)', COLORS.peach, 1, false);
    if (learned) drawArm(learned.q[0], learned.q[1], 'rgba(217,244,91,0.8)', COLORS.lime, 0.62, met.a.touching);
    const b = ARM.base;
    ctx.beginPath(); ctx.arc(view.sx(b[0]), view.sy(b[1]), 0.045 * view.scale, 0, 2 * Math.PI); ctx.fillStyle = COLORS.cream; ctx.fill();
    ctx.fillStyle = COLORS.ink;
    ctx.beginPath(); ctx.arc(view.sx(b[0]), view.sy(b[1]), 0.016 * view.scale, 0, 2 * Math.PI); ctx.fill();
    // C-space inset
    if (!occImg) occImg = buildOcc();
    ctx.imageSmoothingEnabled = false;
    const x0 = csx(ARM.q1[0]), y0 = csy(ARM.q2[1]), w = csx(ARM.q1[1]) - x0, h = csy(ARM.q2[0]) - y0;
    ctx.drawImage(occImg, x0, y0, w, h);
    ctx.strokeStyle = 'rgba(244,239,228,0.4)'; ctx.lineWidth = 1 * d; ctx.strokeRect(x0, y0, w, h);
    ctx.font = `${10 * d}px ui-monospace, Menlo, monospace`;
    ctx.fillStyle = COLORS.creamDim;
    ctx.fillText('C-SPACE  q1 →', x0, y0 + h + 14 * d);
    ctx.fillText('q2 ↑', x0, y0 - 6 * d);
    if (traj) {
      ctx.strokeStyle = COLORS.peach; ctx.lineWidth = 1.5 * d; ctx.beginPath();
      traj.path.forEach((p, i) => (i ? ctx.lineTo(csx(p[0]), csy(p[1])) : ctx.moveTo(csx(p[0]), csy(p[1]))));
      ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(csx(q[0]), csy(q[1]), 4 * d, 0, 2 * Math.PI); ctx.fillStyle = COLORS.peach; ctx.fill();
    if (learned) {
      ctx.beginPath(); ctx.arc(csx(learned.q[0]), csy(learned.q[1]), 3.5 * d, 0, 2 * Math.PI); ctx.fillStyle = COLORS.lime; ctx.fill();
    }
  }

  function updateReadout() {
    for (const k of ['a', 'b']) {
      const m = met[k];
      out(`reach-${k}`).textContent = m.reach !== null ? `${m.reach.toFixed(2)} s`
        : k === 'b' && planInfo === 'blocked' ? 'blocked'
          : target && m.t > REACH_LIMIT_S ? `> ${REACH_LIMIT_S} s` : '–';
      out(`err-${k}`).textContent = m.err === null ? '–' : m.err < 0.1 ? `${(m.err * 1000).toFixed(1)} mm` : `${(m.err * 100).toFixed(0)} cm`;
      out(`path-${k}`).textContent = target ? `${(m.path * 100).toFixed(0)} cm` : '–';
      out(`hit-${k}`).textContent = String(m.contacts);
    }
    out('plan').textContent = planInfo;
    out('msg').textContent = msg;
  }

  /** Names the target sector where the policy did worst offline (sectors are bearings from the base, x axis = 0). */
  function weakest(sectors) {
    if (!sectors || !sectors.length) return '';
    const w = sectors.reduce((a, b) => (b.success / b.trials < a.success / a.trials ? b : a));
    if (w.success === w.trials) return '';
    const where = { '-180': 'down and to the left of', '-90': 'down and to the right of', 0: 'up and to the right of', 90: 'up and to the left of' }[w.from_deg];
    return `It is weakest on targets ${where} the base (${w.success}/${w.trials}), where it would need to swap elbows near a joint limit. `;
  }

  function describe(data) {
    const e = data.eval, tr = data.training, L = e.learned, B = e.ik_min_jerk;
    const pct = (n) => `${Math.round(n / e.trials * 1000) / 10}%`;
    out('policy').textContent =
      `Lime, learned policy: ${data.params} params, trained with evolution strategies for ${tr.episodes.toLocaleString('en')} simulated episodes, seed ${tr.seed}. ` +
      `On ${e.trials} held-out seeded targets it got within 1 cm and stopped in ${pct(L.success)} (median ${L.median_s} s); ` +
      `peach, IK + minimum jerk: ${pct(B.success)} (median ${B.median_s} s). ` +
      weakest(L.by_bearing) +
      `The policy does not see the obstacle and touched it in ${Math.round(e.obstacle.learned_contact_rate * 100)}% of moves, so only the IK planner avoids it.`;
  }

  // pointer: tap = target; drag the obstacle to move it (a move that would overlap the IK arm is refused)
  canvas.addEventListener('pointerdown', (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    const p = view.toWorld(ev);
    const onObs = obs && Math.hypot(p[0] - obs.x, p[1] - obs.y) < obs.r + 0.03;
    drag = { sx: ev.clientX, sy: ev.clientY, moved: false, obs: onObs, off: onObs ? [obs.x - p[0], obs.y - p[1]] : null };
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    if (!drag.moved && Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy) > 8) drag.moved = true;
    if (!drag.moved || !drag.obs) return;
    const p = view.toWorld(ev);
    const cand = { x: clamp(p[0] + drag.off[0], 0.05, CS.x0 - 0.1), y: clamp(p[1] + drag.off[1], 0.05, H - 0.05), r: obs.r };
    if (!collides(q[0], q[1], cand)) { obs.x = cand.x; obs.y = cand.y; }
  });
  const endDrag = (ev) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (d.obs && d.moved) {
      setObstacle(obs);
      if (target) goTo(target[0], target[1], true);
    } else if (!d.moved && ev.type === 'pointerup') {
      const p = view.toWorld(ev);
      if (p[0] < CS.x0 - 0.04) goTo(p[0], p[1]);
    }
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  btn('target').addEventListener('click', () => {
    const here = fk(q[0], q[1]).ee;
    for (let i = 0; i < 200; i++) {
      const x = 0.08 + Math.random() * (CS.x0 - 0.2), y = 0.06 + Math.random() * (H - 0.12);
      const sol = ik(x, y, elbowPref) || ik(x, y, -elbowPref);
      if (sol && !collides(sol[0], sol[1], obs) && Math.hypot(x - here[0], y - here[1]) > 0.2) { goTo(x, y); break; }
    }
  });
  btn('elbow').addEventListener('click', (e) => {
    elbowPref = -elbowPref;
    e.currentTarget.setAttribute('aria-pressed', String(elbowPref < 0));
    e.currentTarget.textContent = `IK elbow: ${elbowPref > 0 ? 'left' : 'right'}`;
    if (target) goTo(target[0], target[1], true);
  });
  btn('obstacle').addEventListener('click', (e) => {
    const on = !obs;
    if (on) {
      if (collides(q[0], q[1], DEFAULT_OBS)) { msg = 'Move the arm away from the obstacle spot first.'; return; }
      setObstacle({ ...DEFAULT_OBS });
    } else setObstacle(null);
    e.currentTarget.setAttribute('aria-pressed', String(on));
  });
  btn('reset').addEventListener('click', reset);

  let uiTimer = 0;
  const loop = new Loop((dt) => {
    updateIK(dt);
    updateLearned(dt);
    draw();
    uiTimer += dt;
    if (uiTimer > 0.12) { updateReadout(); uiTimer = 0; }
  });
  reset();
  draw();
  updateReadout();
  out('policy').textContent = 'Loading policy…';
  loadPolicy().then((data) => {
    learned = new LearnedArm(data);
    learned.reset(q);
    describe(data);
    draw();
  }).catch((err) => {
    console.warn('[demo:scara] policy unavailable:', err);
    out('policy').textContent = 'The learned policy file could not be loaded; only the IK planner is running.';
  });
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
