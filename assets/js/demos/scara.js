// 2-link SCARA arm, seen from above.
// Analytic inverse kinematics (both elbow configurations) with joint limits, a
// configuration-space (C-space) collision map for one circular obstacle, A* in C-space
// when the straight joint-space move would hit it, and minimum-jerk time scaling of
// every segment under joint speed and acceleration limits. No learned policy here.
import { View, Loop, COLORS, clamp } from './common.js?v=20261106c';

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
const TOL_M = 0.001;               // reach tolerance reported as "reached"
const CS = { x0: 1.32, y0: 0.2, w: 0.62, h: 0.8 }; // C-space inset, world coordinates
const DEFAULT_OBS = { x: 0.98, y: 0.86, r: 0.07 };
const HOME = [0.6, 1.2];

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

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, W, H);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);

  let q = HOME.slice(), elbowPref = 1, target = null, traj = null, msg = '', planInfo = '–', moveT = null;
  let obs = null, occ = null, occImg = null, reachImg = null, lastEE = null, drag = null;
  const trail = [];

  function setObstacle(o) {
    obs = o;
    occ = cspace(obs);
    occImg = null;
  }
  function reset() {
    q = HOME.slice();
    elbowPref = 1;
    btn('elbow').setAttribute('aria-pressed', 'false');
    btn('elbow').textContent = 'Elbow: left';
    btn('obstacle').setAttribute('aria-pressed', 'true');
    setObstacle({ ...DEFAULT_OBS });
    target = null; traj = null; trail.length = 0; planInfo = '–'; moveT = null;
    msg = 'Tap inside the shaded workspace to set a target. Drag the obstacle to move it.';
  }

  function goTo(x, y) {
    let tx = x, ty = y, note = '';
    if (!ik(x, y, 1) && !ik(x, y, -1)) {
      const n = nearestReachable(x, y);
      if (!n) { msg = 'Out of reach.'; return; }
      tx = n.x; ty = n.y; note = 'Out of reach: aiming at the nearest reachable point. ';
    }
    target = [tx, ty];
    const r = plan(q, tx, ty, elbowPref, obs, occ);
    if (r.error) { traj = null; planInfo = 'blocked'; msg = note + r.error; return; }
    const segs = [];
    let t0 = 0;
    for (let i = 0; i + 1 < r.path.length; i++) {
      const T = segmentTime(r.path[i], r.path[i + 1]);
      segs.push({ a: r.path[i], b: r.path[i + 1], t0, T });
      t0 += T;
    }
    traj = { segs, t: 0, total: t0, path: r.path };
    moveT = t0;
    planInfo = r.method === 'direct' ? 'direct' : `A* · ${r.path.length - 2} via`;
    const sw = r.elbow !== elbowPref ? `Switched to elbow ${r.elbow > 0 ? 'left' : 'right'}: the other pose is outside the joint limits or blocked. ` : '';
    msg = note + sw + (r.method === 'direct' ? 'Straight joint-space move.' : 'The straight move would hit the obstacle, so A* routed around it in C-space.');
  }

  function update(dt) {
    if (!traj || (drag && drag.obs && drag.moved)) return; // hold still while the obstacle is being moved
    traj.t = Math.min(traj.total, traj.t + dt);
    const sg = traj.segs.find((g) => traj.t <= g.t0 + g.T) || traj.segs[traj.segs.length - 1];
    const s = minJerk(clamp((traj.t - sg.t0) / sg.T, 0, 1));
    q = [sg.a[0] + s * (sg.b[0] - sg.a[0]), sg.a[1] + s * (sg.b[1] - sg.a[1])];
    const { ee } = fk(q[0], q[1]);
    trail.push(ee[0], ee[1]);
    if (trail.length > 240) trail.splice(0, 2);
    if (traj.t >= traj.total) traj = null;
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
    if (trail.length > 4) {
      ctx.beginPath();
      ctx.moveTo(view.sx(trail[0]), view.sy(trail[1]));
      for (let i = 2; i < trail.length; i += 2) ctx.lineTo(view.sx(trail[i]), view.sy(trail[i + 1]));
      ctx.strokeStyle = 'rgba(217,244,91,0.5)'; ctx.lineWidth = 1.5 * d; ctx.stroke();
    }
    if (target) {
      const tx = view.sx(target[0]), ty = view.sy(target[1]);
      ctx.strokeStyle = COLORS.lime; ctx.lineWidth = 2 * d;
      ctx.beginPath(); ctx.arc(tx, ty, 8 * d, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx - 12 * d, ty); ctx.lineTo(tx + 12 * d, ty); ctx.moveTo(tx, ty - 12 * d); ctx.lineTo(tx, ty + 12 * d); ctx.stroke();
    }
    // arm
    const { elbow, ee } = fk(q[0], q[1]);
    const b = ARM.base;
    ctx.lineCap = 'round';
    ctx.strokeStyle = COLORS.violet; ctx.lineWidth = 2 * ARM.linkR * view.scale;
    ctx.beginPath(); ctx.moveTo(view.sx(b[0]), view.sy(b[1])); ctx.lineTo(view.sx(elbow[0]), view.sy(elbow[1])); ctx.stroke();
    ctx.strokeStyle = COLORS.peach; ctx.lineWidth = 1.6 * ARM.linkR * view.scale;
    ctx.beginPath(); ctx.moveTo(view.sx(elbow[0]), view.sy(elbow[1])); ctx.lineTo(view.sx(ee[0]), view.sy(ee[1])); ctx.stroke();
    for (const [p, r, col] of [[b, 0.045, COLORS.cream], [elbow, 0.03, COLORS.cream], [ee, 0.016, COLORS.lime]]) {
      ctx.beginPath(); ctx.arc(view.sx(p[0]), view.sy(p[1]), r * view.scale, 0, 2 * Math.PI); ctx.fillStyle = col; ctx.fill();
    }
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
      ctx.strokeStyle = COLORS.lime; ctx.lineWidth = 1.5 * d; ctx.beginPath();
      traj.path.forEach((p, i) => (i ? ctx.lineTo(csx(p[0]), csy(p[1])) : ctx.moveTo(csx(p[0]), csy(p[1]))));
      ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(csx(q[0]), csy(q[1]), 4 * d, 0, 2 * Math.PI); ctx.fillStyle = COLORS.cream; ctx.fill();
    lastEE = ee;
  }

  function updateReadout() {
    const err = target && lastEE ? Math.hypot(lastEE[0] - target[0], lastEE[1] - target[1]) : null;
    out('err').textContent = err === null ? '–' : `${(err * 1000).toFixed(1)} mm`;
    out('q').textContent = `${(q[0] * 180 / Math.PI).toFixed(0)}° · ${(q[1] * 180 / Math.PI).toFixed(0)}°`;
    out('plan').textContent = !traj && target && err !== null && err < TOL_M ? `reached · ${planInfo}` : planInfo;
    out('t').textContent = moveT === null ? '–' : `${moveT.toFixed(2)} s`;
    out('msg').textContent = msg;
  }

  // pointer: tap = target; drag the obstacle to move it (a move that would overlap the arm is refused)
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
      if (target) goTo(target[0], target[1]);
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
    e.currentTarget.textContent = `Elbow: ${elbowPref > 0 ? 'left' : 'right'}`;
    if (target) goTo(target[0], target[1]);
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
    update(dt);
    draw();
    uiTimer += dt;
    if (uiTimer > 0.12) { updateReadout(); uiTimer = 0; }
  });
  reset();
  draw();
  updateReadout();
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
