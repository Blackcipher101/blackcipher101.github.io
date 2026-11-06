// Sonar obstacle avoidance for a small ROV, top view.
// A forward-looking sonar fan (33 beams, 5 m, 10 Hz, noisy, with dropouts) updates a
// certainty grid (HIMM: +3 at the echo, -1 along the free beam). Every 0.1 s a Vector
// Field Histogram is built from the grid cells near the ROV, each cell spread over the
// sectors its enlarged footprint covers (the VFH+ refinement), and the ROV steers to the
// free valley that best trades goal direction against turning. Planner only, no learning.
import { View, Loop, COLORS, clamp, mulberry32, gaussFrom } from './common.js?v=20261106p';

export const P = {
  W: 16, H: 10,                 // world, m
  dt: 0.02,                     // sim step, s
  robotR: 0.35,                 // m
  safety: 0.25,                 // m, extra clearance in the VFH+ enlargement
  vmax: 0.9, rmax: 1.4,         // m/s, rad/s
  beams: 33, fan: 130 * Math.PI / 180, range: 5, sonarHz: 10, rangeSigma: 0.03, dropout: 0.03,
  cell: 0.2, window: 2.6,       // m, histogram grid cell, VFH active-window radius
  // threshold: smoothed polar density above which a sector is blocked (tuned in a 10-seed sweep)
  sectors: 72, smooth: 2, threshold: 1400, wideValley: 14,
  goalTol: 0.5, goalTimeout: 45, current: 0.15, // m, s, m/s
};
const GW = Math.round(P.W / P.cell), GH = Math.round(P.H / P.cell);
const SECTOR = 2 * Math.PI / P.sectors;
const wrapA = (a) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
const sectorOf = (a) => ((Math.floor((a < 0 ? a + 2 * Math.PI : a) / SECTOR)) % P.sectors + P.sectors) % P.sectors;
const sdiff = (a, b) => { const d = Math.abs(a - b) % P.sectors; return Math.min(d, P.sectors - d); };

/** Headless simulation core; the demo draws it, the tests run it with a seed. */
export class SonarSim {
  constructor(seed = 1) {
    this.rand = mulberry32(seed);
    this.grid = new Uint8Array(GW * GH);
    this.hist = new Float32Array(P.sectors);
    this.blocked = new Uint8Array(P.sectors);
    this.hits = new Float32Array(P.beams * 3); // x, y, isEcho
    this.currentOn = false;
    this.reset();
  }
  reset() {
    this.robot = { x: 1.2, y: 5, psi: 0, v: 0, r: 0 };
    this.grid.fill(0);
    this.pillars = this.randomPillars(14);
    this.goal = this.randomGoal();
    this.goalT = 0; this.t = 0; this.sonarAcc = 0; this.pause = 0;
    this.prevSector = 0; this.inContact = false; this.cmd = null;
    this.stats = { goals: 0, collisions: 0, stalls: 0, minClear: Infinity };
    this.trail = [];
  }
  randomPillars(n) {
    const out = [];
    for (let tries = 0; out.length < n && tries < 500; tries++) {
      const o = { x: 2.6 + this.rand() * (P.W - 3.6), y: 0.8 + this.rand() * (P.H - 1.6), r: 0.3 + this.rand() * 0.4 };
      if (Math.hypot(o.x - this.robot.x, o.y - this.robot.y) < o.r + 1.5) continue;
      if (out.some((q) => Math.hypot(q.x - o.x, q.y - o.y) < q.r + o.r + 1.1)) continue; // leave ROV-sized gaps
      out.push(o);
    }
    return out;
  }
  /** Free distance between the ROV hull at (x, y) and the nearest pillar or tank wall, m. */
  clearance(x, y) {
    let c = Math.min(x, y, P.W - x, P.H - y) - P.robotR;
    for (const o of this.pillars) c = Math.min(c, Math.hypot(x - o.x, y - o.y) - o.r - P.robotR);
    return c;
  }
  randomGoal() {
    for (let i = 0; i < 300; i++) {
      const g = [0.8 + this.rand() * (P.W - 1.6), 0.8 + this.rand() * (P.H - 1.6)];
      if (this.clearance(g[0], g[1]) > 0.8 && Math.hypot(g[0] - this.robot.x, g[1] - this.robot.y) > 5) return g;
    }
    return [P.W - 1.2, P.H / 2];
  }
  setGoal(g) { this.goal = g; this.goalT = 0; this.pause = 0; }
  /** Range along a ray from (x, y) at angle a against pillars and the tank walls. */
  cast(x, y, a) {
    const dx = Math.cos(a), dy = Math.sin(a);
    const tw = (lim, p, dp) => (dp > 1e-9 ? (lim - p) / dp : dp < -1e-9 ? -p / dp : Infinity);
    let best = Math.min(P.range, tw(P.W, x, dx), tw(P.H, y, dy));
    for (const o of this.pillars) {
      const fx = x - o.x, fy = y - o.y, b = fx * dx + fy * dy, c = fx * fx + fy * fy - o.r * o.r, disc = b * b - c;
      if (disc < 0) continue;
      const t = -b - Math.sqrt(disc);
      if (t > 0 && t < best) best = t;
    }
    return best;
  }
  sonarPing() {
    const { x, y, psi } = this.robot;
    for (let k = 0; k < P.beams; k++) {
      const a = psi - P.fan / 2 + (k / (P.beams - 1)) * P.fan;
      const truth = this.cast(x, y, a);
      const drop = this.rand() < P.dropout;
      const rng = drop ? P.range : clamp(truth + P.rangeSigma * gaussFrom(this.rand), 0.05, P.range);
      const echo = !drop && truth < P.range;
      // HIMM update: free cells along the beam -1, the echo cell +3
      const n = Math.floor(rng / P.cell);
      for (let i = 1; i < n; i++) {
        const gx = Math.floor((x + Math.cos(a) * i * P.cell) / P.cell), gy = Math.floor((y + Math.sin(a) * i * P.cell) / P.cell);
        if (gx >= 0 && gy >= 0 && gx < GW && gy < GH && this.grid[gy * GW + gx] > 0) this.grid[gy * GW + gx]--;
      }
      const hx = x + Math.cos(a) * rng, hy = y + Math.sin(a) * rng;
      if (echo) {
        // clamp so echoes off the tank walls land in the border cells
        const gx = clamp(Math.floor(hx / P.cell), 0, GW - 1), gy = clamp(Math.floor(hy / P.cell), 0, GH - 1);
        this.grid[gy * GW + gx] = Math.min(15, this.grid[gy * GW + gx] + 3);
      }
      this.hits[3 * k] = hx; this.hits[3 * k + 1] = hy; this.hits[3 * k + 2] = echo ? 1 : 0;
    }
  }
  /** Build the polar histogram and pick a heading. Returns { dir, density } or null if boxed in. */
  vfh() {
    const { x, y, psi } = this.robot, h = this.hist;
    const wc = Math.ceil(P.window / P.cell), cx = Math.floor(x / P.cell), cy = Math.floor(y / P.cell);
    const raw = new Float32Array(P.sectors);
    for (let gy = cy - wc; gy <= cy + wc; gy++) for (let gx = cx - wc; gx <= cx + wc; gx++) {
      if (gx < 0 || gy < 0 || gx >= GW || gy >= GH) continue;
      const c = this.grid[gy * GW + gx];
      if (!c) continue;
      const ox = (gx + 0.5) * P.cell - x, oy = (gy + 0.5) * P.cell - y, d = Math.hypot(ox, oy);
      if (d > P.window || d < 1e-6) continue;
      const m = c * c * (P.window - d);
      const centre = Math.atan2(oy, ox);
      const gamma = Math.asin(clamp((P.robotR + P.safety) / d, 0, 1)); // VFH+ enlargement
      const k0 = sectorOf(centre - gamma), span = Math.ceil(2 * gamma / SECTOR);
      for (let i = 0; i <= span; i++) raw[(k0 + i) % P.sectors] += m;
    }
    for (let k = 0; k < P.sectors; k++) {
      let s = 0, wsum = 0;
      for (let i = -P.smooth; i <= P.smooth; i++) { const w = P.smooth + 1 - Math.abs(i); s += w * raw[(k + i + P.sectors) % P.sectors]; wsum += w; }
      h[k] = s / wsum;
      this.blocked[k] = h[k] > P.threshold ? 1 : 0;
    }
    const kt = sectorOf(Math.atan2(this.goal[1] - y, this.goal[0] - x)), kh = sectorOf(psi);
    const start = this.blocked.indexOf(1);
    if (start < 0) return { dir: (kt + 0.5) * SECTOR, density: 0 };
    // valleys = runs of free sectors (circular); VFH+ cost picks a candidate direction
    let best = null;
    const consider = (c) => {
      c = ((Math.round(c) % P.sectors) + P.sectors) % P.sectors;
      const cost = 5 * sdiff(c, kt) + 2 * sdiff(c, kh) + 2 * sdiff(c, this.prevSector);
      if (!best || cost < best.cost) best = { c, cost };
    };
    for (let i = 1, runStart = -1; i <= P.sectors; i++) {
      const k = (start + i) % P.sectors;
      if (!this.blocked[k] && runStart < 0) runStart = i;
      if ((this.blocked[k] || i === P.sectors) && runStart >= 0) {
        const lo = start + runStart, hi = start + (this.blocked[k] ? i - 1 : i), width = hi - lo + 1;
        if (width >= 3) {
          if (width > P.wideValley) {
            consider(lo + P.wideValley / 2);
            consider(hi - P.wideValley / 2);
            const rel = ((kt - lo) % P.sectors + P.sectors) % P.sectors;
            if (rel <= width - 1 - P.wideValley / 2 && rel >= P.wideValley / 2) consider(kt);
          } else consider((lo + hi) / 2);
        }
        runStart = -1;
      }
    }
    if (!best) return null;
    this.prevSector = best.c;
    return { dir: (best.c + 0.5) * SECTOR, density: h[kh] };
  }
  step() {
    const r = this.robot, dt = P.dt;
    this.t += dt;
    this.sonarAcc += dt;
    if (this.sonarAcc >= 1 / P.sonarHz) {
      this.sonarAcc -= 1 / P.sonarHz;
      this.sonarPing();
      this.cmd = this.vfh();
    }
    const dGoal = Math.hypot(this.goal[0] - r.x, this.goal[1] - r.y);
    if (this.pause > 0) {
      this.pause -= dt;
      if (this.pause <= 0) this.setGoal(this.randomGoal());
      r.v = 0; r.r = 0;
    } else if (dGoal < P.goalTol) {
      this.stats.goals++;
      this.pause = 0.6;
    } else if (this.goalT > P.goalTimeout) {
      this.stats.stalls++;
      this.setGoal(this.randomGoal());
    } else if (this.cmd) {
      const err = wrapA(this.cmd.dir - r.psi);
      r.r = clamp(2.2 * err, -P.rmax, P.rmax);
      const slow = 1 - Math.min(this.cmd.density, P.threshold) / P.threshold;
      r.v = P.vmax * Math.max(0.15, slow) * Math.max(0, Math.cos(err)) * Math.min(1, dGoal / 1.2 + 0.3);
    } else {
      r.v = 0; r.r = P.rmax * 0.6; // boxed in: turn in place until a valley opens
    }
    this.goalT += dt;
    r.psi = wrapA(r.psi + r.r * dt);
    r.x += r.v * Math.cos(r.psi) * dt;
    r.y += r.v * Math.sin(r.psi) * dt;
    if (this.currentOn) r.y -= P.current * dt; // unmodelled cross-current
    // contact: count once per contact, then push the ROV back out
    const c = this.clearance(r.x, r.y);
    this.stats.minClear = Math.min(this.stats.minClear, c);
    if (c < 0) {
      if (!this.inContact) this.stats.collisions++;
      this.inContact = true;
      r.x = clamp(r.x, P.robotR, P.W - P.robotR);
      r.y = clamp(r.y, P.robotR, P.H - P.robotR);
      for (const o of this.pillars) {
        const dx = r.x - o.x, dy = r.y - o.y, d = Math.hypot(dx, dy), m = o.r + P.robotR;
        if (d < m && d > 1e-9) { r.x = o.x + dx / d * m; r.y = o.y + dy / d * m; }
      }
    } else this.inContact = false;
    this.trail.push(r.x, r.y);
    if (this.trail.length > 600) this.trail.splice(0, 2);
  }
}

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, P.W, P.H);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);
  const sim = new SonarSim((Math.random() * 2 ** 31) | 0);
  let acc = 0;

  function draw() {
    const { ctx } = view, d = view.dpr, r = sim.robot;
    view.clear('#121a22');
    view.grid(1);
    // certainty grid
    for (let gy = 0; gy < GH; gy++) for (let gx = 0; gx < GW; gx++) {
      const c = sim.grid[gy * GW + gx];
      if (!c) continue;
      ctx.fillStyle = `rgba(125,112,255,${0.12 + 0.05 * c})`;
      ctx.fillRect(view.sx(gx * P.cell), view.sy((gy + 1) * P.cell), P.cell * view.scale + 0.5, P.cell * view.scale + 0.5);
    }
    // pillars (ground truth, outline only: the ROV never sees these directly)
    for (const o of sim.pillars) {
      ctx.beginPath(); ctx.arc(view.sx(o.x), view.sy(o.y), o.r * view.scale, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(244,239,228,0.55)'; ctx.lineWidth = 1.5 * d; ctx.stroke();
    }
    if (sim.trail.length > 4) {
      ctx.beginPath(); ctx.moveTo(view.sx(sim.trail[0]), view.sy(sim.trail[1]));
      for (let i = 2; i < sim.trail.length; i += 2) ctx.lineTo(view.sx(sim.trail[i]), view.sy(sim.trail[i + 1]));
      ctx.strokeStyle = 'rgba(255,180,138,0.45)'; ctx.lineWidth = 1.5 * d; ctx.stroke();
    }
    // sonar fan
    ctx.lineWidth = 1 * d;
    ctx.strokeStyle = 'rgba(217,244,91,0.12)';
    ctx.beginPath();
    for (let k = 0; k < P.beams; k++) { ctx.moveTo(view.sx(r.x), view.sy(r.y)); ctx.lineTo(view.sx(sim.hits[3 * k]), view.sy(sim.hits[3 * k + 1])); }
    ctx.stroke();
    ctx.fillStyle = COLORS.lime;
    for (let k = 0; k < P.beams; k++) if (sim.hits[3 * k + 2]) ctx.fillRect(view.sx(sim.hits[3 * k]) - 2 * d, view.sy(sim.hits[3 * k + 1]) - 2 * d, 4 * d, 4 * d);
    // polar histogram ring
    const R0 = 0.75;
    for (let k = 0; k < P.sectors; k++) {
      const a0 = k * SECTOR, len = Math.min(1.1, sim.hist[k] / P.threshold * 0.55);
      if (len < 0.02) continue;
      ctx.beginPath();
      ctx.arc(view.sx(r.x), view.sy(r.y), (R0 + len) * view.scale, -a0 - SECTOR, -a0);
      ctx.arc(view.sx(r.x), view.sy(r.y), R0 * view.scale, -a0, -a0 - SECTOR, true);
      ctx.closePath();
      ctx.fillStyle = sim.blocked[k] ? 'rgba(255,107,87,0.7)' : 'rgba(244,239,228,0.25)';
      ctx.fill();
    }
    if (sim.cmd) {
      const a = sim.cmd.dir;
      ctx.strokeStyle = COLORS.lime; ctx.lineWidth = 2.5 * d;
      ctx.beginPath(); ctx.moveTo(view.sx(r.x), view.sy(r.y)); ctx.lineTo(view.sx(r.x + 1.4 * Math.cos(a)), view.sy(r.y + 1.4 * Math.sin(a))); ctx.stroke();
    }
    // goal
    const gx = view.sx(sim.goal[0]), gy = view.sy(sim.goal[1]);
    ctx.strokeStyle = COLORS.lime; ctx.lineWidth = 2.5 * d;
    ctx.beginPath(); ctx.arc(gx, gy, 9 * d, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([3 * d, 5 * d]); ctx.strokeStyle = 'rgba(217,244,91,0.35)'; ctx.lineWidth = 1 * d;
    ctx.beginPath(); ctx.moveTo(view.sx(r.x), view.sy(r.y)); ctx.lineTo(gx, gy); ctx.stroke(); ctx.setLineDash([]);
    // ROV body
    ctx.save();
    ctx.translate(view.sx(r.x), view.sy(r.y)); ctx.rotate(-r.psi);
    const s = view.scale;
    ctx.fillStyle = sim.inContact ? COLORS.coral : COLORS.violet;
    ctx.beginPath(); ctx.roundRect(-0.38 * s, -0.24 * s, 0.7 * s, 0.48 * s, 0.08 * s); ctx.fill();
    ctx.fillStyle = COLORS.peach;
    ctx.fillRect(-0.45 * s, -0.3 * s, 0.16 * s, 0.1 * s); ctx.fillRect(-0.45 * s, 0.2 * s, 0.16 * s, 0.1 * s);
    ctx.fillStyle = COLORS.cream;
    ctx.beginPath(); ctx.arc(0.26 * s, 0, 0.07 * s, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
    if (sim.currentOn) {
      ctx.font = `${11 * d}px ui-monospace, Menlo, monospace`; ctx.fillStyle = COLORS.peach;
      ctx.fillText(`CURRENT ${P.current} m/s ↓ (not modelled by the planner)`, 18 * d, 26 * d);
    }
  }

  function updateReadout() {
    out('goals').textContent = String(sim.stats.goals);
    out('hits').textContent = String(sim.stats.collisions);
    out('clear').textContent = Number.isFinite(sim.stats.minClear) ? `${Math.max(0, sim.stats.minClear).toFixed(2)} m` : '–';
    out('v').textContent = `${sim.robot.v.toFixed(2)} m/s`;
  }

  let drag = null;
  canvas.addEventListener('pointerdown', (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    const p = view.toWorld(ev);
    const hit = sim.pillars.find((o) => Math.hypot(o.x - p[0], o.y - p[1]) < o.r + 0.15);
    drag = { sx: ev.clientX, sy: ev.clientY, moved: false, pillar: hit || null, off: hit ? [hit.x - p[0], hit.y - p[1]] : null };
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    if (!drag.moved && Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy) > 8) drag.moved = true;
    if (!drag.moved || !drag.pillar) return;
    const p = view.toWorld(ev);
    const nx = clamp(p[0] + drag.off[0], 0.5, P.W - 0.5), ny = clamp(p[1] + drag.off[1], 0.5, P.H - 0.5);
    if (Math.hypot(nx - sim.robot.x, ny - sim.robot.y) > drag.pillar.r + P.robotR + 0.1) { drag.pillar.x = nx; drag.pillar.y = ny; }
  });
  const endDrag = (ev) => {
    if (!drag) return;
    if (!drag.moved && ev.type === 'pointerup') {
      const p = view.toWorld(ev);
      if (sim.clearance(p[0], p[1]) > 0.1) sim.setGoal([p[0], p[1]]);
    }
    drag = null;
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  btn('shuffle').addEventListener('click', () => { sim.pillars = sim.randomPillars(14); sim.grid.fill(0); sim.setGoal(sim.randomGoal()); });
  btn('current').addEventListener('click', (e) => { sim.currentOn = !sim.currentOn; e.currentTarget.setAttribute('aria-pressed', String(sim.currentOn)); });
  btn('reset').addEventListener('click', () => sim.reset());

  let uiTimer = 0;
  const loop = new Loop((dt) => {
    acc += dt;
    let n = 0;
    while (acc >= P.dt && n < 5) { sim.step(); acc -= P.dt; n++; }
    if (n === 5) acc = 0; // fell behind: drop time rather than spiral
    draw();
    uiTimer += dt;
    if (uiTimer > 0.2) { updateReadout(); uiTimer = 0; }
  });
  sim.sonarPing();
  sim.cmd = sim.vfh();
  draw();
  updateReadout();
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
