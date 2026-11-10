// Localisation filters side by side: a linear KF, an EKF and a UKF run on the SAME odometry and the
// SAME noisy beacon range/bearing measurements of a robot driving a figure-eight. The estimators,
// the seeded scenario and the scoring are lib/filters-core.js (no DOM; tools/eval_filters.mjs runs
// it headless). This file only draws and wires the controls. A teaching model, not the estimator
// used on a robot.
import { View, Loop, COLORS, prefersReducedMotion } from './common.js?v=20261106q';
import {
  Sim, DT, WORLD_W, WORLD_H, LANDMARKS, RUN_S, CYCLE_S, SENSE_RANGE_M, BEARING_SIGMAS_DEG, RATES_HZ, truthAt, deg2rad,
} from './lib/filters-core.js?v=20261110a';

const TIME_SCALE = 2;          // sim seconds per wall second
const OFF_S = 5;               // beacon blackout length, s
const HOLD_S = 3;              // pause on the finished run before it restarts, wall s
const TRAIL_S = 20;            // length of the estimate trails, s
const SEED = 7;
const FILTERS = [
  { key: 'kf', label: 'Linear KF', color: COLORS.coral },
  { key: 'ekf', label: 'EKF', color: COLORS.lime },
  { key: 'ukf', label: 'UKF', color: COLORS.violet },
];

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, WORLD_W, WORLD_H);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);
  const noiseBtn = btn('noise'), rateBtn = btn('rate'), offBtn = btn('off');
  const nominal = Array.from({ length: 401 }, (_, i) => { const p = truthAt((i / 400) * CYCLE_S); return [p.x, p.y]; });

  let noiseIdx = 1, rateIdx = 0;
  let sim, trails, acc, hold, lastMsg;

  function restart() {
    sim = new Sim({ seed: SEED, sigmaBearingRad: deg2rad(BEARING_SIGMAS_DEG[noiseIdx]), rateHz: RATES_HZ[rateIdx] });
    trails = { truth: [], kf: [], ekf: [], ukf: [] };
    acc = 0;
    hold = 0;
    lastMsg = '';
    if (prefersReducedMotion()) for (let i = 0; i < 25 / DT; i++) advance();
  }

  function advance() {
    sim.step();
    if (sim.k % 2) return;
    const push = (a, p) => { a.push(p); if (a.length > TRAIL_S / (2 * DT)) a.shift(); };
    push(trails.truth, [sim.truth.x, sim.truth.y]);
    for (const f of FILTERS) push(trails[f.key], sim.filters[f.key].pos());
  }

  function path(points) {
    const { ctx } = view;
    ctx.beginPath();
    points.forEach(([x, y], i) => (i ? ctx.lineTo(view.sx(x), view.sy(y)) : ctx.moveTo(view.sx(x), view.sy(y))));
  }

  function ellipse(pos, P, color) {
    const { ctx } = view;
    const a = P[0][0], b = P[0][1], c = P[1][1];
    const mid = (a + c) / 2, d = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b));
    const l1 = mid + d, l2 = Math.max(0, mid - d);
    const ang = 0.5 * Math.atan2(2 * b, a - c);
    ctx.beginPath();
    ctx.ellipse(view.sx(pos[0]), view.sy(pos[1]), Math.max(2, 2 * Math.sqrt(l1) * view.scale), Math.max(2, 2 * Math.sqrt(l2) * view.scale), -ang, 0, 2 * Math.PI);
    ctx.fillStyle = `${color}22`;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 * view.dpr;
    ctx.stroke();
  }

  function draw() {
    const { ctx } = view;
    const d = view.dpr;
    view.clear();
    view.grid(1);
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    // nominal figure-eight, faint
    path(nominal);
    ctx.setLineDash([4 * d, 6 * d]);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 2 * d;
    ctx.stroke();
    ctx.setLineDash([]);
    // beacons, with the measurement rays of the latest update
    const off = sim.beaconsAreOff;
    const fresh = sim.t - sim.lastEpoch.t < 0.3;
    if (fresh) {
      ctx.strokeStyle = 'rgba(244,239,228,0.28)';
      ctx.lineWidth = 1 * d;
      ctx.beginPath();
      for (const i of sim.lastEpoch.used) {
        ctx.moveTo(view.sx(sim.truth.x), view.sy(sim.truth.y));
        ctx.lineTo(view.sx(LANDMARKS[i][0]), view.sy(LANDMARKS[i][1]));
      }
      ctx.stroke();
    }
    for (const [x, y] of LANDMARKS) {
      const s = 0.3 * view.scale, px = view.sx(x), py = view.sy(y);
      if (!off) {
        ctx.beginPath();
        ctx.arc(px, py, SENSE_RANGE_M * view.scale, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(244,239,228,0.05)';
        ctx.lineWidth = 1 * d;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(px, py - s); ctx.lineTo(px + s, py); ctx.lineTo(px, py + s); ctx.lineTo(px - s, py); ctx.closePath();
      ctx.fillStyle = off ? 'rgba(244,239,228,0.15)' : COLORS.peach;
      ctx.fill();
    }
    // trails: truth under, then the three estimates
    path(trails.truth);
    ctx.strokeStyle = COLORS.cream;
    ctx.lineWidth = 3 * d;
    ctx.stroke();
    for (const f of FILTERS) {
      path(trails[f.key]);
      ctx.strokeStyle = f.color;
      ctx.lineWidth = 1.5 * d;
      ctx.stroke();
    }
    // 2-sigma position ellipses and the current estimates
    for (const f of FILTERS) {
      const filt = sim.filters[f.key];
      ellipse(filt.pos(), filt.posCov(), f.color);
    }
    view.robot(sim.truth.x, sim.truth.y, sim.truth.th, 0.5, null, COLORS.cream);
    for (const f of FILTERS) {
      const p = sim.filters[f.key].pos();
      ctx.beginPath();
      ctx.arc(view.sx(p[0]), view.sy(p[1]), 3.5 * d, 0, 2 * Math.PI);
      ctx.fillStyle = f.color;
      ctx.fill();
    }
    if (off) {
      ctx.fillStyle = COLORS.coral;
      ctx.font = `${12 * d}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.fillText(`beacons off: dead reckoning, ${Math.max(0, sim.offUntil - sim.t).toFixed(0)} s`, view.sx(0.3), view.sy(WORLD_H - 0.6));
    }
    ctx.restore();
  }

  function updateUi() {
    for (const f of FILTERS) {
      const r = sim.summary(f.key);
      const ok = Number.isFinite(r.rmse);
      out(`rmse-${f.key}`).textContent = ok ? `${r.rmse.toFixed(2)} m` : '–';
      out(`cons-${f.key}`).textContent = ok ? `${(100 * r.in2).toFixed(0)}% in 2σ · NEES ${r.nees.toFixed(1)}` : '';
    }
    out('run').textContent = `${Math.min(RUN_S, sim.t).toFixed(0)} / ${RUN_S} s`;
    const resets = sim.ukf.resets;
    out('flag').textContent = resets ? `UKF reset ×${resets}` : sim.beaconsAreOff ? 'beacons off' : 'beacons on';
    noiseBtn.textContent = `Bearing noise: ${BEARING_SIGMAS_DEG[noiseIdx]}°`;
    rateBtn.textContent = `Updates: ${RATES_HZ[rateIdx]} Hz`;
    offBtn.disabled = sim.beaconsAreOff;
    offBtn.textContent = sim.beaconsAreOff ? `Off ${Math.max(0, sim.offUntil - sim.t).toFixed(0)} s` : `Beacons off ${OFF_S} s`;
    const msg = resets
      ? `UKF covariance lost positive-definiteness (${sim.ukf.lastReset}); it was reset ${resets}×.`
      : hold > 0
        ? `Run finished at ${RUN_S} s, restarting. Same seed every time.`
        : '';
    if (msg !== lastMsg) { out('msg').textContent = msg; lastMsg = msg; }
  }

  noiseBtn.addEventListener('click', () => { noiseIdx = (noiseIdx + 1) % BEARING_SIGMAS_DEG.length; restart(); updateUi(); draw(); });
  rateBtn.addEventListener('click', () => { rateIdx = (rateIdx + 1) % RATES_HZ.length; restart(); updateUi(); draw(); });
  offBtn.addEventListener('click', () => { sim.beaconsOff(OFF_S); updateUi(); draw(); });
  btn('reset').addEventListener('click', () => { restart(); updateUi(); draw(); });

  let uiT = 0;
  const loop = new Loop((dt) => {
    if (hold > 0) {
      hold -= dt;
      if (hold <= 0) restart();
    } else {
      acc += dt * TIME_SCALE;
      let n = 0;
      while (acc >= DT && n < 20) { advance(); acc -= DT; n++; }
      if (n === 20) acc = 0;
      if (sim.t >= RUN_S) hold = HOLD_S;
    }
    draw();
    uiT += dt;
    if (uiT > 0.15) { updateUi(); uiT = 0; }
  });

  restart();
  updateUi();
  draw();
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
