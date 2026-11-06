// Underwater dead-reckoning toy: why a DVL matters when the camera goes dark.
// A 6-state linear Kalman filter [x, y, vx, vy, bax, bay] (world position/velocity,
// body-frame accelerometer bias) predicts with a biased, noisy accelerometer and is
// corrected by velocity measurements from the camera (visual odometry) and/or a DVL.
// Neither sensor observes absolute position, so position error always grows; how fast
// depends on which velocity source is alive. This is a teaching model, not the
// estimator in uw-vio-dvl.
import { View, Loop, COLORS, gauss, clamp } from './common.js?v=20261106l';

const W = 30, H = 18;                 // world, metres
const SIM_DT = 0.02;                  // filter / sim step, s
const TIME_SCALE = 2;                 // sim seconds per wall second
const SPEED_MPS = 0.8;
const ACC_NOISE = 0.05;               // accelerometer white noise, m/s^2
const BIAS_RW = 0.003;                // accelerometer bias random walk, m/s^2/sqrt(s)
const CAM_SIGMA = 0.06, CAM_RATE_HZ = 10;   // visual odometry velocity, m/s
const DVL_SIGMA = 0.02, DVL_RATE_HZ = 5;    // DVL body velocity, m/s
const BLACKOUT_S = 20;
const HISTORY_S = 90;

const WAYPOINTS = (() => {
  const pts = [];
  [3, 7, 11, 15].forEach((y, i) => {
    if (i % 2 === 0) pts.push([3, y], [27, y]);
    else pts.push([27, y], [3, y]);
  });
  return pts;
})();

// --- tiny dense matrix helpers (row-major arrays of arrays) ---
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const mul = (A, B) => {
  const r = A.length, c = B[0].length, n = B.length, C = zeros(r, c);
  for (let i = 0; i < r; i++) for (let k = 0; k < n; k++) { const a = A[i][k]; if (a) for (let j = 0; j < c; j++) C[i][j] += a * B[k][j]; }
  return C;
};
const tr = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const inv2 = (S) => {
  const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
  return [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
};

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, W, H + 4); // bottom 4 m of canvas = error strip chart
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);
  const camBtn = btn('camera'), dvlBtn = btn('dvl'), blackBtn = btn('blackout');

  let s;
  function reset() {
    const ang = Math.random() * 2 * Math.PI;
    s = {
      t: 0,
      truth: { x: 3, y: 3, th: 0, vx: SPEED_MPS, vy: 0 },
      wp: 1,
      bias: [0.03 * Math.cos(ang), 0.03 * Math.sin(ang)], // true accelerometer bias, body frame
      headErr: 0,
      x: [3, 3, SPEED_MPS, 0, 0, 0],
      P: [[0.01, 0, 0, 0, 0, 0], [0, 0.01, 0, 0, 0, 0], [0, 0, 0.01, 0, 0, 0], [0, 0, 0, 0.01, 0, 0], [0, 0, 0, 0, 0.0025, 0], [0, 0, 0, 0, 0, 0.0025]],
      camera: s ? s.camera : true,
      dvl: s ? s.dvl : false,
      blackoutLeft: 0,
      camT: 0, dvlT: 0,
      trueTrail: [[3, 3]],
      estTrail: [[3, 3, 0]],             // x, y, dark(1 if camera off)
      errHist: [],
      maxBlackoutErr: 0,
      acc: 0,
    };
    updateUi();
  }

  function stepTruth(dt) {
    const tr_ = s.truth;
    let target = WAYPOINTS[s.wp];
    if (Math.hypot(target[0] - tr_.x, target[1] - tr_.y) < 1.2) {
      s.wp = (s.wp + 1) % WAYPOINTS.length;
      target = WAYPOINTS[s.wp];
    }
    const err = Math.atan2(target[1] - tr_.y, target[0] - tr_.x) - tr_.th;
    const e = Math.atan2(Math.sin(err), Math.cos(err));
    const w = clamp(1.5 * e, -0.7, 0.7);
    tr_.th += w * dt;
    const vx = SPEED_MPS * Math.cos(tr_.th), vy = SPEED_MPS * Math.sin(tr_.th);
    const ax = (vx - tr_.vx) / dt, ay = (vy - tr_.vy) / dt;
    tr_.x += 0.5 * (vx + tr_.vx) * dt;
    tr_.y += 0.5 * (vy + tr_.vy) * dt;
    tr_.vx = vx; tr_.vy = vy;
    return [ax, ay];
  }

  function predict(aWorldTrue, dt) {
    // simulated IMU: body-frame specific force + bias + noise; heading from a slowly drifting gyro
    s.bias[0] += BIAS_RW * Math.sqrt(dt) * gauss();
    s.bias[1] += BIAS_RW * Math.sqrt(dt) * gauss();
    s.headErr += 0.002 * Math.sqrt(dt) * gauss();
    const th = s.truth.th, c = Math.cos(th), sn = Math.sin(th);
    const amx = c * aWorldTrue[0] + sn * aWorldTrue[1] + s.bias[0] + ACC_NOISE * gauss();
    const amy = -sn * aWorldTrue[0] + c * aWorldTrue[1] + s.bias[1] + ACC_NOISE * gauss();
    const the = th + s.headErr, ce = Math.cos(the), se = Math.sin(the);
    const x = s.x;
    const bx = amx - x[4], by = amy - x[5];
    const awx = ce * bx - se * by, awy = se * bx + ce * by;
    x[0] += x[2] * dt + 0.5 * awx * dt * dt;
    x[1] += x[3] * dt + 0.5 * awy * dt * dt;
    x[2] += awx * dt;
    x[3] += awy * dt;
    // F = d(x_next)/d(x)
    const F = zeros(6, 6);
    for (let i = 0; i < 6; i++) F[i][i] = 1;
    F[0][2] = dt; F[1][3] = dt;
    const h = 0.5 * dt * dt;
    F[0][4] = -h * ce; F[0][5] = h * se; F[1][4] = -h * se; F[1][5] = -h * ce;
    F[2][4] = -dt * ce; F[2][5] = dt * se; F[3][4] = -dt * se; F[3][5] = -dt * ce;
    const P = mul(mul(F, s.P), tr(F));
    // discrete process noise: per-sample accelerometer noise enters as G = [h, dt], bias as a random walk
    const sa2 = ACC_NOISE * ACC_NOISE, qb = BIAS_RW * BIAS_RW * dt;
    for (const [p, v] of [[0, 2], [1, 3]]) {
      P[p][p] += sa2 * h * h; P[v][v] += sa2 * dt * dt;
      P[p][v] += sa2 * h * dt; P[v][p] += sa2 * h * dt;
    }
    P[4][4] += qb; P[5][5] += qb;
    s.P = P;
    return the;
  }

  // velocity measurement z = R(theta)^T v + noise (body frame)
  function velocityUpdate(the, sigma) {
    const c = Math.cos(the), sn = Math.sin(the);
    const tv = s.truth;
    const thTrue = tv.th, ct = Math.cos(thTrue), st = Math.sin(thTrue);
    const z = [ct * tv.vx + st * tv.vy + sigma * gauss(), -st * tv.vx + ct * tv.vy + sigma * gauss()];
    const Hm = [[0, 0, c, sn, 0, 0], [0, 0, -sn, c, 0, 0]];
    const x = s.x;
    const zh = [c * x[2] + sn * x[3], -sn * x[2] + c * x[3]];
    const PHt = mul(s.P, tr(Hm));
    const S = mul(Hm, PHt);
    S[0][0] += sigma * sigma; S[1][1] += sigma * sigma;
    const Kg = mul(PHt, inv2(S));
    const y = [z[0] - zh[0], z[1] - zh[1]];
    for (let i = 0; i < 6; i++) x[i] += Kg[i][0] * y[0] + Kg[i][1] * y[1];
    // Joseph form keeps P symmetric positive semi-definite
    const IKH = mul(Kg, Hm).map((row, i) => row.map((v, j) => (i === j ? 1 : 0) - v));
    const KRKt = mul(Kg, tr(Kg)).map((row) => row.map((v) => v * sigma * sigma));
    s.P = mul(mul(IKH, s.P), tr(IKH)).map((row, i) => row.map((v, j) => v + KRKt[i][j]));
  }

  function simStep(dt) {
    const a = stepTruth(dt);
    const the = predict(a, dt);
    s.t += dt;
    if (s.blackoutLeft > 0) s.blackoutLeft = Math.max(0, s.blackoutLeft - dt);
    const camAlive = s.camera && s.blackoutLeft === 0;
    s.camT += dt; s.dvlT += dt;
    if (camAlive && s.camT >= 1 / CAM_RATE_HZ) { velocityUpdate(the, CAM_SIGMA); s.camT = 0; }
    if (s.dvl && s.dvlT >= 1 / DVL_RATE_HZ) { velocityUpdate(the, DVL_SIGMA); s.dvlT = 0; }
    const err = Math.hypot(s.x[0] - s.truth.x, s.x[1] - s.truth.y);
    if (!camAlive) s.maxBlackoutErr = Math.max(s.maxBlackoutErr, err);
    const lt = s.trueTrail[s.trueTrail.length - 1];
    if (Math.hypot(lt[0] - s.truth.x, lt[1] - s.truth.y) > 0.15) {
      s.trueTrail.push([s.truth.x, s.truth.y]);
      s.estTrail.push([s.x[0], s.x[1], camAlive ? 0 : 1]);
      if (s.trueTrail.length > 1500) { s.trueTrail.shift(); s.estTrail.shift(); }
    }
    const hh = s.errHist;
    if (!hh.length || s.t - hh[hh.length - 1][0] > 0.25) {
      hh.push([s.t, err, camAlive ? 0 : 1]);
      while (hh.length && s.t - hh[0][0] > HISTORY_S) hh.shift();
    }
  }

  function draw() {
    const { ctx } = view;
    const d = view.dpr;
    view.clear();
    // survey area (shift world up by 4 m for the chart strip)
    const oy = 4;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath();
    for (let x = 0; x <= W; x += 2) { ctx.moveTo(view.sx(x), view.sy(oy)); ctx.lineTo(view.sx(x), view.sy(H + oy)); }
    for (let y = 0; y <= H; y += 2) { ctx.moveTo(view.sx(0), view.sy(y + oy)); ctx.lineTo(view.sx(W), view.sy(y + oy)); }
    ctx.stroke();
    ctx.strokeStyle = COLORS.gridStrong;
    ctx.strokeRect(view.sx(0), view.sy(H + oy), W * view.scale, H * view.scale);
    // true path
    ctx.beginPath();
    s.trueTrail.forEach(([x, y], i) => (i ? ctx.lineTo(view.sx(x), view.sy(y + oy)) : ctx.moveTo(view.sx(x), view.sy(y + oy))));
    ctx.strokeStyle = COLORS.cream;
    ctx.lineWidth = 2 * d;
    ctx.stroke();
    // estimated path, thick where the camera was dark
    for (const dark of [0, 1]) {
      ctx.beginPath();
      for (let i = 1; i < s.estTrail.length; i++) {
        const p0 = s.estTrail[i - 1], p1 = s.estTrail[i];
        if (p1[2] !== dark) continue;
        ctx.moveTo(view.sx(p0[0]), view.sy(p0[1] + oy));
        ctx.lineTo(view.sx(p1[0]), view.sy(p1[1] + oy));
      }
      ctx.strokeStyle = dark ? 'rgba(255,107,87,0.95)' : 'rgba(255,180,138,0.9)';
      ctx.lineWidth = (dark ? 5 : 2) * d;
      ctx.lineCap = 'round';
      ctx.stroke();
    }
    // 2-sigma position ellipse
    const a = s.P[0][0], b = s.P[0][1], c = s.P[1][1];
    const tr2 = (a + c) / 2, det = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b));
    const l1 = tr2 + det, l2 = Math.max(0, tr2 - det);
    const ang = 0.5 * Math.atan2(2 * b, a - c);
    ctx.beginPath();
    ctx.ellipse(view.sx(s.x[0]), view.sy(s.x[1] + oy), Math.max(2, 2 * Math.sqrt(l1) * view.scale), Math.max(2, 2 * Math.sqrt(l2) * view.scale), -ang, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(125,112,255,0.22)';
    ctx.fill();
    ctx.strokeStyle = COLORS.violet;
    ctx.lineWidth = 1.5 * d;
    ctx.stroke();
    view.robot(s.truth.x, s.truth.y + oy, s.truth.th, 0.7, null, COLORS.cream);
    view.robot(s.x[0], s.x[1] + oy, s.truth.th + s.headErr, 0.7, COLORS.coral, null);
    // error strip chart (bottom 4 m): last HISTORY_S seconds, auto-scaled
    const hh = s.errHist;
    if (hh.length > 1) {
      const maxE = Math.max(1, ...hh.map((h) => h[1]));
      const t0 = s.t - HISTORY_S;
      const px = (t) => view.sx(((t - t0) / HISTORY_S) * W);
      const py = (e) => view.sy(0.4 + (e / maxE) * 2.8);
      ctx.fillStyle = 'rgba(255,107,87,0.12)';
      for (const h of hh) if (h[2]) ctx.fillRect(px(h[0]) - 1, view.sy(3.4), 3 * d, 3 * view.scale);
      ctx.beginPath();
      hh.forEach((h, i) => (i ? ctx.lineTo(px(h[0]), py(h[1])) : ctx.moveTo(px(h[0]), py(h[1]))));
      ctx.strokeStyle = COLORS.lime;
      ctx.lineWidth = 1.5 * d;
      ctx.stroke();
      ctx.fillStyle = COLORS.creamDim;
      ctx.font = `${11 * d}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.fillText(`position error (0–${maxE.toFixed(1)} m), last ${HISTORY_S} s`, view.sx(0.3), view.sy(3.55));
    }
    ctx.restore();
  }

  function updateUi() {
    const err = Math.hypot(s.x[0] - s.truth.x, s.x[1] - s.truth.y);
    out('err').textContent = `${err.toFixed(2)} m`;
    out('sig').textContent = `${(2 * Math.sqrt(Math.max(s.P[0][0], s.P[1][1]))).toFixed(2)} m`;
    out('max').textContent = `${s.maxBlackoutErr.toFixed(2)} m`;
    out('t').textContent = `${s.t.toFixed(0)} s`;
    camBtn.setAttribute('aria-pressed', String(s.camera));
    dvlBtn.setAttribute('aria-pressed', String(s.dvl));
    blackBtn.disabled = s.blackoutLeft > 0 || !s.camera;
    blackBtn.textContent = s.blackoutLeft > 0 ? `Dark ${s.blackoutLeft.toFixed(0)} s` : `${BLACKOUT_S} s blackout`;
  }

  camBtn.addEventListener('click', () => { s.camera = !s.camera; s.blackoutLeft = 0; updateUi(); });
  dvlBtn.addEventListener('click', () => { s.dvl = !s.dvl; updateUi(); });
  blackBtn.addEventListener('click', () => { s.blackoutLeft = BLACKOUT_S; s.maxBlackoutErr = 0; updateUi(); });
  btn('reset').addEventListener('click', () => { reset(); draw(); });

  let uiT = 0;
  const loop = new Loop((dt) => {
    s.acc += dt * TIME_SCALE;
    let n = 0;
    while (s.acc >= SIM_DT && n < 20) { simStep(SIM_DT); s.acc -= SIM_DT; n++; }
    if (n === 20) s.acc = 0;
    draw();
    uiT += dt;
    if (uiT > 0.15) { updateUi(); uiT = 0; }
  });

  reset();
  draw();
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
