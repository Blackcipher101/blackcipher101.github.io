// Kalman-filter variants for 2-D robot localisation: linear KF vs EKF vs UKF, plus the seeded
// scenario that feeds all three the same data. No DOM access: used by demos/filters.js in the
// browser and by tools/eval_filters.mjs in Node.
//
// Frames and units: SI. World frame x east, y north, heading theta in rad from +x (CCW positive).
// Robot = unicycle. Inputs per step: measured speed v (m/s) and yaw rate w (rad/s), both noisy.
// Beacon measurement: range r (m) and the ABSOLUTE bearing alpha (rad) from robot to beacon in the
// world frame (a compass-referenced bearing), so every filter sees exactly the same (r, alpha).
import { mulberry32, gaussFrom } from '../common.js?v=20261106q';

export const DT = 0.05;                       // filter / odometry step, s (20 Hz)
export const WORLD_W = 20, WORLD_H = 12.5;    // m
export const LANDMARKS = [[1.5, 6.25], [18.5, 6.25], [10, 1], [10, 11.5]]; // beacon positions, m
export const SENSE_RANGE_M = 7;               // beacons farther than this are not heard, m
export const RUN_S = 120;                     // one run, s
export const BEARING_SIGMAS_DEG = [3, 10, 30];
export const RATES_HZ = [2, 1, 0.5];
export const SIGMA_RANGE_M = 0.25;            // range noise, m
export const SIGMA_V = 0.3;                   // odometry speed noise per sample, m/s
export const SIGMA_W = 0.2;                   // odometry yaw-rate noise per sample, rad/s
const INIT_SIGMA = { pos: 0.3, th: 0.1 };     // initial pose error, m / rad
const LOOP_R = 3, SPEED = 2;                  // two tangent circles of radius 3 m driven at 2 m/s
export const CYCLE_S = (4 * Math.PI * LOOP_R) / SPEED; // one lap of the figure-eight, s
const KF_ACCEL_PSD = 0.9;                     // linear KF white-acceleration PSD, (m/s^2)^2 * s
const UT = { alpha: 1, beta: 2, kappa: 0 };   // scaled unscented transform parameters

export const deg2rad = (d) => (d * Math.PI) / 180;
export const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// ---- tiny dense matrix helpers (row-major arrays of arrays) ----
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = (n) => { const I = zeros(n, n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; };
const diag = (d) => { const D = zeros(d.length, d.length); d.forEach((v, i) => { D[i][i] = v; }); return D; };
const mul = (A, B) => {
  const r = A.length, c = B[0].length, n = B.length, C = zeros(r, c);
  for (let i = 0; i < r; i++) for (let k = 0; k < n; k++) { const a = A[i][k]; if (a) for (let j = 0; j < c; j++) C[i][j] += a * B[k][j]; }
  return C;
};
const tr = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const add = (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j]));
const sub = (A, B) => A.map((row, i) => row.map((v, j) => v - B[i][j]));
const sym = (A) => A.map((row, i) => row.map((v, j) => 0.5 * (v + A[j][i])));
const finite = (A) => A.every((row) => row.every(Number.isFinite));
/** Inverse of a 2x2, or null when singular / non-finite. */
const inv2 = (S) => {
  const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
  if (!(Math.abs(det) > 1e-18) || !Number.isFinite(det)) return null;
  return [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
};
/** Lower Cholesky factor of a symmetric matrix, or null if it is not positive definite. */
function chol(A) {
  const n = A.length, L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 1e-12)) return null;
        L[i][i] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

const warned = new Set();
function warnOnce(key, msg) {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(msg);
}

/** Ground-truth unicycle pose and the exact inputs at time t: two tangent circles (a figure-eight). */
export function truthAt(t) {
  const cx = WORLD_W / 2, cy = WORLD_H / 2, r = LOOP_R;
  const L = 4 * Math.PI * r;
  const s = (SPEED * t) % L;
  if (s < 2 * Math.PI * r) {
    const a = s / r; // CCW on the left circle, starting at the centre heading +y
    return { x: cx - r + r * Math.cos(a), y: cy + r * Math.sin(a), th: wrap(Math.PI / 2 + a), v: SPEED, w: SPEED / r };
  }
  const b = (s - 2 * Math.PI * r) / r; // CW on the right circle
  return { x: cx + r - r * Math.cos(b), y: cy + r * Math.sin(b), th: wrap(Math.PI / 2 - b), v: SPEED, w: -SPEED / r };
}

/** Unicycle motion model with the heading advanced by half the step (midpoint rule). */
function motion(x, y, th, v, w, dt) {
  const phi = th + 0.5 * w * dt;
  return [x + dt * v * Math.cos(phi), y + dt * v * Math.sin(phi), th + w * dt];
}

/** 2x2 position covariance block as [[a, b], [b, c]]. */
function posBlock(P) { return [[P[0][0], P[0][1]], [P[1][0], P[1][1]]]; }

// ---------------------------------------------------------------------------------------------
// 1. Linear KF: constant-velocity model on [x, y, vx, vy], Cartesian position fixes.
//    A linear filter cannot take a range/bearing pair, so each pair is converted to a position fix
//    p = L - r [cos a, sin a] with covariance R = sr^2 u u' + (r sa)^2 n n' (u radial, n tangential),
//    built from the MEASURED r. That first-order covariance and the banana-shaped true error are what
//    the conversion hides. It also ignores the odometry: a linear CV model has no yaw-rate input.
// ---------------------------------------------------------------------------------------------
export class LinearKF {
  constructor(pose0, posSigma) {
    this.name = 'Linear KF';
    this.x = [pose0[0], pose0[1], 0, 0];
    this.P = diag([posSigma ** 2, posSigma ** 2, 1.5 ** 2, 1.5 ** 2]);
    this.resets = 0;
    this.heading = null;
  }
  predict(_v, _w, dt) {
    const F = eye(4); F[0][2] = dt; F[1][3] = dt;
    this.x = mul(F, this.x.map((v) => [v])).map((r) => r[0]);
    const q = KF_ACCEL_PSD, q11 = q * dt ** 3 / 3, q12 = q * dt ** 2 / 2, q22 = q * dt;
    const Q = [[q11, 0, q12, 0], [0, q11, 0, q12], [q12, 0, q22, 0], [0, q12, 0, q22]];
    this.P = sym(add(mul(mul(F, this.P), tr(F)), Q));
    this.guard();
  }
  update(lm, zr, za, sigR, sigA) {
    const c = Math.cos(za), s = Math.sin(za);
    const z = [lm[0] - zr * c, lm[1] - zr * s];
    const rs = zr * sigA;
    const R = [
      [sigR * sigR * c * c + rs * rs * s * s, (sigR * sigR - rs * rs) * c * s],
      [(sigR * sigR - rs * rs) * c * s, sigR * sigR * s * s + rs * rs * c * c],
    ];
    const H = [[1, 0, 0, 0], [0, 1, 0, 0]];
    const PHt = mul(this.P, tr(H));
    const Sinv = inv2(add(mul(H, PHt), R));
    if (!Sinv) return;
    const K = mul(PHt, Sinv);
    const y = [z[0] - this.x[0], z[1] - this.x[1]];
    for (let i = 0; i < 4; i++) this.x[i] += K[i][0] * y[0] + K[i][1] * y[1];
    const IKH = sub(eye(4), mul(K, H));
    this.P = sym(add(mul(mul(IKH, this.P), tr(IKH)), mul(mul(K, R), tr(K)))); // Joseph form
    this.guard();
  }
  guard() {
    if (this.x.every(Number.isFinite) && finite(this.P)) return;
    this.resets++;
    warnOnce('kf', '[filters] linear KF produced a non-finite state; reset to a broad prior');
    this.x = [WORLD_W / 2, WORLD_H / 2, 0, 0];
    this.P = diag([25, 25, 4, 4]);
  }
  pos() { return [this.x[0], this.x[1]]; }
  posCov() { return posBlock(this.P); }
}

// ---------------------------------------------------------------------------------------------
// 2. EKF on [x, y, theta]: analytic Jacobians of the unicycle model and of range / bearing.
//    Process noise enters through the odometry inputs, Q = G diag(sv^2, sw^2) G'.
// ---------------------------------------------------------------------------------------------
export class EKF {
  constructor(pose0, posSigma, thSigma) {
    this.name = 'EKF';
    this.x = pose0.slice();
    this.P0 = diag([posSigma ** 2, posSigma ** 2, thSigma ** 2]);
    this.P = this.P0.map((r) => r.slice());
    this.resets = 0;
  }
  predict(v, w, dt) {
    const [x, y, th] = this.x;
    const phi = th + 0.5 * w * dt, c = Math.cos(phi), s = Math.sin(phi);
    this.x = motion(x, y, th, v, w, dt);
    const F = [[1, 0, -dt * v * s], [0, 1, dt * v * c], [0, 0, 1]];
    const G = [[dt * c, -0.5 * dt * dt * v * s], [dt * s, 0.5 * dt * dt * v * c], [0, dt]];
    const Q = mul(mul(G, diag([SIGMA_V ** 2, SIGMA_W ** 2])), tr(G));
    this.P = sym(add(mul(mul(F, this.P), tr(F)), Q));
    this.x[2] = wrap(this.x[2]);
    this.guard();
  }
  update(lm, zr, za, sigR, sigA) {
    const dx = lm[0] - this.x[0], dy = lm[1] - this.x[1], q = dx * dx + dy * dy, r = Math.sqrt(q);
    if (r < 1e-6) return; // bearing undefined on top of a beacon
    const H = [[-dx / r, -dy / r, 0], [dy / q, -dx / q, 0]];
    const R = diag([sigR * sigR, sigA * sigA]);
    const PHt = mul(this.P, tr(H));
    const Sinv = inv2(add(mul(H, PHt), R));
    if (!Sinv) return;
    const K = mul(PHt, Sinv);
    const y = [zr - r, wrap(za - Math.atan2(dy, dx))]; // bearing innovation wrapped to (-pi, pi]
    for (let i = 0; i < 3; i++) this.x[i] += K[i][0] * y[0] + K[i][1] * y[1];
    this.x[2] = wrap(this.x[2]);
    const IKH = sub(eye(3), mul(K, H));
    this.P = sym(add(mul(mul(IKH, this.P), tr(IKH)), mul(mul(K, R), tr(K)))); // Joseph form
    this.guard();
  }
  guard() {
    if (this.x.every(Number.isFinite) && finite(this.P)) return;
    this.resets++;
    warnOnce('ekf', '[filters] EKF produced a non-finite state; reset to a broad prior');
    this.x = [WORLD_W / 2, WORLD_H / 2, 0];
    this.P = diag([25, 25, 1]);
  }
  pos() { return [this.x[0], this.x[1]]; }
  posCov() { return posBlock(this.P); }
  get heading() { return this.x[2]; }
}

// ---------------------------------------------------------------------------------------------
// 3. UKF on [x, y, theta]: scaled unscented transform. The prediction uses an augmented state
//    [x, y, theta, noise_v, noise_w] so odometry noise is propagated through the motion model. The
//    heading is an angle: the mean of angular sigma points is a circular mean (atan2 of weighted
//    sin and cos) and every angular residual is wrapped.
// ---------------------------------------------------------------------------------------------
function utWeights(n) {
  const { alpha, beta, kappa } = UT;
  const lambda = alpha * alpha * (n + kappa) - n;
  const wm = new Array(2 * n + 1).fill(1 / (2 * (n + lambda)));
  const wc = wm.slice();
  wm[0] = lambda / (n + lambda);
  wc[0] = wm[0] + 1 - alpha * alpha + beta;
  return { wm, wc, scale: Math.sqrt(n + lambda) };
}

/** 2n+1 sigma points of N(mean, P), or null if P is not positive definite. */
function sigmaPoints(mean, P, scale) {
  const L = chol(P);
  if (!L) return null;
  const n = mean.length, pts = [mean.slice()];
  for (let j = 0; j < n; j++) pts.push(mean.map((m, i) => m + scale * L[i][j]));
  for (let j = 0; j < n; j++) pts.push(mean.map((m, i) => m - scale * L[i][j]));
  return pts;
}

/** Weighted mean of vectors; index `angleIdx` (if >= 0) is averaged on the circle. */
function utMean(pts, wm, angleIdx) {
  const n = pts[0].length, m = new Array(n).fill(0);
  let sn = 0, cs = 0;
  pts.forEach((p, k) => {
    for (let i = 0; i < n; i++) m[i] += wm[k] * p[i];
    if (angleIdx >= 0) { sn += wm[k] * Math.sin(p[angleIdx]); cs += wm[k] * Math.cos(p[angleIdx]); }
  });
  if (angleIdx >= 0) m[angleIdx] = Math.atan2(sn, cs);
  return m;
}

const residual = (p, m, angleIdx) => p.map((v, i) => (i === angleIdx ? wrap(v - m[i]) : v - m[i]));

export class UKF {
  constructor(pose0, posSigma, thSigma) {
    this.name = 'UKF';
    this.x = pose0.slice();
    this.P = diag([posSigma ** 2, posSigma ** 2, thSigma ** 2]);
    this.resets = 0;
    this.lastReset = '';
    this.w3 = utWeights(3);
    this.w5 = utWeights(5);
  }
  predict(v, w, dt) {
    const Pa = zeros(5, 5);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Pa[i][j] = this.P[i][j];
    Pa[3][3] = SIGMA_V ** 2; Pa[4][4] = SIGMA_W ** 2;
    const pts = sigmaPoints([...this.x, 0, 0], Pa, this.w5.scale);
    if (!pts) { this.reset('predict: augmented covariance not positive definite'); return; }
    const prop = pts.map((p) => motion(p[0], p[1], p[2], v + p[3], w + p[4], dt));
    const m = utMean(prop, this.w5.wm, 2);
    const P = zeros(3, 3);
    prop.forEach((p, k) => {
      const d = residual(p, m, 2);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) P[i][j] += this.w5.wc[k] * d[i] * d[j];
    });
    this.x = m;
    this.P = sym(P);
    this.check('predict');
  }
  update(lm, zr, za, sigR, sigA) {
    const pts = sigmaPoints(this.x, this.P, this.w3.scale);
    if (!pts) { this.reset('update: covariance not positive definite'); return; }
    const zs = pts.map((p) => [Math.hypot(lm[0] - p[0], lm[1] - p[1]), Math.atan2(lm[1] - p[1], lm[0] - p[0])]);
    const zm = utMean(zs, this.w3.wm, 1);
    const Pzz = diag([sigR * sigR, sigA * sigA]);
    const Pxz = zeros(3, 2);
    pts.forEach((p, k) => {
      const dx = residual(p, this.x, 2), dz = residual(zs[k], zm, 1);
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) Pzz[i][j] += this.w3.wc[k] * dz[i] * dz[j];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) Pxz[i][j] += this.w3.wc[k] * dx[i] * dz[j];
    });
    const Sinv = inv2(Pzz);
    if (!Sinv) { this.reset('update: innovation covariance singular'); return; }
    const K = mul(Pxz, Sinv);
    const y = [zr - zm[0], wrap(za - zm[1])];
    for (let i = 0; i < 3; i++) this.x[i] += K[i][0] * y[0] + K[i][1] * y[1];
    this.x[2] = wrap(this.x[2]);
    this.P = sym(sub(this.P, mul(mul(K, Pzz), tr(K))));
    this.check('update');
  }
  check(where) {
    if (this.x.every(Number.isFinite) && finite(this.P)) return;
    this.reset(`${where}: non-finite state`);
  }
  /** Visible, counted recovery: keep the state if it is finite, inflate the covariance. */
  reset(reason) {
    this.resets++;
    this.lastReset = reason;
    warnOnce('ukf', `[filters] UKF reset (${reason}); covariance re-initialised`);
    if (!this.x.every(Number.isFinite)) this.x = [WORLD_W / 2, WORLD_H / 2, 0];
    this.x[2] = wrap(this.x[2]);
    this.P = diag([1, 1, 0.25]);
  }
  pos() { return [this.x[0], this.x[1]]; }
  posCov() { return posBlock(this.P); }
  get heading() { return this.x[2]; }
}

// ---------------------------------------------------------------------------------------------
// Scenario: one truth, one set of noisy inputs, three filters.
// ---------------------------------------------------------------------------------------------
const newStat = () => ({ n: 0, sse: 0, in2: 0, nees: 0 });

export class Sim {
  /**
   * @param {{seed?: number, sigmaBearingRad: number, rateHz: number, sigmaRangeM?: number}} cfg
   * The noise draws are fixed by `seed`: odometry, measurements and the initial error use separate
   * streams, and a measurement draw is consumed whether or not the beacon is switched off.
   */
  constructor(cfg) {
    this.cfg = { seed: 1, sigmaRangeM: SIGMA_RANGE_M, ...cfg };
    const { seed } = this.cfg;
    this.rngOdo = mulberry32(seed * 3 + 1);
    this.rngMeas = mulberry32(seed * 3 + 2);
    const rngInit = mulberry32(seed * 3 + 3);
    const g = () => gaussFrom(rngInit);
    this.k = 0;
    this.t = 0;
    this.truth = truthAt(0);
    const pose0 = [this.truth.x + INIT_SIGMA.pos * g(), this.truth.y + INIT_SIGMA.pos * g(), wrap(this.truth.th + INIT_SIGMA.th * g())];
    this.kf = new LinearKF(pose0, INIT_SIGMA.pos);
    this.ekf = new EKF(pose0, INIT_SIGMA.pos, INIT_SIGMA.th);
    this.ukf = new UKF(pose0, INIT_SIGMA.pos, INIT_SIGMA.th);
    this.dead = pose0.slice();
    this.filters = { kf: this.kf, ekf: this.ekf, ukf: this.ukf };
    this.stats = { kf: newStat(), ekf: newStat(), ukf: newStat(), dead: newStat(), fix: newStat() };
    this.offUntil = -1;
    this.lastEpoch = { t: -1, used: [] };
    this.period = Math.max(1, Math.round(1 / (this.cfg.rateHz * DT)));
  }

  /** Switch every beacon off for `seconds` of simulated time from now. */
  beaconsOff(seconds) { this.offUntil = this.t + seconds; }
  get beaconsAreOff() { return this.t < this.offUntil; }

  step() {
    const prev = truthAt(this.t + 0.5 * DT);
    this.k++;
    this.t = this.k * DT;
    this.truth = truthAt(this.t);
    const vm = prev.v + SIGMA_V * gaussFrom(this.rngOdo);
    const wm = prev.w + SIGMA_W * gaussFrom(this.rngOdo);
    for (const f of Object.values(this.filters)) f.predict(vm, wm, DT);
    this.dead = motion(this.dead[0], this.dead[1], this.dead[2], vm, wm, DT);
    this.dead[2] = wrap(this.dead[2]);
    if (this.k % this.period === 0) this.measure();
    this.score();
  }

  measure() {
    const { sigmaBearingRad: sigA, sigmaRangeM: sigR } = this.cfg;
    const off = this.beaconsAreOff;
    const used = [];
    LANDMARKS.forEach((lm, i) => {
      const gr = gaussFrom(this.rngMeas), ga = gaussFrom(this.rngMeas);
      const dx = lm[0] - this.truth.x, dy = lm[1] - this.truth.y;
      if (off || Math.hypot(dx, dy) > SENSE_RANGE_M) return;
      const zr = Math.hypot(dx, dy) + sigR * gr;
      const za = wrap(Math.atan2(dy, dx) + sigA * ga);
      for (const f of Object.values(this.filters)) f.update(lm, zr, za, sigR, sigA);
      const e2 = (lm[0] - zr * Math.cos(za) - this.truth.x) ** 2 + (lm[1] - zr * Math.sin(za) - this.truth.y) ** 2;
      this.stats.fix.n++; this.stats.fix.sse += e2;
      used.push(i);
    });
    this.lastEpoch = { t: this.t, used };
  }

  score() {
    const { x, y } = this.truth;
    for (const key of ['kf', 'ekf', 'ukf']) {
      const f = this.filters[key], st = this.stats[key];
      const p = f.pos(), ex = p[0] - x, ey = p[1] - y;
      st.n++; st.sse += ex * ex + ey * ey;
      const Sinv = inv2(f.posCov());
      if (Sinv) {
        const d2 = ex * (Sinv[0][0] * ex + Sinv[0][1] * ey) + ey * (Sinv[1][0] * ex + Sinv[1][1] * ey);
        st.nees += d2;
        if (d2 <= 4) st.in2++; // squared Mahalanobis distance of the 2-sigma ellipse
      }
    }
    const dd = this.stats.dead;
    dd.n++; dd.sse += (this.dead[0] - x) ** 2 + (this.dead[1] - y) ** 2;
  }

  /** Position RMSE (m), share of steps with the truth inside the 2-sigma ellipse, mean position NEES. */
  summary(key) {
    const st = this.stats[key];
    if (!st.n) return { rmse: NaN, in2: NaN, nees: NaN };
    return { rmse: Math.sqrt(st.sse / st.n), in2: st.in2 / st.n, nees: st.nees / st.n };
  }
  /** RMS error of the raw converted position fixes, i.e. the "measurement noise level" in metres. */
  fixRmse() { return this.stats.fix.n ? Math.sqrt(this.stats.fix.sse / this.stats.fix.n) : NaN; }
}
