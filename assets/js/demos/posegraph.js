// 2D pose-graph optimisation (SE(2)), Gauss-Newton with Levenberg-Marquardt damping.
// Pure math, no DOM: importable from the browser demo and from Node tests.
//
// Poses are flat Float64Array [x0, y0, th0, x1, y1, th1, ...] (metres, radians).
// An edge {i, j, z:[dx, dy, dth], info:[ixx, iyy, ith]} measures pose j expressed
// in the frame of pose i. Information is diagonal (inverse variances).
// Error follows Grisetti et al., "A Tutorial on Graph-Based SLAM" (2010):
//   e_xy = Rz^T (Ri^T (tj - ti) - tz),   e_th = wrap(thj - thi - thz)

export const wrapAngle = (a) => {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
};

/** Pose of b expressed in the frame of a (a^-1 * b). */
export function relativePose(ax, ay, ath, bx, by, bth) {
  const c = Math.cos(ath), s = Math.sin(ath);
  const dx = bx - ax, dy = by - ay;
  return [c * dx + s * dy, -s * dx + c * dy, wrapAngle(bth - ath)];
}

/** Compose a * d where d is expressed in a's frame. */
export function composePose(ax, ay, ath, dx, dy, dth) {
  const c = Math.cos(ath), s = Math.sin(ath);
  return [ax + c * dx - s * dy, ay + s * dx + c * dy, wrapAngle(ath + dth)];
}

function edgeError(p, e, out) {
  const i3 = 3 * e.i, j3 = 3 * e.j;
  const r = relativePose(p[i3], p[i3 + 1], p[i3 + 2], p[j3], p[j3 + 1], p[j3 + 2]);
  const cz = Math.cos(e.z[2]), sz = Math.sin(e.z[2]);
  const ex = r[0] - e.z[0], ey = r[1] - e.z[1];
  out[0] = cz * ex + sz * ey;
  out[1] = -sz * ex + cz * ey;
  out[2] = wrapAngle(r[2] - e.z[2]);
  return out;
}

/** Weighted squared error sum (chi^2) of the graph. */
export function chi2(poses, edges) {
  const err = [0, 0, 0];
  let total = 0;
  for (const e of edges) {
    edgeError(poses, e, err);
    total += e.info[0] * err[0] * err[0] + e.info[1] * err[1] * err[1] + e.info[2] * err[2] * err[2];
  }
  return total;
}

/** In-place dense Cholesky solve of (A) x = b, A symmetric positive definite (n x n, row major). */
export function choleskySolve(A, b, n) {
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j];
    for (let k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (!(d > 0) || !Number.isFinite(d)) return null; // not positive definite / NaN
    d = Math.sqrt(d);
    A[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      const ri = i * n, rj = j * n;
      for (let k = 0; k < j; k++) s -= A[ri + k] * A[rj + k];
      A[i * n + j] = s / d;
    }
  }
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    let s = x[i];
    for (let k = 0; k < i; k++) s -= A[i * n + k] * x[k];
    x[i] = s / A[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= A[k * n + i] * x[k];
    x[i] = s / A[i * n + i];
  }
  return x;
}

/**
 * Build the normal equations H dx = -b around `poses`. Pose 0 is the gauge
 * (held fixed), so the system covers poses 1..N-1 only (size 3(N-1)).
 */
function buildSystem(poses, edges, nPoses) {
  const n = 3 * (nPoses - 1);
  const H = new Float64Array(n * n);
  const g = new Float64Array(n);
  const err = [0, 0, 0];
  const A = new Float64Array(9), B = new Float64Array(9);
  for (const e of edges) {
    const i3 = 3 * e.i, j3 = 3 * e.j;
    const thi = poses[i3 + 2];
    const ci = Math.cos(thi), si = Math.sin(thi);
    const cz = Math.cos(e.z[2]), sz = Math.sin(e.z[2]);
    const dx = poses[j3] - poses[i3], dy = poses[j3 + 1] - poses[i3 + 1];
    // M = Rz^T Ri^T = R(-(thi + thz))
    const m = thi + e.z[2];
    const cm = Math.cos(m), sm = Math.sin(m);
    // dRi^T/dthi * t = [-si*dx + ci*dy, -ci*dx - si*dy]
    const dtx = -si * dx + ci * dy, dty = -ci * dx - si * dy;
    // A = de/dxi
    A[0] = -cm; A[1] = -sm; A[2] = cz * dtx + sz * dty;
    A[3] = sm;  A[4] = -cm; A[5] = -sz * dtx + cz * dty;
    A[6] = 0;   A[7] = 0;   A[8] = -1;
    // B = de/dxj
    B[0] = cm;  B[1] = sm;  B[2] = 0;
    B[3] = -sm; B[4] = cm;  B[5] = 0;
    B[6] = 0;   B[7] = 0;   B[8] = 1;
    edgeError(poses, e, err);
    const w = e.info;
    const blocks = [[e.i, A], [e.j, B]];
    for (const [pa, Ja] of blocks) {
      if (pa === 0) continue;
      const ra = 3 * (pa - 1);
      for (let r = 0; r < 3; r++) {
        // gradient: J^T W e
        let s = 0;
        for (let k = 0; k < 3; k++) s += Ja[k * 3 + r] * w[k] * err[k];
        g[ra + r] += s;
      }
      for (const [pb, Jb] of blocks) {
        if (pb === 0) continue;
        const rb = 3 * (pb - 1);
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            let s = 0;
            for (let k = 0; k < 3; k++) s += Ja[k * 3 + r] * w[k] * Jb[k * 3 + c];
            H[(ra + r) * n + rb + c] += s;
          }
        }
      }
    }
  }
  return { H, g, n };
}

/**
 * One Levenberg-Marquardt iteration. Returns {poses, chi2, accepted, lambda, stepNorm}
 * without mutating the input. Rejected steps return the input poses with larger lambda.
 */
export function lmStep(poses, edges, lambda) {
  const nPoses = poses.length / 3;
  const before = chi2(poses, edges);
  if (nPoses < 2) return { poses, chi2: before, accepted: false, lambda, stepNorm: 0 };
  const { H, g, n } = buildSystem(poses, edges, nPoses);
  for (let tries = 0; tries < 8; tries++) {
    const Hd = Float64Array.from(H);
    for (let k = 0; k < n; k++) Hd[k * n + k] += lambda * (H[k * n + k] + 1e-9);
    const negG = new Float64Array(n);
    for (let k = 0; k < n; k++) negG[k] = -g[k];
    const dx = choleskySolve(Hd, negG, n);
    if (!dx) { lambda *= 10; continue; }
    const next = Float64Array.from(poses);
    let stepNorm = 0;
    for (let k = 0; k < n; k++) {
      next[3 + k] += dx[k];
      stepNorm = Math.max(stepNorm, Math.abs(dx[k]));
    }
    for (let p = 1; p < nPoses; p++) next[3 * p + 2] = wrapAngle(next[3 * p + 2]);
    const after = chi2(next, edges);
    if (Number.isFinite(after) && after <= before) {
      return { poses: next, chi2: after, accepted: true, lambda: Math.max(lambda / 3, 1e-7), stepNorm };
    }
    lambda *= 4;
  }
  return { poses, chi2: before, accepted: false, lambda, stepNorm: 0 };
}

/**
 * Run LM to convergence. Returns the list of iterates (for animation) and a status.
 * Converged: max |dx| < tol. Fails: no accepted step after the damping schedule.
 */
export function optimise(poses, edges, { maxIter = 15, tol = 1e-6, lambda0 = 1e-4 } = {}) {
  const iterates = [{ poses: Float64Array.from(poses), chi2: chi2(poses, edges) }];
  let cur = Float64Array.from(poses);
  let lambda = lambda0;
  let status = 'max-iterations';
  for (let it = 0; it < maxIter; it++) {
    const r = lmStep(cur, edges, lambda);
    lambda = r.lambda;
    if (!r.accepted) { status = iterates.length > 1 ? 'converged' : 'no-progress'; break; }
    cur = r.poses;
    iterates.push({ poses: cur, chi2: r.chi2 });
    if (r.stepNorm < tol) { status = 'converged'; break; }
  }
  return { iterates, status };
}
