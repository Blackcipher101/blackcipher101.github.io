// Pose-graph SLAM playground.
// A robot drives with biased, noisy wheel odometry. Keyframes become graph nodes,
// odometry becomes edges, and revisiting a place adds a loop-closure edge (simulated
// scan match: the true relative pose plus small noise). "Optimise" runs a real
// Levenberg-Marquardt solve (posegraph.js) and animates each iterate.
import { View, Loop, COLORS, gauss, clamp } from './common.js';
import { optimise, chi2, composePose, relativePose, wrapAngle } from './posegraph.js';

const W = 20, H = 12;                 // world, metres
const KEYFRAME_DIST_M = 0.6;
const MAX_KEYFRAMES = 150;            // keeps the dense solve well under 100 ms per iteration
const LOOP_RADIUS_M = 0.9;
const LOOP_MIN_GAP = 20;              // keyframes between the two ends of a loop edge
const LOOP_MIN_SPACING = 4;           // keyframes between successive loop edges
const ODOM_INFO = [1 / 0.03 ** 2, 1 / 0.03 ** 2, 1 / 0.02 ** 2];
const LOOP_INFO = [1 / 0.03 ** 2, 1 / 0.03 ** 2, 1 / 0.01 ** 2];
const SPEED_MPS = 1.6;
const MAX_TURN_RADPS = 1.6;
const TWEEN_S = 0.32;

// Closed reference loop for the autopilot (metres).
const loopPoint = (t) => [10 + 7.2 * Math.cos(t), 6 + 3.9 * Math.sin(t) + 0.7 * Math.sin(3 * t)];

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, W, H);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);
  const playBtn = btn('play'), optBtn = btn('optimise'), resetBtn = btn('reset');

  let worker = null;
  try {
    worker = new Worker(new URL('./posegraph-worker.js', import.meta.url), { type: 'module' });
  } catch (_) {
    worker = null; // fall back to solving on the main thread
  }

  let s;
  function reset() {
    const start = loopPoint(0);
    const ahead = loopPoint(0.05);
    const th0 = Math.atan2(ahead[1] - start[1], ahead[0] - start[0]);
    s = {
      truth: [start[0], start[1], th0],
      kfTrue: [[start[0], start[1], th0]],
      kfEst: [start[0], start[1], th0],      // flat estimate of keyframe poses
      edges: [],
      loops: 0,
      lastLoopK: -1e9,
      acc: [0, 0, 0],                        // odometry since last keyframe (in its frame)
      accTrueDist: 0,
      truthTrail: [[start[0], start[1]]],
      track: 0,
      autopilot: s ? s.autopilot : false,
      steer: null,
      bias: (Math.random() < 0.5 ? -1 : 1) * (0.016 + 0.01 * Math.random()), // rad per metre
      scaleErr: 0.02 * gauss(),
      anim: null,
      optimising: false,
      message: '',
      ateBefore: null,
    };
    updateUi();
  }

  const nKf = () => s.kfEst.length / 3;

  function currentEstimate() {
    const k = nKf() - 1;
    return composePose(s.kfEst[3 * k], s.kfEst[3 * k + 1], s.kfEst[3 * k + 2], s.acc[0], s.acc[1], s.acc[2]);
  }

  function ate(poses) {
    let sum = 0;
    const n = s.kfTrue.length;
    for (let k = 0; k < n; k++) {
      sum += (poses[3 * k] - s.kfTrue[k][0]) ** 2 + (poses[3 * k + 1] - s.kfTrue[k][1]) ** 2;
    }
    return Math.sqrt(sum / n);
  }

  function addKeyframe() {
    const k = nKf();
    const prev = k - 1;
    const est = composePose(s.kfEst[3 * prev], s.kfEst[3 * prev + 1], s.kfEst[3 * prev + 2], s.acc[0], s.acc[1], s.acc[2]);
    s.edges.push({ i: prev, j: k, z: s.acc.slice(), info: ODOM_INFO, loop: false });
    const kf = new Float64Array(s.kfEst.length + 3);
    kf.set(s.kfEst);
    kf.set(est, s.kfEst.length);
    s.kfEst = kf;
    s.kfTrue.push(s.truth.slice());
    s.acc = [0, 0, 0];
    s.accTrueDist = 0;

    // Loop closure: nearest old keyframe by true position (stands in for place recognition
    // + scan matching, which in a real system would also verify geometry).
    if (k - s.lastLoopK >= LOOP_MIN_SPACING) {
      let best = -1, bestD = LOOP_RADIUS_M;
      for (let j = 0; j <= k - LOOP_MIN_GAP; j++) {
        const t = s.kfTrue[j];
        const d = Math.hypot(t[0] - s.truth[0], t[1] - s.truth[1]);
        if (d < bestD && Math.abs(wrapAngle(t[2] - s.truth[2])) < Math.PI / 3) { bestD = d; best = j; }
      }
      if (best >= 0) {
        const t = s.kfTrue[best];
        const z = relativePose(t[0], t[1], t[2], s.truth[0], s.truth[1], s.truth[2]);
        z[0] += 0.02 * gauss(); z[1] += 0.02 * gauss(); z[2] = wrapAngle(z[2] + 0.008 * gauss());
        s.edges.push({ i: best, j: k, z, info: LOOP_INFO, loop: true });
        s.loops++;
        s.lastLoopK = k;
      }
    }
    if (nKf() >= MAX_KEYFRAMES) {
      s.autopilot = false;
      s.steer = null;
      s.message = 'Graph is full. Optimise, then reset to drive again.';
    }
  }

  function drive(dt) {
    if (nKf() >= MAX_KEYFRAMES || s.optimising) return false;
    let target = null;
    if (s.steer) target = s.steer;
    else if (s.autopilot) {
      // advance the carrot along the reference loop
      for (let i = 0; i < 40; i++) {
        const p = loopPoint(s.track);
        if (Math.hypot(p[0] - s.truth[0], p[1] - s.truth[1]) > 1.4) break;
        s.track += 0.02;
      }
      target = loopPoint(s.track);
    }
    if (!target) return false;
    const [x, y, th] = s.truth;
    const dist = Math.hypot(target[0] - x, target[1] - y);
    if (s.steer && dist < 0.3) return false;
    const err = wrapAngle(Math.atan2(target[1] - y, target[0] - x) - th);
    const w = clamp(3 * err, -MAX_TURN_RADPS, MAX_TURN_RADPS);
    const v = SPEED_MPS * clamp(1 - Math.abs(err) / 2, 0.25, 1);
    const ds = v * dt, dth = w * dt;
    // true motion (midpoint heading)
    const mid = th + dth / 2;
    s.truth = [clamp(x + ds * Math.cos(mid), 0.2, W - 0.2), clamp(y + ds * Math.sin(mid), 0.2, H - 0.2), wrapAngle(th + dth)];
    // measured motion: scale error, heading bias per metre, white noise
    const dsm = ds * (1 + s.scaleErr) + 0.004 * Math.sqrt(ds) * gauss();
    const dthm = dth + s.bias * ds + 0.004 * Math.sqrt(ds) * gauss();
    s.acc = composePose(s.acc[0], s.acc[1], s.acc[2], dsm * Math.cos(dthm / 2), dsm * Math.sin(dthm / 2), dthm);
    s.accTrueDist += ds;
    const last = s.truthTrail[s.truthTrail.length - 1];
    if (Math.hypot(last[0] - s.truth[0], last[1] - s.truth[1]) > 0.1) s.truthTrail.push([s.truth[0], s.truth[1]]);
    if (s.accTrueDist >= KEYFRAME_DIST_M) {
      addKeyframe();
      updateUi();
    }
    return true;
  }

  function runOptimise() {
    if (s.optimising || s.loops === 0) return;
    s.optimising = true;
    s.ateBefore = ate(s.kfEst);
    s.message = 'Solving…';
    updateUi();
    const poses = Float64Array.from(s.kfEst);
    const edges = s.edges.map(({ i, j, z, info }) => ({ i, j, z, info }));
    const onResult = (res) => {
      if (!res.ok) {
        s.optimising = false;
        s.message = `Solver error: ${res.error}`;
        updateUi();
        return;
      }
      s.anim = { iterates: res.iterates, idx: 0, t: 0, status: res.status };
    };
    if (worker) {
      const id = Math.random();
      const handler = (ev) => {
        if (ev.data.id !== id) return;
        worker.removeEventListener('message', handler);
        onResult(ev.data);
      };
      worker.addEventListener('message', handler);
      worker.onerror = () => { worker = null; s.optimising = false; runOptimise(); };
      worker.postMessage({ id, poses, edges });
    } else {
      const r = optimise(poses, edges);
      onResult({ ok: true, status: r.status, iterates: r.iterates });
    }
  }

  function stepAnimation(dt) {
    const a = s.anim;
    if (!a) return;
    a.t += dt / TWEEN_S;
    const from = a.iterates[a.idx].poses;
    const to = a.iterates[Math.min(a.idx + 1, a.iterates.length - 1)].poses;
    if (a.t >= 1) {
      a.idx++;
      a.t = 0;
      if (a.idx >= a.iterates.length - 1) {
        s.kfEst = Float64Array.from(a.iterates[a.iterates.length - 1].poses);
        s.anim = null;
        s.optimising = false;
        const after = ate(s.kfEst);
        const iters = a.iterates.length - 1;
        s.message = `${a.status === 'converged' ? 'Converged' : 'Stopped'} after ${iters} iteration${iters === 1 ? '' : 's'}. ATE ${s.ateBefore.toFixed(2)} m → ${after.toFixed(2)} m`;
        updateUi();
        return;
      }
    }
    const e = a.t < 0.5 ? 2 * a.t * a.t : 1 - (-2 * a.t + 2) ** 2 / 2;
    const cur = new Float64Array(from.length);
    for (let i = 0; i < from.length; i++) {
      cur[i] = (i % 3 === 2) ? from[i] + wrapAngle(to[i] - from[i]) * e : from[i] + (to[i] - from[i]) * e;
    }
    s.kfEst = cur;
    out('chi2').textContent = fmtChi(a.iterates[a.idx].chi2);
    out('iter').textContent = String(a.idx + 1);
  }

  const fmtChi = (v) => (v >= 1e4 ? v.toExponential(2) : v.toFixed(1));

  function updateUi() {
    out('kf').textContent = String(nKf());
    out('loops').textContent = String(s.loops);
    out('ate').textContent = `${ate(s.kfEst).toFixed(2)} m`;
    out('chi2').textContent = fmtChi(chi2(s.kfEst, s.edges));
    out('iter').textContent = '–';
    out('msg').textContent = s.message ||
      (s.loops ? 'Loop closed. Press Optimise to solve the graph.' : 'Press Drive, or drag on the map to steer. Come back to the start to close the loop.');
    playBtn.textContent = s.autopilot ? 'Pause' : 'Drive';
    playBtn.setAttribute('aria-pressed', String(s.autopilot));
    optBtn.disabled = s.loops === 0 || s.optimising;
  }

  function draw() {
    const { ctx } = view;
    const d = view.dpr;
    view.clear();
    view.grid(1);
    // ground truth trail
    ctx.beginPath();
    s.truthTrail.forEach(([x, y], i) => (i ? ctx.lineTo(view.sx(x), view.sy(y)) : ctx.moveTo(view.sx(x), view.sy(y))));
    ctx.setLineDash([4 * d, 5 * d]);
    ctx.strokeStyle = COLORS.creamDim;
    ctx.lineWidth = 1.5 * d;
    ctx.stroke();
    ctx.setLineDash([]);
    // loop closure edges
    const P = s.kfEst;
    ctx.beginPath();
    for (const e of s.edges) {
      if (!e.loop) continue;
      ctx.moveTo(view.sx(P[3 * e.i]), view.sy(P[3 * e.i + 1]));
      ctx.lineTo(view.sx(P[3 * e.j]), view.sy(P[3 * e.j + 1]));
    }
    ctx.strokeStyle = COLORS.lime;
    ctx.lineWidth = 2 * d;
    ctx.stroke();
    // estimated trajectory through keyframes
    const n = nKf();
    const cur = s.anim ? null : currentEstimate();
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const x = view.sx(P[3 * k]), y = view.sy(P[3 * k + 1]);
      k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    if (cur) ctx.lineTo(view.sx(cur[0]), view.sy(cur[1]));
    ctx.strokeStyle = COLORS.coral;
    ctx.lineWidth = 2.5 * d;
    ctx.lineJoin = 'round';
    ctx.stroke();
    // keyframe nodes
    ctx.fillStyle = COLORS.peach;
    const r = 2.2 * d;
    for (let k = 0; k < n; k++) {
      ctx.fillRect(view.sx(P[3 * k]) - r, view.sy(P[3 * k + 1]) - r, 2 * r, 2 * r);
    }
    // robots: true (outline) and estimated (filled)
    view.robot(s.truth[0], s.truth[1], s.truth[2], 0.42, null, COLORS.cream);
    const est = cur || [P[3 * (n - 1)], P[3 * (n - 1) + 1], P[3 * (n - 1) + 2]];
    view.robot(est[0], est[1], est[2], 0.42, COLORS.coral, null);
    if (s.steer) {
      ctx.beginPath();
      ctx.arc(view.sx(s.steer[0]), view.sy(s.steer[1]), 7 * d, 0, 2 * Math.PI);
      ctx.strokeStyle = COLORS.lime;
      ctx.lineWidth = 2 * d;
      ctx.stroke();
    }
  }

  const loop = new Loop((dt) => {
    const moved = drive(dt);
    stepAnimation(dt);
    if (moved || s.anim || view.dirty) {
      draw();
      view.dirty = false;
      if (moved) out('ate').textContent = `${ate(s.kfEst).toFixed(2)} m`;
    }
  });

  // Input
  playBtn.addEventListener('click', () => {
    s.autopilot = !s.autopilot;
    if (s.autopilot && nKf() >= MAX_KEYFRAMES) s.autopilot = false;
    s.message = '';
    updateUi();
  });
  optBtn.addEventListener('click', runOptimise);
  resetBtn.addEventListener('click', () => { s.autopilot = false; reset(); view.dirty = true; });
  canvas.addEventListener('pointerdown', (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    s.steer = view.toWorld(ev);
    s.autopilot = false;
    s.message = '';
    updateUi();
  });
  canvas.addEventListener('pointermove', (ev) => { if (s.steer) s.steer = view.toWorld(ev); });
  const endSteer = () => { s.steer = null; view.dirty = true; };
  canvas.addEventListener('pointerup', endSteer);
  canvas.addEventListener('pointercancel', endSteer);

  reset();
  draw();
  return {
    resume: () => loop.start(),
    pause: () => loop.stop(),
  };
}
