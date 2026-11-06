// One held-out SCARA task run by the demo's classical planner (shared by tools/eval_scara_ik.mjs and the
// browser equivalence check; no Node APIs). Positions in the task are relative to the arm base.
import { ARM, fk, plan, cspace, segmentTime, minJerk, collides } from '../assets/js/demos/scara.js';

const TOL_M = 0.01, STOP_MPS = 0.02, LIMIT_S = 5, TRIAL_S = 6, DT = 0.02;
const [bx, by] = ARM.base;

/** Joint pose along a planned path at time t (s); holds the final pose after the last segment. */
function poseAt(segs, t) {
  const sg = segs.find((g) => t <= g.t0 + g.T) || segs[segs.length - 1];
  const s = minJerk(Math.min(1, Math.max(0, (t - sg.t0) / sg.T)));
  return [sg.a[0] + s * (sg.b[0] - sg.a[0]), sg.a[1] + s * (sg.b[1] - sg.a[1])];
}

export function runTask(q0, tg, ob) {
  const obs = ob[0] > 0.5 ? { x: bx + ob[1], y: by + ob[2], r: ob[3] } : null;
  const x = bx + tg[0], y = by + tg[1];
  const r = plan(q0, x, y, 1, obs, cspace(obs));
  if (r.error) return { success: false, touch: false, planned: false, time: null };
  const segs = [];
  let t0 = 0;
  for (let i = 0; i + 1 < r.path.length; i++) {
    const T = segmentTime(r.path[i], r.path[i + 1]);
    segs.push({ a: r.path[i], b: r.path[i + 1], t0, T });
    t0 += T;
  }
  let reach = null, touch = obs ? collides(q0[0], q0[1], obs) : false, prev = fk(q0[0], q0[1]).ee;
  const steps = Math.round(TRIAL_S / DT);
  let ee = prev;
  for (let k = 1; k <= steps; k++) {
    const q = poseAt(segs, k * DT);
    ee = fk(q[0], q[1]).ee;
    const d = Math.hypot(ee[0] - x, ee[1] - y), v = Math.hypot(ee[0] - prev[0], ee[1] - prev[1]) / DT;
    if (reach === null && d < TOL_M && v < STOP_MPS) reach = k * DT;
    if (obs && collides(q[0], q[1], obs)) touch = true;
    prev = ee;
  }
  const fin = Math.hypot(ee[0] - x, ee[1] - y);
  return { success: reach !== null && reach <= LIMIT_S + 1e-9 && fin < TOL_M && !touch, touch, planned: true, time: reach };
}
