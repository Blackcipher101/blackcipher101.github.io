// Cart-pole swing-up: a learned policy and a classical controller in the same run.
// The policy is a 5-16-16-1 tanh MLP trained offline with evolution strategies in
// numpy (tools/train_pendulum.py); its weights and measured success rate ship in
// assets/data/cartpole-policy.json. The plant, RK4 integrator, observation scaling and
// wall model here mirror that script exactly, so the browser runs the same system the
// policy was evaluated on. The classical contrast is energy-pumping swing-up + LQR.
// Two identical plants run side by side (top lane: learned, bottom lane: classical). They
// start from the same state and get the same disturbances: Push kicks both poles by the
// same angular velocity, a tap flicks both the same way, and a drag pulls each pole tip
// toward the same point relative to its own rail.
// theta = 0 is upright; positive theta leans the pole toward +x.
import { View, Loop, COLORS, clamp, setHeadline } from './common.js?v=20261106p';

const POLICY_URL = new URL('../../data/cartpole-policy.json', import.meta.url);
POLICY_URL.search = new URL(import.meta.url).search;

const UPRIGHT_RAD = 12 * Math.PI / 180;
const KICK_RADPS = 3;            // same magnitude as the push-recovery evaluation
const DRAG_K = 60, DRAG_C = 6;   // spring/damper of the "grab the pole" interaction (1/s^2, 1/s)
const W = 5;                     // world width, metres
const LANE_H = 2.2;              // height of one lane, metres (pole 1 m up or down from the rail)
const H = 2 * LANE_H;

export const wrap = (a) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));

/** State derivative; s = [x, xd, th, thd], f = cart force (N), ext = extra pole angular accel (rad/s^2). */
function deriv(p, s, f, ext, out) {
  const xd = s[1], th = s[2], thd = s[3];
  const sin = Math.sin(th), cos = Math.cos(th);
  const mt = p.m_cart + p.m_pole;
  const tmp = (f + p.m_pole * p.l_half * thd * thd * sin) / mt;
  const thdd = (p.g * sin - cos * tmp) / (p.l_half * (4 / 3 - p.m_pole * cos * cos / mt)) + ext;
  out[0] = xd;
  out[1] = tmp - p.m_pole * p.l_half * thdd * cos / mt;
  out[2] = thd;
  out[3] = thdd;
}

const K1 = new Float64Array(4), K2 = new Float64Array(4), K3 = new Float64Array(4), K4 = new Float64Array(4), TMP = new Float64Array(4);

/** One RK4 step of length p.dt in place. Returns true if the cart hit a wall (inelastic stop). */
export function step(p, s, f, ext = 0) {
  const dt = p.dt;
  deriv(p, s, f, ext, K1);
  for (let i = 0; i < 4; i++) TMP[i] = s[i] + 0.5 * dt * K1[i];
  deriv(p, TMP, f, ext, K2);
  for (let i = 0; i < 4; i++) TMP[i] = s[i] + 0.5 * dt * K2[i];
  deriv(p, TMP, f, ext, K3);
  for (let i = 0; i < 4; i++) TMP[i] = s[i] + dt * K3[i];
  deriv(p, TMP, f, ext, K4);
  for (let i = 0; i < 4; i++) s[i] += dt / 6 * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]);
  if (Math.abs(s[0]) > p.x_max) { s[0] = clamp(s[0], -p.x_max, p.x_max); s[1] = 0; return true; }
  return false;
}

/** Learned policy force (N). */
export function learnedForce(data, s) {
  const p = data.plant, sc = p.obs_scale, dims = data.arch;
  let h = [s[0] * sc[0], s[1] * sc[1], Math.cos(s[2]) * sc[2], Math.sin(s[2]) * sc[3], s[3] * sc[4]];
  for (let l = 0; l < dims.length - 1; l++) {
    const w = data.weights[2 * l], b = data.weights[2 * l + 1], nIn = dims[l], nOut = dims[l + 1];
    const o = new Array(nOut);
    for (let j = 0; j < nOut; j++) {
      let a = b[j];
      for (let i = 0; i < nIn; i++) a += h[i] * w[i * nOut + j];
      o[j] = Math.tanh(a);
    }
    h = o;
  }
  return h[0] * p.f_max;
}

/** Classical baseline: energy-pumping swing-up, LQR inside the capture region. */
export function classicalForce(data, s) {
  const p = data.plant, c = data.classical;
  const x = s[0], xd = s[1], th = wrap(s[2]), thd = s[3];
  if (Math.abs(th) < c.switch_rad && Math.abs(thd) < c.switch_radps) {
    const k = c.k_lqr;
    return clamp(-(k[0] * x + k[1] * xd + k[2] * th + k[3] * thd), -p.f_max, p.f_max);
  }
  const j = 4 / 3 * p.m_pole * p.l_half * p.l_half;
  const energy = 0.5 * j * thd * thd + p.m_pole * p.g * p.l_half * (Math.cos(th) - 1);
  const a = c.k_energy * energy * thd * Math.cos(th) - c.k_x * x - c.k_v * xd;
  return clamp((p.m_cart + p.m_pole) * a * 4, -p.f_max, p.f_max);
}

/**
 * Headless swing-up trial with the same criterion as the training script: start hanging
 * (theta = pi + offset), success = upright and off the walls for the final 3 s of 10 s.
 * Returns the start time of the final upright run (s), or null on failure.
 */
export function swingUpTrial(data, mode, offsetRad) {
  const p = data.plant, s = new Float64Array([0, 0, Math.PI + offsetRad, 0]);
  const steps = Math.round(10 / p.dt), hold = Math.round(3 / p.dt);
  const ctrl = mode === 'classical' ? classicalForce : learnedForce;
  let lastBad = -1;
  for (let t = 0; t < steps; t++) {
    const f = clamp(ctrl(data, s), -p.f_max, p.f_max);
    const hit = step(p, s, f);
    if (hit || Math.abs(wrap(s[2])) >= UPRIGHT_RAD) lastBad = t;
  }
  return steps - 1 - lastBad >= hold ? (lastBad + 1) * p.dt : null;
}

export async function loadPolicy() {
  const r = await fetch(POLICY_URL);
  if (!r.ok) throw new Error(`policy fetch failed: HTTP ${r.status}`);
  return r.json();
}

/**
 * One plant + controller + live metrics. Lanes share nothing but the data file, so two of
 * them can be stepped in lock-step from the same state.
 */
export function makeLane(mode) {
  return {
    mode, s: new Float64Array(4), force: 0,
    upFor: 0, wasUp: false, swingStart: 0, lastSwing: null, peakF: 0, effort: 0,
    trail: [], flash: 0, wallFlash: 0,
  };
}

/** Put every lane in the same state and clear its metrics. */
export function resetLanes(lanes, state) {
  for (const ln of lanes) {
    ln.s.set(state);
    Object.assign(ln, { force: 0, upFor: 0, wasUp: false, swingStart: 0, lastSwing: null, peakF: 0, effort: 0 });
    ln.trail.length = 0;
  }
}

/** Identical push for every lane: pole angular velocity += dir * KICK_RADPS. */
export function kickLanes(lanes, dir) {
  for (const ln of lanes) {
    ln.s[3] += dir * KICK_RADPS;
    ln.flash = 0.35;
    ln.upFor = 0;
  }
}

/**
 * Advance one lane by one control period (p.dt). simT = time since reset at the end of the step.
 * pull = { dx, dy } pointer position relative to the cart's rail centre (m), or null.
 */
export function stepLane(data, ln, simT, pull) {
  const p = data.plant, s = ln.s;
  let ext = 0;
  if (pull) ext = DRAG_K * wrap(Math.atan2(pull.dx - s[0], pull.dy) - s[2]) - DRAG_C * s[3];
  ln.force = clamp(ln.mode === 'learned' ? learnedForce(data, s) : classicalForce(data, s), -p.f_max, p.f_max);
  if (step(p, s, ln.force, ext)) ln.wallFlash = 0.3;
  ln.peakF = Math.max(ln.peakF, Math.abs(ln.force));
  ln.effort += Math.abs(ln.force) * p.dt;
  const up = Math.abs(wrap(s[2])) < UPRIGHT_RAD;
  if (up && !ln.wasUp) ln.swingStart = simT - p.dt;
  if (!up) ln.upFor = 0;
  if (up) {
    ln.upFor += p.dt;
    // same rule as the offline evaluation: swing-up time = start of an upright run that lasts 3 s
    if (ln.upFor >= 3 && ln.lastSwing === null) ln.lastSwing = ln.swingStart;
  }
  ln.wasUp = up;
}

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, W, H);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);

  let data = null, acc = 0, simT = 0, drag = null;
  // top lane: learned policy; bottom lane: classical controller
  const lanes = [makeLane('learned'), makeLane('classical')];
  const LANE_STYLE = {
    learned: { railY: LANE_H + LANE_H / 2, color: COLORS.lime, trail: 'rgba(217,244,91,0.4)', label: 'LEARNED POLICY (ES)' },
    classical: { railY: LANE_H / 2, color: COLORS.peach, trail: 'rgba(255,180,138,0.4)', label: 'ENERGY SWING-UP + LQR' },
  };

  function reset() {
    resetLanes(lanes, [0, 0, Math.PI + 0.02 * (Math.random() - 0.5), 0]);
    simT = 0;
  }

  const tipPos = (ln) => {
    const L = 2 * data.plant.l_half, railY = LANE_STYLE[ln.mode].railY;
    return [W / 2 + ln.s[0] + L * Math.sin(ln.s[2]), railY + L * Math.cos(ln.s[2])];
  };
  const laneAt = (y) => (y >= LANE_H ? lanes[0] : lanes[1]);

  function physics() {
    const pull = drag && drag.moved ? { dx: drag.dx, dy: drag.dy } : null;
    simT += data.plant.dt;
    for (const ln of lanes) {
      stepLane(data, ln, simT, pull);
      const tip = tipPos(ln);
      ln.trail.push(tip[0], tip[1]);
      if (ln.trail.length > 120) ln.trail.splice(0, 2);
    }
  }

  function drawLane(ln) {
    const { ctx } = view, d = view.dpr, p = data ? data.plant : { x_max: 2, l_half: 0.5, f_max: 20 };
    const st = LANE_STYLE[ln.mode], RAIL_Y = st.railY, s = ln.s;
    const cx = W / 2 + s[0];
    // rail + walls
    ctx.strokeStyle = 'rgba(244,239,228,0.35)';
    ctx.lineWidth = 2 * d;
    ctx.beginPath();
    ctx.moveTo(view.sx(W / 2 - p.x_max - 0.25), view.sy(RAIL_Y));
    ctx.lineTo(view.sx(W / 2 + p.x_max + 0.25), view.sy(RAIL_Y));
    ctx.stroke();
    for (const side of [-1, 1]) {
      const wx = view.sx(W / 2 + side * (p.x_max + 0.25));
      ctx.fillStyle = ln.wallFlash > 0 ? COLORS.coral : 'rgba(244,239,228,0.5)';
      ctx.fillRect(wx - 3 * d, view.sy(RAIL_Y + 0.18), 6 * d, 0.36 * view.scale);
    }
    // success cone around upright
    ctx.beginPath();
    ctx.moveTo(view.sx(cx), view.sy(RAIL_Y));
    ctx.arc(view.sx(cx), view.sy(RAIL_Y), 1.05 * view.scale, -Math.PI / 2 - UPRIGHT_RAD, -Math.PI / 2 + UPRIGHT_RAD);
    ctx.closePath();
    ctx.fillStyle = ln.upFor > 0 ? 'rgba(217,244,91,0.16)' : 'rgba(244,239,228,0.05)';
    ctx.fill();
    // tip trail
    const tr = ln.trail;
    if (tr.length > 4) {
      ctx.beginPath();
      ctx.moveTo(view.sx(tr[0]), view.sy(tr[1]));
      for (let i = 2; i < tr.length; i += 2) ctx.lineTo(view.sx(tr[i]), view.sy(tr[i + 1]));
      ctx.strokeStyle = st.trail;
      ctx.lineWidth = 1.5 * d;
      ctx.stroke();
    }
    // cart
    const cw = 0.42 * view.scale, ch = 0.2 * view.scale;
    ctx.fillStyle = st.color;
    ctx.beginPath();
    ctx.roundRect(view.sx(cx) - cw / 2, view.sy(RAIL_Y) - ch / 2, cw, ch, 6 * d);
    ctx.fill();
    ctx.fillStyle = COLORS.ink;
    for (const wx of [-0.13, 0.13]) {
      ctx.beginPath();
      ctx.arc(view.sx(cx + wx), view.sy(RAIL_Y - 0.1), 0.045 * view.scale, 0, 2 * Math.PI);
      ctx.fill();
    }
    // force arrow
    if (data && Math.abs(ln.force) > 0.3) {
      const len = (ln.force / p.f_max) * 0.6 * view.scale, y = view.sy(RAIL_Y - 0.24);
      ctx.strokeStyle = COLORS.violet; ctx.fillStyle = COLORS.violet; ctx.lineWidth = 3 * d;
      ctx.beginPath(); ctx.moveTo(view.sx(cx), y); ctx.lineTo(view.sx(cx) + len, y); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(view.sx(cx) + len + Math.sign(len) * 7 * d, y);
      ctx.lineTo(view.sx(cx) + len, y - 5 * d);
      ctx.lineTo(view.sx(cx) + len, y + 5 * d);
      ctx.fill();
    }
    // pole
    const tip = data ? tipPos(ln) : [cx, RAIL_Y - 1];
    ctx.strokeStyle = ln.flash > 0 ? COLORS.coral : COLORS.cream;
    ctx.lineWidth = 7 * d;
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(view.sx(cx), view.sy(RAIL_Y)); ctx.lineTo(view.sx(tip[0]), view.sy(tip[1])); ctx.stroke();
    ctx.fillStyle = COLORS.peach;
    ctx.beginPath(); ctx.arc(view.sx(tip[0]), view.sy(tip[1]), 0.07 * view.scale, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = COLORS.ink;
    ctx.beginPath(); ctx.arc(view.sx(cx), view.sy(RAIL_Y), 4 * d, 0, 2 * Math.PI); ctx.fill();
    // drag spring (the same pointer, mirrored into both lanes)
    if (drag && drag.moved) {
      ctx.setLineDash([4 * d, 4 * d]);
      ctx.strokeStyle = COLORS.coral; ctx.lineWidth = 1.5 * d;
      ctx.beginPath(); ctx.moveTo(view.sx(tip[0]), view.sy(tip[1])); ctx.lineTo(view.sx(drag.dx + W / 2), view.sy(drag.dy + RAIL_Y)); ctx.stroke();
      ctx.setLineDash([]);
    }
    // lane tag
    ctx.font = `${11 * d}px ui-monospace, Menlo, monospace`;
    ctx.fillStyle = st.color;
    ctx.fillText(st.label, 18 * d, view.sy(RAIL_Y + LANE_H / 2) + 26 * d);
  }

  function draw() {
    const { ctx } = view, d = view.dpr;
    view.clear();
    view.grid(0.5);
    // lane divider
    ctx.strokeStyle = 'rgba(244,239,228,0.22)';
    ctx.lineWidth = 1 * d;
    ctx.beginPath(); ctx.moveTo(view.sx(0), view.sy(LANE_H)); ctx.lineTo(view.sx(W), view.sy(LANE_H)); ctx.stroke();
    for (const ln of lanes) drawLane(ln);
  }

  function updateReadout() {
    if (!data) return;
    const nz = (v, dp) => (Math.abs(v) < 0.5 * 10 ** -dp ? 0 : v).toFixed(dp); // no "-0"
    lanes.forEach((ln, i) => {
      const k = i === 0 ? 'a' : 'b';
      out(`swing-${k}`).textContent = ln.lastSwing !== null ? `${ln.lastSwing.toFixed(1)} s` : '–';
      out(`up-${k}`).textContent = ln.upFor > 0 ? `${ln.upFor.toFixed(1)} s` : '–';
      out(`peak-${k}`).textContent = `${nz(ln.peakF, 1)} N`;
      out(`effort-${k}`).textContent = `${nz(ln.effort, 0)} N·s`;
    });
  }

  function describe() {
    const e = data.eval, tr = data.training;
    const res = (r) => `${r.success}/${e.trials}` + (r.median_s !== null ? ` (median ${r.median_s} s, worst ${r.max_s} s)` : '') +
      `, ${Math.round(r.push_recovery * 100)}% recovered from a ${e.push_kick_radps} rad/s push`;
    out('policy').textContent =
      `Top, learned policy: ${data.params} params, trained with evolution strategies for ${tr.episodes.toLocaleString('en')} simulated episodes, seed ${tr.seed}. ` +
      `Bottom, classical energy swing-up then LQR near the top. Offline over ${e.trials} seeded swing-ups from hanging: learned ${res(e.learned)}; classical ${res(e.classical)}.`;
    const L = e.learned, C = e.classical;
    setHeadline(root, [`swing-up ${L.median_s} s vs ${C.median_s} s`, `${L.success}/${e.trials} vs ${C.success}/${e.trials} swing-ups`,
      `push recovery ${Math.round(L.push_recovery * 100)}% vs ${Math.round(C.push_recovery * 100)}%`]);
  }

  // pointer: a drag pulls both pole tips toward the same point relative to their rails; a tap flicks
  // both poles the same way (away from the pointer, judged on the lane that was tapped)
  canvas.addEventListener('pointerdown', (ev) => {
    if (!data) return;
    canvas.setPointerCapture(ev.pointerId);
    const [px, py] = view.toWorld(ev);
    const ln = laneAt(py);
    drag = { lane: ln, dx: px - W / 2, dy: py - LANE_STYLE[ln.mode].railY, sx: ev.clientX, sy: ev.clientY, moved: false };
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const [px, py] = view.toWorld(ev);
    drag.dx = px - W / 2; drag.dy = py - LANE_STYLE[drag.lane.mode].railY;
    if (!drag.moved && Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy) > 8) {
      drag.moved = true;
      for (const ln of lanes) ln.upFor = 0;
    }
  });
  const endDrag = (ev) => {
    if (!drag) return;
    if (!drag.moved && ev.type === 'pointerup') kickLanes(lanes, tipPos(drag.lane)[0] - W / 2 >= drag.dx ? 1 : -1);
    drag = null;
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  btn('push').addEventListener('click', () => data && kickLanes(lanes, Math.random() < 0.5 ? -1 : 1));
  btn('reset').addEventListener('click', () => data && reset());

  let uiTimer = 0;
  const loop = new Loop((dt) => {
    if (!data) { draw(); return; }
    acc += dt;
    let n = 0;
    while (acc >= data.plant.dt && n < 4) { physics(); acc -= data.plant.dt; n++; }
    if (n === 4) acc = 0; // fell behind: drop time rather than spiral
    for (const ln of lanes) {
      ln.flash = Math.max(0, ln.flash - dt);
      ln.wallFlash = Math.max(0, ln.wallFlash - dt);
    }
    draw();
    uiTimer += dt;
    if (uiTimer > 0.15) { updateReadout(); uiTimer = 0; }
  });

  for (const ln of lanes) ln.s.set([0, 0, Math.PI, 0]);
  draw();
  out('policy').textContent = 'Loading policy…';
  loadPolicy().then((d) => {
    data = d;
    reset();
    describe();
    updateReadout();
    draw();
  }).catch((err) => {
    console.warn('[demo:cartpole] policy unavailable:', err);
    out('policy').textContent = 'The policy file could not be loaded.';
  });
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
