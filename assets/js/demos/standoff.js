// Stand-off tracking: hold a set distance from a manoeuvring beacon. A Double-DQN agent (learned) and
// a tuned proportional controller (classical) drive identical point-mass followers in two copies of
// the same episode: same beacon path, same start, same sensor-noise sequence.
// Simulation, controllers and network are lib/standoff-core.js (unchanged from the original
// artifact). The agent was trained offline with tools/train_standoff.mjs; its weights, the tuned P
// gains and the held-out test suite ship in assets/data/standoff-policy.json.
import { View, Loop, COLORS } from './common.js?v=20261106l';
import './lib/standoff-core.js?v=20261106l';

const C = window.TrackCore;
const POLICY_URL = new URL('../../data/standoff-policy.json', import.meta.url);
POLICY_URL.search = new URL(import.meta.url).search;

const PLAYBACK = 2;                // simulated seconds per real second
const TRAIL_PTS = 260;
const ERR_STRIP = 0.24;            // share of the canvas height used by the range-error strip

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, 16, 10);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);
  const families = [...C.TRAIN_FAMILIES, ...C.HELDOUT_FAMILIES];

  let data = null, rlNet = null, famIdx = 0, run = null, acc = 0, runNo = 0;

  /** Two copies of one seeded episode; the controllers differ, nothing else does. */
  function newRun() {
    if (!data) return;
    const fam = families[famIdx];
    const seed = C.SEED_BASE.test + famIdx * 1000 + (runNo++ % data.test.episodes_per_family);
    const cfg = { family: fam, seed, standoff_m: data.test.standoff_m, episode_s: data.test.episode_s, ...data.test.noise };
    const mk = (ctrl) => ({ env: new C.TrackingEnv(cfg), ctrl, trail: [], err: [], dead: null, se: 0, n: 0, inBand: 0 });
    run = { cfg, beacon: [], lanes: [mk(new C.RLController(rlNet)), mk(new C.PController(data.p_gains.Kp, data.p_gains.Kv, true))] };
    acc = 0;
    btn('path').textContent = `Path: ${C.FAMILIES[fam].label}${C.FAMILIES[fam].split === 'held-out' ? ' (never trained on)' : ''}`;
    readout();
  }

  function stepRun() {
    const d = run.cfg.standoff_m;
    for (const ln of run.lanes) {
      if (ln.env.done) continue;
      const [aR, aT] = ln.ctrl.act(ln.env.meas);
      const r = ln.env.step(aR, aT);
      ln.err.push([ln.env.t, r.rho - d]);
      ln.trail.push(ln.env.agent.x, ln.env.agent.y);
      if (ln.trail.length > 2 * TRAIL_PTS) ln.trail.splice(0, 2);
      ln.ax = r.ax; ln.ay = r.ay;
      if (ln.env.t > C.SETTLE_S + 1e-9) {
        ln.n++; ln.se += (r.rho - d) ** 2;
        if (Math.abs(r.rho - d) <= 0.1 * d) ln.inBand++;
      }
      if (r.terminal) ln.dead = r.reason;
    }
    const b = run.lanes[0].env.beacon; // identical in both copies
    run.beacon.push(b.x, b.y);
    if (run.beacon.length > 2 * TRAIL_PTS) run.beacon.splice(0, 2);
  }

  function draw() {
    const { ctx } = view, dp = view.dpr, W = canvas.width, Hc = canvas.height;
    view.clear();
    if (!run) return;
    const d = run.cfg.standoff_m, lead = run.lanes.reduce((a, l) => (l.env.t > a.env.t ? l : a));
    const beacon = lead.env.beacon, worldH = Hc * (1 - ERR_STRIP);
    const scale = Math.min(W, worldH) / (2 * 2.6 * d), cx = beacon.x, cy = beacon.y;
    const X = (x) => W / 2 + (x - cx) * scale, Y = (y) => worldH / 2 - (y - cy) * scale;
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, W, worldH); ctx.clip();
    // 5 m grid moving under the camera
    const g = 5;
    ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1; ctx.beginPath();
    for (let gx = Math.floor((cx - W / 2 / scale) / g) * g; X(gx) < W; gx += g) { ctx.moveTo(X(gx), 0); ctx.lineTo(X(gx), worldH); }
    for (let gy = Math.floor((cy - worldH / 2 / scale) / g) * g; Y(gy) > 0; gy += g) { ctx.moveTo(0, Y(gy)); ctx.lineTo(W, Y(gy)); }
    ctx.stroke();
    // stand-off ring and contact ring
    ctx.setLineDash([5 * dp, 5 * dp]); ctx.strokeStyle = COLORS.creamDim; ctx.lineWidth = 1.2 * dp;
    ctx.beginPath(); ctx.arc(X(cx), Y(cy), d * scale, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = COLORS.coral; ctx.setLineDash([2 * dp, 4 * dp]);
    ctx.beginPath(); ctx.arc(X(cx), Y(cy), C.PHYS.contactFrac * d * scale, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);
    const trail = (pts, color) => {
      if (pts.length < 4) return;
      ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dp; ctx.beginPath();
      ctx.moveTo(X(pts[0]), Y(pts[1]));
      for (let i = 2; i < pts.length; i += 2) ctx.lineTo(X(pts[i]), Y(pts[i + 1]));
      ctx.stroke();
    };
    trail(run.beacon, 'rgba(244,239,228,0.35)');
    const colors = [COLORS.lime, COLORS.peach], fade = ['rgba(217,244,91,0.45)', 'rgba(255,180,138,0.45)'];
    run.lanes.forEach((ln, i) => trail(ln.trail, fade[i]));
    // beacon
    ctx.save(); ctx.translate(X(beacon.x), Y(beacon.y)); ctx.rotate(-beacon.psi);
    ctx.fillStyle = COLORS.cream; ctx.beginPath();
    ctx.moveTo(10 * dp, 0); ctx.lineTo(-7 * dp, 6 * dp); ctx.lineTo(-7 * dp, -6 * dp); ctx.closePath(); ctx.fill();
    ctx.restore();
    // followers + commanded acceleration (1 m/s^2 drawn as 0.6 m)
    run.lanes.forEach((ln, i) => {
      const a = ln.env.agent;
      ctx.fillStyle = colors[i]; ctx.beginPath(); ctx.arc(X(a.x), Y(a.y), 6 * dp, 0, 2 * Math.PI); ctx.fill();
      ctx.lineWidth = 2 * dp; ctx.strokeStyle = COLORS.panel; ctx.stroke();
      if (ln.ax !== undefined) {
        ctx.strokeStyle = colors[i]; ctx.lineWidth = 1.5 * dp; ctx.beginPath();
        ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(a.x + 0.6 * ln.ax), Y(a.y + 0.6 * ln.ay)); ctx.stroke();
      }
      if (ln.dead) {
        ctx.fillStyle = COLORS.coral; ctx.font = `${11 * dp}px ui-monospace, Menlo, monospace`;
        ctx.fillText(ln.dead.toUpperCase(), X(a.x) + 9 * dp, Y(a.y) - 9 * dp);
      }
    });
    ctx.restore();
    ctx.font = `${10 * dp}px ui-monospace, Menlo, monospace`; ctx.fillStyle = COLORS.creamDim;
    ctx.fillText(`T ${lead.env.t.toFixed(1)} S · D ${d} M · GRID 5 M · ${PLAYBACK}× SPEED`, 12 * dp, 18 * dp);
    // range-error strip: rho - d over the episode, band +-10 % d
    const y0 = worldH, h = Hc - worldH, pad = 8 * dp, T = run.cfg.episode_s, lim = 0.5 * d;
    const sx = (t) => pad + (t / T) * (W - 2 * pad), sy = (e) => y0 + h / 2 - (Math.max(-lim, Math.min(lim, e)) / lim) * (h / 2 - pad);
    ctx.fillStyle = '#121016'; ctx.fillRect(0, y0, W, h);
    ctx.fillStyle = 'rgba(217,244,91,0.08)'; ctx.fillRect(pad, sy(0.1 * d), W - 2 * pad, sy(-0.1 * d) - sy(0.1 * d));
    ctx.strokeStyle = COLORS.gridStrong; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, sy(0)); ctx.lineTo(W - pad, sy(0)); ctx.stroke();
    const sxSettle = sx(C.SETTLE_S);
    ctx.setLineDash([3 * dp, 4 * dp]); ctx.beginPath(); ctx.moveTo(sxSettle, y0 + pad); ctx.lineTo(sxSettle, Hc - pad); ctx.stroke(); ctx.setLineDash([]);
    run.lanes.forEach((ln, i) => {
      if (ln.err.length < 2) return;
      ctx.strokeStyle = colors[i]; ctx.lineWidth = 1.5 * dp; ctx.beginPath();
      ln.err.forEach(([t, e], k) => (k ? ctx.lineTo(sx(t), sy(e)) : ctx.moveTo(sx(t), sy(e))));
      ctx.stroke();
    });
    ctx.fillStyle = COLORS.creamDim;
    ctx.fillText(`RANGE ERROR ±${lim.toFixed(0)} M · BAND ±10 % · SCORED AFTER ${C.SETTLE_S} S`, 12 * dp, Hc - 5 * dp);
  }

  function readout() {
    if (!run) return;
    run.lanes.forEach((ln, i) => {
      const k = i === 0 ? 'a' : 'b';
      const nTot = Math.round((Math.min(ln.env.t, run.cfg.episode_s) - C.SETTLE_S) / C.PHYS.dt_s);
      out(`rms-${k}`).textContent = ln.n ? `${Math.sqrt(ln.se / ln.n).toFixed(2)} m` : '–';
      out(`band-${k}`).textContent = nTot > 0 ? `${Math.round((100 * ln.inBand) / nTot)}%` : '–';
      out(`min-${k}`).textContent = `${(ln.env.minRange / run.cfg.standoff_m).toFixed(2)} d`;
      out(`state-${k}`).textContent = ln.dead ? ln.dead : ln.env.done ? 'done' : 'tracking';
    });
  }

  function describe() {
    const t = data.test, f = t.families, tr = data.training;
    const avg = (keys, who, m) => keys.reduce((s, k) => s + f[k][who][m], 0) / keys.length;
    const fails = (keys, who) => keys.reduce((s, k) => s + f[k][who].failures, 0);
    const n = (keys) => keys.length * t.episodes_per_family;
    const seen = C.TRAIN_FAMILIES, unseen = C.HELDOUT_FAMILIES;
    const part = (keys, label) => `${label}: RMS ${avg(keys, 'rl', 'rms_m').toFixed(2)} vs ${avg(keys, 'p', 'rms_m').toFixed(2)} m, ` +
      `in band ${Math.round(100 * avg(keys, 'rl', 'in_band'))}% vs ${Math.round(100 * avg(keys, 'p', 'in_band'))}%, ` +
      `lost or too close ${fails(keys, 'rl')}/${n(keys)} vs ${fails(keys, 'p')}/${n(keys)}`;
    out('policy').textContent =
      `Lime, learned: Double DQN, ${data.params.toLocaleString('en')} params, 25 discrete accelerations, best of ${tr.episodes.toLocaleString('en')} ` +
      `training episodes (seed ${tr.seed}). Peach, classical: proportional control on range error and relative velocity, gains tuned on the same ` +
      `validation runs (Kp ${data.p_gains.Kp}, Kv ${data.p_gains.Kv}${data.p_gains.at_grid_edge ? ', at the edge of the tuning grid, so a wider search might do better' : ''}). Held-out test, ${t.episodes_per_family} seeded runs per path at d = ${t.standoff_m} m, ` +
      `learned vs classical. ${part(seen, 'Paths it trained on')}. ${part(unseen, 'Paths it never saw')}.`;
  }

  btn('new').addEventListener('click', newRun);
  btn('path').addEventListener('click', () => { famIdx = (famIdx + 1) % families.length; newRun(); });
  canvas.addEventListener('pointerup', (ev) => { if (ev.pointerType !== 'mouse' || ev.button === 0) newRun(); });

  let uiTimer = 0;
  const loop = new Loop((dt) => {
    if (run && !run.lanes.every((l) => l.env.done)) {
      acc += dt * PLAYBACK;
      let n = 0;
      while (acc >= C.PHYS.dt_s && n < 8) { stepRun(); acc -= C.PHYS.dt_s; n++; }
      if (n === 8) acc = 0;
    } else if (run) {
      acc += dt;
      if (acc > 2.5) newRun(); // pause on the finished run, then start the next seed
    }
    draw();
    uiTimer += dt;
    if (uiTimer > 0.15) { readout(); uiTimer = 0; }
  });

  draw();
  out('policy').textContent = 'Loading policy…';
  fetch(POLICY_URL).then((r) => {
    if (!r.ok) throw new Error(`policy fetch failed: HTTP ${r.status}`);
    return r.json();
  }).then((d) => {
    data = d;
    rlNet = C.MLP.fromJSON(d.net);
    describe();
    newRun();
  }).catch((err) => {
    console.warn('[demo:standoff] policy unavailable:', err);
    out('policy').textContent = 'The policy file could not be loaded.';
  });
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
