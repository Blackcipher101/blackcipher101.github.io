// Box-track prediction in a long-range camera: a small conditional diffusion model (learned) vs the
// constant-velocity prediction a standard tracker makes (classical).
// The model, synthetic track generator, baseline and calibrated probability regions live in
// lib/boxtrack-core.js (unchanged from the original artifact). Weights and the held-out scorecard
// ship in assets/data/boxtrack-model.json; tools/eval_boxtrack.mjs measures that scorecard.
// Everything here is synthetic: a 1280 x 720 px camera with a 30 deg field of view, 10 Hz boxes,
// 0.8 s observed, 1.2 s predicted. No real sensor data is involved.
import { View, Loop, COLORS, prefersReducedMotion } from './common.js?v=20261106l';
import './lib/boxtrack-core.js?v=20261106l';

const B = window.BBD;
const MODEL_URL = new URL('../../data/boxtrack-model.json', import.meta.url);
MODEL_URL.search = new URL(import.meta.url).search;

const SAMPLES = 64, DDIM_STEPS = 10, ETA = 1;       // same sampler settings as the scorecard (which uses 32 samples)
const MOTIONS = [-1, 0, 1, 2, 3];                    // -1 = any; indices into B.MODES
const MOTION_LABEL = ['any', 'straight', 'steady turn', 'weave', 'late turn'];
const AGILITY = [['any', null], ['low', [2, 5]], ['medium', [5, 12]], ['high', [12, 30]]]; // max sideways accel, m/s^2
const RANGES_M = [500, 1000, 2000];
const G_MPS2 = 9.81;
const HOLD_NOISE_S = 0.35, PER_STEP_S = 0.17;        // denoising animation timing
const GRID_W = 160, GRID_H = 90;                     // density grid for the region contours

// Marching squares: line segments where a gridded field crosses thr (grid units).
const MS_CASES = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]],
  [[2, 3]], [[0, 2]], [[0, 1], [2, 3]], [[1, 2]], [[1, 3]], [[0, 1]], [[0, 3]], []];
function contour(vals, gw, gh, thr) {
  const segs = [];
  const at = (i, j) => vals[j * gw + i];
  const lerp = (a, b) => (thr - a) / ((b - a) || 1e-30);
  for (let j = 0; j < gh - 1; j++) for (let i = 0; i < gw - 1; i++) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    const idx = (a >= thr ? 1 : 0) | (b >= thr ? 2 : 0) | (c >= thr ? 4 : 0) | (d >= thr ? 8 : 0);
    if (idx === 0 || idx === 15) continue;
    const pt = (e) => (e === 0 ? [i + lerp(a, b), j] : e === 1 ? [i + 1, j + lerp(b, c)] : e === 2 ? [i + lerp(d, c), j + 1] : [i, j + lerp(a, d)]);
    for (const [e1, e2] of MS_CASES[idx]) segs.push([pt(e1), pt(e2)]);
  }
  return segs;
}

export function mount(root) {
  const canvas = root.querySelector('canvas');
  const view = new View(canvas, 16, 9);
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = (k) => root.querySelector(`[data-act="${k}"]`);
  const reduce = prefersReducedMotion();
  const rng = B.rng(20261106);

  let model = null, data = null, cur = null, animT = null, frame = null;
  let motionIdx = 0, agilIdx = 0, rangeIdx = 0;

  function cycle(name, label) { btn(name).textContent = label; }

  /** Crop of the image (px) that keeps the past boxes, the truth, both predictions and most samples in view. */
  function cropFor(c) {
    const { tr, ft, mean } = c, k = B.H - 1, xs = [], ys = [];
    let smax = 0;
    for (let i = 0; i < B.P; i++) { xs.push(tr.ou[i]); ys.push(tr.ov[i]); smax = Math.max(smax, tr.os[i]); }
    for (let i = 0; i < B.H; i++) {
      xs.push(tr.u[B.P + i], ft.cvU[i], mean.u[i]); ys.push(tr.v[B.P + i], ft.cvV[i], mean.v[i]);
      smax = Math.max(smax, tr.s[B.P + i]);
    }
    const su = c.samples.map((s) => s.u[k]).sort((p, q) => p - q), sv = c.samples.map((s) => s.v[k]).sort((p, q) => p - q);
    const q = (arr, f) => arr[Math.min(arr.length - 1, Math.floor(f * arr.length))];
    xs.push(q(su, 0.05), q(su, 0.95)); ys.push(q(sv, 0.05), q(sv, 0.95));
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = smax * 1.4 + 10;
    x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
    const aspect = canvas.width / canvas.height;
    let w = Math.max(x1 - x0, 120), h = Math.max(y1 - y0, 120 / aspect);
    if (w / h > aspect) h = w / aspect; else w = h * aspect;
    return { x0: (x0 + x1) / 2 - w / 2, y0: (y0 + y1) / 2 - h / 2, w, h };
  }

  function newTrack() {
    if (!model) return;
    const mode = MOTIONS[motionIdx], band = AGILITY[agilIdx][1], range = RANGES_M[rangeIdx];
    const env = B.sampleEnvelope(rng);
    if (band) env.aLat_mps2 = Math.exp(Math.log(band[0]) + (Math.log(band[1]) - Math.log(band[0])) * rng());
    const tr = B.genTrack(rng, { range_m: range, mode: mode < 0 ? undefined : mode, env });
    const ft = B.featurize(tr);
    const res = B.sample(model, ft, rng, SAMPLES, DDIM_STEPS, { eta: ETA, keepPath: true });
    const samples = res.samples, mean = B.meanOf(samples), k = B.H - 1;
    const scores = B.regionScores(tr, ft, samples, model, k, true); // metres at the true range, calibrated levels
    const mpp = tr.Z[B.P + k] / B.CAM.f;
    cur = { tr, ft, path: res.path, samples, mean, scores, range,
      errCv: Math.hypot(ft.cvU[k] - tr.u[B.P + k], ft.cvV[k] - tr.v[B.P + k]) * mpp,
      errDf: Math.hypot(mean.u[k] - tr.u[B.P + k], mean.v[k] - tr.v[B.P + k]) * mpp };
    cur.crop = cropFor(cur);
    frame = null;
    animT = reduce ? null : 0;
    readout();
  }

  /** Static layer for the finished prediction: density heat, calibrated 50/90% contours, CV 90% ellipse. */
  function buildRegionLayer(c, X, Y, z) {
    const k = B.H - 1, cr = c.crop, { tr, ft } = c;
    const layer = document.createElement('canvas');
    layer.width = canvas.width; layer.height = canvas.height;
    const g = layer.getContext('2d'), d = view.dpr;
    const xs = c.samples.map((s) => s.u[k]), ys = c.samples.map((s) => s.v[k]);
    const kd = B.kde2(xs, ys);
    const vals = new Float64Array(GRID_W * GRID_H);
    let vmax = 0;
    for (let j = 0; j < GRID_H; j++) for (let i = 0; i < GRID_W; i++) {
      const v = kd.density(cr.x0 + cr.w * i / (GRID_W - 1), cr.y0 + cr.h * j / (GRID_H - 1));
      vals[j * GRID_W + i] = v; vmax = Math.max(vmax, v);
    }
    const off = document.createElement('canvas');
    off.width = GRID_W; off.height = GRID_H;
    const og = off.getContext('2d'), img = og.createImageData(GRID_W, GRID_H);
    for (let p = 0; p < GRID_W * GRID_H; p++) {
      const a = Math.pow(vals[p] / (vmax || 1), 0.6);
      img.data[4 * p] = 217; img.data[4 * p + 1] = 244; img.data[4 * p + 2] = 91; img.data[4 * p + 3] = Math.round(110 * a);
    }
    og.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    const cw = layer.width / (GRID_W - 1), ch = layer.height / (GRID_H - 1);
    g.drawImage(off, -cw / 2, -ch / 2, layer.width + cw, layer.height + ch);
    const gx = (i) => X(cr.x0 + cr.w * i / (GRID_W - 1)), gy = (j) => Y(cr.y0 + cr.h * j / (GRID_H - 1));
    for (const [p, dash, lw] of [[0.9, [10 * d, 7 * d], 1.5], [0.5, [], 2.5]]) {
      g.strokeStyle = COLORS.lime; g.lineWidth = lw * d; g.setLineDash(dash);
      g.beginPath();
      const thr = kd.threshold(B.dfLevel(model, xs.length, k, p));
      for (const [[i1, j1], [i2, j2]] of contour(vals, GRID_W, GRID_H, thr)) { g.moveTo(gx(i1), gy(j1)); g.lineTo(gx(i2), gy(j2)); }
      g.stroke();
    }
    // constant-velocity 90% ellipse, calibrated radius (pixel covariance: one "metre" per pixel)
    const S = B.cvGaussian(ft, k, model.norm, 1), r2 = B.cvR2(model, k, 0.9);
    const tm = (S.xx + S.yy) / 2, dd = Math.sqrt(((S.xx - S.yy) / 2) ** 2 + S.xy ** 2);
    g.strokeStyle = COLORS.peach; g.lineWidth = 2 * d; g.setLineDash([8 * d, 6 * d]);
    g.beginPath();
    g.ellipse(X(ft.cvU[k]), Y(ft.cvV[k]), Math.sqrt((tm + dd) * r2) * z, Math.sqrt(Math.max(tm - dd, 1e-9) * r2) * z,
      0.5 * Math.atan2(2 * S.xy, S.xx - S.yy), 0, 2 * Math.PI);
    g.stroke(); g.setLineDash([]);
    // true position at 1.2 s
    const tu = X(tr.u[B.P + k]), tv = Y(tr.v[B.P + k]);
    g.strokeStyle = COLORS.cream; g.lineWidth = 2.5 * d;
    g.beginPath(); g.moveTo(tu - 10 * d, tv); g.lineTo(tu + 10 * d, tv); g.moveTo(tu, tv - 10 * d); g.lineTo(tu, tv + 10 * d); g.stroke();
    return layer;
  }

  function lerpPaths(a, b, f) {
    return a.map((sa, n) => {
      const sb = b[n], u = new Float64Array(B.H), v = new Float64Array(B.H);
      for (let i = 0; i < B.H; i++) { u[i] = sa.u[i] + (sb.u[i] - sa.u[i]) * f; v[i] = sa.v[i] + (sb.v[i] - sa.v[i]) * f; }
      return { u, v, s: sb.s };
    });
  }

  function draw() {
    const { ctx } = view, d = view.dpr, W = canvas.width, Hc = canvas.height;
    view.clear('#0d1412');
    if (!cur) {
      ctx.fillStyle = COLORS.creamDim; ctx.font = `${12 * d}px ui-monospace, Menlo, monospace`;
      ctx.fillText(model ? '' : 'LOADING MODEL…', 18 * d, 28 * d);
      return;
    }
    const c = cur, cr = c.crop, z = W / cr.w;
    const X = (u) => (u - cr.x0) * z, Y = (v) => (v - cr.y0) * z;
    // image-pixel grid
    const stepPx = [5, 10, 20, 50, 100, 200].find((s) => s * z >= 90 * d) || 200;
    ctx.strokeStyle = '#1d2925'; ctx.lineWidth = 1; ctx.fillStyle = '#7f948d';
    ctx.font = `${10 * d}px ui-monospace, Menlo, monospace`;
    ctx.beginPath();
    for (let gx = Math.ceil(cr.x0 / stepPx) * stepPx; gx <= cr.x0 + cr.w; gx += stepPx) { ctx.moveTo(X(gx), 0); ctx.lineTo(X(gx), Hc); }
    for (let gy = Math.ceil(cr.y0 / stepPx) * stepPx; gy <= cr.y0 + cr.h; gy += stepPx) { ctx.moveTo(0, Y(gy)); ctx.lineTo(W, Y(gy)); }
    ctx.stroke();
    const nSnap = c.path.length, total = HOLD_NOISE_S + PER_STEP_S * (nSnap - 1);
    const done = animT === null || animT >= total;
    if (done) {
      if (!frame) frame = buildRegionLayer(c, X, Y, z);
      ctx.drawImage(frame, 0, 0);
    }
    const box = (u, v, s, color, lw, dash, alpha = 1) => {
      const w = s * z, h = s * 0.6 * z;
      ctx.globalAlpha = alpha; ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = lw * d;
      ctx.strokeRect(X(u) - w / 2, Y(v) - h / 2, w, h);
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    };
    const line = (us, vs, color, lw, dash, from) => {
      ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = lw * d; ctx.lineJoin = 'round';
      ctx.beginPath();
      if (from) ctx.moveTo(X(from[0]), Y(from[1]));
      for (let i = 0; i < us.length; i++) (i === 0 && !from) ? ctx.moveTo(X(us[i]), Y(vs[i])) : ctx.lineTo(X(us[i]), Y(vs[i]));
      ctx.stroke(); ctx.setLineDash([]);
    };
    const { tr, ft } = c, now = [tr.ou[B.P - 1], tr.ov[B.P - 1]], dash = (a, b) => [a * d, b * d];
    for (let i = 0; i < B.P; i++) box(tr.ou[i], tr.ov[i], tr.os[i], '#9fb1ab', 1.5, null, 0.25 + 0.75 * (i + 1) / B.P);
    line(Array.from(tr.ou), Array.from(tr.ov), '#9fb1ab', 1.5, dash(2, 5));
    let snaps = c.samples, step = DDIM_STEPS;
    if (!done) {
      const p = Math.max(0, animT - HOLD_NOISE_S) / PER_STEP_S, i = Math.min(Math.floor(p), nSnap - 2);
      snaps = lerpPaths(c.path[i], c.path[i + 1], p - i);
      step = i;
    }
    for (const s of snaps) line(s.u, s.v, 'rgba(217,244,91,0.22)', 1.5, null, now);
    const tu = Array.from(tr.u.slice(B.P)), tv = Array.from(tr.v.slice(B.P));
    line(tu, tv, COLORS.cream, 2.5, null, now);
    box(tu[B.H - 1], tv[B.H - 1], tr.s[B.P + B.H - 1], COLORS.cream, 2.5);
    line(Array.from(ft.cvU), Array.from(ft.cvV), COLORS.peach, 2.5, dash(10, 7), now);
    box(ft.cvU[B.H - 1], ft.cvV[B.H - 1], Math.exp(ft.cvL[B.H - 1]), COLORS.peach, 2.5, dash(10, 7));
    const m = done ? c.mean : B.meanOf(snaps);
    line(Array.from(m.u), Array.from(m.v), COLORS.lime, 3, null, now);
    if (done) box(m.u[B.H - 1], m.v[B.H - 1], m.s[B.H - 1], COLORS.lime, 2.5);
    ctx.fillStyle = '#7f948d'; ctx.font = `${10 * d}px ui-monospace, Menlo, monospace`;
    ctx.fillText(`CROP ${cr.w.toFixed(0)} × ${cr.h.toFixed(0)} PX · ${c.range} M`, 12 * d, 18 * d);
    const right = done ? `${c.samples.length} SAMPLES · REGIONS AT 1.2 S` : `DENOISING ${step}/${DDIM_STEPS}`;
    const left = `CROP ${cr.w.toFixed(0)} × ${cr.h.toFixed(0)} PX · ${c.range} M`;
    const fits = ctx.measureText(left).width + ctx.measureText(right).width + 36 * d <= W;
    ctx.fillText(right, fits ? W - ctx.measureText(right).width - 12 * d : 12 * d, fits ? 18 * d : 32 * d); // narrow: second line
  }

  function readout() {
    if (!cur) return;
    const s = cur.scores, e = cur.tr.env;
    out('err-a').textContent = `${cur.errDf.toFixed(1)} m`;
    out('err-b').textContent = `${cur.errCv.toFixed(1)} m`;
    out('area-a').textContent = `${s.df.area[0.9].toFixed(0)} m²`;
    out('area-b').textContent = `${s.cv.area[0.9].toFixed(0)} m²`;
    out('in-a').textContent = s.df.inside[0.9] ? 'yes' : 'no';
    out('in-b').textContent = s.cv.inside[0.9] ? 'yes' : 'no';
    out('target').textContent = `${B.MODES[cur.tr.mode]} · ${(e.aLat_mps2 / G_MPS2).toFixed(1)} g`;
  }

  function describe() {
    const meta = data.model.meta, ev = data.eval, r = ev.by_range.find((x) => x.range_m === RANGES_M[rangeIdx]) || ev.by_range[0];
    const rc = r.regions_1p2s, bm = Object.fromEntries(r.by_motion.map((x) => [x.motion, x]));
    // Comparative words follow the measured numbers: within 10 % counts as "about the same".
    const rel = (a, b) => (a - b) / b;
    const errRel = rel(r.err_1p2s_m.diffusion, r.err_1p2s_m.cv);
    const point = Math.abs(errRel) < 0.1 ? 'about as accurate as' : errRel < 0 ? 'more accurate than' : 'less accurate than';
    const ratio = rc.cv.area90_m2_median / rc.diffusion.area90_m2_median;
    const size = ratio >= 1.1 ? `${ratio.toFixed(1)}× smaller` : ratio <= 1 / 1.1 ? `${(1 / ratio).toFixed(1)}× larger` : 'about the same size';
    out('policy').textContent =
      `Lime, learned: a ${data.model.sizes.join('-')} denoising network, ${meta.trainedSteps.toLocaleString('en')} training steps on ` +
      `${meta.opts.nData.toLocaleString('en')} synthetic tracks. Peach, classical: constant-velocity extrapolation, as in a SORT-style tracker. ` +
      `On ${ev.protocol.tracks_per_range} held-out tracks at ${r.range_m} m, the average of the samples is ${point} constant velocity ` +
      `(${r.err_1p2s_m.diffusion} vs ${r.err_1p2s_m.cv} m off at 1.2 s; ${bm['steady turn'].diffusion_m} vs ${bm['steady turn'].cv_m} m in steady turns, ` +
      `${bm.straight.diffusion_m} vs ${bm.straight.cv_m} m on straight runs). Its 90% region held the truth ` +
      `${Math.round(rc.diffusion.in90 * 100)}% of the time (constant-velocity ellipse ${Math.round(rc.cv.in90 * 100)}%) and was ` +
      `${size} (median ${rc.diffusion.area90_m2_median.toFixed(0)} vs ${rc.cv.area90_m2_median.toFixed(0)} m²).`;
  }

  btn('new').addEventListener('click', newTrack);
  btn('replay').addEventListener('click', () => { if (cur && !reduce) { animT = 0; } });
  btn('motion').addEventListener('click', () => {
    motionIdx = (motionIdx + 1) % MOTIONS.length;
    cycle('motion', `Motion: ${MOTION_LABEL[motionIdx]}`);
    newTrack();
  });
  btn('agility').addEventListener('click', () => {
    agilIdx = (agilIdx + 1) % AGILITY.length;
    cycle('agility', `Agility: ${AGILITY[agilIdx][0]}`);
    newTrack();
  });
  btn('range').addEventListener('click', () => {
    rangeIdx = (rangeIdx + 1) % RANGES_M.length;
    cycle('range', `Range: ${RANGES_M[rangeIdx]} m`);
    if (data) describe(); // the held-out scorecard is quoted for the selected range
    newTrack();
  });
  canvas.addEventListener('pointerup', (ev) => { if (ev.pointerType !== 'mouse' || ev.button === 0) newTrack(); });

  let lastW = 0, lastH = 0;
  const loop = new Loop((dt) => {
    const resized = canvas.width !== lastW || canvas.height !== lastH;
    if (resized) { lastW = canvas.width; lastH = canvas.height; frame = null; if (cur) cur.crop = cropFor(cur); }
    const animating = cur && animT !== null && animT < HOLD_NOISE_S + PER_STEP_S * (cur.path.length - 1);
    if (animating) animT += dt;
    if (animating || resized || !frame) draw(); // the finished frame is static: redraw only when it changes
  });

  draw();
  out('policy').textContent = 'Loading model…';
  fetch(MODEL_URL).then((r) => {
    if (!r.ok) throw new Error(`model fetch failed: HTTP ${r.status}`);
    return r.json();
  }).then((d) => {
    data = d;
    model = B.importModel(d.model);
    describe();
    newTrack();
    draw();
  }).catch((err) => {
    console.warn('[demo:boxtrack] model unavailable:', err);
    out('policy').textContent = 'The model file could not be loaded.';
  });
  return { resume: () => loop.start(), pause: () => loop.stop() };
}
