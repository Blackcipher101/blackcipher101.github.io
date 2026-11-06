#!/usr/bin/env node
// Score the shipped box-track diffusion model against the constant-velocity baseline on held-out
// synthetic tracks, and store the numbers next to the weights for the "Where will it be" demo.
// Node >= 18, no dependencies. Uses assets/js/demos/lib/boxtrack-core.js unchanged.
//
// Protocol (same as the original artifact's in-browser scorecard): test tracks from seed 12345,
// sampler seed 999, 200 tracks per range, 32 diffusion samples, 10 DDIM steps (eta 1). The test
// stream is never used for training or for calibrating the regions.
//
//   node tools/eval_boxtrack.mjs            # updates the "eval" field of assets/data/boxtrack-model.json
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const B = require(path.join(ROOT, 'assets/js/demos/lib/boxtrack-core.js'));
const FILE = path.join(ROOT, 'assets/data/boxtrack-model.json');

const TEST_SEED = 12345, SAMPLER_SEED = 999, TRACKS = 200, SAMPLES = 32, DDIM_STEPS = 10, ETA = 1;
const RANGES_M = [500, 1000, 2000];
const data = JSON.parse(fs.readFileSync(FILE, 'utf8'));
const model = B.importModel(data.model);
const median = (a) => { const s = a.slice().sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
const k = B.H - 1;

const byRange = [];
for (const range of RANGES_M) {
  const t0 = Date.now();
  const r = B.rng(TEST_SEED), rs = B.rng(SAMPLER_SEED);
  const tracks = Array.from({ length: TRACKS }, () => B.genTrack(r, { range_m: range }));
  const acc = { cv: 0, df: 0, perH: { cv: new Array(B.H).fill(0), df: new Array(B.H).fill(0) },
    reg: { cv: { in50: 0, in90: 0, nll: 0, area: [] }, df: { in50: 0, in90: 0, nll: 0, area: [] } },
    modes: B.MODES.map(() => ({ n: 0, cv: 0, df: 0 })) };
  for (const tr of tracks) {
    const ev = B.evalTrack(model, tr, rs, SAMPLES, DDIM_STEPS, { eta: ETA, areaKs: [k] });
    ev.rows.forEach((e, h) => { acc.perH.cv[h] += Math.hypot(e.cvX, e.cvY) / TRACKS; acc.perH.df[h] += Math.hypot(e.dfX, e.dfY) / TRACKS; });
    const e = ev.rows[k], ce = Math.hypot(e.cvX, e.cvY), de = Math.hypot(e.dfX, e.dfY);
    acc.cv += ce / TRACKS; acc.df += de / TRACKS;
    const m = acc.modes[tr.mode]; m.n++; m.cv += ce; m.df += de;
    for (const key of ['cv', 'df']) {
      const g = ev.region[k][key], a = acc.reg[key];
      a.in50 += g.inside[0.5] ? 1 : 0; a.in90 += g.inside[0.9] ? 1 : 0; a.nll += g.nll / TRACKS; a.area.push(g.area[0.9]);
    }
  }
  const reg = (a) => ({ in50: +(a.in50 / TRACKS).toFixed(3), in90: +(a.in90 / TRACKS).toFixed(3),
    area90_m2_median: +median(a.area).toFixed(1), loglik: +(-a.nll).toFixed(2) });
  const row = {
    range_m: range,
    err_1p2s_m: { cv: +acc.cv.toFixed(2), diffusion: +acc.df.toFixed(2) },
    err_by_horizon_m: { cv: acc.perH.cv.map((v) => +v.toFixed(2)), diffusion: acc.perH.df.map((v) => +v.toFixed(2)) },
    regions_1p2s: { cv: reg(acc.reg.cv), diffusion: reg(acc.reg.df) },
    by_motion: acc.modes.map((m, i) => ({ motion: B.MODES[i], n: m.n, cv_m: +(m.cv / m.n).toFixed(2), diffusion_m: +(m.df / m.n).toFixed(2) })),
  };
  byRange.push(row);
  console.log(`${range} m: error at 1.2 s CV ${row.err_1p2s_m.cv} m, diffusion ${row.err_1p2s_m.diffusion} m | ` +
    `90% region holds truth CV ${(100 * row.regions_1p2s.cv.in90).toFixed(0)}% (median ${row.regions_1p2s.cv.area90_m2_median} m2), ` +
    `diffusion ${(100 * row.regions_1p2s.diffusion.in90).toFixed(0)}% (median ${row.regions_1p2s.diffusion.area90_m2_median} m2) | ` +
    row.by_motion.map((m) => `${m.motion} ${m.cv_m}/${m.diffusion_m}`).join(', ') + ` | ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}

data.eval = { protocol: { test_seed: TEST_SEED, sampler_seed: SAMPLER_SEED, tracks_per_range: TRACKS, samples: SAMPLES,
  ddim_steps: DDIM_STEPS, eta: ETA, horizon_s: B.H * B.DT_S, error: 'distance between predicted (sample mean) and true box ' +
  'centre at 1.2 s, converted to metres at the true range' }, by_range: byRange };
fs.writeFileSync(FILE, JSON.stringify(data));
console.log(`updated ${FILE} (${fs.statSync(FILE).size} bytes)`);
