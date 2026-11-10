#!/usr/bin/env node
// Headless run of the linear KF / EKF / UKF localisation demo for fixed seeds. Prints, per
// scenario, position RMSE, the share of time the truth is inside the 2-sigma position ellipse
// (ideal for a consistent 2-D Gaussian: 1 - exp(-2) = 86.5 %) and the mean position NEES (ideal 2).
// Node >= 22 (imports the browser ES module lib/filters-core.js unchanged).
//
//   node tools/eval_filters.mjs
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const F = await import(path.join(ROOT, 'assets/js/demos/lib/filters-core.js'));

const SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);
const SCENARIOS = [
  ...F.BEARING_SIGMAS_DEG.map((d) => ({ name: `bearing ${d} deg, 2 Hz`, bearingDeg: d, rateHz: 2 })),
  { name: 'bearing 10 deg, 0.5 Hz', bearingDeg: 10, rateHz: 0.5 },
  { name: 'bearing 10 deg, 2 Hz, beacons off 5 s at 40/70/100 s', bearingDeg: 10, rateHz: 2, off: [40, 70, 100] },
];

function run(sc, seed) {
  const sim = new F.Sim({ seed, sigmaBearingRad: F.deg2rad(sc.bearingDeg), rateHz: sc.rateHz });
  const off = new Set((sc.off || []).map((t) => Math.round(t / F.DT)));
  const n = Math.round(F.RUN_S / F.DT);
  for (let i = 0; i < n; i++) {
    if (off.has(sim.k)) sim.beaconsOff(5);
    sim.step();
  }
  return sim;
}

const pct = (v) => `${(100 * v).toFixed(1)}%`;
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const line = (label, r) => console.log(`  ${label.padEnd(10)} RMSE ${r.rmse.toFixed(3)} m   in 2-sigma ${pct(r.in2)}   NEES ${r.nees.toFixed(2)}`);

for (const sc of SCENARIOS) {
  console.log(`\n== ${sc.name} (${F.RUN_S} s, ${SEEDS.length} seeds) ==`);
  const sims = SEEDS.map((s) => run(sc, s));
  const first = sims[0];
  console.log(`seed 1: raw converted-fix RMSE ${first.fixRmse().toFixed(3)} m, odometry-only RMSE ${first.summary('dead').rmse.toFixed(2)} m, UKF resets ${first.ukf.resets}`);
  for (const key of ['kf', 'ekf', 'ukf']) line(`${first.filters[key].name} s1`, first.summary(key));
  console.log('mean over seeds:');
  console.log(`  raw fix RMSE ${mean(sims.map((s) => s.fixRmse())).toFixed(3)} m`);
  for (const key of ['kf', 'ekf', 'ukf']) {
    const ss = sims.map((s) => s.summary(key));
    line(sims[0].filters[key].name, { rmse: mean(ss.map((r) => r.rmse)), in2: mean(ss.map((r) => r.in2)), nees: mean(ss.map((r) => r.nees)) });
  }
  const nonfinite = sims.filter((s) => !Object.values(s.filters).every((f) => f.pos().every(Number.isFinite))).length;
  console.log(`  UKF resets (total over seeds) ${sims.reduce((a, s) => a + s.ukf.resets, 0)}, EKF resets ${sims.reduce((a, s) => a + s.ekf.resets, 0)}, KF resets ${sims.reduce((a, s) => a + s.kf.resets, 0)}, non-finite runs ${nonfinite}`);
}
