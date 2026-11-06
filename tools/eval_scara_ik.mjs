#!/usr/bin/env node
// Score the demo's classical SCARA planner (analytic IK, C-space A* around the obstacle when the
// straight joint move would hit it, minimum-jerk timing) on the same held-out task sets as the learned
// policy, and store the result as eval.ik_astar in assets/data/scara-policy.json.
// Node >= 18. Imports the planner from assets/js/demos/scara.js itself, so this is the code the page runs.
//
//   python3 tools/train_scara.py --eval-only --dump-tasks /tmp/heldout.json
//   node tools/eval_scara_ik.mjs /tmp/heldout.json
//
// Same criterion as the policy: tip within 1 cm and slower than 2 cm/s within 5 s, still within 1 cm
// at 6 s, and no obstacle contact, checked every 20 ms along the executed trajectory. A task the
// planner refuses (no IK pose clear of the obstacle, or A* finds no path on its grid) is a failure.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { runTask } from './scara_ik_task.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FILE = path.join(ROOT, 'assets/data/scara-policy.json');
const tasksPath = process.argv[2];
if (!tasksPath) throw new Error('usage: node tools/eval_scara_ik.mjs <held-out tasks JSON from train_scara.py --dump-tasks>');
const sets = JSON.parse(fs.readFileSync(tasksPath, 'utf8'));
const data = JSON.parse(fs.readFileSync(FILE, 'utf8'));

const out = {};
for (const mode of ['free', 'random', 'demo']) {
  const s = sets[mode], t0 = Date.now(), times = [];
  let ok = 0, touch = 0, refused = 0;
  for (let i = 0; i < s.q0.length; i++) {
    const r = runTask(s.q0[i], s.tg[i], s.ob[i]);
    ok += r.success; touch += r.touch; refused += !r.planned;
    if (r.success) times.push(r.time);
  }
  times.sort((a, b) => a - b);
  out[mode] = { trials: s.q0.length, success: ok, touch, refused,
    median_s: times.length ? +times[Math.floor(times.length / 2)].toFixed(2) : null,
    p90_s: times.length ? +times[Math.floor(times.length * 0.9)].toFixed(2) : null };
  console.log(`[${mode}] IK + A* + min-jerk: ${ok}/${s.q0.length} reached, touch ${touch}, refused ${refused}, ` +
    `median ${out[mode].median_s} s, p90 ${out[mode].p90_s} s (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
data.eval.ik_astar = { method: 'analytic IK (elbow-left preferred), C-space A* when the straight move hits the obstacle, ' +
  'minimum-jerk timing; tools/eval_scara_ik.mjs on the same held-out tasks', ...out };
fs.writeFileSync(FILE, JSON.stringify(data));
console.log(`updated ${FILE} (${fs.statSync(FILE).size} bytes)`);
