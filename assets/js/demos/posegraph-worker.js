// Runs the pose-graph solve off the main thread so the page keeps animating.
import { optimise } from './posegraph.js?v=20261106e';

self.onmessage = (ev) => {
  const { id, poses, edges } = ev.data;
  try {
    const r = optimise(new Float64Array(poses), edges);
    self.postMessage({ id, ok: true, status: r.status, iterates: r.iterates.map((it) => ({ poses: it.poses, chi2: it.chi2 })) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err) });
  }
};
