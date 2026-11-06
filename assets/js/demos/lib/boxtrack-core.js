// Box-track diffusion core (synthetic tracks, DDPM/DDIM model, constant-velocity baseline, regions),
// from the "Box Track Diffusion" artifact; unchanged below this header. Used by demos/boxtrack.js and
// tools/eval_boxtrack.mjs.
// Tiny conditional diffusion model (DDPM training, DDIM sampling) for
// bounding-box trajectory prediction, plus a constant-velocity baseline.
// Plain JS, no dependencies. Works in the browser (window.BBD) and in Node.
(function (root) {
  'use strict';

  // ---- Problem dimensions -------------------------------------------------
  const P = 8;            // past frames observed (last one is "now")
  const H = 12;           // future frames predicted
  const DT_S = 0.1;       // frame period, 10 Hz
  const CV_FIT_FRAMES = 5;// frames used by the constant-velocity least-squares fit
  const XD = H * 3;       // predicted dims: H x (du, dv) residuals + H x log-size residual
  const N_ENV = 6;        // envelope features appended to the condition (see featurize)
  const CD = (P - 1) * 3 + 1 + N_ENV; // 7 past (du,dv) + 7 past log-size + log speed + envelope
  const ENV0 = CD - N_ENV;            // index of the first envelope feature
  const TE = 16;          // timestep embedding size
  const HID = 96;         // default hidden width (Trainer opts.hid overrides)
  const DEPTH = 3;        // default number of hidden layers (Trainer opts.depth overrides)
  const T_DIFF = 100;     // diffusion steps used in training
  const SIZE_SCALE = 10;  // log-size residuals are multiplied by this before whitening
  const MODES = ['straight', 'steady turn', 'weave', 'late turn'];

  // Camera: 1280 px wide, 30 deg horizontal field of view.
  const CAM = { W: 1280, H: 720, hfovDeg: 30 };
  CAM.f = (CAM.W / 2) / Math.tan((CAM.hfovDeg * Math.PI / 180) / 2);

  // ---- RNG ------------------------------------------------------------------
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0;
    while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  const uni = (r, a, b) => a + (b - a) * r();
  const sgn = (r) => (r() < 0.5 ? -1 : 1);

  // ---- Synthetic target tracks ---------------------------------------------
  // A target moves in the plane facing the camera (X right, Y up, metres)
  // while its range Z drifts. Each target has a performance envelope: the most
  // sideways acceleration it can pull (aLat_mps2), the most along-track
  // acceleration (aLong_mps2), its largest range rate (vz_mps) and its length
  // (L_m). These are what a tracker knows from the target class (a truck vs a
  // small drone vs a jet). Every manoeuvre stays inside the envelope:
  // turn rate = sideways acceleration / speed.
  const ENV_RANGES = {
    aLat_mps2: [2, 30],   // log-uniform: ~0.2 g truck to ~3 g agile aircraft
    aLong_mps2: [0.5, 6], // log-uniform
    vz_mps: [2, 20],      // uniform
    L_m: [4, 8],          // uniform
  };
  function sampleEnvelope(r) {
    const lu = (lo, hi) => Math.exp(uni(r, Math.log(lo), Math.log(hi)));
    return {
      aLat_mps2: lu(...ENV_RANGES.aLat_mps2),
      aLong_mps2: lu(...ENV_RANGES.aLong_mps2),
      vz_mps: uni(r, ...ENV_RANGES.vz_mps),
      L_m: uni(r, ...ENV_RANGES.L_m),
    };
  }
  function genTrack(r, opts) {
    const o = opts || {};
    const range = o.range_m || 500;
    const pxNoise = o.pxNoise == null ? 1.0 : o.pxNoise;
    const mode = o.mode != null ? o.mode : Math.floor(r() * MODES.length);
    const env = o.env || sampleEnvelope(r);
    let Z = range * uni(r, 0.9, 1.1);
    const vz = uni(r, -1, 1) * env.vz_mps;
    let V = uni(r, 10, 40);
    let psi = r() * 2 * Math.PI;
    const L = env.L_m;
    const acc = uni(r, -1, 1) * env.aLong_mps2;
    const aMax = env.aLat_mps2;
    let a0 = 0, A = 0, Tp = 1, ph = 0, ts = 0;  // sideways accelerations in m/s^2
    if (mode === 1) a0 = sgn(r) * uni(r, 0.3, 1.0) * aMax;
    if (mode === 2) { A = uni(r, 0.4, 1.0) * aMax; Tp = uni(r, 1.5, 3.0); ph = r() * 2 * Math.PI; }
    if (mode === 3) { ts = uni(r, (P - 3) * DT_S, (P + H - 3) * DT_S); a0 = sgn(r) * uni(r, 0.5, 1.0) * aMax; }
    const wander = gauss(r) * 0.03 * aMax;
    const N = P + H, SUB = 5, dt = DT_S / SUB;
    const u = new Float64Array(N), v = new Float64Array(N), s = new Float64Array(N), Zs = new Float64Array(N);
    let X = 0, Y = 0, t = 0;
    for (let k = 0; k < N; k++) {
      u[k] = CAM.W / 2 + CAM.f * X / Z;
      v[k] = CAM.H / 2 - CAM.f * Y / Z;
      s[k] = CAM.f * L / Z;
      Zs[k] = Z;
      for (let j = 0; j < SUB; j++) {
        let aLat = wander;
        if (mode === 1) aLat += a0;
        else if (mode === 2) aLat += A * Math.sin(2 * Math.PI * t / Tp + ph);
        else if (mode === 3 && t >= ts) aLat += a0;
        aLat = Math.max(-aMax, Math.min(aMax, aLat));
        psi += (aLat / V) * dt;
        V = Math.max(3, V + acc * dt);
        X += V * Math.cos(psi) * dt;
        Y += V * Math.sin(psi) * dt;
        Z = Math.max(50, Z + vz * dt);
        t += dt;
      }
    }
    // Detector noise on the observed (past) boxes only; the future is ground truth.
    const ou = new Float64Array(P), ov = new Float64Array(P), os = new Float64Array(P);
    for (let k = 0; k < P; k++) {
      ou[k] = u[k] + gauss(r) * pxNoise;
      ov[k] = v[k] + gauss(r) * pxNoise;
      os[k] = s[k] * Math.exp(gauss(r) * 0.03);
    }
    return { u, v, s, Z: Zs, ou, ov, os, mode, env };
  }

  // ---- Constant-velocity baseline + canonical frame ------------------------
  // Least-squares line through the last CV_FIT_FRAMES observations, extrapolated.
  // This is what a steady-state constant-velocity Kalman filter (SORT-style) does.
  function lsFit(arr, n) {
    const k0 = P - n;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
      const x = i - (n - 1), y = arr[k0 + i];
      sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const icpt = (sy - slope * sx) / n; // value at the last frame (x = 0)
    return [icpt, slope];
  }

  function featurize(tr) {
    const [u0, vu] = lsFit(tr.ou, CV_FIT_FRAMES);
    const [v0, vv] = lsFit(tr.ov, CV_FIT_FRAMES);
    const ls = new Float64Array(P);
    for (let k = 0; k < P; k++) ls[k] = Math.log(tr.os[k]);
    const [l0, lr] = lsFit(ls, CV_FIT_FRAMES);
    const speed = Math.hypot(vu, vv);
    const th = speed > 1e-6 ? Math.atan2(vv, vu) : 0;
    const c = Math.max(speed * H, 3); // px: distance covered over the horizon
    const cs = Math.cos(th), sn = Math.sin(th);
    const cond = new Float64Array(CD);
    for (let k = 0; k < P - 1; k++) {
      const du = tr.ou[k] - u0, dv = tr.ov[k] - v0;
      cond[2 * k] = (cs * du + sn * dv) / c;
      cond[2 * k + 1] = (-sn * du + cs * dv) / c;
      cond[2 * (P - 1) + k] = (ls[k] - l0) * SIZE_SCALE;
    }
    cond[ENV0 - 1] = Math.log(speed + 0.1);
    // Envelope features. Range comes from the known target length and the
    // observed box size (Z = f L / s), so metres-per-pixel = L / s_now.
    // The two "reach" features say how far (in the canonical horizon units the
    // model predicts in) full sideways / along-track acceleration could move
    // the target off the constant-velocity line by the end of the horizon.
    const e = tr.env;
    const mpp = e.L_m / Math.exp(l0);
    const T_s = H * DT_S;
    const reach = (a) => 0.5 * a * T_s * T_s / mpp / c;
    cond[ENV0] = Math.log(e.aLat_mps2);
    cond[ENV0 + 1] = Math.log(e.aLong_mps2);
    cond[ENV0 + 2] = Math.log(e.vz_mps);
    cond[ENV0 + 3] = Math.log(mpp);
    cond[ENV0 + 4] = Math.log(reach(e.aLat_mps2) + 1e-3);
    cond[ENV0 + 5] = Math.log(reach(e.aLong_mps2) + 1e-3);
    const cvU = new Float64Array(H), cvV = new Float64Array(H), cvL = new Float64Array(H);
    for (let k = 1; k <= H; k++) {
      cvU[k - 1] = u0 + vu * k; cvV[k - 1] = v0 + vv * k; cvL[k - 1] = l0 + lr * k;
    }
    return { cond, cs, sn, c, cvU, cvV, cvL };
  }

  // Residual of the true future w.r.t. the CV prediction, in the canonical frame.
  function encodeTarget(tr, ft) {
    const x = new Float64Array(XD);
    for (let k = 0; k < H; k++) {
      const du = tr.u[P + k] - ft.cvU[k], dv = tr.v[P + k] - ft.cvV[k];
      x[2 * k] = (ft.cs * du + ft.sn * dv) / ft.c;
      x[2 * k + 1] = (-ft.sn * du + ft.cs * dv) / ft.c;
      x[2 * H + k] = (Math.log(tr.s[P + k]) - ft.cvL[k]) * SIZE_SCALE;
    }
    return x;
  }
  function decode(x, ft) { // x in canonical (un-whitened) units -> pixel boxes
    const u = new Float64Array(H), v = new Float64Array(H), s = new Float64Array(H);
    for (let k = 0; k < H; k++) {
      const a = x[2 * k] * ft.c, b = x[2 * k + 1] * ft.c;
      u[k] = ft.cvU[k] + ft.cs * a - ft.sn * b;
      v[k] = ft.cvV[k] + ft.sn * a + ft.cs * b;
      s[k] = Math.exp(ft.cvL[k] + x[2 * H + k] / SIZE_SCALE);
    }
    return { u, v, s };
  }

  // ---- Diffusion schedule (cosine) ------------------------------------------
  const alphaBar = new Float64Array(T_DIFF + 1);
  {
    const sOff = 0.008, f = (t) => Math.cos(((t / T_DIFF + sOff) / (1 + sOff)) * Math.PI / 2) ** 2;
    for (let t = 0; t <= T_DIFF; t++) alphaBar[t] = Math.max(f(t) / f(0), 1e-5);
  }
  function tEmbed(t, out, off) { // t in [1, T_DIFF]
    const x = t / T_DIFF;
    for (let i = 0; i < TE / 2; i++) {
      const w = Math.pow(2, i) * Math.PI;
      out[off + 2 * i] = Math.sin(w * x);
      out[off + 2 * i + 1] = Math.cos(w * x);
    }
  }

  // ---- MLP with manual backprop ---------------------------------------------
  const IN = XD + CD + TE;
  function netSizes(hid, depth) {
    const s = [IN];
    for (let i = 0; i < depth; i++) s.push(hid);
    s.push(XD);
    return s;
  }
  const SIZES = netSizes(HID, DEPTH); // default architecture
  const sizesOf = (layers) => [layers[0].nin].concat(layers.map((L) => L.nout));
  function newParams(r, sizes) {
    const layers = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const W = new Float32Array(nin * nout), b = new Float32Array(nout);
      const sc = (l === sizes.length - 2 ? 0.3 : 1.4) / Math.sqrt(nin);
      for (let i = 0; i < W.length; i++) W[i] = gauss(r) * sc;
      layers.push({ W, b, nin, nout });
    }
    return layers;
  }
  const sigm = (z) => 1 / (1 + Math.exp(-z));

  function makeWorkspace(B, layers) {
    const sizes = sizesOf(layers);
    return {
      B,
      X: new Float32Array(B * IN),
      Z: sizes.slice(1).map((n) => new Float32Array(B * n)),
      A: sizes.slice(1).map((n) => new Float32Array(B * n)),
      dA: sizes.map((n) => new Float32Array(B * n)),
    };
  }
  function forward(layers, ws, B) {
    let inp = ws.X;
    for (let l = 0; l < layers.length; l++) {
      const { W, b, nin, nout } = layers[l];
      const Z = ws.Z[l], A = ws.A[l], last = l === layers.length - 1;
      for (let n = 0; n < B; n++) {
        const zo = n * nout, io = n * nin;
        for (let j = 0; j < nout; j++) Z[zo + j] = b[j];
        for (let i = 0; i < nin; i++) {
          const xi = inp[io + i];
          if (xi === 0) continue;
          const wo = i * nout;
          for (let j = 0; j < nout; j++) Z[zo + j] += xi * W[wo + j];
        }
        if (last) for (let j = 0; j < nout; j++) A[zo + j] = Z[zo + j];
        else for (let j = 0; j < nout; j++) { const z = Z[zo + j]; A[zo + j] = z * sigm(z); }
      }
      inp = A;
    }
    return ws.A[layers.length - 1];
  }
  function backward(layers, ws, B, grads) { // ws.dA[last+1] must hold dLoss/dOut
    for (let l = layers.length - 1; l >= 0; l--) {
      const { W, nin, nout } = layers[l];
      const inp = l === 0 ? ws.X : ws.A[l - 1];
      const dOut = ws.dA[l + 1], Z = ws.Z[l], last = l === layers.length - 1;
      const dIn = ws.dA[l];
      const gW = grads[l].W, gb = grads[l].b;
      for (let n = 0; n < B; n++) {
        const zo = n * nout, io = n * nin;
        if (!last) for (let j = 0; j < nout; j++) {
          const z = Z[zo + j], s = sigm(z);
          dOut[zo + j] *= s * (1 + z * (1 - s));
        }
        for (let j = 0; j < nout; j++) gb[j] += dOut[zo + j];
        for (let i = 0; i < nin; i++) {
          const xi = inp[io + i], wo = i * nout;
          let acc = 0;
          for (let j = 0; j < nout; j++) {
            const d = dOut[zo + j];
            gW[wo + j] += xi * d;
            acc += d * W[wo + j];
          }
          if (l > 0) dIn[io + i] = acc;
        }
      }
    }
  }

  // ---- Dataset + normalisation ----------------------------------------------
  // opts.rangeMin_m/rangeMax_m: each track's nominal range is drawn log-uniformly
  // from that band, so one model covers the demo's range slider.
  function buildDataset(n, seed, opts) {
    const o = opts || {};
    const r = rng(seed);
    const X = new Float32Array(n * XD), C = new Float32Array(n * CD);
    const lo = Math.log(o.rangeMin_m || 500), hi = Math.log(o.rangeMax_m || o.rangeMin_m || 500);
    for (let i = 0; i < n; i++) {
      const tr = genTrack(r, { range_m: Math.exp(uni(r, lo, hi)) });
      const ft = featurize(tr);
      X.set(encodeTarget(tr, ft), i * XD);
      C.set(ft.cond, i * CD);
      if (o.useEnv === false) C.fill(0, i * CD + ENV0, (i + 1) * CD);
    }
    const stats = (A, d) => {
      const m = new Float32Array(d), sd = new Float32Array(d);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += A[i * d + j] / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) sd[j] += (A[i * d + j] - m[j]) ** 2 / n;
      for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) + 1e-4;
      return { m, sd };
    };
    const norm = { x: stats(X, XD), c: stats(C, CD) };
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < XD; j++) X[i * XD + j] = (X[i * XD + j] - norm.x.m[j]) / norm.x.sd[j];
      for (let j = 0; j < CD; j++) C[i * CD + j] = (C[i * CD + j] - norm.c.m[j]) / norm.c.sd[j];
    }
    return { n, X, C, norm };
  }

  // ---- Trainer ---------------------------------------------------------------
  function Trainer(opts) {
    const o = Object.assign({ seed: 7, nData: 20000, batch: 64, steps: 4000, lr: 2e-3,
      hid: HID, depth: DEPTH, rangeMin_m: 500, rangeMax_m: 500, emaDecay: 0.999,
      useEnv: true }, opts || {});
    const r = rng(o.seed);
    const data = buildDataset(o.nData, o.seed + 1, o);
    const layers = newParams(r, netSizes(o.hid, o.depth));
    const grads = layers.map((L) => ({ W: new Float32Array(L.W.length), b: new Float32Array(L.b.length) }));
    const mom = layers.map((L) => ({ W: new Float32Array(L.W.length), b: new Float32Array(L.b.length) }));
    const vel = layers.map((L) => ({ W: new Float32Array(L.W.length), b: new Float32Array(L.b.length) }));
    // Exponential moving average of the weights; sampling uses these. The decay
    // ramps up from ~0.1 so short in-browser runs are not dominated by the init.
    const ema = layers.map((L) => ({ W: Float32Array.from(L.W), b: Float32Array.from(L.b), nin: L.nin, nout: L.nout }));
    const ws = makeWorkspace(o.batch, layers);
    const eps = new Float32Array(o.batch * XD);
    let step = 0;
    function trainStep() {
      const B = o.batch;
      for (let n = 0; n < B; n++) {
        const i = Math.floor(r() * data.n);
        const t = 1 + Math.floor(r() * T_DIFF);
        const ab = alphaBar[t], sa = Math.sqrt(ab), sb = Math.sqrt(1 - ab);
        const xo = n * IN;
        for (let j = 0; j < XD; j++) {
          const e = gauss(r);
          eps[n * XD + j] = e;
          ws.X[xo + j] = sa * data.X[i * XD + j] + sb * e;
        }
        for (let j = 0; j < CD; j++) ws.X[xo + XD + j] = data.C[i * CD + j];
        tEmbed(t, ws.X, xo + XD + CD);
      }
      const out = forward(layers, ws, B);
      const dOut = ws.dA[layers.length];
      let loss = 0;
      for (let k = 0; k < B * XD; k++) {
        const d = out[k] - eps[k];
        loss += d * d;
        dOut[k] = 2 * d / (B * XD);
      }
      for (const g of grads) { g.W.fill(0); g.b.fill(0); }
      backward(layers, ws, B, grads);
      // Adam with cosine learning-rate decay
      step++;
      const lr = 1e-4 + (o.lr - 1e-4) * 0.5 * (1 + Math.cos(Math.PI * Math.min(step / o.steps, 1)));
      const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, step), c2 = 1 - Math.pow(b2, step);
      for (let l = 0; l < layers.length; l++) for (const key of ['W', 'b']) {
        const p = layers[l][key], g = grads[l][key], m = mom[l][key], v = vel[l][key];
        for (let i = 0; i < p.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i];
          v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
        }
      }
      const d = Math.min(o.emaDecay, (1 + step) / (10 + step));
      for (let l = 0; l < layers.length; l++) for (const key of ['W', 'b']) {
        const p = layers[l][key], e = ema[l][key];
        for (let i = 0; i < p.length; i++) e[i] = d * e[i] + (1 - d) * p[i];
      }
      return loss / (B * XD);
    }
    return {
      opts: o,
      get step() { return step; },
      trainStep,
      model() { return { layers: ema, norm: data.norm, useEnv: o.useEnv }; },
      rawModel() { return { layers, norm: data.norm, useEnv: o.useEnv }; },
      sizes: sizesOf(layers),
    };
  }

  // ---- Sampling (DDIM, eta = 1 by default) ----------------------------------
  // Returns nSamples predicted futures for one track, plus optional per-step
  // snapshots (for the denoising animation).
  function sample(model, ft, r, nSamples, steps, opts) {
    const o = Object.assign({ eta: 1, keepPath: false }, opts || {});
    const { layers, norm } = model;
    const ws = makeWorkspace(nSamples, layers);
    const x = new Float32Array(nSamples * XD);
    for (let i = 0; i < x.length; i++) x[i] = gauss(r);
    const condN = new Float32Array(CD);
    for (let j = 0; j < CD; j++) condN[j] = (ft.cond[j] - norm.c.m[j]) / norm.c.sd[j];
    if (model.useEnv === false) condN.fill(0, ENV0); // model trained without envelope inputs
    const ts = [];
    for (let i = 0; i < steps; i++) ts.push(Math.round(1 + (T_DIFF - 1) * i / Math.max(steps - 1, 1)));
    const path = [];
    const snap = () => {
      if (o.keepPath) path.push(unwhiten(x, nSamples, norm).map((xi) => decode(xi, ft)));
    };
    snap();
    for (let s = steps - 1; s >= 0; s--) {
      const t = ts[s], abT = alphaBar[t], abP = s > 0 ? alphaBar[ts[s - 1]] : 1;
      for (let n = 0; n < nSamples; n++) {
        const xo = n * IN;
        for (let j = 0; j < XD; j++) ws.X[xo + j] = x[n * XD + j];
        ws.X.set(condN, xo + XD);
        tEmbed(t, ws.X, xo + XD + CD);
      }
      const e = forward(layers, ws, nSamples);
      const sig = o.eta * Math.sqrt((1 - abP) / (1 - abT)) * Math.sqrt(Math.max(1 - abT / abP, 0));
      const dirC = Math.sqrt(Math.max(1 - abP - sig * sig, 0));
      for (let k = 0; k < x.length; k++) {
        let x0 = (x[k] - Math.sqrt(1 - abT) * e[k]) / Math.sqrt(abT);
        x0 = Math.max(-5, Math.min(5, x0));
        const epsHat = (x[k] - Math.sqrt(abT) * x0) / Math.sqrt(1 - abT);
        x[k] = Math.sqrt(abP) * x0 + dirC * epsHat + (s > 0 ? sig * gauss(r) : 0);
      }
      snap();
    }
    const samples = unwhiten(x, nSamples, norm).map((xi) => decode(xi, ft));
    return { samples, path };
  }
  function unwhiten(x, nS, norm) {
    const out = [];
    for (let n = 0; n < nS; n++) {
      const xi = new Float64Array(XD);
      for (let j = 0; j < XD; j++) xi[j] = x[n * XD + j] * norm.x.sd[j] + norm.x.m[j];
      out.push(xi);
    }
    return out;
  }
  function meanOf(samples) {
    const u = new Float64Array(H), v = new Float64Array(H), s = new Float64Array(H);
    for (const b of samples) for (let k = 0; k < H; k++) { u[k] += b.u[k]; v[k] += b.v[k]; s[k] += Math.log(b.s[k]); }
    const n = samples.length;
    for (let k = 0; k < H; k++) { u[k] /= n; v[k] /= n; s[k] = Math.exp(s[k] / n); }
    return { u, v, s };
  }

  // ---- Evaluation: errors in metres at the target's true range --------------
  // Pixel error -> angle (px / f, radians) -> cross-range metres (angle * Z).
  function evalTrack(model, tr, r, nSamples, steps, sampleOpts) {
    const ft = featurize(tr);
    const { samples } = sample(model, ft, r, nSamples, steps, sampleOpts);
    const dm = meanOf(samples);
    const cv = { u: ft.cvU, v: ft.cvV };
    const rows = [];
    for (let k = 0; k < H; k++) {
      const m = tr.Z[P + k] / CAM.f;
      rows.push({
        cvX: (cv.u[k] - tr.u[P + k]) * m, cvY: (cv.v[k] - tr.v[P + k]) * m,
        dfX: (dm.u[k] - tr.u[P + k]) * m, dfY: (dm.v[k] - tr.v[P + k]) * m,
      });
    }
    const areaKs = (sampleOpts && sampleOpts.areaKs) || [H - 1];
    const region = [];
    for (let k = 0; k < H; k++) region.push(regionScores(tr, ft, samples, model, k, areaKs.includes(k)));
    return { ft, samples, mean: dm, rows, region };
  }


  // ---- Probability regions ---------------------------------------------------
  // All region maths is in metres in the image plane at the target's true range
  // for that frame (pixels * Z / f), the same units as the point errors.

  // Kernel density estimate of a 2-D point set: Gaussian kernels with Scott's
  // bandwidth (sample covariance * n^(-1/3) in 2-D), plus a small ridge so a
  // tight cluster never gives a singular kernel.
  function kde2(xs, ys) {
    const n = xs.length;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += xs[i] / n; my += ys[i] / n; }
    let sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - mx, dy = ys[i] - my;
      sxx += dx * dx / (n - 1); syy += dy * dy / (n - 1); sxy += dx * dy / (n - 1);
    }
    const f = Math.pow(n, -1 / 3), ridge = 1e-4 + 1e-3 * (sxx + syy);
    const a = sxx * f + ridge, d = syy * f + ridge, b = sxy * f;
    const det = a * d - b * b;
    const ia = d / det, id = a / det, ib = -b / det, norm = 1 / (2 * Math.PI * Math.sqrt(det));
    const kern = (dx, dy) => norm * Math.exp(-0.5 * (ia * dx * dx + 2 * ib * dx * dy + id * dy * dy));
    const density = (x, y) => { let q = 0; for (let i = 0; i < n; i++) q += kern(x - xs[i], y - ys[i]); return q / n; };
    // Leave-one-out density at each sample, used to set region thresholds.
    const dens = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let q = 0;
      for (let j = 0; j < n; j++) if (j !== i) q += kern(xs[i] - xs[j], ys[i] - ys[j]);
      dens[i] = q / (n - 1);
    }
    const sorted = Array.from(dens).sort((p, q) => p - q);
    // Highest-density region holding a share p of the probability: density >= threshold(p).
    // p >= 1 (only reachable after calibration) reaches past the least dense sample.
    const threshold = (p) => (p >= 1 ? 0.5 * sorted[0] : sorted[Math.min(n - 1, Math.max(0, Math.floor((1 - p) * n)))]);
    const sx = Math.sqrt(a), sy = Math.sqrt(d);
    function area(p, grid) {
      const G = grid || 48, thr = threshold(p);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < n; i++) { x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]); y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]); }
      x0 -= 3 * sx; x1 += 3 * sx; y0 -= 3 * sy; y1 += 3 * sy;
      const cw = (x1 - x0) / G, ch = (y1 - y0) / G;
      let cells = 0;
      for (let gi = 0; gi < G; gi++) for (let gj = 0; gj < G; gj++) {
        if (density(x0 + (gi + 0.5) * cw, y0 + (gj + 0.5) * ch) >= thr) cells++;
      }
      return cells * cw * ch;
    }
    return { density, threshold, area, kernelSd: [sx, sy] };
  }

  // Constant-velocity baseline as a distribution: a Gaussian around the CV
  // prediction. Its spread at each horizon step is the RMS of the true future's
  // offset from CV over the training set, in the same along/cross-track frame
  // and speed scaling the model uses (taken from the training normalisation
  // stats). This is a tuned constant-velocity Kalman prediction in spirit.
  function cvGaussian(ft, k, norm, mpp) {
    const qa = norm.x.m[2 * k] ** 2 + norm.x.sd[2 * k] ** 2;       // along-track
    const qb = norm.x.m[2 * k + 1] ** 2 + norm.x.sd[2 * k + 1] ** 2; // cross-track
    const sa = qa * (ft.c * mpp) ** 2, sb = qb * (ft.c * mpp) ** 2;
    const cs = ft.cs, sn = ft.sn; // canonical -> image rotation (see decode)
    return {
      xx: cs * cs * sa + sn * sn * sb, yy: sn * sn * sa + cs * cs * sb, xy: cs * sn * (sa - sb),
    };
  }
  // r2For(p): squared Mahalanobis radius of the p-region (chi-square 2 dof
  // unless calibrated).
  function gaussStats(S, dx, dy, ps, r2For) {
    const det = S.xx * S.yy - S.xy * S.xy;
    const m2 = (S.yy * dx * dx - 2 * S.xy * dx * dy + S.xx * dy * dy) / det; // squared Mahalanobis
    const out = { nll: Math.log(2 * Math.PI) + 0.5 * Math.log(det) + 0.5 * m2, m2, inside: {}, area: {} };
    for (const p of ps) {
      const r2 = r2For ? r2For(p) : -2 * Math.log(1 - p);
      out.inside[p] = m2 <= r2;
      out.area[p] = Math.PI * Math.sqrt(det) * r2;
    }
    return out;
  }

  const REGION_P = [0.5, 0.9];

  // Conformal calibration (see calibrate.js). Without it a "90% region" is
  // only nominal: the sample density can be too tight and the CV Gaussian too
  // loose. model.calib maps each nominal share p to the level that really
  // holds the truth p of the time on calibration tracks (separate from both
  // training and test tracks):
  //   calib.df[nSamples][k][pi] = KDE level to use at horizon step k
  //   calib.cv[k][pi]           = squared Mahalanobis radius for the CV ellipse
  function dfLevel(model, nSamples, k, p) {
    const c = model.calib, pi = REGION_P.indexOf(p);
    if (!c || pi < 0) return p;
    const keys = Object.keys(c.df).map(Number);
    const nn = keys.reduce((b, x) => (Math.abs(Math.log(x / nSamples)) < Math.abs(Math.log(b / nSamples)) ? x : b), keys[0]);
    return c.df[nn][k][pi];
  }
  function cvR2(model, k, p) {
    const c = model.calib, pi = REGION_P.indexOf(p);
    return c && pi >= 0 ? c.cv[k][pi] : -2 * Math.log(1 - p);
  }
  // Region scores for one track at horizon step k. Positions are offsets from
  // the true position in metres (x right, y down in the image).
  function regionScores(tr, ft, samples, model, k, withArea) {
    const mpp = tr.Z[P + k] / CAM.f;
    const n = samples.length;
    const xs = new Float64Array(n), ys = new Float64Array(n);
    let best = Infinity;
    for (let i = 0; i < n; i++) {
      xs[i] = (samples[i].u[k] - tr.u[P + k]) * mpp;
      ys[i] = (samples[i].v[k] - tr.v[P + k]) * mpp;
      best = Math.min(best, Math.hypot(xs[i], ys[i]));
    }
    const kd = kde2(xs, ys);
    const dTrue = kd.density(0, 0);
    const df = { nll: -Math.log(Math.max(dTrue, 1e-12)), inside: {}, area: {}, bestOfN: best, dTrue, kd };
    for (const p of REGION_P) {
      const lv = dfLevel(model, n, k, p);
      df.inside[p] = dTrue >= kd.threshold(lv);
      if (withArea) df.area[p] = kd.area(lv);
    }
    const S = cvGaussian(ft, k, model.norm, mpp);
    const cv = gaussStats(S, (ft.cvU[k] - tr.u[P + k]) * mpp, (ft.cvV[k] - tr.v[P + k]) * mpp, REGION_P, (p) => cvR2(model, k, p));
    return { df, cv };
  }

  // ---- Weight (de)serialisation ---------------------------------------------
  function f32ToB64(arr) {
    const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function b64ToF32(b64) {
    let bytes;
    if (typeof Buffer !== 'undefined') bytes = new Uint8Array(Buffer.from(b64, 'base64'));
    else { const s = atob(b64); bytes = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i); }
    return new Float32Array(bytes.buffer, 0, bytes.length / 4);
  }
  function exportModel(model) {
    return {
      sizes: sizesOf(model.layers),
      layers: model.layers.map((L) => ({ W: f32ToB64(L.W), b: f32ToB64(L.b), nin: L.nin, nout: L.nout })),
      useEnv: model.useEnv !== false,
      calib: model.calib || null,
      norm: { xm: f32ToB64(model.norm.x.m), xs: f32ToB64(model.norm.x.sd), cm: f32ToB64(model.norm.c.m), cs: f32ToB64(model.norm.c.sd) },
    };
  }
  function importModel(j) {
    const s = j.sizes;
    if (!Array.isArray(s) || s.length < 3 || s[0] !== IN || s[s.length - 1] !== XD) {
      throw new Error('weights do not fit this model: need input ' + IN + ' and output ' + XD + ', got sizes ' + JSON.stringify(s));
    }
    const layers = j.layers.map((L) => ({ W: b64ToF32(L.W), b: b64ToF32(L.b), nin: L.nin, nout: L.nout }));
    layers.forEach((L, l) => {
      if (L.nin !== s[l] || L.nout !== s[l + 1] || L.W.length !== L.nin * L.nout || L.b.length !== L.nout) {
        throw new Error('weights layer ' + l + ' is malformed');
      }
    });
    return {
      useEnv: j.useEnv !== false,
      calib: j.calib || null,
      layers,
      norm: { x: { m: b64ToF32(j.norm.xm), sd: b64ToF32(j.norm.xs) }, c: { m: b64ToF32(j.norm.cm), sd: b64ToF32(j.norm.cs) } },
    };
  }

  const api = {
    P, H, DT_S, XD, CD, HID, DEPTH, T_DIFF, MODES, CAM, SIZES, netSizes,
    rng, gauss, genTrack, sampleEnvelope, ENV_RANGES, featurize, decode, sample, meanOf, evalTrack,
    kde2, cvGaussian, gaussStats, regionScores, REGION_P, N_ENV, dfLevel, cvR2,
    Trainer, exportModel, importModel,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BBD = api;
})(typeof window !== 'undefined' ? window : globalThis);
