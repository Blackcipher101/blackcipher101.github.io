// Stand-off tracking core (simulation, P controller, Double DQN), from the "Stand-off Tracking Lab"
// artifact; unchanged below this header. Used by demos/standoff.js and tools/train_standoff.mjs.
/*
 * core.js — simulation, controllers and learning agent for the stand-off
 * tracking study. No DOM access here: the same file runs in the browser
 * (exposes window.TrackCore) and in Node (module.exports) for headless tests.
 *
 * Units: SI throughout. Positions in m, velocities in m/s, accelerations in
 * m/s^2, angles in rad, time in s. World frame is a flat 2-D plane, x east,
 * y north. "LOS frame" = line-of-sight frame attached to the follower:
 *   r_hat points from follower to beacon, t_hat = r_hat rotated +90 deg.
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Deterministic randomness. Every stochastic source takes its own seeded
  // stream so that two controllers can be tested on bit-identical conditions.
  // ---------------------------------------------------------------------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  const uniform = (rng, lo, hi) => lo + (hi - lo) * rng();
  const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

  // ---------------------------------------------------------------------------
  // Physical constants shared by both controllers (identical plant).
  // ---------------------------------------------------------------------------
  const PHYS = Object.freeze({
    dt_s: 0.1, // control period (controller output held constant: zero-order hold)
    substeps: 5, // physics/metrics resolution inside one control period
    aMax_mps2: 4.0, // follower acceleration limit (norm)
    kDrag_per_s: 0.5, // linear drag -> terminal speed aMax/kDrag = 8 m/s
    beaconLatAccMax_mps2: 2.5, // cap on beacon centripetal acceleration v*omega
    beaconSpeedTau_s: 1.0, // first-order lag of beacon speed toward its command
    vScale_mps: 8.0, // observation normalisers
    aScale_mps2: 4.0,
    dRef_m: 20.0,
    contactFrac: 0.3, // range < 0.3 d  -> safety violation, episode terminates
    lostFrac: 4.0, // range > 4 d    -> track lost, episode terminates
    terminalPenalty: -10.0,
  });

  // ---------------------------------------------------------------------------
  // Beacon trajectory generators. Each is a small stateful process producing a
  // turn-rate command omega (rad/s) and a speed command v (m/s). They are
  // stepped at the physics substep rate h. "train" families are used for
  // learning; "held-out" families are never seen during training.
  // ---------------------------------------------------------------------------
  const FAMILIES = {
    weave: {
      label: 'Smooth weave', split: 'train',
      make(rng) {
        const A = uniform(rng, 0.3, 0.8), f = uniform(rng, 0.08, 0.25);
        const phi = uniform(rng, 0, 2 * Math.PI), v = uniform(rng, 2.0, 4.5);
        let t = 0;
        return (h) => { t += h; return { omega: A * Math.sin(2 * Math.PI * f * t + phi), v }; };
      },
    },
    sweep: {
      label: 'Wide sweeps', split: 'train',
      make(rng) {
        // Alternating wide turns, smoothed with a first-order filter (tau 1 s).
        const W = uniform(rng, 0.12, 0.35), v = uniform(rng, 2.0, 4.5);
        let period = uniform(rng, 6, 12), sign = rng() < 0.5 ? -1 : 1, tSeg = 0, w = 0;
        return (h) => {
          tSeg += h;
          if (tSeg > period) { tSeg = 0; sign = -sign; period = uniform(rng, 6, 12); }
          w += (sign * W - w) * (1 - Math.exp(-h / 1.0));
          return { omega: w, v };
        };
      },
    },
    turns: {
      label: 'Changing turns', split: 'train',
      make(rng) {
        // Ornstein-Uhlenbeck turn rate and speed, discretised exactly:
        // x' = x e^{-th h} + sigma sqrt((1-e^{-2 th h})/(2 th)) xi
        const thW = 0.5, sW = 0.45, thV = 0.2, sV = 0.6, vMean = uniform(rng, 2.5, 3.8);
        let w = 0, v = vMean;
        return (h) => {
          const ew = Math.exp(-thW * h), ev = Math.exp(-thV * h);
          w = w * ew + sW * Math.sqrt((1 - ew * ew) / (2 * thW)) * gauss(rng);
          v = vMean + (v - vMean) * ev + sV * Math.sqrt((1 - ev * ev) / (2 * thV)) * gauss(rng);
          return { omega: w, v: clamp(v, 1.5, 4.5) };
        };
      },
    },
    chirp: {
      label: 'Chirp weave', split: 'held-out',
      make(rng) {
        // Weave whose frequency sweeps 0.05 -> 0.4 Hz over 40 s: faster than
        // anything in the training families by the end of the episode.
        const A = uniform(rng, 0.5, 0.8), v = uniform(rng, 2.5, 4.0), phi = uniform(rng, 0, 6.28);
        let t = 0, phase = phi;
        return (h) => {
          t += h;
          const f = 0.05 + (0.35 * Math.min(t, 40)) / 40;
          phase += 2 * Math.PI * f * h;
          return { omega: A * Math.sin(phase), v };
        };
      },
    },
    zigzag: {
      label: 'Zig-zag + bursts', split: 'held-out',
      make(rng) {
        // Abrupt (tau 0.15 s) turn reversals and abrupt speed changes
        // 1.5 <-> 4.5 m/s. Training families never change speed this fast.
        let tSeg = 0, period = uniform(rng, 2, 5), wCmd = uniform(rng, 0.4, 0.8), w = 0;
        let vCmd = 2.5, tV = 0, periodV = uniform(rng, 3, 7);
        return (h) => {
          tSeg += h; tV += h;
          if (tSeg > period) { tSeg = 0; period = uniform(rng, 2, 5); wCmd = -Math.sign(wCmd) * uniform(rng, 0.4, 0.8); }
          if (tV > periodV) { tV = 0; periodV = uniform(rng, 3, 7); vCmd = vCmd > 3 ? 1.5 : 4.5; }
          w += (wCmd - w) * (1 - Math.exp(-h / 0.15));
          return { omega: w, v: vCmd };
        };
      },
    },
  };
  const TRAIN_FAMILIES = Object.keys(FAMILIES).filter((k) => FAMILIES[k].split === 'train');
  const HELDOUT_FAMILIES = Object.keys(FAMILIES).filter((k) => FAMILIES[k].split === 'held-out');

  // ---------------------------------------------------------------------------
  // Beacon: kinematic unicycle (x, y, heading psi, speed v). Integrated with the
  // exact constant-turn-rate arc over each substep, so there is no integration
  // drift in curvature. Speed follows its command with a first-order lag.
  // ---------------------------------------------------------------------------
  class Beacon {
    constructor(family, rng) {
      this.gen = FAMILIES[family].make(rng);
      this.x = 0; this.y = 0;
      this.psi = uniform(rng, -Math.PI, Math.PI);
      this.v = 3.0;
      this.omega = 0;
    }
    step(h) {
      const cmd = this.gen(h);
      const vPrev = this.v;
      this.v += (cmd.v - this.v) * (1 - Math.exp(-h / PHYS.beaconSpeedTau_s));
      const vMid = 0.5 * (vPrev + this.v);
      // Lateral-acceleration cap keeps the target physically trackable.
      const wMax = PHYS.beaconLatAccMax_mps2 / Math.max(vMid, 0.5);
      const w = clamp(cmd.omega, -wMax, wMax);
      this.omega = w;
      if (Math.abs(w) > 1e-9) {
        const psi1 = this.psi + w * h;
        this.x += (vMid / w) * (Math.sin(psi1) - Math.sin(this.psi));
        this.y += (vMid / w) * (Math.cos(this.psi) - Math.cos(psi1));
        this.psi = psi1;
      } else {
        this.x += vMid * h * Math.cos(this.psi);
        this.y += vMid * h * Math.sin(this.psi);
      }
    }
    get vx() { return this.v * Math.cos(this.psi); }
    get vy() { return this.v * Math.sin(this.psi); }
  }

  // ---------------------------------------------------------------------------
  // Follower: planar point mass with linear drag,  p'' = a - k p'.
  // For a constant acceleration a held over h the solution is exact (ZOH):
  //   v(h) = a/k + (v0 - a/k) e^{-k h}
  //   p(h) = p0 + (a/k) h + (v0 - a/k)(1 - e^{-k h})/k
  // so the plant has no discretisation error at any step size.
  // ---------------------------------------------------------------------------
  class Follower {
    constructor() { this.x = 0; this.y = 0; this.vx = 0; this.vy = 0; }
    step(ax, ay, h) {
      const k = PHYS.kDrag_per_s, e = Math.exp(-k * h), g = (1 - e) / k;
      const vtx = ax / k, vty = ay / k; // terminal velocity under this input
      this.x += vtx * h + (this.vx - vtx) * g;
      this.y += vty * h + (this.vy - vty) * g;
      this.vx = vtx + (this.vx - vtx) * e;
      this.vy = vty + (this.vy - vty) * e;
    }
  }

  // ---------------------------------------------------------------------------
  // Environment. One instance = one episode with one seed. Two instances built
  // with the same (family, seed, cfg) produce identical beacon motion, initial
  // conditions and sensor noise, whatever the controllers do, because each
  // random stream is independent of the follower's behaviour.
  // ---------------------------------------------------------------------------
  const OBS_DIM = 8;

  class TrackingEnv {
    /**
     * @param {object} cfg
     *   family: key of FAMILIES; seed: integer; standoff_m: desired range d;
     *   episode_s: time limit; noisePos_m, noiseVel_mps: 1-sigma sensor noise.
     */
    constructor(cfg) {
      this.cfg = Object.assign({ standoff_m: 10, episode_s: 30, noisePos_m: 0.1, noiseVel_mps: 0.05 }, cfg);
      if (!(this.cfg.standoff_m > 0)) throw new Error('standoff_m must be > 0, got ' + this.cfg.standoff_m);
      if (!FAMILIES[this.cfg.family]) throw new Error('unknown trajectory family: ' + this.cfg.family);
      const s = this.cfg.seed >>> 0;
      this.rngBeacon = mulberry32(s * 3 + 11);
      this.rngInit = mulberry32(s * 7 + 5);
      this.rngNoise = mulberry32(s * 13 + 1);
      this.beacon = new Beacon(this.cfg.family, this.rngBeacon);
      this.agent = new Follower();
      this.t = 0;
      this.done = false;
      this.prevAx = 0; this.prevAy = 0;
      this.abEst = [0, 0]; // low-passed beacon acceleration estimate (world frame)
      this.vbPrev = null;
      // Initial condition: follower somewhere between 0.6 d and 1.8 d from the
      // beacon at a random bearing, with a small random velocity.
      const d = this.cfg.standoff_m;
      const rho0 = uniform(this.rngInit, 0.6, 1.8) * d, th = uniform(this.rngInit, -Math.PI, Math.PI);
      this.agent.x = this.beacon.x - rho0 * Math.cos(th);
      this.agent.y = this.beacon.y - rho0 * Math.sin(th);
      this.agent.vx = uniform(this.rngInit, -1, 1);
      this.agent.vy = uniform(this.rngInit, -1, 1);
      this.minRange = rho0;
      this.meas = this._measure();
    }

    get trueRange() { return Math.hypot(this.beacon.x - this.agent.x, this.beacon.y - this.agent.y); }

    // Noisy relative measurement + LOS-frame features (same for both controllers).
    _measure() {
      const n = this.rngNoise, sp = this.cfg.noisePos_m, sv = this.cfg.noiseVel_mps, d = this.cfg.standoff_m;
      const rx = this.beacon.x - this.agent.x + sp * gauss(n);
      const ry = this.beacon.y - this.agent.y + sp * gauss(n);
      const vrx = this.beacon.vx - this.agent.vx + sv * gauss(n);
      const vry = this.beacon.vy - this.agent.vy + sv * gauss(n);
      const rho = Math.max(Math.hypot(rx, ry), 1e-6);
      const rh = [rx / rho, ry / rho], th = [-rh[1], rh[0]];
      // Beacon velocity estimate = measured relative velocity + own velocity
      // (own velocity is assumed known from proprioception). Acceleration is a
      // first-order low-passed finite difference (alpha 0.3) to limit noise gain.
      const vb = [vrx + this.agent.vx, vry + this.agent.vy];
      if (this.vbPrev) {
        const alpha = 0.3;
        this.abEst[0] += alpha * ((vb[0] - this.vbPrev[0]) / PHYS.dt_s - this.abEst[0]);
        this.abEst[1] += alpha * ((vb[1] - this.vbPrev[1]) / PHYS.dt_s - this.abEst[1]);
      }
      this.vbPrev = vb;
      const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
      const vr = [vrx, vry], va = [this.agent.vx, this.agent.vy];
      const obs = new Float64Array(OBS_DIM);
      obs[0] = clamp((rho - d) / d, -1, 3); // normalised range error
      obs[1] = dot(vr, rh) / PHYS.vScale_mps; // range rate
      obs[2] = dot(vr, th) / PHYS.vScale_mps; // tangential relative velocity
      obs[3] = dot(va, rh) / PHYS.vScale_mps; // own velocity, LOS frame
      obs[4] = dot(va, th) / PHYS.vScale_mps;
      obs[5] = clamp(dot(this.abEst, rh) / PHYS.aScale_mps2, -3, 3); // beacon accel, LOS frame
      obs[6] = clamp(dot(this.abEst, th) / PHYS.aScale_mps2, -3, 3);
      obs[7] = d / PHYS.dRef_m;
      return { obs, rho, rhat: rh, that: th, vrel: vr, vown: va, vbeacon: vb, d };
    }

    /**
     * Advance one control period with LOS-frame command (aR, aT) in m/s^2.
     * The command is saturated to |a| <= aMax here, for every controller.
     */
    step(aR, aT) {
      if (this.done) throw new Error('step() called on a finished episode');
      const m = this.meas;
      let ax = aR * m.rhat[0] + aT * m.that[0], ay = aR * m.rhat[1] + aT * m.that[1];
      if (!Number.isFinite(ax) || !Number.isFinite(ay)) {
        // A controller that emits NaN/Inf is a defect; fail loudly, do not coast.
        throw new Error('non-finite acceleration command (' + aR + ', ' + aT + ') at t=' + this.t.toFixed(2));
      }
      const n = Math.hypot(ax, ay);
      if (n > PHYS.aMax_mps2) { ax *= PHYS.aMax_mps2 / n; ay *= PHYS.aMax_mps2 / n; }

      const d = this.cfg.standoff_m, h = PHYS.dt_s / PHYS.substeps;
      let terminal = null;
      for (let i = 0; i < PHYS.substeps; i++) {
        this.beacon.step(h);
        this.agent.step(ax, ay, h);
        const r = this.trueRange;
        if (r < this.minRange) this.minRange = r;
        if (r < PHYS.contactFrac * d) { terminal = 'contact'; break; }
        if (r > PHYS.lostFrac * d) { terminal = 'lost'; break; }
      }
      this.t += PHYS.dt_s;

      // ---- reward (uses ground-truth range; see write-up) ----
      const rho = this.trueRange, e = (rho - d) / d;
      const rdot = ((this.beacon.x - this.agent.x) * (this.beacon.vx - this.agent.vx) +
                    (this.beacon.y - this.agent.y) * (this.beacon.vy - this.agent.vy)) / Math.max(rho, 1e-6);
      const aN = Math.hypot(ax, ay) / PHYS.aMax_mps2;
      const daN = Math.hypot(ax - this.prevAx, ay - this.prevAy) / (2 * PHYS.aMax_mps2);
      const parts = {
        track: 0.5 / (1 + 5 * Math.abs(e)) + 0.5 * Math.exp(-((e / 0.08) ** 2)),
        rate: -0.15 * Math.min((rdot / 2.0) ** 2, 1),
        effort: -0.02 * aN * aN,
        smooth: -0.05 * daN * daN,
      };
      let reward = parts.track + parts.rate + parts.effort + parts.smooth;
      if (terminal) reward += PHYS.terminalPenalty;
      this.prevAx = ax; this.prevAy = ay;

      const truncated = !terminal && this.t >= this.cfg.episode_s - 1e-9;
      this.done = !!terminal || truncated;
      this.meas = this._measure();
      return { obs: this.meas.obs, reward, terminal: !!terminal, truncated, reason: terminal, e, rho, ax, ay, parts };
    }
  }

  // ---------------------------------------------------------------------------
  // Controller 1: proportional feedback in the LOS frame.
  //   a = Kp (rho - d) r_hat + Kv (v_beacon - v_own) [+ k v_own drag feed-forward]
  // Kp acts on range error, Kv on relative velocity (a proportional law on each
  // error signal; with Kv = 0 the closed loop is a lightly damped oscillator).
  // ---------------------------------------------------------------------------
  class PController {
    constructor(Kp = 0.6, Kv = 1.2, dragFF = true) { this.Kp = Kp; this.Kv = Kv; this.dragFF = dragFF; this.name = 'P'; }
    act(meas) {
      const ff = this.dragFF ? PHYS.kDrag_per_s : 0;
      const ax = this.Kp * (meas.rho - meas.d) * meas.rhat[0] + this.Kv * meas.vrel[0] + ff * meas.vown[0];
      const ay = this.Kp * (meas.rho - meas.d) * meas.rhat[1] + this.Kv * meas.vrel[1] + ff * meas.vown[1];
      return [ax * meas.rhat[0] + ay * meas.rhat[1], ax * meas.that[0] + ay * meas.that[1]];
    }
  }

  // ---------------------------------------------------------------------------
  // Discrete action set for the learner: radial x tangential accelerations in
  // the LOS frame, 5 x 5 = 25 actions, saturated to the same aMax as P.
  // ---------------------------------------------------------------------------
  const LEVELS = [-1, -0.4, 0, 0.4, 1];
  const ACTIONS = [];
  for (const r of LEVELS) for (const t of LEVELS) ACTIONS.push([r * PHYS.aMax_mps2, t * PHYS.aMax_mps2]);

  // ---------------------------------------------------------------------------
  // Minimal dense network (ReLU hidden layers, linear output) with manual
  // backprop. Single-sample forward/backward; gradients accumulate over a batch.
  // ---------------------------------------------------------------------------
  class MLP {
    constructor(sizes, rng) {
      this.sizes = sizes.slice();
      this.W = []; this.b = []; this.gW = []; this.gb = [];
      this.z = []; this.a = [new Float64Array(sizes[0])];
      for (let l = 0; l < sizes.length - 1; l++) {
        const nin = sizes[l], nout = sizes[l + 1];
        const W = new Float64Array(nin * nout);
        // He initialisation for ReLU layers; small output layer so initial Q ~ 0.
        const s = (l === sizes.length - 2 ? 0.1 : 1) * Math.sqrt(2 / nin);
        if (rng) for (let i = 0; i < W.length; i++) W[i] = s * gauss(rng);
        this.W.push(W); this.b.push(new Float64Array(nout));
        this.gW.push(new Float64Array(nin * nout)); this.gb.push(new Float64Array(nout));
        this.z.push(new Float64Array(nout)); this.a.push(new Float64Array(nout));
      }
      this.delta = this.z.map((z) => new Float64Array(z.length));
    }
    forward(x) {
      const L = this.W.length;
      this.a[0].set(x);
      for (let l = 0; l < L; l++) {
        const W = this.W[l], b = this.b[l], ain = this.a[l], z = this.z[l], aout = this.a[l + 1];
        const nin = ain.length, nout = z.length, last = l === L - 1;
        for (let j = 0; j < nout; j++) {
          let s = b[j]; const off = j * nin;
          for (let i = 0; i < nin; i++) s += W[off + i] * ain[i];
          z[j] = s; aout[j] = last ? s : s > 0 ? s : 0;
        }
      }
      return this.a[L];
    }
    /** Accumulate gradients for dLoss/dOutput = gOut, using the last forward(). */
    backward(gOut) {
      const L = this.W.length;
      this.delta[L - 1].set(gOut);
      for (let l = L - 1; l >= 0; l--) {
        const W = this.W[l], ain = this.a[l], dl = this.delta[l], gW = this.gW[l], gb = this.gb[l];
        const nin = ain.length, nout = dl.length;
        for (let j = 0; j < nout; j++) {
          const dj = dl[j]; if (dj === 0) continue;
          gb[j] += dj; const off = j * nin;
          for (let i = 0; i < nin; i++) gW[off + i] += dj * ain[i];
        }
        if (l > 0) {
          const dprev = this.delta[l - 1], zprev = this.z[l - 1];
          for (let i = 0; i < nin; i++) {
            if (zprev[i] <= 0) { dprev[i] = 0; continue; } // ReLU derivative
            let s = 0;
            for (let j = 0; j < nout; j++) s += W[j * nin + i] * dl[j];
            dprev[i] = s;
          }
        }
      }
    }
    zeroGrad() { for (const g of this.gW) g.fill(0); for (const g of this.gb) g.fill(0); }
    params() { const p = []; for (let l = 0; l < this.W.length; l++) p.push(this.W[l], this.b[l]); return p; }
    grads() { const p = []; for (let l = 0; l < this.W.length; l++) p.push(this.gW[l], this.gb[l]); return p; }
    copyFrom(o) { const a = this.params(), b = o.params(); for (let i = 0; i < a.length; i++) a[i].set(b[i]); }
    softUpdate(o, tau) {
      const a = this.params(), b = o.params();
      for (let i = 0; i < a.length; i++) { const x = a[i], y = b[i]; for (let k = 0; k < x.length; k++) x[k] += tau * (y[k] - x[k]); }
    }
    toJSON() { return { sizes: this.sizes, W: this.W.map((w) => Array.from(w)), b: this.b.map((w) => Array.from(w)) }; }
    static fromJSON(j) {
      const m = new MLP(j.sizes, null);
      j.W.forEach((w, l) => m.W[l].set(w)); j.b.forEach((w, l) => m.b[l].set(w));
      return m;
    }
  }

  class Adam {
    constructor(params, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
      this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps; this.t = 0;
      this.m = params.map((p) => new Float64Array(p.length));
      this.v = params.map((p) => new Float64Array(p.length));
    }
    step(params, grads, scale) {
      this.t++;
      const { b1, b2, eps } = this, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t, lr = this.lr;
      for (let i = 0; i < params.length; i++) {
        const p = params[i], g = grads[i], m = this.m[i], v = this.v[i];
        for (let k = 0; k < p.length; k++) {
          const gk = g[k] * scale;
          m[k] = b1 * m[k] + (1 - b1) * gk;
          v[k] = b2 * v[k] + (1 - b2) * gk * gk;
          p[k] -= (lr * (m[k] / c1)) / (Math.sqrt(v[k] / c2) + eps);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Uniform experience replay in a fixed ring buffer (no per-step allocation).
  // ---------------------------------------------------------------------------
  class Replay {
    constructor(cap, dim) {
      this.cap = cap; this.dim = dim; this.n = 0; this.i = 0;
      this.S = new Float32Array(cap * dim); this.S2 = new Float32Array(cap * dim);
      this.A = new Int32Array(cap); this.R = new Float32Array(cap); this.T = new Uint8Array(cap);
    }
    push(s, a, r, s2, terminal) {
      const k = this.i;
      this.S.set(s, k * this.dim); this.S2.set(s2, k * this.dim);
      this.A[k] = a; this.R[k] = r; this.T[k] = terminal ? 1 : 0;
      this.i = (k + 1) % this.cap; this.n = Math.min(this.n + 1, this.cap);
    }
  }

  // ---------------------------------------------------------------------------
  // Double DQN (van Hasselt, Guez & Silver 2016):
  //   y = r + gamma (1 - terminal) Q_target(s', argmax_a Q_online(s', a))
  // Online net selects, target net evaluates -> removes the max-operator
  // overestimation bias of standard Q-learning. Time-limit truncation is NOT
  // terminal: the target still bootstraps (Pardo et al. 2018).
  // ---------------------------------------------------------------------------
  const DQN_DEFAULTS = {
    hidden: [64, 64], lr: 5e-4, gamma: 0.98, batch: 64, bufferSize: 100000, warmup: 3000,
    trainEvery: 2, tau: 0.005, epsStart: 1.0, epsEnd: 0.05, epsDecaySteps: 60000,
    huberDelta: 1.0, gradClip: 10.0, maxHold: 4, seed: 1,
  };

  class DoubleDQN {
    constructor(cfg) {
      this.cfg = Object.assign({}, DQN_DEFAULTS, cfg);
      this.rng = mulberry32(this.cfg.seed);
      const sizes = [OBS_DIM, ...this.cfg.hidden, ACTIONS.length];
      this.online = new MLP(sizes, this.rng);
      this.target = new MLP(sizes, null);
      this.target.copyFrom(this.online);
      this.opt = new Adam(this.online.params(), this.cfg.lr);
      this.replay = new Replay(this.cfg.bufferSize, OBS_DIM);
      this.steps = 0; this.updates = 0;
      this.holdA = -1; this.holdLeft = 0;
      this.gOut = new Float64Array(ACTIONS.length);
      this.lastLoss = NaN; this.lastQ = NaN;
    }
    get epsilon() {
      const c = this.cfg, f = Math.min(this.steps / c.epsDecaySteps, 1);
      return c.epsStart + f * (c.epsEnd - c.epsStart);
    }
    qValues(obs) { return Array.from(this.online.forward(obs)); }
    greedy(obs) {
      const q = this.online.forward(obs); let best = 0;
      for (let k = 1; k < q.length; k++) if (q[k] > q[best]) best = k;
      return best;
    }
    /**
     * Behaviour policy: epsilon-greedy with temporally extended exploration.
     * A random action is held for 1..maxHold control steps, because single-step
     * random accelerations at 10 Hz largely cancel out and barely move a
     * massive body (a cheap stand-in for correlated/OU exploration noise).
     */
    explore(obs) {
      if (this.holdLeft > 0) { this.holdLeft--; return this.holdA; }
      if (this.rng() < this.epsilon) {
        this.holdA = Math.floor(this.rng() * ACTIONS.length);
        this.holdLeft = Math.floor(this.rng() * this.cfg.maxHold);
        return this.holdA;
      }
      return this.greedy(obs);
    }
    resetEpisode() { this.holdLeft = 0; }
    /** Store a transition; train when due. Returns the TD loss if an update ran. */
    observe(s, a, r, s2, terminal) {
      this.replay.push(s, a, r, s2, terminal);
      this.steps++;
      if (this.replay.n >= this.cfg.warmup && this.steps % this.cfg.trainEvery === 0) return this.update();
      return null;
    }
    update() {
      const c = this.cfg, R = this.replay, D = OBS_DIM, B = c.batch, net = this.online, g = this.gOut;
      const s = new Float64Array(D), s2 = new Float64Array(D);
      net.zeroGrad();
      let lossSum = 0, qSum = 0;
      for (let n = 0; n < B; n++) {
        const k = Math.floor(this.rng() * R.n);
        for (let i = 0; i < D; i++) { s[i] = R.S[k * D + i]; s2[i] = R.S2[k * D + i]; }
        let y = R.R[k];
        if (!R.T[k]) {
          const qo = net.forward(s2); let aStar = 0;
          for (let j = 1; j < qo.length; j++) if (qo[j] > qo[aStar]) aStar = j;
          y += c.gamma * this.target.forward(s2)[aStar];
        }
        const q = net.forward(s), a = R.A[k], td = q[a] - y;
        const ad = Math.abs(td);
        lossSum += ad <= c.huberDelta ? 0.5 * td * td : c.huberDelta * (ad - 0.5 * c.huberDelta);
        qSum += q[a];
        g.fill(0); g[a] = clamp(td, -c.huberDelta, c.huberDelta) / B; // dHuber/dq
        net.backward(g);
      }
      // Global-norm gradient clipping.
      let sq = 0; for (const gr of net.grads()) for (let i = 0; i < gr.length; i++) sq += gr[i] * gr[i];
      const norm = Math.sqrt(sq), scale = norm > c.gradClip ? c.gradClip / norm : 1;
      if (!Number.isFinite(norm)) throw new Error('DQN gradient became non-finite at update ' + this.updates);
      this.opt.step(net.params(), net.grads(), scale);
      this.target.softUpdate(net, c.tau);
      this.updates++;
      this.lastLoss = lossSum / B; this.lastQ = qSum / B;
      return this.lastLoss;
    }
    toJSON() { return { cfg: this.cfg, net: this.online.toJSON(), steps: this.steps }; }
    loadNet(json) { const m = MLP.fromJSON(json); this.online.copyFrom(m); this.target.copyFrom(m); }
  }

  /** Greedy wrapper exposing the same act(meas) interface as PController. */
  class RLController {
    constructor(net) { this.net = net; this.name = 'RL'; this.lastA = 0; }
    act(meas) {
      const q = this.net.forward(meas.obs); let best = 0;
      for (let k = 1; k < q.length; k++) if (q[k] > q[best]) best = k;
      this.lastA = best;
      return ACTIONS[best];
    }
  }

  // ---------------------------------------------------------------------------
  // Evaluation. Metrics are computed after a 5 s settling window so that the
  // random initial offset does not dominate steady-state tracking numbers.
  // ---------------------------------------------------------------------------
  const SETTLE_S = 5.0;
  function runEpisode(controller, envCfg, onStep) {
    const env = new TrackingEnv(envCfg), d = env.cfg.standoff_m;
    let ret = 0, n = 0, se = 0, ae = 0, maxE = 0, inBand = 0, effort = 0, chatter = 0;
    while (!env.done) {
      const [aR, aT] = controller.act(env.meas);
      const r = env.step(aR, aT);
      ret += r.reward;
      if (onStep) onStep(env, r);
      if (env.t > SETTLE_S + 1e-9) {
        n++; const em = r.rho - d;
        se += em * em; ae += Math.abs(em); maxE = Math.max(maxE, Math.abs(em));
        if (Math.abs(em) <= 0.1 * d) inBand++;
        effort += Math.hypot(r.ax, r.ay);
        chatter += r.parts.smooth;
      }
    }
    // A terminated episode counts the remaining time as out-of-band.
    const nTot = Math.round((env.cfg.episode_s - SETTLE_S) / PHYS.dt_s);
    return {
      ret, reason: env.done && env.t < env.cfg.episode_s - 1e-9 ? 'terminated' : 'ok',
      rms_m: n ? Math.sqrt(se / n) : NaN, mae_m: n ? ae / n : NaN, maxErr_m: maxE,
      inBand: nTot ? inBand / nTot : 0, minRange_frac: env.minRange / d,
      effort_mps2: n ? effort / n : NaN, chatter: n ? (-chatter / 0.05) / n : NaN,
      terminal: env.t < env.cfg.episode_s - 1e-9,
    };
  }

  function mean(xs) { const v = xs.filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; }
  function summarize(results) {
    return {
      ret: mean(results.map((r) => r.ret)), rms_m: mean(results.map((r) => r.rms_m)),
      mae_m: mean(results.map((r) => r.mae_m)), maxErr_m: Math.max(...results.map((r) => r.maxErr_m)),
      inBand: mean(results.map((r) => r.inBand)), minRange_frac: Math.min(...results.map((r) => r.minRange_frac)),
      effort_mps2: mean(results.map((r) => r.effort_mps2)), chatter: mean(results.map((r) => r.chatter)),
      failures: results.filter((r) => r.terminal).length, n: results.length,
    };
  }

  /** Seeds: training uses 1..1e6, validation 2e6+, test 3e6+ — disjoint sets. */
  const SEED_BASE = { train: 1, val: 2000000, test: 3000000 };

  /**
   * Fair tuning of the P baseline: grid search on the same validation set and
   * the same reward the RL agent is selected on.
   */
  function tuneP(valSet) {
    const KP = [0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0], KV = [0.5, 0.8, 1.2, 1.6, 2.0, 3.0];
    let best = null;
    for (const Kp of KP) for (const Kv of KV) {
      const c = new PController(Kp, Kv, true);
      const s = summarize(valSet.map((cfg) => runEpisode(c, cfg))).ret;
      if (!best || s > best.ret) best = { Kp, Kv, ret: s };
    }
    return best;
  }

  /** Validation configurations: fixed seeds, training families, mixed standoffs. */
  function makeValSet(n = 9, episode_s = 30, noise = {}) {
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push(Object.assign({ family: TRAIN_FAMILIES[i % TRAIN_FAMILIES.length], seed: SEED_BASE.val + i,
        standoff_m: [7, 10, 14][Math.floor(i / TRAIN_FAMILIES.length) % 3], episode_s }, noise));
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Trainer: drives episodes for the agent; sampled families and standoffs are
  // randomised (domain randomisation over d in [6, 16] m).
  // ---------------------------------------------------------------------------
  class Trainer {
    constructor(agentCfg, opts = {}) {
      this.opts = Object.assign({ episode_s: 30, noisePos_m: 0.1, noiseVel_mps: 0.05, evalEvery: 10, seed: 1 }, opts);
      this.agent = new DoubleDQN(Object.assign({ seed: this.opts.seed }, agentCfg));
      this.rng = mulberry32(this.opts.seed * 31 + 7);
      this.valSet = makeValSet(9, this.opts.episode_s, { noisePos_m: this.opts.noisePos_m, noiseVel_mps: this.opts.noiseVel_mps });
      this.episode = 0; this.env = null; this.epReturn = 0; this.epLen = 0;
      this.history = []; // per-episode {ep, ret, len, eps, loss, q, reason, family}
      this.evals = []; // {ep, steps, ret, rms}
      this.best = null; // {ep, ret, net(JSON)}
      this.lossAcc = 0; this.lossN = 0; this.qAcc = 0;
      this._newEpisode();
    }
    _newEpisode() {
      const fam = TRAIN_FAMILIES[Math.floor(this.rng() * TRAIN_FAMILIES.length)];
      this.env = new TrackingEnv({ family: fam, seed: SEED_BASE.train + this.episode,
        standoff_m: uniform(this.rng, 6, 16), episode_s: this.opts.episode_s,
        noisePos_m: this.opts.noisePos_m, noiseVel_mps: this.opts.noiseVel_mps });
      this.agent.resetEpisode();
      this.epReturn = 0; this.epLen = 0; this.lossAcc = 0; this.lossN = 0; this.qAcc = 0;
    }
    /** One environment step. Returns an episode record when an episode ends. */
    step() {
      const env = this.env, s = env.meas.obs.slice();
      const a = this.agent.explore(s);
      const [aR, aT] = ACTIONS[a];
      const r = env.step(aR, aT);
      const loss = this.agent.observe(s, a, r.reward, r.obs, r.terminal);
      if (loss !== null) { this.lossAcc += loss; this.qAcc += this.agent.lastQ; this.lossN++; }
      this.epReturn += r.reward; this.epLen++;
      if (!env.done) return null;
      const rec = { ep: this.episode, ret: this.epReturn, len: this.epLen, eps: this.agent.epsilon,
        loss: this.lossN ? this.lossAcc / this.lossN : NaN, q: this.lossN ? this.qAcc / this.lossN : NaN,
        reason: r.reason || 'time', family: env.cfg.family, steps: this.agent.steps };
      this.history.push(rec);
      this.episode++;
      if (this.episode % this.opts.evalEvery === 0) this.evaluate();
      this._newEpisode();
      return rec;
    }
    evaluate() {
      const ctrl = new RLController(this.agent.online);
      const s = summarize(this.valSet.map((cfg) => runEpisode(ctrl, cfg)));
      const rec = { ep: this.episode, steps: this.agent.steps, ret: s.ret, rms: s.rms_m, failures: s.failures };
      this.evals.push(rec);
      if (!this.best || s.ret > this.best.ret) this.best = { ep: this.episode, ret: s.ret, net: this.agent.online.toJSON() };
      return rec;
    }
    bestNet() { return this.best ? MLP.fromJSON(this.best.net) : this.agent.online; }
  }

  const api = { PHYS, FAMILIES, TRAIN_FAMILIES, HELDOUT_FAMILIES, ACTIONS, LEVELS, OBS_DIM, SETTLE_S, SEED_BASE,
    DQN_DEFAULTS, mulberry32, TrackingEnv, PController, RLController, DoubleDQN, MLP, Trainer,
    runEpisode, summarize, tuneP, makeValSet };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TrackCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
