// Shared helpers for the canvas demos: HiDPI canvas, world<->screen view, RNG, run loop.

export const COLORS = {
  ink: '#0e0d12',
  panel: '#17151d',
  grid: 'rgba(244,239,228,0.06)',
  gridStrong: 'rgba(244,239,228,0.12)',
  cream: '#f4efe4',
  creamDim: 'rgba(244,239,228,0.45)',
  violet: '#7d70ff',
  coral: '#ff6b57',
  peach: '#ffb48a',
  lime: '#d9f45b',
};

/** Standard normal sample (Box-Muller). */
export function gauss() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Keeps a canvas sized to its CSS box at devicePixelRatio (capped at 2) and maps a
 * world rectangle [0,W]x[0,H] (metres, y up) into it with uniform scale.
 */
export class View {
  constructor(canvas, worldW, worldH) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.worldW = worldW;
    this.worldH = worldH;
    this.resize();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.dpr = dpr;
    this.cssW = r.width;
    this.cssH = r.height;
    const pad = 12 * dpr;
    this.scale = Math.min((w - 2 * pad) / this.worldW, (h - 2 * pad) / this.worldH);
    this.ox = (w - this.scale * this.worldW) / 2;
    this.oy = (h + this.scale * this.worldH) / 2;
    this.dirty = true;
  }
  sx(x) { return this.ox + x * this.scale; }
  sy(y) { return this.oy - y * this.scale; }
  /** Pointer event -> world coordinates. */
  toWorld(ev) {
    const r = this.canvas.getBoundingClientRect();
    const px = (ev.clientX - r.left) * this.dpr;
    const py = (ev.clientY - r.top) * this.dpr;
    return [(px - this.ox) / this.scale, (this.oy - py) / this.scale];
  }
  clear(bg = COLORS.panel) {
    const { ctx, canvas } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  grid(step = 1) {
    const { ctx } = this;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= this.worldW + 1e-9; x += step) {
      ctx.moveTo(Math.round(this.sx(x)) + 0.5, this.sy(0));
      ctx.lineTo(Math.round(this.sx(x)) + 0.5, this.sy(this.worldH));
    }
    for (let y = 0; y <= this.worldH + 1e-9; y += step) {
      ctx.moveTo(this.sx(0), Math.round(this.sy(y)) + 0.5);
      ctx.lineTo(this.sx(this.worldW), Math.round(this.sy(y)) + 0.5);
    }
    ctx.strokeStyle = COLORS.grid;
    ctx.stroke();
    ctx.strokeStyle = COLORS.gridStrong;
    ctx.strokeRect(this.sx(0), this.sy(this.worldH), this.worldW * this.scale, this.worldH * this.scale);
  }
  /** Small triangle robot glyph at world pose. */
  robot(x, y, th, size, fill, stroke) {
    const { ctx } = this;
    const s = size * this.scale;
    ctx.save();
    ctx.translate(this.sx(x), this.sy(y));
    ctx.rotate(-th);
    ctx.beginPath();
    ctx.moveTo(s, 0);
    ctx.lineTo(-0.7 * s, 0.62 * s);
    ctx.lineTo(-0.35 * s, 0);
    ctx.lineTo(-0.7 * s, -0.62 * s);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.lineWidth = 1.5 * this.dpr; ctx.strokeStyle = stroke; ctx.stroke(); }
    ctx.restore();
  }
  destroy() { this.ro.disconnect(); }
}

/**
 * requestAnimationFrame loop that only runs while `active()` is true.
 * step(dtSeconds) is called with dt clamped to 50 ms so a background tab or a
 * long frame never produces a huge simulation jump.
 */
export class Loop {
  constructor(step) {
    this.step = step;
    this.running = false;
    this.last = 0;
    this.raf = 0;
    this.tick = this.tick.bind(this);
  }
  start() {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
  }
  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }
  tick(now) {
    if (!this.running) return;
    const dt = Math.min(0.05, Math.max(0, (now - this.last) / 1000));
    this.last = now;
    this.step(dt);
    this.raf = requestAnimationFrame(this.tick);
  }
}

export const prefersReducedMotion = () =>
  window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Small seeded PRNG (mulberry32): returns a function giving uniform [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample from a given uniform generator (Box-Muller). */
export function gaussFrom(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
