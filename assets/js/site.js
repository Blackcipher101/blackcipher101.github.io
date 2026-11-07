// Page behaviour: scroll reveals, card tilt, and lazy demo loading.
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

// ---- scroll reveals ----
const reveals = document.querySelectorAll('.reveal');
if (reduceMotion || !('IntersectionObserver' in window)) {
  reveals.forEach((el) => el.classList.add('in'));
} else {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }
  }, { rootMargin: '0px 0px -24px 0px', threshold: 0 }); // reveal as soon as the top edge is on screen: tall cards left holes in the grid with an area threshold
  reveals.forEach((el) => io.observe(el));
}

// ---- hover tilt ----
if (finePointer && !reduceMotion) {
  const MAX_DEG = 4;
  document.querySelectorAll('[data-tilt]').forEach((el) => {
    let raf = 0;
    el.addEventListener('pointermove', (ev) => {
      const r = el.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width - 0.5;
      const py = (ev.clientY - r.top) / r.height - 0.5;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        el.classList.add('is-tilting');
        el.style.setProperty('--ry', `${(px * 2 * MAX_DEG).toFixed(2)}deg`);
        el.style.setProperty('--rx', `${(-py * 2 * MAX_DEG).toFixed(2)}deg`);
      });
    });
    el.addEventListener('pointerleave', () => {
      cancelAnimationFrame(raf);
      el.classList.remove('is-tilting');
      el.style.setProperty('--rx', '0deg');
      el.style.setProperty('--ry', '0deg');
    });
  });
}

// Cache-busting query of this module (index.html sets ?v=...), forwarded to lazily imported demo modules so a
// deploy never mixes new and old files from the browser cache.
const ASSET_QUERY = new URL(import.meta.url).search;

// ---- demos: import on approach, run only while visible ----
const demos = document.querySelectorAll('[data-demo]');
const instances = new Map();
const visible = new Set();

function showFallback(root, text) {
  const stage = root.querySelector('.demo-stage');
  if (!stage || stage.querySelector('.demo-fallback')) return;
  const div = document.createElement('div');
  div.className = 'demo-fallback';
  div.textContent = text;
  stage.appendChild(div);
}

function sync(root) {
  const inst = instances.get(root);
  if (!inst || inst === 'loading') return;
  const allowed = !reduceMotion || root.dataset.userStarted === '1';
  if (visible.has(root) && !document.hidden && allowed) inst.resume();
  else inst.pause();
}

async function load(root) {
  if (instances.has(root)) return;
  instances.set(root, 'loading');
  try {
    const canvas = root.querySelector('canvas');
    if (!canvas || !canvas.getContext('2d')) throw new Error('Canvas 2D is not available');
    const mod = await import(`./demos/${root.dataset.demo}.js${ASSET_QUERY}`);
    instances.set(root, mod.mount(root));
    root.querySelector('.demo-hint')?.remove();
    sync(root);
  } catch (err) {
    instances.delete(root);
    console.warn(`[demo:${root.dataset.demo}] failed to start:`, err);
    showFallback(root, 'This demo could not start in your browser. The source is on GitHub.');
  }
}

if (demos.length) {
  if (reduceMotion) {
    // Do not animate until the visitor interacts with a demo.
    demos.forEach((root) => {
      const start = () => { root.dataset.userStarted = '1'; sync(root); };
      root.addEventListener('pointerdown', start, { once: true });
      root.addEventListener('click', start, { once: true });
      root.addEventListener('keydown', start, { once: true });
    });
  }
  if ('IntersectionObserver' in window) {
    const near = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { load(e.target); near.unobserve(e.target); }
    }, { rootMargin: '300px 0px' });
    const vis = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) visible.add(e.target); else visible.delete(e.target);
        sync(e.target);
      }
    }, { threshold: 0.15 });
    demos.forEach((d) => { near.observe(d); vis.observe(d); });
  } else {
    demos.forEach((d) => { visible.add(d); load(d); });
  }
  document.addEventListener('visibilitychange', () => demos.forEach(sync));
}

// ---- looping videos: play only while on screen; with reduced motion, stay paused on the poster with controls ----
const loopVideos = document.querySelectorAll('video[loop]');
if (reduceMotion) {
  loopVideos.forEach((v) => { v.removeAttribute('autoplay'); v.pause(); v.controls = true; });
} else if ('IntersectionObserver' in window) {
  const vio = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) e.target.play().catch(() => {}); // autoplay can be refused (e.g. data saver); the poster stays
      else e.target.pause();
    }
  }, { threshold: 0.1 });
  loopVideos.forEach((v) => vio.observe(v));
}
