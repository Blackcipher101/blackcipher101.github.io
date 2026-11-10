// Audience tailoring: the visitor says who they are (free text or a chip) and the page
// re-orders itself. Everything runs in the browser; nothing is sent anywhere.
//
// - Choice comes from ?for=<id> first, then localStorage, else the general page.
// - Free text is matched to a preset by keyword scoring (see KEYWORDS and score()).
// - A preset changes: hero pill/headline/lede, the projects intro, which project cards
//   show and in what order (max 6), which demos show and in what order, and which experience cards and
//   bullets are marked as relevant. All copy repeats claims already made on the page.
// - Section order: `sections` picks a group (SECTION_ORDER). The inline script in index.html's <head>
//   applies the same group before first paint (no layout shift); init() checks the two agree.
// Rationale for the preset list and keyword sets: AUDIENCES.md.

const STORAGE_KEY = 'nn-audience';
const MAX_PROJECTS = 6;
/** Visible order of the reorderable sections per group; CSS (html[data-order]) places and numbers them. */
export const SECTION_ORDER = {
  base: ['experience', 'demos', 'work'],
  proof: ['experience', 'work', 'demos'],
  play: ['demos', 'experience', 'work'],
};
const hasDom = typeof document !== 'undefined';
const reduceMotion = hasDom && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Hero "Now" card variants shared by several presets (static, trusted markup). */
const NOW_FASTLIO = {
  nowart: '<svg class="now-vid now-chart" viewBox="0 0 640 480" role="img" aria-label="Bar chart of CPU use, in percent of one core, measured live on a Jetson Orin before and after moving FAST-LIO2 and the localizer to the GPU: FAST-LIO2 46 to 16, localizer 28 to 16, lidar pipeline 40.9 to 10.2, cloud merger 8.9 to 7.0; FAST-LIO2 alone 0.46 to 0.16 cores, all four 1.24 to 0.49 cores"><rect width="640" height="480" fill="#0e0d12"/><text x="32" y="50" fill="#f4efe4" font-size="22" font-weight="600">CPU, % of one core</text><text x="32" y="76" fill="#9b95a6" font-size="15">measured live on a Jetson Orin · CPU build vs exact GPU build</text><text x="32" y="116" fill="#d6d1de" font-size="16">FAST-LIO2</text><rect x="32" y="126" width="350" height="14" rx="4" fill="#4a4556"/><text x="392" y="138" fill="#9b95a6" font-size="14">46</text><rect class="bar" style="animation-delay:0.00s" x="32" y="144" width="122" height="14" rx="4" fill="#d9f552"/><text x="164" y="156" fill="#d9f552" font-size="14" font-weight="600">16</text><text x="32" y="188" fill="#d6d1de" font-size="16">Localizer</text><rect x="32" y="198" width="213" height="14" rx="4" fill="#4a4556"/><text x="255" y="210" fill="#9b95a6" font-size="14">28</text><rect class="bar" style="animation-delay:0.15s" x="32" y="216" width="122" height="14" rx="4" fill="#d9f552"/><text x="164" y="228" fill="#d9f552" font-size="14" font-weight="600">16</text><text x="32" y="260" fill="#d6d1de" font-size="16">Lidar pipeline</text><rect x="32" y="270" width="311" height="14" rx="4" fill="#4a4556"/><text x="353" y="282" fill="#9b95a6" font-size="14">40.9</text><rect class="bar" style="animation-delay:0.30s" x="32" y="288" width="78" height="14" rx="4" fill="#d9f552"/><text x="120" y="300" fill="#d9f552" font-size="14" font-weight="600">10.2</text><text x="32" y="332" fill="#d6d1de" font-size="16">Cloud merger</text><rect x="32" y="342" width="68" height="14" rx="4" fill="#4a4556"/><text x="110" y="354" fill="#9b95a6" font-size="14">8.9</text><rect class="bar" style="animation-delay:0.45s" x="32" y="360" width="53" height="14" rx="4" fill="#d9f552"/><text x="95" y="372" fill="#d9f552" font-size="14" font-weight="600">7</text><text x="32" y="434" fill="#f4efe4" font-size="34" font-weight="700">0.46 → 0.16 cores</text><text x="32" y="460" fill="#9b95a6" font-size="15">FAST-LIO2 alone · all four nodes 1.24 → 0.49 · same results as CPU</text></svg>',
  nowtitle: 'Exact GPU FAST-LIO2',
  nowtext: 'FAST-LIO2 and the map localizer on a Jetson Orin GPU, same results as the CPU code. <b>2.9× less CPU</b> for FAST-LIO2 (0.46 → 0.16 cores), <b>2.5× less</b> for the whole localization chain, <b>2.2× faster</b> per scan and <b>7× lower</b> cloud-merge latency.',
};
const NOW_NAV_RL = {
  nowart: '<video class="now-vid now-wide" src="assets/vid/rov-nav-overhead.mp4" poster="assets/vid/rov-nav-overhead.webp" width="960" height="320" autoplay muted loop playsinline preload="metadata" aria-label="Overhead view of one ROV episode through crossing traffic with sonar only, driven by the fine-tuned RL policy"></video>',
  nowtitle: 'Navigation RL in MuJoCo',
  nowtext: 'An RL policy steering an ROV through crossing traffic on sonar only. One episode, picked for illustration; the results are being re-validated.',
};

/** Presets. headline may contain <b>; it is static, trusted copy. */
export const PRESETS = {
  startup: {
    label: 'Startup founder / CTO',
    pill: 'Founding engineer · 0 → 1 · Full stack',
    headline: 'I take autonomy from <b>zero</b> to <b>launch</b>.',
    lede: 'Founding engineer at Minus Zero: took ZPod, India\'s first autonomous vehicle, from zero to 100k+ lines of production code, its zDay 2023 launch and 10,000 km driven, from pre-seed to Series A. Founding member at BYTES: raised $500K, built the product, ran a public demo; selected for Entrepreneur First and YC Startup School.',
    pitch: 'Founding-team experience twice: pre-seed to Series A on ZPod, then a $500K raise, product and public demo at BYTES (EF, YC Startup School).',
    work: 'Shipped, measured side projects across estimation, RL and simulation.',
    projects: ['vio', 'humanoid', 'rov', 'q14', 'inspect', 'klt'],
    demos: ['mppi', 'cartpole', 'scara', 'slam'],
    sections: 'play',
    jobs: ['mz', 'bytes', 'griffin'],
    points: ['zero', 'series', 'raise', 'demo', 'accel'],
  },
  enterprise: {
    ...NOW_FASTLIO,
    label: 'Big company / enterprise',
    pill: 'Production code · CI/CD · Leadership',
    headline: 'Autonomy code that <b>runs in production</b>, with CI behind it.',
    lede: '100k+ lines of production code on ZPod, backed by a simulation pipeline and closed-loop CI/CD. Head of Planning & Prediction at Minus Zero, Senior Robotics Engineer at Aeroarc, now Autonomy Engineer at Griffin Labs.',
    pitch: 'Led planning & prediction, wrote production-scale autonomy code, and builds evaluation you can reproduce.',
    work: 'Benchmarked against ground truth, with the evaluation scripts in each repo.',
    projects: ['vio', 'klt', 'bench', 'calib', 'inspect', 'tb3'],
    demos: ['scara', 'filters', 'slam', 'mppi'],
    sections: 'proof',
    jobs: ['mz', 'griffin', 'aeroarc'],
    points: ['zero', 'series', 'ci', 'sim'],
  },
  underwater: {
    label: 'Underwater / marine',
    pill: 'Underwater VIO · DVL · Inspection',
    headline: 'Knowing <b>where you are</b> underwater, even when the camera goes dark.',
    lede: 'VIO with a tightly coupled DVL (20 s camera blackout: ATE 3.88 m → 0.66 m with simulated DVL), a CUDA KLT front end, in-water calibration monitoring, automated inspection reports and an RL ROV controller. Benchmarked, negative results included.',
    pitch: 'Five underwater perception and estimation repos plus an RL ROV controller, all measured against ground truth.',
    nowart: '<img src="assets/img/rov.svg" width="580" height="650" alt="Original illustration of an underwater ROV with thrusters, a camera dome, a tether and Doppler velocity log beams">',
    nowtitle: 'Underwater autonomy',
    nowtext: 'Visual-inertial odometry, DVL fusion and inspection tooling for robots that cannot see GPS.',
    work: 'Underwater perception and estimation, with results measured against ground truth. Negative results are published too.',
    projects: ['vio', 'rov', 'klt', 'calib', 'inspect', 'bench'],
    demos: ['dvl', 'filters', 'sonar', 'slam'],
    sections: 'base',
    jobs: [],
    points: [],
  },
  embodied: {
    label: 'Embodied AI / humanoids',
    pill: 'Embodied AI · RL · Real robots',
    headline: 'Embodied AI, from <b>simulation</b> to <b>real robots</b>.',
    lede: 'PPO with domain randomisation for an ROV that beats a tuned PID outside its training range, a quadruped simulator with hand-written multibody dynamics and a trained trot, H-GAIL and PPO for driving at Minus Zero, and a VLM-verified inspection pipeline. In MuJoCo, humanoids that walk, get up (127/128) and spin-kick on a Unitree G1 at real torque limits (64/64, 0 falls).',
    pitch: 'RL that holds up beyond its training range, physics written from scratch, and deployment on real robots.',
    work: 'RL, simulation and VLM projects, each with its evaluation.',
    projects: ['humanoid', 'rov', 'q14', 'inspect', 'vio', 'tb3'],
    demos: ['cartpole', 'standoff', 'scara', 'mppi'],
    sections: 'play',
    jobs: ['griffin', 'mz', 'aeroarc', 'edu'],
    points: ['rl', 'ue', 'sim', 'mlss', 'eeml'],
  },
  av: {
    ...NOW_NAV_RL,
    label: 'Autonomous vehicles',
    pill: 'Planning · Prediction · Simulation',
    headline: 'Planning and prediction for <b>autonomous vehicles</b>.',
    lede: 'Head of Planning & Prediction at Minus Zero, where ZPod, India\'s first autonomous vehicle, went from zero to 100k+ lines of production code and 10,000 km driven. Simulation pipeline and an India-specific AV simulator, closed-loop CI/CD, RL driving with H-GAIL and PPO, NVIDIA DriveWorks.',
    pitch: 'ZPod end to end: planning, prediction, simulation and CI for India\'s first autonomous vehicle.',
    work: 'Planning, SLAM and estimation projects outside work.',
    projects: ['tb3', 'vio', 'klt', 'bench', 'q14', 'rov'],
    demos: ['mppi', 'filters', 'slam', 'dvl'],
    sections: 'base',
    jobs: ['mz', 'bytes'],
    points: ['zero', 'sim', 'ue', 'ci', 'ga', 'rl', 'dw'],
  },
  defence: {
    ...NOW_FASTLIO,
    label: 'Defence',
    pill: 'Defence robotics · UGV · Quadruped',
    headline: 'Autonomy for <b>defence robotics</b>.',
    lede: 'Senior Robotics Engineer at Aeroarc, on India\'s defence robotics: the VIRA tracked UGV (15 units in active deployment) and the ARCV MULE quadruped (10 in active deployment). Plus state estimation that survives camera blackouts and in-water calibration checks with zero false alarms.',
    pitch: 'Defence robotics at Aeroarc (VIRA, ARCV MULE), plus estimation built for degraded sensing.',
    work: 'Estimation that survives sensor dropout, measured against ground truth.',
    projects: ['vio', 'calib', 'klt', 'q14', 'tb3', 'bench'],
    demos: ['boxtrack', 'filters', 'sonar', 'dvl'],
    sections: 'base',
    jobs: ['aeroarc', 'mz'],
    points: ['sim', 'ci'],
  },
  inspection: {
    ...NOW_FASTLIO,
    label: 'Inspection / industrial',
    pill: 'Inspection · Reality capture · Facilities',
    headline: 'Robots that <b>inspect</b>, scan and report.',
    lede: 'The PhiSCAN 360° LiDAR scanner at Aeroarc, an inspection pipeline where a detector plus a VLM drafts the report (8/8 true findings confirmed, precision 0.89), and a calibration monitor that caught every ≥ 0.2° change with 0 false alarms. Now working on TCR and Mega at Griffin Labs.',
    pitch: 'Reality-capture hardware, automated inspection reports and calibration health checks.',
    work: 'Inspection and calibration tooling, measured on real dive data.',
    projects: ['inspect', 'calib', 'vio', 'bench', 'klt', 'tb3'],
    demos: ['scara', 'slam', 'sonar', 'dvl'],
    sections: 'base',
    jobs: ['phi', 'griffin', 'clutter'],
    points: ['yolo', 'fsm'],
  },
  legged: {
    label: 'Legged robots',
    pill: 'Quadrupeds · Locomotion · Dynamics',
    headline: 'Quadrupeds in the field and <b>in simulation</b>.',
    lede: 'Aeroarc\'s ARCV MULE (a Ghost Robotics quadruped) and VIRA UGV in defence robotics, and Q14 Dynamics Lab: a browser robot-dog simulator with hand-written multibody dynamics and a trained trot over 40 dynamic rocks. Humanoids in MuJoCo: PPO walk and get-up, and a Unitree G1 spin kick at real torque limits.',
    pitch: 'Legged robots from both sides: real quadrupeds, and the dynamics written from scratch.',
    work: 'Locomotion, dynamics and RL projects.',
    projects: ['humanoid', 'q14', 'rov', 'tb3', 'vio', 'klt'],
    demos: ['cartpole', 'scara', 'mppi', 'slam'],
    sections: 'play',
    jobs: ['aeroarc', 'mz'],
    points: ['rl', 'ga'],
  },
  recruiter: {
    ...NOW_FASTLIO,
    label: 'Recruiter / hiring manager',
    pill: 'Exact GPU SLAM · Estimation · Planning · RL',
    headline: '<b>FAST-LIO2</b> on the GPU: exact, <b>2.9× less CPU</b>, <b>2.2× faster</b>.',
    lede: 'Autonomy Engineer at Griffin Labs (now): moved FAST-LIO2 and the map localizer onto the Jetson Orin GPU with results that match the CPU code (0 mismatches over 7.27 M nearest-neighbour searches); <b>2.9× less CPU</b> for FAST-LIO2 live on the robot (0.46 → 0.16 cores), <b>2.5× less</b> for the whole localization chain (1.24 → 0.49), <b>2.2× faster</b> per scan (42.9 → 19.8 ms) and <b>7× lower</b> cloud-merge latency (104 → 14 ms). Before: Senior Robotics Engineer at Aeroarc, Robotics & AI at BYTES (led tech and product), Head of Planning & Prediction and Founding Engineer at Minus Zero (ZPod), Robotics Intern at Clutterbot. B.Tech ELC (Robotics), Amrita.',
    pitch: 'Skills: state estimation, SLAM, GPU acceleration, planning & prediction, RL, simulation, CI/CD · C++, Python, CUDA, ROS 2 · nehalnevle@gmail.com',
    work: 'A short list of public projects, each with measured results.',
    projects: ['vio', 'humanoid', 'rov', 'q14', 'klt', 'inspect'],
    demos: ['filters', 'slam', 'cartpole', 'mppi'],
    sections: 'proof',
    jobs: ['griffin', 'aeroarc', 'bytes', 'mz'],
    points: ['lio', 'zero', 'series', 'raise', 'accel', 'ci', 'lead', 'mlss', 'eeml'],
  },
};

/**
 * Keyword weights per preset. Terms of 5+ characters match as word prefixes
 * ("inspect" matches "inspection"); shorter terms and phrases must match whole
 * words/phrases, so "av" does not fire on "available". Highest total wins; ties go to
 * the earlier preset in PRESETS order; zero means no match.
 */
export const KEYWORDS = {
  startup: { startup: 3, 'start-up': 3, founder: 3, 'co-founder': 3, cofounder: 3, founding: 2, cto: 3, ceo: 1, seed: 2, 'series a': 2, 'early stage': 3, 'early-stage': 3, 'small team': 2, mvp: 2, stealth: 1, yc: 2, 'y combinator': 2, 'entrepreneur first': 2, ef: 1, fundraising: 2, investor: 2, vc: 2, venture: 1, '0 to 1': 2, 'zero to one': 2 },
  enterprise: { enterprise: 3, corporation: 3, corp: 3, 'big company': 3, 'large company': 3, 'big tech': 3, fortune: 2, multinational: 3, mnc: 3, oem: 2, 'tier 1': 2, 'tier-1': 2, faang: 3, reliability: 2, production: 1, scale: 1, bosch: 2, siemens: 2, nvidia: 1, amazon: 1, google: 1 },
  underwater: { underwater: 4, subsea: 4, auv: 4, auvs: 4, dvl: 4, marine: 3, ocean: 3, rov: 3, rovs: 3, sonar: 3, aquatic: 3, offshore: 2, maritime: 2, sea: 2, hull: 2, naval: 1, vessel: 1, diver: 1, water: 1 },
  embodied: { embodied: 4, humanoid: 4, 'physical ai': 4, vla: 4, 'vision-language-action': 4, 'robot learning': 4, 'foundation model': 3, 'reinforcement learning': 3, imitation: 3, sim2real: 3, 'sim-to-real': 3, mujoco: 3, dexterous: 3, rl: 2, isaac: 2, manipulation: 2, 'general purpose': 2, generalist: 2, 'ai lab': 2, policy: 1 },
  av: { 'autonomous vehicle': 4, 'self-driving': 4, 'self driving': 4, driverless: 4, robotaxi: 4, adas: 4, automotive: 4, av: 3, avs: 3, trucking: 3, 'two-wheeler': 3, car: 2, cars: 2, driving: 2, mobility: 2, truck: 2, shuttle: 2, scooter: 2, motorcycle: 2, pod: 1 },
  defence: { defence: 4, defense: 4, military: 4, army: 4, drdo: 4, 'unmanned ground': 4, tactical: 3, ugv: 3, ugvs: 3, 'air force': 3, 'dual-use': 3, 'dual use': 3, navy: 2, surveillance: 2, border: 2, security: 1, government: 1 },
  inspection: { inspection: 4, inspect: 3, industrial: 3, facilities: 3, cleaning: 3, 'digital twin': 3, scanner: 3, scanning: 3, infrastructure: 3, workcell: 3, ndt: 3, 'non-destructive': 3, facility: 2, lidar: 2, survey: 2, plant: 2, factory: 2, warehouse: 2, maintenance: 2, manufacturing: 2, cobot: 2, logistics: 1 },
  legged: { legged: 4, quadruped: 4, 'robot dog': 4, 'robotic dog': 4, locomotion: 4, biped: 3, gait: 3, unitree: 3, anymal: 3, 'boston dynamics': 3, 'ghost robotics': 3, dog: 2, walking: 2, legs: 2, spot: 2 },
  recruiter: { recruiter: 4, recruiting: 4, headhunter: 4, 'hiring manager': 4, talent: 3, hr: 3, cv: 3, resume: 3, 'résumé': 3, sourcing: 3, hiring: 2, hire: 2, candidate: 2, role: 1, position: 1, job: 1, opening: 1 },
};

const escapeRe = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const MATCHERS = Object.fromEntries(Object.entries(KEYWORDS).map(([id, terms]) => [
  id,
  Object.entries(terms).map(([term, weight]) => ({
    weight,
    re: new RegExp(term.length >= 5 && !term.includes(' ')
      ? `(^|[^\\p{L}\\p{N}])${escapeRe(term)}`
      : `(^|[^\\p{L}\\p{N}])${escapeRe(term)}(?=$|[^\\p{L}\\p{N}])`, 'u'),
  })),
]));

/** Score free text against every preset. Returns {id, score, scores} (id null if no match). */
export function score(text) {
  const t = ` ${String(text || '').toLowerCase()} `;
  const scores = {};
  let best = null, bestScore = 0;
  for (const id of Object.keys(PRESETS)) {
    let s = 0;
    for (const m of MATCHERS[id]) if (m.re.test(t)) s += m.weight;
    scores[id] = s;
    if (s > bestScore) { bestScore = s; best = id; }
  }
  return { id: best, score: bestScore, scores };
}

// ---------------------------------------------------------------- DOM wiring
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const form = hasDom ? $('#tailor') : null;
if (form) init();

function init() {
  const input = $('#tailor-input');
  const chipsBox = $('#tailor-chips');
  const ask = $('.tailor-ask', form);
  const now = $('#tailor-now');
  const status = $('#tailor-status');
  const projGrid = $('#work .bento');
  const demoGrid = $('#demos .bento');
  const textSlots = $$('[data-aud]');
  const workIntro = $('#work .sec-head p');
  if (workIntro) { workIntro.dataset.aud = 'work'; textSlots.push(workIntro); }

  // Snapshot of the general page, restored by "General view".
  const general = {
    text: Object.fromEntries(textSlots.map((el) => [el.dataset.aud, el.innerHTML])),
    projects: $$('[data-proj]', projGrid).map((el) => el.dataset.proj),
    demos: $$('[data-demo]', demoGrid).map((el) => el.dataset.demo),
    wide: $('.proj-wide', projGrid)?.dataset.proj,
  };

  // "Relevant" badges, shown only on featured experience cards.
  $$('[data-job]').forEach((job) => {
    const when = $('.when', job);
    if (!when) return;
    const b = document.createElement('span');
    b.className = 'rel-badge mono';
    b.textContent = 'Relevant';
    when.prepend(b);
  });

  // Chips are in the HTML (Recruiter second, so it is visible without scrolling on phones) so the bar does not grow
  // after first paint; create any that are missing (e.g. a preset added in JS only) and wire them all up here.
  for (const id of Object.keys(PRESETS)) {
    let c = chipsBox.querySelector(`.chip-btn[data-preset="${id}"]`);
    if (!c) {
      c = document.createElement('button');
      c.type = 'button';
      c.className = 'chip-btn';
      c.dataset.preset = id;
      c.textContent = PRESETS[id].label;
      c.setAttribute('aria-pressed', 'false');
      chipsBox.appendChild(c);
    }
    c.addEventListener('click', () => choose(id, { source: 'chip' }));
  }

  form.addEventListener('submit', (ev) => {
    ev.preventDefault();
    const r = score(input.value);
    if (!r.id) {
      status.textContent = input.value.trim()
        ? 'No clear match. Pick the closest chip below.'
        : 'Type a few words about yourself, or pick a chip.';
      return;
    }
    choose(r.id, { source: 'text' });
  });
  $('#tailor-change').addEventListener('click', () => { showAsk(true); input.focus(); });
  $('#tailor-reset').addEventListener('click', () => choose(null, { source: 'reset' }));
  $('#tailor-share').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      status.textContent = 'Link copied.';
    } catch (_) {
      status.textContent = `Share this link: ${location.href}`;
    }
  });

  function showAsk(open) {
    ask.hidden = !open;
    now.hidden = open || !current;
  }

  let current = null;

  function choose(id, { source, animate = true } = {}) {
    if (id && !PRESETS[id]) id = null;
    current = id;
    persist(id);
    apply(id, animate && !reduceMotion);
    $$('.chip-btn', chipsBox).forEach((c) => c.setAttribute('aria-pressed', String(c.dataset.preset === id)));
    if (id) {
      $('#tailor-name').textContent = PRESETS[id].label;
      $('#tailor-pitch').textContent = PRESETS[id].pitch;
      status.textContent = source === 'text'
        ? `Matched "${PRESETS[id].label}". Page tailored.`
        : source === 'load' ? '' : `Page tailored for ${PRESETS[id].label}.`;
      showAsk(false);
      if (source !== 'load') $('#tailor-change').focus({ preventScroll: true });
    } else {
      status.textContent = source === 'reset' ? 'Showing the general page.' : '';
      showAsk(true);
    }
  }

  function apply(id, animate) {
    const p = id ? PRESETS[id] : null;
    // 1. copy (crossfade)
    for (const el of textSlots) {
      const key = el.dataset.aud;
      const html = p ? (p[key] ?? general.text[key]) : general.text[key];
      if (el.innerHTML === html) continue;
      if (animate) {
        el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140, easing: 'ease-in' }).finished
          .then(() => { el.innerHTML = html; el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: 'ease-out' }); })
          .catch(() => { el.innerHTML = html; });
      } else {
        el.innerHTML = html;
      }
    }
    // 2. projects: selection + order. Presets show six equal cards (two full rows of
    //    three); only the general page keeps its wide lead card.
    const order = p ? p.projects.slice(0, MAX_PROJECTS) : general.projects;
    const wide = p ? null : general.wide;
    reorder(projGrid, '[data-proj]', 'proj', order, animate, (el) => {
      el.classList.toggle('proj-wide', el.dataset.proj === wide);
    });
    // 3. demos: selection + order (featured demo first; the general page shows all)
    reorder(demoGrid, '[data-demo]', 'demo', p ? p.demos : general.demos, animate);
    // 4. experience emphasis
    $$('[data-job]').forEach((el) => el.classList.toggle('is-feature', !!p && p.jobs.includes(el.dataset.job)));
    $$('[data-pt]').forEach((el) => el.classList.toggle('is-key', !!p && p.points.includes(el.dataset.pt)));
    // 5. section order (+ numbers and nav, via CSS)
    setSections(p ? p.sections : 'base', animate);
    document.documentElement.dataset.audience = id || 'general';
  }

  /** Apply a section-order group: CSS places the sections; the DOM is moved to match so tab order follows. */
  function setSections(group, animate) {
    const ids = SECTION_ORDER[group] || SECTION_ORDER.base;
    const secs = ids.map((k) => document.getElementById(k)).filter(Boolean);
    const contact = document.getElementById('contact');
    const before = new Map(secs.map((el) => [el, el.getBoundingClientRect().top]));
    const html = document.documentElement;
    const inOrder = secs.every((el, i) => i === 0 || secs[i - 1].compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
    if (html.dataset.order === group && inOrder) return;
    html.dataset.order = group;
    for (const el of secs) el.parentNode.insertBefore(el, contact);
    if (!animate) return;
    for (const el of secs) {
      const dy = before.get(el) - el.getBoundingClientRect().top;
      if (Math.abs(dy) < 1) continue;
      el.animate([{ transform: `translateY(${dy}px)`, opacity: 0.6 }, { transform: 'none', opacity: 1 }],
        { duration: 520, easing: 'cubic-bezier(.2,.7,.2,1)' });
    }
  }

  /** Move children into `order` (others hidden), animating position changes (FLIP). */
  function reorder(grid, sel, key, order, animate, each) {
    const items = $$(sel, grid);
    const before = new Map(items.map((el) => [el, el.hidden ? null : el.getBoundingClientRect()]));
    const byId = new Map(items.map((el) => [el.dataset[key], el]));
    const shown = order.map((k) => byId.get(k)).filter(Boolean);
    for (const el of items) {
      el.hidden = !shown.includes(el);
      if (each) each(el);
    }
    // keep hidden items at the end, shown ones in the requested order
    shown.forEach((el) => grid.appendChild(el));
    items.filter((el) => el.hidden).forEach((el) => grid.appendChild(el));
    const tail = $$(':scope > :not(' + sel + ')', grid); // anything else (none today) stays last
    tail.forEach((el) => grid.appendChild(el));
    if (!animate) return;
    for (const el of shown) {
      el.classList.add('in'); // revealed state; avoids a second reveal transition
      const a = before.get(el);
      const b = el.getBoundingClientRect();
      if (!a) {
        el.animate([{ opacity: 0, transform: 'scale(.97)' }, { opacity: 1, transform: 'none' }], { duration: 380, easing: 'cubic-bezier(.2,.7,.2,1)' });
        continue;
      }
      const dx = a.left - b.left, dy = a.top - b.top;
      const sx = a.width / (b.width || 1), sy = a.height / (b.height || 1);
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1 && Math.abs(sx - 1) < 0.01 && Math.abs(sy - 1) < 0.01) continue;
      el.animate([
        { transformOrigin: 'top left', transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` },
        { transformOrigin: 'top left', transform: 'none' },
      ], { duration: 520, easing: 'cubic-bezier(.2,.7,.2,1)' });
    }
  }

  function persist(id) {
    const url = new URL(location.href);
    if (id) url.searchParams.set('for', id); else url.searchParams.delete('for');
    history.replaceState(null, '', url);
    try {
      if (id) localStorage.setItem(STORAGE_KEY, id); else localStorage.removeItem(STORAGE_KEY);
    } catch (_) { /* storage blocked: URL still carries the choice */ }
  }

  // Initial choice: URL wins over storage.
  let initial = new URLSearchParams(location.search).get('for');
  if (initial && !PRESETS[initial]) initial = null;
  if (!initial) {
    try { initial = localStorage.getItem(STORAGE_KEY); } catch (_) { initial = null; }
    if (initial && !PRESETS[initial]) initial = null;
  }
  const expected = initial ? PRESETS[initial].sections : 'base';
  if (document.documentElement.dataset.order !== expected) {
    console.warn(`[audience] <head> chose section order "${document.documentElement.dataset.order}", expected "${expected}"; correcting`);
  }
  if (initial) choose(initial, { source: 'load', animate: false });
  else { setSections('base', false); showAsk(true); }

}
