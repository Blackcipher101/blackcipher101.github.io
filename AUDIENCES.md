# Audience presets

The "Who's visiting?" bar re-tailors the page in the browser. It has no backend and no LLM. The code is in `assets/js/audience.js`.

## How it works

- **Choosing:** a chip picks a preset directly. Free text is scored against the keyword lists in `KEYWORDS` and the highest total wins. Ties go to the earlier preset. A score of zero changes nothing and asks the visitor to pick a chip.
- **Matching rules:** terms of 5+ characters match as word prefixes ("inspect" matches "inspection"). Shorter terms and phrases must match whole words, so "av" does not fire on "available".
- **Saving the choice:** it goes into the URL as `?for=<id>`, so each version is a shareable link, and into `localStorage` (`nn-audience`) when storage is allowed. The URL wins over storage. No choice means the general page.
- **What a preset changes:**
  - the hero pill, headline and lede
  - the projects intro
  - which project cards show and in what order (at most 6, most relevant first)
  - which demos show and in what order (4 of the 8, the featured one first; the general page shows all 8)
  - which experience cards get a "Relevant" badge, and which bullets are bolded
  - a one-line pitch in the "Viewing as" bar
- **Copy rule:** all preset copy repeats claims already on the page or in the repo READMEs. Nothing is added.

## Presets and why they exist

| id | Who | Leads with |
|---|---|---|
| `startup` | Startup founder / CTO | 0 → 1 on ZPod, tech + product lead at BYTES, full-stack breadth |
| `enterprise` | Big company / enterprise | production-scale code, closed-loop CI/CD, leadership roles, reproducible evaluation |
| `underwater` | Underwater / marine | the five `uw-*` repos and rov-rl-mujoco |
| `embodied` | Embodied AI / humanoid labs | RL (rov-rl-mujoco, H-GAIL/PPO), Q14 dynamics, humanoid RL in progress, Griffin Labs |
| `av` | Autonomous vehicles | Minus Zero planning & prediction, simulators, DriveWorks |
| `defence` | Defence | Aeroarc VIRA and ARCV MULE, estimation under sensor dropout |
| `inspection` | Inspection / industrial / facilities | PhiSCAN, uw-inspect-report, uw-calib-health, Griffin Labs (TCR, Mega), Clutterbot |
| `legged` | Legged robots | ARCV MULE, Q14 Dynamics Lab, humanoid RL in progress |
| `recruiter` | Recruiter / hiring manager | the timeline in one paragraph, a skills line, contact |

These nine cover the kinds of reader the work actually serves: company stage (startup, enterprise), domain (underwater, AV, defence, inspection), robot type (embodied/humanoid, legged), and role (recruiting). A domain that has no matching work behind it has no preset.

## Featured demos

| id | Demos, in order | Why the first one |
|---|---|---|
| `startup` | MPPI, cart-pole RL, SCARA, SLAM | planning was the ZPod job; then breadth |
| `enterprise` | SCARA, SLAM, MPPI, DVL | deterministic, limit-aware motion planning (IK + A*), measured against a learned policy |
| `underwater` | DVL, sonar VFH, SLAM, MPPI | the idea behind uw-vio-dvl |
| `embodied` | cart-pole RL, stand-off DQN, SCARA, MPPI | a learned policy (`tools/train_pendulum.py`) racing LQR in the same run; then a Double DQN against a tuned P controller (`tools/train_standoff.mjs`) |
| `av` | MPPI, SLAM, cart-pole RL, DVL | planning & prediction at Minus Zero |
| `defence` | box-track diffusion, sonar VFH, stand-off DQN, DVL | tracking a manoeuvring target in a long-range camera with calibrated uncertainty (synthetic tracks); then navigating and estimating with degraded sensing |
| `inspection` | SCARA, SLAM, sonar VFH, DVL | manipulation for workcells such as Mega: learned reaching vs an IK planner |
| `legged` | cart-pole RL, SCARA, MPPI, SLAM | balance and recovery from pushes |
| `recruiter` | SLAM, cart-pole RL, SCARA, MPPI | a mix across estimation, learning, manipulation, planning |

The `embodied` and `recruiter` presets also bold the two summer schools in the Education card.

## Keyword lists (summary; the exact weights are in `KEYWORDS`)

- **startup:** startup, founder, co-founder, founding, cto, seed, series a, early stage, mvp, yc
- **enterprise:** enterprise, corporation, big company, multinational, mnc, oem, tier 1, faang, reliability
- **underwater:** underwater, subsea, auv, dvl, marine, ocean, rov, sonar, offshore, maritime
- **embodied:** embodied, humanoid, physical ai, vla, robot learning, reinforcement learning, imitation, sim2real, mujoco
- **av:** autonomous vehicle, self-driving, driverless, robotaxi, adas, automotive, av, trucking, two-wheeler
- **defence:** defence/defense, military, army, drdo, unmanned ground, tactical, ugv, dual-use
- **inspection:** inspection, industrial, facilities, cleaning, digital twin, scanner, infrastructure, workcell, ndt, lidar
- **legged:** legged, quadruped, robot dog, locomotion, biped, gait, unitree, anymal
- **recruiter:** recruiter, headhunter, hiring manager, talent, hr, cv, resume, sourcing, hiring
