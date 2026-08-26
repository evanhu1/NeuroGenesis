# 2026-08-25: co-evolutionary text-forge ratchet + learnability frontier

Status: completed
Date: 2026-08-25
Slug: 2026-08-25-coev-text-forge-ratchet
Code revision: uncommitted working tree atop befdf23 (co_evolution machinery added this session;
diff snapshot in artifact directory)

## Question

1. **Ratchet (E3).** Does an evolving adversarial text-forge population keep
   pushing brain competence past the static-environment plateau?
2. **Frontier (E4).** Where does open-ended acquisition break as environment
   complexity scales (snippet length 48) at matched lifetime budget?
3. **Dose confirmation (E5).** Does the linear self-supervised dose-response
   continue at passes 192 toward the BRIEF gate?

## Contract

As preregistered. All arms p1024/g200, generalize, len-48, lp12, ti8,
di/si=8/8, predictive coding; E5 hard passage PC p512/g150/seed 101 @192p.

## Result

### E3 — co-evolution vs static control at len-48/matched budget

| Arm | Sealed acc | Dev trajectory | Notes |
|---|---:|---|---|
| static s7 | **73.72%** (289/392) | plateau after gen ~100 | selected g199 |
| co-evolved s7 | 52.04% (204/392) | flat after gen ~75 | forge hardness saturated |
| co-evolved s27 | 59.95% (235/392) | **still climbing at gen 199** (64.5% dev) | |

**Preregistered "harmful" branch fired**: co-evolved arms trail the static
control by 14–22 points at matched budget. Diagnosis is clean and instrumented:
champion forge hardness hit **1.0 by generation ~1–2 and stayed there** —
forged len-48 text was so far beyond current acquisition ability that even the
best brain scored ~0% on it after 12 passes. With all forges tied at maximum
failure, forge selection had no gradient (noise drift), and half of every
brain's lifetime budget went to unlearnable text. The neutral panel half
preserved ticket flow (no extinction), exactly its design purpose.

The one positive signal: co-evolved s27's dev curve kept climbing through
gen 199 while static plateaus — weak evidence that *some* pressure persists,
but confounded by the budget penalty.

### E5 — deep dose extension

Passes 32/64/96/**192** → sealed **95/101/105/112** per 150. Near-linear
(+4 to +7 per +32p), dev-audit still rising at g149 (74.7%). The
self-supervised ladder continues to convert lifetime data into competence with
no saturation through 4× the original dose.

### Ceiling analysis (new instrumentation insight)

Exact-task n-char Markov ceilings on the hard passage: k=1 → **64/150**,
k=2 → 120, k=3 → 141, k=4 → 145. Champions at 95–112 already exceed the
k=1 ceiling via leaky-membrane context integration ("0% recurrent edges" ≠ no
temporal computation). Reaching 135+ requires reliable k≥3 integration —
this is what any difficulty ratchet should push on.

## Decision rule outcomes

- E3: harmful branch fired as preregistered; saturation diagnosis confirmed by
  trajectory telemetry.
- E5: linearity held (+7); gate chase remains viable (~480p projected for
  135/150) but each step costs more compute than the last.

## Interpretation and next decision

**What was built:** full co-evolutionary EA machinery (`evolution::co_ecology`):
dual populations, champion-forge panel biasing (50/50 neutral/forged split),
forge fitness from best-brain acquisition failure, deterministic generation
advancement (verified bit-identical across duplicate runs, both populations),
trajectory sidecar telemetry. Plus instance-panel flattening optimization.

**Why v1 ratchet failed:** absolute-failure fitness saturates whenever forge
difficulty exceeds single-lifetime learnability — a forge that maximizes
failure destroys its own gradient. This is the same trap as PowerPlay round 11.

**V2 requirements (next iteration):**
1. **Relative forge ranking** — forges compete against each other's sub-panel
   accuracy, not against absolute zero.
2. **Adaptive length knob** — forges control snippet length within a band
   anchored to demonstrated competence (zone-of-proximal-development / minimal
   criterion): hard enough to demand growth, easy enough to be learnable.
3. **Fitness = progress signal**, not failure rate: e.g., accuracy delta
   between early and late learning passes on forge text (measures whether the
   forge text is *learnable-with-effort*, the only regime that grows brains).

Also next: E4 frontier point at len-64/lp16 to complete the F(L,B) map; E5
continuation toward the gate at ~480 passes if compute allows.

Commands, stderr logs, result checksums, and forge trajectories preserved in
`artifacts/research/runs/completed/2026-08-25-coev-ratchet/`.
