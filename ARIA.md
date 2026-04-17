# ARIA — Agentic Routing & Intelligence for Amenities

**Living document. Updated after every strategic session.**
**Last updated: April 2026 — Phase 1 scaffold complete**

---

## Agent Map — What Every Agent Does and Why It Exists

This section explains every LLM-powered agent in the system. Each agent has exactly one job. If something can be done with a SQL query or a keyword match, it is NOT an agent.

### Why so many agents?

A single LLM prompt evaluating multiple hypotheses has a known failure mode: **anchoring**. Whichever hypothesis it considers first influences all subsequent scores. Isolated agents with isolated evidence produce genuinely independent assessments. This is the core architectural reason — not complexity for its own sake.

---

### Pipeline Nodes (deterministic, no LLM)

| Node | What it does | Why no LLM |
|---|---|---|
| **Intake Normalizer** | Maps source schema → unified complaint object | Pure data transformation |
| **Domain Classifier** | Keywords → domain (water_plumbing, electrical, etc.) | Rules cover 77% of cases correctly; LLM only for low-confidence fallback |
| **Complexity Assessor** | Assigns Tier 1/2/3 | Safety/recurrence/ambiguity rules are deterministic |
| **Context Assembler** | 3 parallel SQL queries → flat history, adjacent history, building pattern | Pure DB retrieval, ~65ms |
| **Pattern State Query** | DBSCAN over Redis sliding window → active complaint clusters | Deterministic clustering algorithm |
| **Execution Layer** | Dispatch vendor, notify resident, create ticket | Pure API calls (stubbed in Phase 1) |
| **Audit Logger** | Persist full reasoning trace | Pure DB write |

---

### Reasoning Agents (LLM-powered)

#### Tier 2: Single Reasoning Agent
**Model:** Claude Haiku (cost-efficient)
**When:** Complaints that need context but aren't high-stakes ambiguous
**What it does:** Takes complaint + full context package → produces routing decision in one call
**Why an agent:** The complaint text + flat history + building pattern together suggest a routing that no keyword rule can produce. Example: "water in kitchen" + flat has 3 prior plumbing calls + building has no structural issues → send plumber confidently.

---

#### Tier 3: Hypothesis Agents (run in parallel)

Each agent receives ONLY the evidence relevant to its hypothesis. This prevents the anchoring problem.

**Water / Plumbing domain — up to 4 agents:**

| Agent | Job | Cost-of-error weight | Why isolated |
|---|---|---|---|
| `pipe_failure` | Is this a fixture/pipe issue in THIS unit? | 1.0× (baseline) | Must not see structural evidence — would anchor toward structural if it did |
| `structural_seepage` | Is this waterproofing/building envelope failure? | **10.0×** | Receives adjacent/stack context. Missing this costs ₹5,000-50,000 |
| `environmental` | Is this rain/roof/external ingress? | 2.0× | Receives seasonal/roof data only |
| `hvac_condensate` | Is this AC condensate drain? | 1.5× | Only spawned for ceiling complaints |

**Electrical domain — up to 4 agents:**

| Agent | Job | Cost-of-error weight |
|---|---|---|
| `internal_wiring` | Fault inside this flat's circuit | 1.0× |
| `external_supply` | Building/MSEB supply failure | 2.0× |
| `equipment_specific` | Inverter/geyser/fan/VDP failure | 0.8× |
| `safety_hazard` | Sparking/shock/fire risk — **escalate if likelihood > 0.3** | **20.0×** |

**Other domains** (structural_civil, carpentry, HVAC, lift, safety_security) follow the same pattern — see `src/config/hypothesis_library.yaml` for full config.

---

#### Pattern Interpretation Agent
**Model:** Claude Sonnet
**When:** Pattern State Query returns active clusters for this building
**What it does:** Receives cluster data + hypothesis scores → interprets whether the spatial pattern corroborates or contradicts each hypothesis. Example: vertical stack seepage pattern → increase structural_seepage likelihood by +0.2, decrease pipe_failure by -0.1
**Why an agent:** The *meaning* of a cluster depends on its relationship to the current hypotheses. "6 seepage complaints in a vertical stack" could be coincidence, plumbing cascade, or systemic waterproofing failure. No deterministic rule captures this — it requires reasoning about the specific combination.

---

#### Arbiter Agent
**Model:** Claude Opus (highest reasoning quality — this is the decision-maker)
**When:** Always in Tier 3, after hypothesis agents + pattern interpreter have run
**What it does:** Receives all hypothesis scores (with cost-of-error weights), pattern interpretation, complaint history → applies asymmetric cost-of-error logic → produces final routing decision, which may be multi-action
**Why an agent (and why Opus):** This is the integration step. The arbiter must reason about tradeoffs: "structural_seepage has only 35% likelihood but 10× cost weight = adjusted score of 3.5. pipe_failure has 60% likelihood but 1× cost weight = score of 0.6. Route structural despite lower raw probability." This is asymmetric expected-value reasoning that requires the strongest model.

---

### Agent Cost Per Complaint

| Tier | Agents called | Approx tokens | Approx cost (INR) |
|---|---|---|---|
| Tier 1 | 0 | 0 | ₹0 |
| Tier 2 | 1 (Haiku) | ~2,000 | ₹0.50-1 |
| Tier 3 | 4-6 (Sonnet) + 1 (Opus) | ~8,000-12,000 | ₹8-15 |
| Blended (20/64/16 split) | — | ~2,500 avg | ₹1.5-3 avg |

**Full eval cost estimate:** 17,029 complaints × ₹2 avg = ~₹34,000 (~$400). Run 100-complaint sample first.

---

## What ARIA Is

ARIA is the intelligence layer inside Resolv. It is not a complaint ticketing system. It is not a CRM. It is a decision engine that takes an ambiguous complaint, reasons about it using spatial and historical context, and produces a routing + scheduling decision with a full reasoning trace.

The name matters: **Agentic** (it reasons, not just classifies) · **Routing** (dispatches to the right vendor) · **Intelligence** (learns from every resolution) · **for Amenities** (FM-first, but the pattern is portable).

Resolv is the product. ARIA is the brain.

---

## The Problem — Precise Framing

### What is actually broken in FM complaint management today

Residents in large residential complexes (Godrej, Prestige, DLF, Brigade) submit complaints through MyGate, NoBrokerHood, or direct CRM. What happens next:

1. Complaint sits in a queue (Excel dump or dashboard)
2. FM team manually reads it, decides vendor category
3. FM team sends to project team
4. Project team assigns vendor
5. Vendor gets a phone call or WhatsApp
6. Vendor visits — often without knowing what to bring, what floor, what the history is
7. **48–72 hour TAT. No acknowledgment to resident. No time slot. No tracking.**

The resident experience: you complain and you wait. You don't know if anyone has seen it. You don't know when someone is coming. If something is urgent — a leak, no power — you have no recourse except to call the FM helpline and wait on hold.

### Why it breaks NPS

The resident doesn't care about routing accuracy. They care about two things:
1. **Someone acknowledged my problem** (within minutes, not hours)
2. **I know when it will be fixed** (a confirmed time, not "soon")

Every CEO tag on LinkedIn, every housing.com review, every RWA escalation traces back to one of those two failures. Not the actual fix time — the silence before it.

### Why it breaks operations

- Wrong vendor dispatched → vendor can't fix it → reassignment → TAT doubles
- Vendor visits one flat at a time → 5 complaints in the same tower = 5 separate visits
- Numbers not shared with vendor (security policy) → vendor can't reach resident → visit fails
- No materials prediction → vendor arrives unprepared → visit fails → T+1 delay again
- No feedback loop → same wrong decision made repeatedly

### The asymmetric cost problem

Not all routing errors are equal. This is the core insight most FM software misses.

| Complaint type | Cost if wrong vendor sent |
|---|---|
| Broken door hinge | ₹500 wasted, reschedule |
| Plumbing leak | ₹800-1,500 wasted, reschedule |
| Structural seepage (sent plumber instead of structural) | ₹5,000-50,000 waterproofing job delayed, damage compounds |
| Electrical safety hazard (underestimated) | Life safety risk |

A system that optimises for uniform accuracy is solving the wrong problem. ARIA encodes **cost-of-error weights** per hypothesis — structural and safety hypotheses require far less confidence to escalate because the cost of underestimating them is catastrophic.

---

## The Solution — What ARIA Does

### The intelligence layer (routing)

ARIA processes every complaint through a tiered pipeline:

**Tier 1 — Rule engine (no LLM, ~0 cost):**
Unambiguous complaints matched against top patterns from historical data.
- "Flush not working" → Plumbing, P2, send plumber
- "MCB tripped, no power in flat" → Electrical, P1, send electrician
- ~35% of all complaints land here. Resolved in seconds.

**Tier 2 — Single reasoning agent:**
Complaints that need context but have a dominant hypothesis.
- Pulls flat history, adjacent complaints, building pattern from PostgreSQL
- One LLM call synthesises context + complaint → routing decision + reasoning trace
- ~40% of complaints. Resolved in under 10 seconds.

**Tier 3 — Multi-agent deliberation:**
Ambiguous complaints, high-cost domains, or spatial cluster detected.
- Multiple hypothesis agents run in parallel, each with isolated evidence
- Pattern interpretation agent checks if spatial/temporal cluster changes the diagnosis
- Arbiter integrates all signals with cost-of-error weights → final decision
- May produce multi-action: "plumber today + schedule structural assessment if not resolved in 48hrs"
- ~25% of complaints. Most important ones.

### Why agents, not a classifier

A single LLM evaluating multiple hypotheses anchors on whichever it considers first. Isolated hypothesis agents with filtered evidence produce independent assessments. The separation is the mechanism, not a design choice.

More importantly: a classifier forces a binary call. A classifier cannot say "I'm 60% sure it's plumbing and 35% sure it's structural — given the cost asymmetry, do both." The agent + arbiter pattern models uncertainty and makes portfolio decisions. This is not a feature. This is the product.

### How ARIA knows it's structural, not plumbing

This is the right question. "Seepage from ceiling" tells you almost nothing by itself. The differentiating evidence is external to the complaint text:

**Signals toward structural:**
- Multiple flats in the same vertical stack have seepage in the last 30 days (water is coming from outside the building envelope, not from one pipe)
- Complaint peaks during monsoon — July to September, dies in November
- Same complaint recurs in the same flat after a plumbing "fix" (the fix that doesn't stick is Bayesian evidence of structural cause)
- Exterior-facing wall or ceiling, no plumbing fixtures nearby
- Building 8–12 years old — typical waterproofing lifecycle failure point

**Signals toward plumbing:**
- Flat directly above has an active or recent plumbing complaint
- Wet spot is localised and circular, near pipe chase
- Complaint appeared suddenly, not gradually
- Only this one flat affected — no vertical pattern

This evidence lives in context: adjacent flat history, building-wide pattern, seasonal data, resolution outcomes. ARIA retrieves this via structured SQL queries before any hypothesis agent runs.

**The fresh complaint problem:** When there's no history (new building, first complaint), ARIA cannot be certain. This is handled correctly: uncertainty + high cost-of-error → multi-action decision. Send plumber today (cheap, rules out plumbing fast). If unresolved in 48 hours, structural assessment follows. You don't need certainty to act sensibly under uncertainty.

**The recurring complaint signal:** A plumbing "fix" that doesn't stick is the strongest structural indicator. ARIA tracks: resolution → recurrence within 90 days → automatic Bayesian update to structural hypothesis. By month 6 in a building, ARIA knows which towers have chronic seepage issues, which floors are problematic, which season triggers it. The accuracy compounds. Human triage stays flat.

---

## The Experience Layer (Scheduling + Dispatch)

Routing alone is not enough. The experience layer is what turns a correct routing decision into a resident who feels heard.

### The full flow

```
T+0       Complaint received
          → Auto-acknowledge via WhatsApp/SMS in < 2 minutes
          "Your complaint #1234 is received. We're on it."
          (This alone kills the anxiety that drives CEO tags)

T+0–5min  ARIA routes it
          → Resident gets time slot options
          "A plumber will visit. Choose: [Today 3–5pm] [Tomorrow 9–11am] [Tomorrow 2–4pm]"
          (Giving choice costs nothing. It converts passive frustration into active participation)

T+15min   Resident confirms slot
          → Work order created: flat number, floor, complaint summary,
            predicted materials, time slot. No phone number shared.

T=slot    Vendor dispatched
          → Resident gets tracking: "Ramesh is on the way, 12 minutes out"
          (The Uber moment. Once you have this, the complaint becomes a service experience)

T+slot    Work completed
          → Vendor marks done, resident rates in 2 taps
          → Resolution logged, feeds ARIA's learning loop
```

### The scheduling intelligence

The same spatial model ARIA builds for structural diagnosis is the model that makes vendor scheduling efficient. This is not a coincidence — it's the same data.

**Route batching:** 5 plumbing complaints in Tower 1 today → one vendor, floors 5 → 8 → 12 in sequence → same-day resolution for all five vs. 5 separate T+1 visits. TAT drops from 48 hours to 4 hours without a single vendor working faster.

**Skill matching:** Seepage complaint with 35% structural likelihood → don't send a junior plumber. Send a senior who can assess structural too. One visit that either resolves or correctly escalates vs. two visits where the first fails.

**Predictive materials:** Based on complaint type and flat history, predict what the vendor needs. Running toilet → 70% chance of a specific valve. Vendor arrives prepared → first-visit resolution rate goes up → TAT goes down → NPS goes up.

**Vendor exclusion:** Recurring complaint in same flat → check who was last dispatched. If same vendor "resolved" it before, route to a different vendor. The history knows what the queue does not.

### The numbers problem (security)

Vendors don't get resident phone numbers. Communication is masked — vendor gets flat number and work order, any calls route through the platform. Same model as Ola/Uber. Resident privacy maintained, vendor has everything they need to do the job.

---

## The Vendor / Partner Question

### Why partner with Urban Company / Snabbit now

Vendor management is a different business: hiring, training, certification, insurance, geography, surge capacity. UC and Snabbit have already done this. They have background-checked vendors, a mobile app for tracking, work order management, and payment rails.

The integration is clean: ARIA sends a structured work order (vendor type, skill level, building, flat, time slot, predicted materials, complaint context) via API. UC/Snabbit handle execution. ARIA gets back: vendor assigned, en-route time, completion status.

**What ARIA buys from the partnership:** Their vendor supply and their tech stack for resident-facing tracking. Not their intelligence — they have none on the building side.

### Why you own the intelligence layer regardless

UC/Snabbit don't have:
- The building's spatial model (which floors are adjacent, which stacks have seepage history)
- The complaint history (what was fixed, what recurred, which vendors succeeded)
- The cost-of-error framework (structural vs. plumbing distinction)
- The cluster detection (systemic vs. isolated)

They route a work order. ARIA decides what the work order should contain and whether the problem behind it is the one the resident described.

### The medium-term migration

UC/Snabbit data reveals which vendor types have best first-visit resolution rates, which complaint categories take longest, where materials prediction fails. Use that data to build direct relationships with the top-performing vendors. Over 12–18 months, reduce marketplace dependence. At Godrej scale (23 sites, 17,000+ complaints/year), the volume justifies dedicated vendor relationships.

### The long-term answer to "why not own the whole thing"

At Godrej scale, you eventually should. A trained, Godrej-certified FM workforce is a competitive moat JLL and Cushman can't replicate. But you build to it through data — know exactly what skills, what geographies, what response times you need before you hire for them. Don't build the workforce before you have the intelligence. Build the intelligence, let it tell you what workforce to build.

---

## Why This Is Defensible

### The compounding data moat

Month 1: ARIA is uncertain on ambiguous complaints. Makes cautious multi-action decisions.
Month 6: ARIA knows which towers have monsoon seepage patterns. Which floors are chronically problematic. Which vendor consistently fails on AC complaints. Which complaint types always need structural assessment.
Year 2: ARIA is predicting complaints before they're filed — scheduling proactive inspections based on seasonal patterns and building age.

Human triage never compounds. ARIA does. Every resolved complaint makes the next decision more accurate.

### The spatial model is dual-use

The same adjacency table and cluster detection that powers structural diagnosis powers vendor route optimisation. The same complaint history that powers hypothesis scoring powers vendor exclusion and materials prediction. One data structure, two products.

### The feedback loop is the moat

Resolution outcome → recurrence detection → hypothesis recalibration → better routing → fewer reassignments → less cost → measurable ROI. This loop is only possible if you own the routing AND the resolution data. Competitors who only do ticketing never close this loop.

---

## Expansion: What ARIA Becomes

The FM implementation is the proof of concept. The architecture is domain-agnostic.

**Same pattern, different verticals:**

| Vertical | Complaint type | Key hypotheses | Cost asymmetry |
|---|---|---|---|
| Commercial real estate | Office facility complaints | Tenant vs. landlord vs. shared infrastructure | Lease obligation risk |
| Hospital facilities | Clinical environment maintenance | Infection risk vs. equipment vs. structural | Patient safety (extreme asymmetry) |
| Data centre operations | Infrastructure incidents | Power vs. cooling vs. network vs. hardware | Downtime cost per minute |
| Insurance claims | First notice of loss | Fraud vs. genuine vs. ambiguous | False denial vs. fraudulent payout |

The hypothesis library is a YAML configuration. The orchestration layer doesn't change. Each new vertical is: new domain config + new prompts + new cost-of-error weights.

**The platform play:** ARIA as a configurable decision intelligence platform. FM is vertical 1. Healthcare facilities is vertical 2. The moat is the pattern — hypothesis-based reasoning with asymmetric cost-of-error — not any single vertical's data.

---

## Key Metrics That Matter

| Metric | Definition | Target |
|---|---|---|
| First-time-right rate | Complaints resolved without reassignment | Baseline from Godrej data → 20% improvement |
| Acknowledgment TAT | Time from complaint submission to first resident notification | < 2 minutes |
| Scheduling TAT | Time from complaint to confirmed time slot | < 15 minutes |
| Vendor visit TAT | Time from confirmed slot to visit | Same-day for P1/P2 |
| First-visit resolution rate | Vendor resolves on first visit | Track vs. baseline |
| Recurrence rate | Same complaint reopened within 90 days | Reduction = structural misrouting rate |
| NPS delta | Resident satisfaction before/after | Primary business metric |

---

## What Resolv / ARIA Is Not

- Not a ticketing system (ServiceNow, Freshdesk solve that — ARIA sits above them)
- Not a chatbot (residents don't want to chat, they want acknowledgment and a time slot)
- Not a workforce management tool (that's UC/Snabbit's job)
- Not a generic AI assistant (domain-specific reasoning with cost-of-error encoding)

---

## Interview Anchors (Quick Reference)

**"How do you know it's structural?"**
You don't know from the complaint text. You know from context: vertical stack pattern, monsoon timing, recurring failed plumbing fixes. In the absence of context (fresh complaint), you make a multi-action decision under uncertainty: plumber today + structural assessment if unresolved in 48 hours. The cost-of-error framework handles uncertainty without requiring certainty.

**"Why not just use a classifier?"**
A classifier forces a binary call and optimises for uniform accuracy. Structural seepage misrouted as plumbing isn't 1x worse — it's 10x worse. ARIA encodes asymmetric costs and can produce portfolio decisions ("do both, in sequence"). That's not a classifier. That's a decision engine.

**"What happens after routing?"**
Acknowledgment in 2 minutes. Time slot options sent to resident. Work order created with predicted materials and masked contact. Vendor tracking shared at dispatch time. Resolution logged and fed back to recalibrate future decisions. The routing and scheduling intelligence come from the same building model — they're not two separate systems.

**"Why partner with UC/Snabbit instead of owning the vendor layer?"**
We own the intelligence. UC/Snabbit own the execution. Vendor management is a different business — hiring, training, certification, insurance. They've solved it. We use their API and their vendor supply. The medium-term play is building direct vendor relationships as we accumulate data on which vendor types perform best in which complaint categories. At scale, we reduce marketplace dependence.

**"What's the moat?"**
The feedback loop. Resolution outcome → recurrence detection → hypothesis recalibration → better routing. This loop only closes if you own routing AND resolution data. Every resolved complaint makes the next decision more accurate. Competitors who only do ticketing never close this loop. After 12 months of data, the accuracy gap between ARIA and a fresh competitor is 12 months of resolution outcomes they don't have.

**"How does this scale beyond Godrej?"**
ARIA's hypothesis library is a YAML config. The orchestration layer doesn't change per vertical. FM is vertical 1. The same pattern — hypothesis agents, asymmetric cost-of-error, spatial context, feedback loop — applies to hospital facilities, commercial real estate, data centre operations, insurance claims intake. New vertical = new config + new prompts + new cost weights.
