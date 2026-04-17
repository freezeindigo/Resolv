# ARIA BUILD JOURNAL
### Agentic Routing & Intelligence for Amenities
#### Architecture Decisions, Reasoning, and Interview Defense

---

## What ARIA Is (30-second version)

ARIA is the intelligence layer between a resident's complaint and the FM company's dispatch. When someone says "water leaking from ceiling," ARIA doesn't just label it "Plumbing." It checks if the flat above had a recent complaint, whether the building has a monsoon seepage pattern, whether there's a cluster of similar complaints in the same vertical stack — then decides: "This is structural seepage, not a pipe leak. Send waterproofing assessment team, not a plumber. Also send a plumber today for immediate mitigation."

**It's not a chatbot** (we don't talk to residents). **It's not a classifier** (we don't assign labels). **It's a decision engine** that reasons about root causes using building-level evidence and routes based on cost-of-error, not just likelihood.

---

## The Problem (with proof from real data)

We analyzed 17,029 real complaints from Godrej Properties (23 sites, 4 zones, ~40,000 residents).

### What the data shows:

**Taxonomy chaos:** 398 complaints with "seepage" keyword scattered across 16 different categories — Plumbing (141), Seepage (60), Project (49), Civil Work (47), Mason (36), Leakage (25), and 10 others. The same physical phenomenon classified differently depending on who logs it.

**Cross-unit causality ignored:** 674 complaints explicitly reference other flats — "leakage from flat 403 above," "nany trap leakage from B 206." A single-complaint classifier cannot reason about inter-unit causal chains.

**Systemic failures invisible:** One flat (V1006, Golf Link Crest) logged 243 complaints across 15 categories over 2 years. The system keeps routing individual tickets instead of triggering a comprehensive flat inspection. 212 complaints open for over 200 days.

**Category contamination:** The "FM" category contains subcategories called "PLUMBER" (149), "ELECTRICAL" (86), "GENRAL" (59). The "Civil Work" category contains "Carpenter" (45) and "plumbing" (6). The taxonomy itself is broken.

### The cost:

Every wrong dispatch costs ₹500–₹5,000. At 25–30% misrouting across ~660 complaints/month, that's 165–200 wrong dispatches monthly, costing ₹1–10 lakh/month per developer.

---

## Why Not Just a Classifier?

This is the most important question in interviews. Here's the answer:

**A classifier optimizes for: "What's most likely?"**
**ARIA optimizes for: "What's the most expensive mistake to make?"**

Example: "Water leaking from ceiling."

A classifier says: "70% plumbing, 25% structural, 5% other → route to plumbing." It's right 70% of the time. But the 25% it gets wrong costs 10-100x more than the 70% it gets right.

ARIA's formula: `adjusted_score = likelihood × cost_of_error_weight`

- Plumbing: 0.70 likelihood × 1.0 weight = **0.70 adjusted**
- Structural: 0.25 likelihood × 10.0 weight = **2.50 adjusted**

Structural wins. Not because it's more likely — but because missing it is catastrophically more expensive. The arbiter routes toward structural assessment, possibly with a plumber sent in parallel for immediate mitigation.

**Interview line:** "A classifier optimizes uniform accuracy. ARIA encodes asymmetric costs and produces portfolio decisions. That's a decision engine, not a classifier."

---

## The Tiered Architecture (why not multi-agent for everything?)

### The challenge from interviewers:

"If multi-agent is better, why not use it for every complaint?"

### The answer:

Running 4 hypothesis agents on "flush not working" wastes ₹15-35 of compute on a problem that's plumbing 99% of the time. Intelligence should be proportional to the cost of error.

| Tier | % Volume | When | Reasoning | LLM Calls | Cost |
|------|----------|------|-----------|-----------|------|
| Tier 1 | ~35% | Unambiguous: "flush not working" → Plumbing | Rule-based | 0 | ₹0 |
| Tier 2 | ~40% | Some ambiguity, not high-stakes | Single agent + context | 1 | ₹2-10 |
| Tier 3 | ~25% | Ambiguous AND high cost-of-error | Multi-agent deliberation | 3-5 | ₹15-35 |

**Blended average: ~1.4 LLM calls/complaint, ~₹5-12/complaint.**

The complexity assessor decides the tier based on three signals:
1. **Textual ambiguity:** Does the complaint text contain multi-cause symptoms (leak, seepage, water, crack)?
2. **Historical signal:** Does this flat/building have active complaint clusters or repeat patterns?
3. **Cost-of-error:** Does the likely domain carry high misrouting cost (structural, safety)?

**Key design decision:** Tier 2 can be UPGRADED to Tier 3 mid-pipeline. If the pattern state query detects a vertical stack cluster after context assembly, the complaint gets promoted. The system adapts to new information, not just the initial assessment.

**Interview line:** "I scale intelligence to the cost of error. Simple complaints bypass agents entirely. High-stakes ambiguous complaints get independent hypothesis evaluation with arbitration. The system spends intelligence where it matters."

---

## Why Independent Hypothesis Agents (The Anchoring Bias Argument)

This was debated extensively. The key question: why not evaluate all hypotheses in one LLM call?

### The argument FOR one prompt:
- Simpler, cheaper, one LLM call
- The model sees all evidence simultaneously
- You can structure the prompt: "evaluate hypothesis A, then B, then C"

### The argument AGAINST (and why we chose independent agents):

**Anchoring bias.** When a single LLM evaluates Hypothesis A, then B, then C in one forward pass, the evaluation of B is contaminated by having already formed an opinion about A. The model's attention weights are shared across the entire sequence. If the first 200 tokens lean toward "plumbing," the next 200 tokens evaluating "structural" fight upstream against that framing.

**Why this matters specifically for our problem:** Plumbing is the most common complaint category (36% of all complaints). In a single-pass evaluation, the model will anchor on plumbing because it's the most statistically likely answer — even when the evidence for structural is stronger in this specific case. The anchoring effect systematically under-evaluates the expensive-to-miss hypotheses.

**What independent agents do:** Each hypothesis agent gets:
- Its own system prompt focused on ONE hypothesis
- ONLY evidence relevant to its hypothesis (via evidence filter)
- No knowledge of what other hypotheses exist or their scores

Three separate calls, running in parallel. Latency = one call (since they're parallel). Cost = 3 calls. The independence produces measurably less biased assessments.

**When this is worth the cost:** Only for Tier 3 complaints (25% of volume) where the cost difference between hypotheses is 10x or more. For Tier 2 (40%), a single reasoning call is fine because the stakes don't justify the extra cost.

**Interview line:** "I use agents to model competing intelligence, not to model functions. Each hypothesis agent produces an independent assessment without anchoring on other hypotheses. This matters when the cost difference between plumbing (₹500) and structural (₹5,000-50,000) is 10-100x."

---

## The Dynamic Hypothesis Library (Why Not Fixed at 3 Agents?)

### The problem with hardcoding agents:

"Water from ceiling" needs: plumbing, structural, environmental, HVAC condensate hypotheses.
"AC not cooling" needs: compressor, electrical, post-service, external factor hypotheses.
"Door not closing" needs: mechanical wear, structural movement, installation defect hypotheses.

Different domains need different hypothesis sets. A fixed 3-agent system would either miss relevant hypotheses or waste compute on irrelevant ones.

### How the dynamic library works:

The hypothesis library is a YAML configuration. Each domain (water_plumbing, electrical, structural_civil, carpentry, HVAC, lift_elevator, safety_security) defines 2-4 hypothesis types with:
- System prompt template (isolated, focused on one hypothesis)
- Evidence filter (which context fields this agent sees)
- Cost-of-error weight (how expensive is it to miss this?)
- Optional trigger condition (e.g., HVAC condensate only spawns for ceiling complaints)

For Tier 3 complaints, the system:
1. Identifies the domain (via domain classifier)
2. Loads the hypothesis set for that domain from YAML
3. Spawns 2-4 hypothesis agents in parallel

### Current library:

| Domain | % of Data | Hypothesis Agents |
|--------|-----------|-------------------|
| Water/Plumbing | 36% | pipe_failure (1.0×), structural_seepage (10.0×), environmental (2.0×), hvac_condensate (1.5×) |
| Electrical | 19.7% | internal_wiring (1.0×), external_supply (2.0×), equipment_specific (0.8×), safety_hazard (20.0×) |
| Structural/Civil | 9.2% | settlement (5.0×), waterproofing (10.0×), installation_defect (3.0×), environmental_damage (2.0×) |
| Carpentry | 15.2% | mechanical_wear (1.0×), structural_movement (4.0×), installation_defect (2.0×) |
| HVAC | 3.3% | compressor_gas (2.0×), electrical_fault (1.5×), post_service (3.0×) |
| Lift/Elevator | 2.8% | electrical_fault (5.0×), mechanical_wear (3.0×), maintenance_overdue (2.0×) |
| Safety/Security | 3.5% | fire_system (50.0×), gas_leak (50.0×), structural_hazard (20.0×), security_equipment (3.0×) |

**Safety domain special rule:** ALL hypothesis agents always spawn, regardless of tier. Missing a gas leak or fire hazard is never acceptable.

### Platform extensibility:

When commercial properties onboard → add "commercial_bms" domain with BMS failure, chiller, fire suppression hypotheses. When industrial onboards → add "process_water" domain. New domains are YAML config + prompt files. The orchestration code doesn't change.

**Interview line:** "The hypothesis library is config-driven, not hardcoded. Each new property type or domain adds a YAML entry and prompt files. The orchestration layer stays the same. That's how we scale from residential to commercial to industrial without rewriting the pipeline."

---

## The Pattern Interpreter (Why Building-Level Intelligence Matters)

### The problem it solves:

Individual complaints don't have enough signal for structural diagnosis. "Water from ceiling" could be anything. But CLUSTERS tell a story.

If Flats 1003, 1103, 1203, 1303 (vertical stack) all report seepage within 72 hours — that's not 4 independent plumbing problems. That's one structural problem: waterproofing failure in the building envelope affecting an entire vertical line.

### How it works:

1. **Pattern State (Redis):** Maintains sliding-window aggregation of complaints. Uses DBSCAN over spatial-temporal features (building, tower, floor, domain, timestamp). Deterministic clustering — no LLM needed for detection.

2. **Pattern Interpretation Agent (LLM):** Takes cluster data + hypothesis scores and reasons: "Does this cluster corroborate or contradict the top hypothesis?" This IS an LLM task because the meaning of a cluster depends on context — 4 plumbing complaints on the same floor might be a coincidence, while 4 seepage complaints in a vertical stack almost certainly indicate structural failure.

3. **Output:** Adjusts hypothesis likelihoods. Vertical stack seepage pattern → structural_seepage gets +0.2, pipe_failure gets -0.1. This adjusted signal feeds into the arbiter.

### Why it's not a batch job:

I initially proposed the pattern detector as a nightly batch analytics job. That was wrong. When complaint #7 arrives from the same vertical stack that already has 6 seepage complaints in 48 hours, the pattern should influence the routing of complaint #7 RIGHT NOW — not in tomorrow's report.

The pattern state layer is live (Redis sliding windows). The pattern interpretation agent runs per Tier-3 complaint in real-time. The cross-complaint intelligence is immediate, not delayed.

**Interview line:** "The unit of intelligence isn't the complaint — it's the building graph. A single complaint has low signal. The relationship between complaints, flats, floors, and time is where the diagnostic power lives."

---

## The Arbiter (Why Multi-Action Decisions Matter)

### What the arbiter does:

Takes ALL inputs — hypothesis scores, pattern interpretation, cost-of-error weights, vendor availability, SLA constraints — and produces a final routing decision.

### The key capability: multi-action decisions.

Real FM routing isn't always "send one vendor." Example:

- Structural hypothesis: 0.35 likelihood × 10.0 weight = 3.5 adjusted
- Plumbing hypothesis: 0.60 likelihood × 1.0 weight = 0.6 adjusted
- Pattern: vertical stack cluster corroborates structural

Arbiter decision: "Send plumber TODAY for immediate leak mitigation (the resident needs the water stopped). Schedule structural assessment team for TOMORROW (to diagnose the root cause). If plumber fix doesn't hold in 48 hours, confirm structural and dispatch waterproofing team."

That's three actions, staggered, with an escalation trigger. A classifier produces one label. The arbiter produces an operational plan.

### The safety override:

If ANY safety hypothesis (fire, gas leak, electrical hazard) has likelihood > 0.3, that action goes FIRST regardless of other scores. No cost-benefit analysis for life safety — just act.

**Interview line:** "The arbiter doesn't pick a label. It produces a routing plan with primary action, secondary action, escalation triggers, and SLA targets. It can say 'send both' when uncertainty is genuine, rather than forcing a binary decision."

---

## Pipeline Nodes vs Agents (The Principle)

### The rule:

**If it can be done with SQL, keywords, or API calls → pipeline node (no LLM).**
**If it requires evaluating evidence and forming a judgment → agent (LLM).**

### Pipeline nodes (no LLM):
| Node | What it does | Why no LLM |
|------|-------------|------------|
| Intake Normalizer | Maps MyGate/NoBrokerHood schema to unified format | Schema mapping is deterministic |
| Domain Classifier | Keywords → domain (water, electrical, etc.) | Keyword rules, small model fallback |
| Complexity Assessor | Assigns Tier 1/2/3 | Rule-based on ambiguity + history signals |
| Context Assembler | 3 parallel SQL queries (~65ms) | Database retrieval, not reasoning |
| Pattern State Query | DBSCAN cluster detection on Redis | Deterministic spatial-temporal clustering |
| Execution Layer | Vendor dispatch, notifications, scheduling | API calls, not reasoning |
| Audit Logger | Persist reasoning trace to database | Database write, not reasoning |

### Agents (LLM-powered):
| Agent | What it does | Why it needs LLM |
|-------|-------------|-----------------|
| Tier 2 Single Reasoner | Complaint + context → routing decision | Needs to weigh ambiguous evidence |
| Hypothesis Agents (2-4 per Tier 3) | Evaluate one hypothesis with isolated evidence | Independent evidence evaluation |
| Pattern Interpreter | "Does this cluster change the diagnosis?" | Interpreting spatial patterns requires judgment |
| Arbiter | Integrate all signals → multi-action decision | Cost-weighted multi-factor decision under uncertainty |

**Interview line:** "I have 4-6 LLM-powered agents and 7 pipeline nodes. The nodes handle retrieval, execution, and logging. The agents handle root cause disambiguation and cost-weighted routing. If you ask me to justify any agent, I can tell you exactly what degrades if I remove it. If you ask about any node, I can explain why an LLM adds nothing there."

---

## How ARIA Knows It's Structural, Not Plumbing

This is the walkthrough for interviews. Use the "water leaking from ceiling" example.

### Complaint arrives: "Seepage from ceiling in master bedroom"

**Step 1 — Domain classifier:** Keywords "seepage" + "ceiling" → domain: water_plumbing. Confidence: high.

**Step 2 — Complexity assessor:** "Seepage" is an ambiguous symptom (Tier 3 trigger). Check history: this flat has 3 prior complaints. Check building: 4 similar complaints in this tower in last 30 days. → **Tier 3**.

**Step 3 — Context assembler (parallel, ~65ms):**
- Flat history: 3 prior complaints — 2 plumbing (resolved), 1 seepage (marked resolved but recurred)
- Adjacent flats: Flat above (1903) had seepage complaint 2 weeks ago. Flat below (1703) reported "damp patch on ceiling" last week.
- Building pattern: 6 seepage complaints across floors 14-19 in this tower, last 45 days

**Step 4 — Pattern state query:** DBSCAN detects cluster: 4 seepage complaints in vertical stack (floors 15-19), last 2 weeks. → Stack pattern confirmed. Tier stays at 3.

**Step 5 — Hypothesis agents spawn (parallel, ~2s):**

| Agent | Sees | Score | Reasoning |
|-------|------|-------|-----------|
| pipe_failure | Flat history only | 0.25 | Prior plumbing fixes resolved. No active fixture issue described. Flat above has seepage (supports upstream leak) but flat below ALSO has it → less likely individual pipe. |
| structural_seepage | Stack pattern + adjacent flats + building age | 0.75 | Vertical stack pattern (4 flats). Prior fix recurred in 90 days. Monsoon season. Multiple floors affected. Classic waterproofing failure signature. |
| environmental | Weather + roof data | 0.15 | Monsoon active, but pattern is too localized to one vertical stack for weather alone. |
| hvac_condensate | AC data + ceiling location | 0.10 | Ceiling complaint but no AC service history, no condensate pattern. |

**Step 6 — Pattern interpreter:** "Vertical stack cluster of 4 seepage complaints across floors 15-19 strongly corroborates structural_seepage hypothesis. This is not 4 independent plumbing problems — it's one building envelope failure. Adjust structural_seepage +0.15, pipe_failure -0.10."

**Step 7 — Arbiter:**
- pipe_failure: (0.25 - 0.10) × 1.0 = **0.15 adjusted**
- structural_seepage: (0.75 + 0.15) × 10.0 = **9.00 adjusted**
- environmental: 0.15 × 2.0 = **0.30 adjusted**
- hvac_condensate: 0.10 × 1.5 = **0.15 adjusted**

Decision: structural_seepage wins decisively.

**Routing:** "PRIMARY: Send senior plumber today for immediate leak mitigation in Flat 1803, P2, 24h SLA. SECONDARY: Schedule structural/waterproofing assessment team for Floors 14-19 west stack, P1, 48h SLA. ESCALATION: If plumber fix doesn't hold in 48h, confirm structural and dispatch full waterproofing remediation team."

**Total pipeline time:** ~3-4 seconds. **Total LLM cost:** ~₹20-30 for this Tier 3 complaint.

---

## The Feedback Loop (How ARIA Gets Smarter)

When the plumber visits Flat 1803 and says "not a pipe issue — it's coming from the building envelope," that resolution outcome feeds back:

1. The pipe_failure hypothesis for this building type gets a negative signal
2. The structural_seepage hypothesis for vertical-stack-pattern complaints gets a positive signal
3. Over time, the complexity assessor learns: "seepage complaints in buildings >7 years old with monsoon timing → go straight to Tier 3"

This is NOT autonomous learning (dangerous in production). It's:
- System surfaces patterns monthly: "Building T6 plumbing hypothesis overestimates by 15% for seepage cases"
- Human reviews and approves threshold adjustments
- Config updated in YAML

**The compounding advantage:** Month 1, ARIA is uncertain on ambiguous complaints. Month 6, it knows which towers have monsoon seepage patterns. Year 2, it predicts complaints before they're filed and schedules proactive inspections.

**Interview line:** "The feedback loop only closes if you own BOTH routing AND resolution data. Competitors who only do ticketing (MyGate, NoBrokerHood) never close this loop. After 12 months, our accuracy gap is 12 months of resolution outcomes they don't have."

---

## The Moat

### Why this is hard to replicate:

1. **Cross-developer pattern intelligence:** When we serve Godrej, DLF, and Prestige simultaneously, we see patterns no individual developer sees. "Buildings constructed 2018-2021 using Method X have 3.2x seepage recurrence rate." That insight improves routing for ALL customers.

2. **Data network effect:** Every resolved complaint makes future decisions more accurate. Every new developer adds episodic memory. Every FM company adds vendor performance data. The platform gets smarter with every customer.

3. **Hypothesis library depth:** 24 prompt files, 7 domains, each calibrated against real complaint data. A competitor starting from scratch needs the same 17,000+ complaints, the same domain expertise, and the same architectural iteration to reach this quality.

4. **The feedback loop:** Resolution outcomes → recurrence detection → hypothesis recalibration → better routing → fewer reassignments. Competitors without resolution data can't close this loop.

---

## Cost Model

| Tier | Volume | Agents | Tokens | Cost/complaint |
|------|--------|--------|--------|---------------|
| Tier 1 | 35% | 0 | 0 | ₹0 |
| Tier 2 | 40% | 1 (Haiku) | ~2,000 | ₹0.50-1 |
| Tier 3 | 25% | 4-6 (Sonnet) + 1 (Opus) | ~10,000 | ₹8-15 |
| **Blended** | **100%** | **~1.4** | **~2,500** | **₹1.5-3** |

Full eval (17K complaints): ~₹34,000 ($400). Run 100-sample first.

### Cost levers:
- **Prompt caching:** Common system prompt prefixes cached across calls (~20-30% savings)
- **Early exit:** Tier 1 = zero LLM cost
- **Model routing:** Haiku for Tier 2 (cheap), Sonnet for hypotheses, Opus only for arbiter
- **Parallel execution:** Hypothesis agents run simultaneously. Latency = 1 call. Cost = N calls.
- **Feedback-driven tier migration:** As system learns building patterns, more complaints shift Tier 3 → Tier 2 → Tier 1

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Orchestration | LangGraph | Stateful DAG with conditional branching, human-in-the-loop checkpoints |
| Database | PostgreSQL 16 | Structured queries, joins for adjacency, JSONB flexibility |
| Pattern state | Redis 7 | In-memory speed for sliding windows, DBSCAN clustering |
| Semantic memory | ChromaDB | Vector retrieval for FM rules and domain knowledge |
| LLM | Anthropic API | Haiku (Tier 2), Sonnet (hypotheses), Opus (arbiter) |
| API | FastAPI | Async Python, auto-docs, WebSocket support |
| Language | Python 3.9+ | LangGraph/Anthropic SDK ecosystem |

---

## Customer Acquisition — Layered GTM

| Layer | Customer | Role | Revenue | Timeline |
|-------|----------|------|---------|----------|
| 1 | Developers (Godrej first) | Proof of concept, case study | Free/pilot | Month 1-6 |
| 2 | FM Companies (Quess, CBRE, JLL) | Revenue engine. 1 contract = 50-200 properties | ₹1-5L/month per portfolio | Month 6-18 |
| 3 | Gate Platforms (MyGate, NoBrokerHood) | Scale lever. 1 integration = 25,000+ societies | ₹5-15/complaint or rev share | Month 12-24 |
| 4 | RWAs / Societies | Indirect beneficiaries | Never the direct buyer | Ongoing |

**Key insight:** FM companies are the real distribution channel. Each FM company services 50+ developers. One partnership gives access to 50-200 properties without selling to individual RWAs. They have budgets, understand ROI, and their margin directly improves with better routing.

**Gate platforms are the scale play.** MyGate has 25,000+ societies. If ARIA becomes the intelligence layer inside their complaint module, one integration = nationwide distribution. They won't build it themselves — multi-agent reasoning for complaint routing is a deep AI problem, not their core competency (society ERPs).

---

## Interview Quick-Fire Answers

**"Why not just use GPT/Claude directly with a good prompt?"**
We do use Claude — but the architecture around it is what creates value. A single Claude call with "classify this complaint" gives you a label. ARIA gives you independent hypothesis evaluation, spatial pattern detection, cost-weighted arbitration, and multi-action routing decisions. The LLM is the reasoning engine; the architecture is the decision framework.

**"How do you handle cold start for a new developer?"**
Tier 1 rules work immediately (keyword-based). Tier 2 reasoning works with minimal context (one LLM call). Tier 3 benefits from building history — so for new buildings, more complaints initially go to Tier 2 and get upgraded to Tier 3 only when cluster patterns emerge. As complaints accumulate, the system gets sharper. Cross-developer patterns from existing customers provide priors even for new buildings.

**"What if the LLM hallucinates?"**
Three safeguards: (1) Structured output — agents must return JSON with specific fields, not free text. (2) Confidence scores — low confidence triggers human escalation, not autonomous routing. (3) The arbiter sees all hypothesis scores and can detect when agents disagree wildly, flagging for human review. Plus, worst case, a wrong routing decision means the wrong vendor visits — it's a ₹500-5,000 error, not a life-safety issue (safety domain has its own override).

**"What's the accuracy?"**
Human classification is already ~70-75% first-time-right (25-30% reassignment rate). Our target is 85-90%. But accuracy alone isn't the right metric — cost-weighted accuracy is. Getting structural right matters 10x more than getting plumbing right. We measure: first-time-right rate, cost-weighted accuracy, reassignment rate, and resolution time.

**"Why multi-agent and not fine-tuning?"**
Fine-tuning gives you a better classifier. We don't want a classifier — we want a decision engine that reasons about competing hypotheses under uncertainty. Fine-tuning also requires labeled data (ours is inconsistent) and produces a static model (can't adapt per-building without retraining). The multi-agent architecture adapts per-building through context retrieval and per-domain through the hypothesis library — no retraining needed.

**"How does this scale beyond residential FM?"**
The hypothesis library is YAML config. Orchestration doesn't change per vertical. FM is proof of concept. Same pattern applies to: hospital facilities (patient safety asymmetry), data centres (downtime cost per minute), insurance claims (fraud vs genuine), commercial real estate. New vertical = new YAML + new prompts.

---

## Build Status

| Day | Task | Status |
|-----|------|--------|
| Day 1-2 | PostgreSQL schema, ETL (17,029 complaints), flat adjacency | ✅ |
| Day 3 | Domain classifier + complexity assessor (keyword rules) | ✅ |
| Day 4 | Context assembler (3 parallel async SQL, ~65ms) | ✅ |
| Day 5 | Hypothesis library YAML + 24 prompt files | ✅ |
| Day 6 | Redis pattern state (DBSCAN) + arbiter agent | ✅ |
| Day 7 | LangGraph full graph compiled and verified | ✅ |
| Day 8 | FastAPI — 5 endpoints | ✅ |
| Day 9 | Eval framework (cost gate before running) | ✅ |
| Day 10 | Demo UI + smoke test | In progress |

**Known bugs being fixed:**
1. Pattern interpreter not wired into pipeline (spawn_hypotheses → arbitrate, should go through interpret_patterns)
2. pipe_failure prompt incorrectly ignores adjacent flat data
3. README shows Landed content instead of ARIA
4. No error handling around API calls
5. Pattern interpreter prompt has hardcoded hypothesis IDs

**GitHub:** https://github.com/freezeindigo/Resolv

---

## Tool Routing (How We Build)

| Task | Tool | Why |
|------|------|-----|
| Write code, fix bugs | Cursor | Uses subscription, not API $ |
| Architecture, prompts, strategy | Claude.ai chat | Pro subscription, unlimited |
| High-IQ architecture decisions needing file reads | Claude Code (Opus) | Short sessions only, expensive |
| Running eval scripts | Terminal directly | Controls API spend |

**Rule:** Never write large files in Claude Code. That's Cursor's job.
**Rule:** Every Claude Code session starts with "Read ARIA.md" for context.
**Rule:** After any session, update this journal with new decisions.

---

*Last updated: April 2026*
*Kartheek | IIT Bombay M.Tech | Senior PM, Godrej Living*
