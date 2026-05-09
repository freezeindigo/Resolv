# ARIA BUILD JOURNAL
### Agentic Routing & Intelligence for Amenities
#### Architecture Decisions, Reasoning, and Interview Defense

---

## What ARIA Is (30-second version)

ARIA is the intelligence layer between a resident's complaint and the FM company's dispatch. It doesn't classify — it reasons about root causes using building-level evidence and routes based on cost-of-error, not just likelihood.

**It's not a chatbot.** **It's not a classifier.** **It's a decision engine.**

---

## The Problem (proof from 17,029 real complaints)

- **Taxonomy chaos:** "seepage" across 16 categories. "leak" across 22 categories.
- **Cross-unit causality ignored:** 674 complaints reference other flats, treated as independent.
- **Systemic failures invisible:** Flat V1006 — 243 complaints, 15 categories, 2 years. Still getting individual tickets.
- **Category contamination:** FM category has "PLUMBER" and "ELECTRICAL" as subcategories.
- **Cost:** 25-30% misrouting = ₹1-10 lakh/month per developer.

---

## Core Architecture

### Tiered Reasoning

| Tier | Volume | When | LLM Calls | Cost |
|------|--------|------|-----------|------|
| Tier 1 | ~35% | Unambiguous ("flush not working") | 0 | ₹0 |
| Tier 2 | ~40% | Some ambiguity, not high-stakes | 1 | ₹2-10 |
| Tier 3 | ~25% | Ambiguous + high cost-of-error | 3-5 | ₹15-35 |
| **Blended** | **100%** | | **~1.4** | **₹1.5-3** |

**Interview line:** "I scale intelligence to the cost of error. Simple complaints bypass agents entirely. High-stakes get independent hypothesis evaluation."

### Cost-of-Error Formula

`adjusted_score = likelihood × cost_of_error_weight`

Plumbing: 0.70 × 1.0 = 0.70. Structural: 0.25 × 10.0 = 2.50. **Structural wins.** Not more likely — more expensive to miss.

**Interview line:** "A classifier optimizes uniform accuracy. ARIA encodes asymmetric costs and produces portfolio decisions."

### Independent Hypothesis Agents

Each agent gets isolated system prompt + isolated evidence. No anchoring bias across hypotheses.

**Why not one multi-hypothesis prompt:** Single forward pass anchors on first hypothesis generated. Plumbing is most common → model defaults to plumbing → structural gets systematically under-diagnosed. Separate calls = genuinely independent assessments.

### Dynamic Hypothesis Library

7 domains, 24 hypotheses. Config-driven (YAML + prompts). New property type = new YAML entry. Orchestration unchanged.

| Domain | Hypotheses | Highest Cost Weight |
|--------|-----------|-------------------|
| Water/Plumbing | pipe, structural, environmental, hvac_condensate | structural: 10.0× |
| Electrical | wiring, supply, equipment, safety_hazard | safety: 20.0× |
| Structural | settlement, waterproofing, installation, environmental | waterproofing: 10.0× |
| Carpentry | wear, structural_movement, installation | movement: 4.0× |
| HVAC | compressor, electrical, post_service | post_service: 3.0× |
| Lift | electrical, mechanical, maintenance | electrical: 5.0× |
| Safety | fire, gas, structural_hazard, security_equipment | fire/gas: 50.0× |

### Pattern Interpreter

Detects building-level patterns invisible to per-complaint routing. Vertical stack cluster → "not 4 plumbing problems, one structural problem." Feeds adjusted likelihoods to arbiter.

### Arbiter (Multi-Action Decisions)

Can produce: "Send plumber today + schedule structural assessment tomorrow." Uses cost-of-error weighting. Safety override: any safety hypothesis > 0.3 likelihood = always first.

### Pipeline Nodes vs Agents

| Agents (LLM reasoning) | Nodes (no LLM) |
|------------------------|-----------------|
| Hypothesis evaluators (2-4 per Tier 3) | Intake normalizer |
| Pattern interpreter | Domain classifier |
| Arbiter | Complexity assessor |
| Tier 2 single reasoner | Context assembler (65ms) |
| | Pattern state query (DBSCAN) |
| | Execution layer |
| | Audit logger |

**Interview line:** "I have 4-6 agents and 7 nodes. If you ask me to justify any agent, I can tell you what degrades if I remove it."

---

## Smoke Test Results (April 17, 2026)

### Forced Tier 3 — "Seepage from ceiling, same issue 3 months ago"
- **Tier 3** ✓ | **Domain: structural_civil** ✓
- **Action: send_structural_team + senior plumber in parallel** ✓
- **Recurrence signal used:** "prior fix was cosmetic patch" ✓
- **Adjusted score:** waterproofing 7.50 ✓
- **Reasoning:** Identified flat below terrace/roof slab, monsoon timing, prior fix failure. Multi-action with escalation trigger.
- Tokens: 6,233 | Confidence: medium

**This routing decision is the demo. The reasoning trace IS the product.**

---

## Three-Phase Product Vision

**Phase 1 (DONE): Routing Intelligence — "What's wrong?"**
- Tiered complaint processing, hypothesis agents, pattern detection, cost-weighted arbitration
- Proven on real data with correct multi-action routing

**Phase 2 (NEXT): Execution Intelligence — "Who fixes it, when, how?"**
- Vendor DB onboarding (their vendors, not ours — lock-in through optimization intelligence)
- Vendor matching (performance × diagnosis alignment × availability)
- Internal team vs. external vendor routing
- Schedule optimization (batch visits using spatial patterns — same data, two products)
- Vendor quote arbitration (agent-powered: compare quotes against ARIA's diagnosis)
- Acknowledgment → time slot → tracking → resolution feedback
- Vendor masking (Uber model)

**Phase 3 (FUTURE): Predictive Intelligence — "What's about to break?"**
- Proactive maintenance from 12 months of complaint + resolution data
- Cross-developer pattern intelligence (the ultimate moat)

**Interview line:** "Phase 1 is the brain. Phase 2 is the body. Phase 3 is foresight. Each compounds the data moat."

---

## Customer Acquisition — Layered GTM

| Layer | Customer | Role | Revenue | Timeline |
|-------|----------|------|---------|----------|
| 1 | Developers (pilot first) | Proof of concept | Free/pilot | Month 1-6 |
| 2 | FM Companies (Quess, CBRE, JLL) | Revenue engine (1 contract = 50-200 properties) | ₹1-5L/month | Month 6-18 |
| 3 | Gate Platforms (MyGate, NoBrokerHood) | Scale lever (1 integration = 25,000+ societies) | ₹5-15/complaint | Month 12-24 |
| 4 | RWAs | Indirect beneficiaries | Never direct buyer | Ongoing |

---

## The Moat

1. **Cross-developer intelligence:** Patterns no individual developer sees
2. **Data network effect:** Every complaint makes future decisions better
3. **Feedback loop:** Resolution → recurrence detection → recalibration. Only closes with both routing + resolution data.
4. **Hypothesis library depth:** 24 prompts, 7 domains, calibrated against real data

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Orchestration | LangGraph | Stateful DAG, conditional branching |
| Database | PostgreSQL 16 | Joins for adjacency, JSONB flexibility |
| Pattern state | Redis 7 | Sliding windows, DBSCAN |
| LLM | Anthropic API | Haiku (T2), Sonnet (hypotheses), Opus (arbiter) |
| API | FastAPI | Async, auto-docs |

---

## Build Status

| Day | Task | Status |
|-----|------|--------|
| 1-2 | PostgreSQL + ETL (17,029 complaints) + adjacency | ✅ |
| 3 | Domain classifier + complexity assessor | ✅ |
| 4 | Context assembler (65ms parallel queries) | ✅ |
| 5 | Hypothesis library + 24 prompts | ✅ |
| 6 | Redis pattern state + arbiter | ✅ |
| 7 | LangGraph full graph | ✅ |
| 8 | FastAPI (5 endpoints) | ✅ |
| 9 | Eval framework | ✅ |
| 10 | Demo UI + smoke test | In progress |

**Bug fixes applied:** Pattern interpreter wired into graph, pipe_failure evidence fixed, README replaced, API error handling added, pattern interpreter generalized. Total Claude Code cost: $0.57.

---

## Known Issues

| Issue | Status | Owner |
|-------|--------|-------|
| Tier 1 rules too narrow | Cursor fixing | Cursor |
| Ground truth labels (100) | TODO | Kartheek |
| Demo UI | TODO | Cursor |
| Insights report script | TODO | Cursor |
| 50-complaint eval | TODO | Terminal ($3-5) |

---

## Tool Routing

| Task | Tool | Cost |
|------|------|------|
| Code writing | Cursor | Subscription (₹0 API) |
| Architecture + strategy | Claude.ai | Pro subscription (₹0 API) |
| High-IQ debugging | Claude Code | ≤$1 per session |
| Eval runs | Terminal | API balance |

**API balance: ~$4.43 remaining** (started $5.04, spent $0.57 Claude Code + $0.04 eval)

---

*GitHub: https://github.com/freezeindigo/Resolv*
*Last updated: April 17, 2026*
