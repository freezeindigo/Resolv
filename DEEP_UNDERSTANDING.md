# Resolv / ARIA — Deep Understanding Guide
## Think through these, don't memorize them
### v2 — April 18, 2026

---

## 0. SYSTEM ARCHITECTURE

```
                         ┌─────────────────────┐
                         │  Complaint arrives   │ ← MyGate / CRM / WhatsApp
                         └─────────┬───────────┘
                                   ▼
                         ┌─────────────────────┐
                         │  Intake normalizer   │  No LLM
                         └─────────┬───────────┘
                                   ▼
                         ┌─────────────────────┐
                         │  Domain classifier   │  Keywords → 8 domains
                         │                     │  water, electrical, structural,
                         │                     │  carpentry, HVAC, lift, safety, pest
                         └─────────┬───────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Complexity assessor  │  Assigns Tier 1 / 2 / 3
                         └─────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              │
          ┌──────────────┐  ┌───────────────┐     │
          │  TIER 1      │  │ Context       │     │
          │  Rule route  │  │ assembler     │     │
          │  0 LLM, 0ms  │  │ ~65ms         │     │
          │  ~35% volume │  └───────┬───────┘     │
          └──────┬───────┘          │              │
                 │         ┌────────┤  ┌───────────┘
                 │         │PostgreSQL│  ChromaDB RAG
                 │         │flat hist │  audit reports
                 │         │adjacent  │  society MoM
                 │         │patterns  │  daily reports
                 │         └────────┬─┘  building specs
                 │                  ▼    vendor history
                 │         ┌───────────────┐
                 │         │ Pattern state  │
                 │         │ Redis DBSCAN   │  Can upgrade T2→T3
                 │         └───────┬───────┘
                 │                 │
                 │         ┌───────┴──────────────┐
                 │         ▼                      ▼
                 │  ┌──────────────┐    ┌──────────────────┐
                 │  │  TIER 2      │    │  TIER 3           │
                 │  │  Single agent│    │  Hypothesis scoring│
                 │  │  1 LLM call  │    │  YAML triggers     │
                 │  │  Groq 70B    │    │  → top 2-4         │
                 │  │  ~40% volume │    │  ~25% volume       │
                 │  └──────┬───────┘    └────────┬──────────┘
                 │         │                     ▼
                 │         │           ┌──────────────────────┐
                 │         │           │ Parallel hypothesis   │
                 │         │           │ agents (isolated)     │
                 │         │           │ Groq 70B per agent    │
                 │         │           │ [pipe] [struct] [env] │
                 │         │           └────────┬─────────────┘
                 │         │                    ▼
                 │         │           ┌──────────────────┐
                 │         │           │Pattern interpreter│
                 │         │           │Groq 70B           │
                 │         │           └────────┬──────────┘
                 │         │                    ▼
                 │         │           ┌──────────────────┐
                 │         │           │    Arbiter        │
                 │         │           │ Claude Sonnet     │
                 │         │           │ score = likelihood│
                 │         │           │   × cost weight   │
                 │         │           └────────┬──────────┘
                 │         │                    │
                 │         └────────┬───────────┘
                 │                  ▼
                 │         ┌───────────────┐
                 │         │ LLM-as-Judge  │
                 │         │ Groq 8B       │
                 │         │ approve/flag/ │
                 │         │ override      │
                 │         └───────┬───────┘
                 │                 │
                 └────────┬────────┘
                          ▼
                 ┌───────────────────┐
                 │Ownership classifr │  FM / Project
                 │hypothesis + text  │
                 └────────┬──────────┘
                          ▼
                 ┌───────────────────┐
                 │ Execution + Audit │
                 └────────┬──────────┘
                          ▼
              ┌──────────────────────────┐
              │    ROUTING DECISION      │
              │ ownership + action +     │
              │ reasoning + judge verdict│
              └──────────────────────────┘
                          │
                    ▼ feedback loop
               Resolution outcome
               → updates complaint DB
               → improves future routing
```

### 14 nodes total
- **7 use zero LLM**: intake, domain classifier, complexity assessor, context assembler, pattern state, ownership classifier, execution
- **5-6 use LLM**: Tier 2 reasoning, hypothesis agents, pattern interpreter, arbiter, judge
- **Only 1 uses Claude API**: the arbiter, for ~25% of complaints

### Model routing

| Role | Model | Provider | Cost |
|------|-------|----------|------|
| Domain classifier | keyword rules | — | ₹0 |
| Complexity assessor | rules | — | ₹0 |
| Context assembler | SQL + RAG | — | ₹0 |
| Pattern state | DBSCAN | — | ₹0 |
| RAG embeddings | MiniLM-L6-v2 | Local CPU | ₹0 |
| Tier 2 reasoning | Llama 3.1 70B | Groq free | ₹0 |
| Hypothesis agents | Llama 3.1 70B | Groq free | ₹0 |
| Pattern interpreter | Llama 3.1 70B | Groq free | ₹0 |
| **Arbiter** | **Claude Sonnet** | **Anthropic** | **~₹1-2** |
| Judge | Llama 3.1 8B | Groq free | ₹0 |
| Ownership classifier | rules + hypothesis | — | ₹0 |

Blended cost: ~₹0.25 per complaint

---

## 1. THE FUNDAMENTAL QUESTION

**"Why is this hard? Can't you just use an LLM with a good prompt?"**

Open Claude and paste: "Water leaking from ceiling in Flat 1803, Tower T6. What should the FM team do?"

Claude says: "This could be plumbing or structural. Send a plumber to investigate."

That's USELESS for routing because:

- It doesn't know Flat 1903 above had a seepage complaint 2 weeks ago
- It doesn't know the Q3 audit flagged waterproofing degradation on this tower
- It doesn't know the RWA rejected the waterproofing tender in March
- It doesn't know a plumber was sent 3 months ago and the problem came back
- It doesn't know structural misclassification costs 10x more
- It doesn't know whether this is FM or Project responsibility
- It can't dispatch both a plumber AND structural team in parallel

ARIA is a SYSTEM that:
- Retrieves building-level evidence before reasoning (PostgreSQL + RAG)
- Evaluates competing hypotheses independently (isolated agents)
- Weighs decisions by cost of error, not just likelihood (arbiter)
- Detects spatial patterns across complaints (DBSCAN)
- Determines FM vs Project ownership from hypothesis results
- Validates every decision through an LLM-as-judge guardrail
- Produces multi-action plans with escalation triggers

---

## 2. TRACE THE DATA FLOW

**Complaint: "Water leaking from ceiling, same issue 3 months ago"**

At each step ask: what DECISION is being made, what INFORMATION is needed?

**Domain classification** → Keywords "water" + "ceiling" → structural_civil. No LLM.

**Complexity assessment** → "Ceiling" + "same issue" (recurrence) → ambiguous + high stakes → Tier 3.

**Context assembly** → Parallel: PostgreSQL (flat history, adjacent flats, patterns) + ChromaDB RAG (audit excerpts, MoM decisions, building specs). ~65ms, no LLM.

**Pattern state** → DBSCAN over (building, floor, domain, timestamp). Can upgrade T2→T3.

**Hypothesis scoring** → YAML triggers checked against text + context. pipe_failure + structural_seepage + environmental spawn. hvac_condensate filtered (no AC signal). 3 agents, not all 4.

**Hypothesis evaluation** → 3 parallel LLM calls (Groq 70B), each with ISOLATED evidence. pipe_failure sees consumption data. structural_seepage sees audit reports + building specs. No anchoring.

**Pattern interpretation** → Groq 70B interprets cluster data + hypothesis scores. "4 seepage in vertical stack = structural, not 4 independent plumbing issues."

**Arbitration** → Claude Sonnet. adjusted = likelihood × cost_weight. Plumbing: 0.70 × 1.0 = 0.70. Structural: 0.25 × 10.0 = 2.50. Structural wins. Multi-action: structural team + plumber in parallel.

**Judge** → Groq 8B validates. "Routing is sound. Dual dispatch appropriate. P2 proportionate."

**Ownership** → Structural seepage won → Project (waterproofing = construction defect).

---

## 3. COST-OF-ERROR MATH

A classifier picks highest probability: plumbing (70%). Right 70% of the time.

But costs are asymmetric:
- Plumbing misclassified as structural: overspend ₹4,500
- Structural misclassified as plumbing: ₹6,000 + weeks delay + reputation damage (10x worse)

ARIA's formula: adjusted_score = likelihood × cost_of_error_weight
- Plumbing: 0.70 × 1.0 = 0.70
- Structural: 0.25 × 10.0 = 2.50
- Structural wins despite lower likelihood

"What happens when I'm wrong in EACH direction?" Symmetric → classifier. Asymmetric → cost-weighted.

---

## 4. FM vs PROJECT OWNERSHIP

The real operational pain. Not "what domain" but "who owns it."

**Today:** Manual decision → wrong 30% → teams fight → 3-5 days delay.
**With Resolv:** Evidence-based ownership from hypothesis + text signals → reasoning trace settles disputes.

**From 17,098 real complaints:**
- Security: 96% FM | Elevator: 93% FM | HK: 93% FM
- Plumbing: 87% FM | Electrical: 93% FM
- Carpentry: 58% Project | Civil work: 53% Project | VDP: 74% Project

**Ambiguous cases resolved by hypothesis winners:**
- structural_seepage wins → Project
- pipe_failure wins → FM
- "since possession" text signal → Project

**The reasoning trace becomes the arbiter between teams.** Not a person's opinion. Evidence.

---

## 5. RAG — OPERATIONAL INTELLIGENCE

Complaint text is 10% of the signal. The other 90% is in operational documents.

| Source | What it adds |
|--------|-------------|
| Audit reports | "Q3 audit flagged waterproofing degradation on T6 west wall" |
| Society MoM | "RWA rejected waterproofing tender — explains recurring seepage" |
| Daily reports | "Plumber visited yesterday, said not pipe issue" |
| Building specs | "Cantilever slab, waterproofing warranty valid until 2030" |
| Vendor history | "ABC Plumbing 78% first-time-fix. BuildRight Structural 87%" |
| AMC contracts | "DG AMC expired — urgent renewal before next outage" |

RAG is hypothesis-aware: structural agent sees audit + specs. Pipe agent sees consumption + vendor data. Not the same documents dumped into every prompt.

---

## 6. LLM-AS-JUDGE

Every Tier 2/3 decision passes through a judge. Checks:
- Domain mismatch (door lock → structural team is WRONG)
- Ownership error (routine plumbing → Project is SUSPICIOUS)
- Priority inflation (paint peeling → P1 is WRONG)
- Evidence mismatch (cites "vertical stack" when no adjacent complaints exist)

Verdicts: approve / flag (→ human review) / override.

"The system never autonomously routes a complaint it's uncertain about."

---

## 7. THE DELETION TEST

**Remove Tier 1?** → Cost goes from ₹0.25 to ₹5+ for ALL complaints.
**Remove Tier 3?** → Anchoring bias, structural under-diagnosed.
**Remove pattern interpreter?** → 4 seepage = 4 plumbing calls, not 1 structural.
**Remove cost-of-error?** → Plumbing always wins. Wrong objective.
**Remove RAG?** → Hypothesis agents guess instead of reason.
**Remove judge?** → Hallucinated decisions go to execution.
**Remove ownership?** → FM vs Project blame game continues.

---

## 8. DEMO RESULTS — 8 COMPLAINTS (April 18, 2026)

Full pipeline, Groq + Claude arbiter. **All 8 correct.**

| # | Complaint | Tier | Domain | Owner | Action | Key signal |
|---|-----------|------|--------|-------|--------|------------|
| 1 | Seepage, recurring | 3 | structural | **Project** | structural_team + plumber | Cost-of-error: 7.50 vs 0.70 |
| 2 | Color on car in parking | 3 | **safety_security** | FM | security_team | Domain fix: was common_area |
| 3 | Door lock since possession | 2 | carpentry | **Project** | carpenter | Text signal: "since possession" |
| 4 | Flush not working | 1 | plumbing | FM | plumber | Rule-based, zero LLM |
| 5 | Burning smell switchboard | 3 | electrical | FM | **emergency** | Safety: 20x cost multiplier |
| 6 | Lift stuck with people | 1 | lift | FM | **emergency** | Rule-based, zero LLM |
| 7 | Wall cracks since monsoon | 3 | structural | **Project** | structural_team | Cost-weight breaks 50/50 tie |
| 8 | AC post-service recurrence | 1 | hvac | FM | hvac_tech | Rule-based, zero LLM |

- Tier 1: 3 (37.5%) — zero LLM
- Tier 2: 1 (12.5%) — Groq free
- Tier 3: 4 (50%) — Groq + Claude arbiter
- All 8 ownership correct. All 8 actions from constrained vocabulary.

---

## 9. PLATFORM + MOAT

**Not just an internal tool:** Every developer, every FM company, every gate platform has the same problem. Hypothesis library is config — new domains are YAML, not code.

**Moat compounds:** Month 6: monsoon patterns. Month 12: cross-developer intelligence. Year 2: predictive maintenance.

**Competitors can't catch up:** Need BOTH routing AND resolution data. MyGate has routing only. FM companies have resolution only. Resolv connects both.

---

## 10. THREE PHASES

**Phase 1 (DONE):** Routing Intelligence — what's wrong, who owns it
**Phase 2 (NEXT):** Execution Intelligence — who fixes it, when, how (vendor matching, scheduling)
**Phase 3 (FUTURE):** Predictive Intelligence — what's about to break

---

## 11. HONEST WEAKNESSES

**Accuracy at scale?** — 8/8 demo, need 100-complaint eval against ground truth.
**Why not fine-tune?** — Labels are broken (16 categories for "seepage"). Fine-tuning on bad labels = confidently wrong. Also, we want reasoning, not classification.
**You're a PM?** — Designed architecture, debugged pipeline, built cost-of-error from operational data. Production ML engineering is the gap — what the raise is for.

---

## 12. THE 3-MINUTE VERSION

"I manage facility operations for 40,000 residents across 23 sites. 660 complaints/month. 25-30% misrouted. FM and Project teams fight over ownership for 3-5 days.

I analyzed 17,000 complaints. Found 'water from ceiling' in 16 categories. 674 cross-flat references treated as isolated. The taxonomy is broken and the ownership decision is wrong 30% of the time.

I built Resolv — an agentic system that reasons about complaints. Simple ones route instantly with rules (zero AI). Ambiguous ones spawn independent hypothesis agents, each with isolated evidence. An arbiter weighs by cost of error. A judge validates every decision. And the ownership classifier determines FM or Project from the winning hypothesis.

8 diverse complaints, all correct. Seepage correctly routed to Project. Parking vandalism to FM security. Door lock 'since possession' to Project. Burning smell triggered immediate emergency. Cost: ₹0.25 per complaint blended."

---

*When you can explain WHY each component exists (not WHAT it does), you're ready.*
*v2 — April 18, 2026*
