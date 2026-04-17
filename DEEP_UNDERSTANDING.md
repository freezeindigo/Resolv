# ARIA — Deep Understanding Guide
## Think through these, don't memorize them

---

## 1. THE FUNDAMENTAL QUESTION

**"Why is this hard? Can't you just use an LLM with a good prompt?"**

Think about it this way. Open Claude right now and paste: "Water leaking from ceiling in Flat 1803, Tower T6, Godrej Woods. What should the FM team do?"

Claude will say something like: "This could be plumbing or structural. Send a plumber to investigate."

That's a reasonable answer. But it's USELESS for routing because:

- It doesn't know that Flat 1903 above had a seepage complaint 2 weeks ago
- It doesn't know that 4 other flats in this vertical stack reported similar issues
- It doesn't know that a plumber was sent 3 months ago and the problem came back
- It doesn't know that structural misclassification costs 10x more than plumbing misclassification
- It can't dispatch a plumber AND a structural team in parallel
- It has no memory — tomorrow, the same complaint gets the same generic answer

ARIA is not "Claude with a good prompt." ARIA is a SYSTEM that:
- Retrieves building-level evidence before reasoning
- Evaluates competing hypotheses independently
- Weighs decisions by cost of error, not just likelihood
- Detects spatial patterns across complaints
- Produces multi-action operational plans
- Learns from resolution outcomes

The LLM is the reasoning engine inside the system. The system is what creates value.

---

## 2. TRACE THE DATA FLOW

Don't memorize the architecture. Trace a complaint through it mentally.

**Complaint: "Water leaking from ceiling in master bedroom"**

Ask yourself at each step: what DECISION is being made, and what INFORMATION is needed?

**Step: Domain classification**
- Decision: Which domain does this belong to?
- Information needed: Just the complaint text
- Method: Keywords "water" + "leaking" + "ceiling" → water_plumbing
- Why no LLM: Keywords are sufficient. "Water leaking" is never electrical.

**Step: Complexity assessment**
- Decision: How much reasoning does this need?
- Information needed: The text + what we know about this flat/building
- Method: "Leaking" is ambiguous (could be pipe or seepage). Check: does this flat have history? Does this building have clusters?
- Key insight: We CAN'T fully assess complexity without checking history. That's why Tier 2 can upgrade to Tier 3 AFTER context assembly.

**Step: Context assembly**
- Decision: None — this is pure retrieval
- Information needed: flat_id, tower, site_name
- Method: 3 parallel SQL queries. ~65ms.
- Why no LLM: These are database lookups. The schema is known. The queries are deterministic.

**Step: Pattern state query**
- Decision: Is there a spatial/temporal cluster?
- Information needed: Building complaints in sliding window
- Method: DBSCAN over (building, floor, domain, timestamp)
- Key insight: If a vertical stack pattern is found and we're in Tier 2, UPGRADE to Tier 3. The system adapts mid-pipeline.

**Step: Hypothesis evaluation (Tier 3)**
- Decision: How likely is each possible root cause?
- Information needed: DIFFERENT evidence for EACH hypothesis
- Method: Parallel independent LLM calls, each with isolated evidence
- WHY INDEPENDENT: Ask yourself — if you gave one doctor ALL the test results and asked "what's wrong?", they'd anchor on their first intuition. If you gave three specialists each ONLY their relevant tests and asked each "is this your specialty?", you'd get more honest assessments. Same principle.

**Step: Pattern interpretation**
- Decision: Does the spatial pattern change the diagnosis?
- Information needed: Cluster data + hypothesis scores
- WHY THIS IS LLM, NOT RULES: "4 seepage complaints in vertical stack" COULD mean structural failure OR could mean 4 independent plumbing issues that happen to be in the same stack. Interpreting which requires judgment — not just counting.

**Step: Arbitration**
- Decision: What's the final routing plan?
- Information needed: ALL hypothesis scores + pattern interpretation + cost weights
- Key formula: adjusted_score = likelihood × cost_of_error_weight
- WHY MULTI-ACTION: Real FM decisions aren't binary. "Send plumber today AND schedule structural assessment for tomorrow" is often the RIGHT answer when you're uncertain.

---

## 3. UNDERSTAND THE COST-OF-ERROR MATH

This is the core insight. Don't memorize it — understand WHY.

**Scenario A: Classifier approach**
- "Water from ceiling" → 70% plumbing, 25% structural, 5% other
- Classifier picks: Plumbing (highest probability)
- Right 70% of the time. Wrong 30% of the time.

**But what does "wrong" cost?**
- Plumbing classified as structural: Send structural team (₹5,000) when plumber (₹500) would've worked. Overspend: ₹4,500.
- Structural classified as plumbing: Send plumber (₹500), doesn't fix it. Send plumber again (₹500). Finally send structural team (₹5,000). Total: ₹6,000 + weeks of delay + angry resident. Actual cost: ₹6,000 + reputation damage.

The second error is 10x worse than the first. A classifier that treats both errors equally is optimizing the wrong objective.

**Scenario B: ARIA approach**
- Plumbing: 0.70 × 1.0 weight = 0.70
- Structural: 0.25 × 10.0 weight = 2.50
- Structural wins despite lower likelihood

This means ARIA will sometimes send a structural team when it turns out to be plumbing. That's a ₹4,500 overspend. But it CATCHES the structural cases that a classifier misses — saving ₹6,000+ per catch. Net expected value is positive.

**The key question to ask yourself:** "What happens when I'm wrong in EACH direction?" If the costs are symmetric, use a classifier. If they're asymmetric, use cost-weighted decision-making.

---

## 4. WHY EACH COMPONENT EXISTS (THE DELETION TEST)

For every component, ask: "What breaks if I remove this?"

**Remove Tier 1 (rule-based routing)?**
→ Every complaint hits the LLM. Cost goes from ₹1.5-3 blended to ₹8-15 for ALL complaints. 35% of complaints are paying for reasoning they don't need. "Flush not working" doesn't need a hypothesis debate.

**Remove Tier 3 (multi-agent)?**
→ Ambiguous high-stakes complaints get one LLM call instead of independent hypothesis evaluation. Anchoring bias means structural seepage gets systematically under-diagnosed. The 25% of complaints that cost the most to get wrong are now being decided with the least rigor.

**Remove the pattern interpreter?**
→ The arbiter makes decisions complaint-by-complaint without seeing building-level patterns. 4 seepage complaints in a vertical stack get routed as 4 independent plumbing calls instead of one structural intervention. You lose the most powerful diagnostic signal in the system.

**Remove cost-of-error weighting?**
→ The arbiter picks the most likely hypothesis, not the most important to get right. Structural seepage (25% likely, ₹5,000-50,000 to miss) loses to plumbing (70% likely, ₹500 to miss) every time. You're optimizing accuracy instead of business outcomes.

**Remove context assembly?**
→ Every complaint is evaluated in isolation. "Water from ceiling" with no knowledge of adjacent flats, no building history, no recurrence data. The LLM is guessing, not reasoning. You're back to "Claude with a good prompt."

**Remove the hypothesis library (hardcode 3 agents)?**
→ Works for water domain. Fails for electrical (need different hypotheses). Fails for HVAC. Fails for carpentry. Can't onboard new property types without code changes. You lose the extensibility that makes this a platform.

---

## 5. THE PLATFORM QUESTION

**"This is impressive for Godrej. But is it a startup or an internal tool?"**

Trace the logic:

**Why it's not just an internal tool:**
- Every residential developer in India has the same problem (same misrouting rates, same taxonomy chaos)
- Every FM company services multiple developers (one FM contract = 50-200 properties)
- MyGate/NoBrokerHood have 50,000+ societies with zero routing intelligence
- The hypothesis library is config-driven — new domains are YAML + prompts, not code rewrites

**Why the moat compounds:**
- Month 1: ARIA is uncertain on ambiguous complaints (limited history)
- Month 6: Knows which towers have monsoon seepage patterns
- Month 12: Cross-developer intelligence — "buildings constructed 2018-2021 with Method X have 3.2x seepage recurrence"
- Year 2: Predicts complaints before they're filed, schedules proactive inspections

**Why competitors can't catch up:**
- The feedback loop (resolution outcome → recurrence detection → hypothesis recalibration) only closes if you own BOTH routing AND resolution data
- MyGate/NoBrokerHood have routing data but no resolution data
- FM companies have resolution data but no cross-developer routing data
- ARIA connects both

**The data network effect:**
- Every new developer adds building-specific episodic memory
- Every new FM company adds vendor performance data
- Every resolved complaint makes the next decision more accurate
- A competitor starting today needs the same years of resolution data to reach the same accuracy

---

## 6. HONEST WEAKNESSES (KNOW THESE BEFORE THEY'RE ASKED)

**"What if the Godrej data isn't representative?"**
Honest answer: It might not be. Godrej builds mid-to-premium residential. A budget developer's complaint patterns might differ. But the ARCHITECTURE is domain-agnostic — the hypothesis library adapts. What changes is the training data and prompt calibration, not the system design.

**"You haven't proven accuracy improvement yet."**
Honest answer: Correct. We have a working prototype that produces reasonable routing decisions (the seepage test proves this). We need a 100-complaint eval against hand-labeled ground truth to claim specific accuracy numbers. The human baseline (25-30% misrouting) is well-documented from operational data.

**"Multi-agent is expensive. Why not fine-tune a model?"**
Honest answer: Fine-tuning gives you a better classifier. We explicitly chose NOT to build a classifier — we want cost-weighted multi-hypothesis reasoning. Also, fine-tuning requires consistent labeled data. Our data has 16 categories for "seepage." The labels are the problem, not the solution. Fine-tuning on bad labels produces a model that's confidently wrong.

**"What if LLM costs drop and this whole architecture becomes over-engineered?"**
Honest answer: If LLM costs drop 10x, the Tier 1 rule engine becomes unnecessary — you'd just LLM everything. But the hypothesis isolation, cost-of-error weighting, and pattern detection remain valuable regardless of cost. Those are architectural decisions about decision quality, not cost optimization.

**"You're a PM, not an ML engineer. Can you actually build this?"**
Honest answer: I designed the architecture. My co-founder (Arunabh) and Claude Code built the implementation. I own the problem deeply (I run FM operations at Godrej), I understand the technical architecture well enough to debug it and make design decisions, and I have enough ML background (IIT Bombay M.Tech, CS-229, hands-on with LangGraph/ChromaDB) to be a credible technical co-founder at the pre-seed stage. The gap is in production ML engineering — which is what the ₹2 Cr raise is for.

---

## 7. THE 3-MINUTE VERSION (FOR WHEN YOU HAVE LIMITED TIME)

"I manage facility operations for 40,000 residents across 23 Godrej sites. Every month, 660 complaints come in. 25-30% get routed to the wrong vendor — wrong plumber for a structural problem, wrong electrician for an equipment issue.

I analyzed 17,000 real complaints and found that the same physical symptom — 'water leaking from ceiling' — gets classified into 16 different categories depending on who logs it. 674 complaints reference other flats but get treated as isolated incidents. The taxonomy itself is broken.

So I built ARIA — an agentic routing system that doesn't classify complaints, it reasons about them. For simple complaints like 'flush not working,' it routes instantly with rules — zero AI cost. For ambiguous complaints like 'seepage from ceiling,' it spawns independent hypothesis agents — one evaluates plumbing evidence, another evaluates structural evidence, another evaluates environmental factors — each with isolated context to prevent anchoring bias. Then an arbiter weighs the results by cost of error, not just likelihood, because missing a structural problem costs 10x more than missing a plumbing problem.

The system runs on real data. I just tested it — 'seepage from ceiling, same issue reported 3 months ago' — and it correctly identified waterproofing failure as the primary cause, dispatched a structural team with a plumber in parallel for immediate mitigation, and flagged the recurrence as evidence that the prior fix was inadequate.

The platform scales because the hypothesis library is configuration, not code. New property types — commercial, industrial — add YAML files and prompt templates. The orchestration doesn't change. And every resolved complaint makes future decisions more accurate. After 12 months, a competitor starting from scratch needs 12 months of resolution data we already have."

---

## 8. READ THESE FILES IN THIS ORDER

1. This document (you're reading it)
2. BUILD_JOURNAL.md — full architecture with interview lines
3. ARIA.md in the repo — the living context document
4. src/pipeline/resolv_graph.py — trace the actual code flow
5. src/config/hypothesis_library.yaml — see all 7 domains and 24 hypotheses
6. src/agents/prompts/structural_seepage.md — the most important prompt
7. src/agents/prompts/arbiter.md — how decisions get made

Don't read everything. Read enough to be able to trace a complaint through the system in your head without looking at the code. That's when you're ready.

---

*When you can explain WHY each component exists (not WHAT it does), you're ready for any interview.*
