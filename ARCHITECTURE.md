# ARCHITECTURE — Resolv.AI

## System Overview

Resolv is a LangGraph-orchestrated pipeline that processes complaints through a tiered reasoning system. Complaints enter the pipeline, get assessed for complexity and domain, and are routed through progressively deeper reasoning based on their characteristics.

## Pipeline Nodes (Deterministic, No LLM)

### 1. Intake Normalizer
**Purpose:** Map any source schema to unified complaint object.

**Input:** Raw complaint from MyGate / NoBrokerHood / CRM webhook / direct API.

**Output:**
```python
{
    "complaint_id": str,
    "source": str,              # "mygate" | "nobroker" | "direct" | etc.
    "raw_text": str,
    "language_detected": str,   # "en" | "hi" | "hinglish"
    "flat_id": str,
    "building_id": str,
    "tower": str,
    "floor": int,
    "timestamp": datetime,
    "priority_requested": str,  # Optional, if source provides
}
```

**Implementation:** Pure function. Language detection via langdetect or fasttext. No LLM.

### 2. Domain Classifier
**Purpose:** Classify complaint into one of 8+ domains for hypothesis library selection.

**Domains:**
- `water_plumbing`
- `electrical`
- `structural_civil`
- `carpentry`
- `hvac`
- `lift_elevator`
- `common_area`
- `safety_security`
- `pest_hygiene`
- `other` (fallback)

**Implementation:** Keyword-based rules first, with fallback to small LLM (distilled classifier or Groq-hosted small model) for unmatched cases. Store rules in `src/config/domain_rules.yaml`.

### 3. Complexity Assessor
**Purpose:** Assign Tier 1/2/3 based on ambiguity and cost-of-error.

**Tier 1 triggers (deterministic routing, no LLM):**
- Exact high-confidence keyword match within domain
- Example: "flush not working" → Plumbing, P2, send plumber
- Build from top 50 unambiguous patterns in historical complaint data

**Tier 3 triggers (multi-agent deliberation):**
- Multi-domain keyword overlap (e.g., both "leak" and "wall crack")
- Flat has >5 prior complaints in last 90 days
- Building has >3 similar-category complaints in last 30 days
- Active cluster detected in same vertical stack
- Domain is high-cost (structural, safety, safety-security)

**Tier 2 (default):** Everything else. Single reasoning agent.

### 4. Context Assembler
**Purpose:** Parallel async retrieval of all context needed for reasoning.

**Three parallel queries:**

1. **Flat context:**
```sql
SELECT * FROM complaints 
WHERE flat_id = ? AND created_date > NOW() - INTERVAL '365 days'
ORDER BY created_date DESC LIMIT 20
```

2. **Adjacent context:**
```sql
-- Uses flat_adjacency table
SELECT c.* FROM complaints c
JOIN flat_adjacency a ON c.flat_id IN (a.above_flat, a.below_flat, a.lateral_flats)
WHERE a.flat_id = ? AND c.created_date > NOW() - INTERVAL '90 days'
ORDER BY c.created_date DESC LIMIT 10
```

3. **Building context:**
```sql
SELECT category, floor, COUNT(*) as count
FROM complaints
WHERE tower = ? AND created_date > NOW() - INTERVAL '90 days'
GROUP BY category, floor
```

**Returns structured context in ~200ms. No LLM.**

### 5. Pattern State Query
**Purpose:** Check for active clusters in real-time.

**Implementation:** Redis sliding-window aggregation. On each new complaint ingestion, update the cluster state. Query returns:

```python
{
    "active_clusters": [
        {
            "cluster_id": str,
            "complaint_count": int,
            "spatial_pattern": "vertical_stack" | "floor_range" | "scattered",
            "temporal_pattern": "last_24h" | "last_7d",
            "dominant_category": str,
            "confidence": float,  # 0.0 - 1.0
        }
    ]
}
```

**Clustering algorithm:** DBSCAN over (building_id, floor, category, timestamp) features. Deterministic clustering. An LLM will interpret the cluster's meaning separately (see Pattern Interpretation Agent).

### 6. Execution Layer
**Purpose:** Dispatch vendor, notify, schedule, monitor TAT.

**Operations:**
- Vendor dispatch API call (stub for MVP, real integration later)
- Resident notification (SMS/WhatsApp/app push — stub for MVP)
- Ticket creation in source system (webhook back to MyGate etc.)
- TAT monitoring job

**Implementation:** Pure API orchestration. No LLM.

### 7. Audit Logger
**Purpose:** Persist complete reasoning trace for post-hoc analysis.

**Stores:**
- Full complaint object
- Tier assigned and why
- Domain classified
- Context retrieved (summaries)
- Pattern signals
- Each agent's input, output, tokens, latency, cost
- Arbiter decision and reasoning
- Final routing action
- Later: resolution outcome (feedback loop)

**Implementation:** PostgreSQL write. No LLM.

## Reasoning Agents (LLM-Powered)

See `HYPOTHESIS_LIBRARY.md` for the domain → hypothesis agent configurations.

### System-Level Agents

#### Pattern Interpretation Agent
**Triggers when:** Pattern State Query returns active clusters for the complaint's location.

**System prompt template:** `src/agents/prompts/pattern_interpreter.md`

**Input:** Cluster data + hypothesis scores (from parallel hypothesis agents).

**Output:** Interpretation of whether the cluster corroborates or contradicts each hypothesis, and whether it suggests systemic intervention beyond per-complaint routing.

**Why it's an agent:** The meaning of a cluster depends on its relationship to the current hypotheses. "6 seepage complaints in vertical stack" could be coincidence, plumbing cascade, or systemic waterproofing failure — a deterministic rule cannot capture this.

#### Arbiter Agent
**Triggers when:** Tier 3 complaints, multiple hypothesis agents have returned, or pattern signals conflict with hypothesis scores.

**System prompt template:** `src/agents/prompts/arbiter.md`

**Input:** All hypothesis scores, pattern interpretation, cost-of-error weights, vendor availability, SLA constraints.

**Output:** Final routing decision (may be multi-action: "plumber today + structural assessment tomorrow").

**Why it's an agent:** Integrates asymmetric cost-of-error with competing signals. Not a simple threshold — requires reasoning about tradeoffs.

### Domain-Specific Hypothesis Agents

See `HYPOTHESIS_LIBRARY.md` for complete list.

Each hypothesis agent:
- Has an isolated system prompt focused on ONE hypothesis
- Receives ONLY evidence relevant to its hypothesis (via evidence_filter)
- Returns: likelihood (0.0-1.0), evidence summary, confidence level

**Why isolated calls, not one multi-hypothesis prompt:** A single LLM evaluating hypotheses sequentially anchors on whichever it generates first. Shared forward pass creates latent bias. Separate calls with isolated evidence produce measurably more independent assessments.

## Orchestration — LangGraph Topology

```python
from langgraph.graph import StateGraph, END

class ResolvState(TypedDict):
    complaint: ComplaintObject
    domain: str
    tier: int
    context: ContextPackage
    pattern_signal: Optional[PatternSignal]
    hypotheses: List[HypothesisResult]
    pattern_interpretation: Optional[str]
    routing_decision: Optional[RoutingDecision]

graph = StateGraph(ResolvState)

# Nodes
graph.add_node("intake", intake_normalizer)
graph.add_node("classify_domain", domain_classifier)
graph.add_node("assess_complexity", complexity_assessor)
graph.add_node("assemble_context", context_assembler)
graph.add_node("query_patterns", pattern_state_query)
graph.add_node("spawn_hypotheses", spawn_hypothesis_agents)
graph.add_node("single_reasoning", tier2_reasoning_agent)
graph.add_node("rule_route", tier1_rule_router)
graph.add_node("interpret_patterns", pattern_interpretation_agent)
graph.add_node("arbitrate", arbiter_agent)
graph.add_node("execute", execution_layer)
graph.add_node("audit", audit_logger)

# Conditional edges based on tier
def route_by_tier(state):
    if state["tier"] == 1:
        return "rule_route"
    elif state["tier"] == 2:
        return "single_reasoning"
    else:  # Tier 3
        return "spawn_hypotheses"

graph.add_edge("intake", "classify_domain")
graph.add_edge("classify_domain", "assess_complexity")
graph.add_conditional_edges(
    "assess_complexity",
    lambda s: "rule_route" if s["tier"] == 1 else "assemble_context"
)
graph.add_edge("assemble_context", "query_patterns")
graph.add_conditional_edges("query_patterns", route_by_tier)
graph.add_edge("rule_route",        "execute")
graph.add_edge("single_reasoning",  "judge")    # Tier 2 validated by judge before execute
graph.add_edge("spawn_hypotheses",  "interpret_patterns")
graph.add_edge("interpret_patterns","arbitrate")
graph.add_edge("arbitrate",         "judge")    # Tier 3 validated by judge before execute
graph.add_edge("judge",             "execute")
graph.add_edge("execute",           "audit")
graph.add_edge("audit",             END)
# Note: judge uses Groq llama-3.1-8b-instant; validates output format and
# flags low-confidence decisions. Tier 1 (rule_route) bypasses judge.
```

## Memory Architecture

| Layer | Stores | Update Frequency | Implementation |
|---|---|---|---|
| Working | Current LangGraph state | Per-complaint | In-memory during pipeline |
| Episodic | Flat complaint history, resolutions | Per-resolution | PostgreSQL with indexes |
| Semantic | FM rules, best practices, domain knowledge | Monthly calibration | ChromaDB (vector) |
| Relational | Flat ↔ Building ↔ Tower ↔ Vendor graph | On onboarding | PostgreSQL graph queries |
| Pattern State | Active complaint clusters | Continuous | Redis sliding windows |

## Model Strategy

| Stage | Model Tier | Recommendation | Cost/Call |
|---|---|---|---|
| Domain classification (fallback) | Small | Groq Llama 3.1 8B | ~₹0.50 |
| Complexity assessment (edge cases) | Small | Groq Llama 3.1 8B | ~₹0.50 |
| Tier 2 reasoning | Mid | Claude Haiku or Groq Llama 3.1 70B | ₹2-5 |
| Tier 3 hypothesis agents | Mid-high | Claude Sonnet 4 or GPT-4o | ₹5-10 each |
| Pattern Interpretation | Mid-high | Claude Sonnet 4 | ₹5-10 |
| Arbiter | High | Claude Opus or GPT-4o structured | ₹10-15 |

**Cost levers:**
- Prompt caching (Anthropic native caching for common prefixes)
- Early exit (Tier 1 complaints never touch LLM)
- Model routing by tier
- Parallel hypothesis execution (latency of 1 call, cost of N)
- Feedback-driven tier migration (complaints shift T3 → T2 → T1 as system learns building patterns)

## Evaluation Framework

Run all loaded sample complaints through the pipeline. Measure:

1. **First-time-right rate:** Resolv's routing matches what would have avoided reassignment
2. **Tier distribution:** What percentage ended up in each tier (target: ~35/40/25)
3. **Latency per tier:** p50, p95, p99
4. **Cost per complaint:** blended average and per-tier
5. **Accuracy on known-ambiguous cases:** 398 seepage-keyword complaints and 674 cross-flat complaints
6. **Cluster detection accuracy:** For building-level pattern cases

Baseline comparison: the current human-assigned category in the data. Note: human classification is already inconsistent, so we'll need to establish ground truth for a sample (manual labeling of 500 complaints).
