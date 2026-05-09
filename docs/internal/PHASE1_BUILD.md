# PHASE 1 BUILD PLAN — Resolv.AI Foundation

**Duration:** Week 1-2 (10 working days)
**Goal:** Working prototype that processes real facility complaints through the tiered pipeline and produces measurable accuracy metrics.

## Deliverables

1. PostgreSQL database with ~17k sample complaints loaded and indexed
2. Working LangGraph pipeline with all tiers functional
3. At least 2 domains (water_plumbing, electrical) with full hypothesis libraries
4. Evaluation script that runs all complaints and produces accuracy report
5. FastAPI endpoint to submit new complaints and see reasoning traces

## Day-by-Day Plan

### Day 1-2: Data Foundation

**Tasks:**
- Set up project structure (per README.md)
- Initialize PostgreSQL locally (Docker: `postgres:16`)
- Create schema:
  ```sql
  CREATE TABLE complaints (
      id SERIAL PRIMARY KEY,
      ticket_id VARCHAR(50) UNIQUE NOT NULL,
      site_name VARCHAR(100),
      zone VARCHAR(50),
      created_date TIMESTAMP,
      complaint_title TEXT NOT NULL,
      status VARCHAR(20),
      category VARCHAR(100),
      sub_category VARCHAR(200),
      issue_type VARCHAR(50),
      created_by VARCHAR(200),
      tower VARCHAR(50),
      flat VARCHAR(50),
      aging_days INTEGER,
      priority VARCHAR(10),
      resolution_tat_minutes INTEGER,
      closed_date TIMESTAMP,
      raw_data JSONB  -- Store full original row for flexibility
  );
  
  CREATE INDEX idx_flat ON complaints(site_name, flat);
  CREATE INDEX idx_tower ON complaints(site_name, tower);
  CREATE INDEX idx_category ON complaints(category);
  CREATE INDEX idx_created ON complaints(created_date);
  CREATE INDEX idx_text ON complaints USING gin(to_tsvector('english', complaint_title));
  ```
- Write ETL script: `scripts/load_complaints_xlsx.py` — reads Excel, normalizes, inserts

**Flat Adjacency Table:**
- Parse tower + flat to extract floor and unit position
- Example: "T6-1803" → tower T6, floor 18, unit 03
- Build adjacency:
  ```sql
  CREATE TABLE flat_adjacency (
      site_name VARCHAR(100),
      flat VARCHAR(50),
      above_flat VARCHAR(50),
      below_flat VARCHAR(50),
      lateral_flats VARCHAR(50)[],  -- Array of adjacent flats on same floor
      PRIMARY KEY (site_name, flat)
  );
  ```
- Script: `scripts/build_adjacency.py` — infers above/below/lateral from flat numbering patterns per site

**Done when:** Can run `SELECT * FROM complaints WHERE flat = 'V1006'` and get 243 rows.

### Day 3: Domain Classifier + Complexity Assessor

**Tasks:**
- Create `src/config/domain_rules.yaml` with keyword patterns for 8 domains
- Implement `src/nodes/domain_classifier.py`:
  - Check keyword matches against YAML rules
  - Return top domain match + confidence
  - Fallback to LLM call if no strong match (use Groq Llama 8B for cost)
- Create `src/config/tier_rules.yaml` with Tier 1/2/3 triggers
- Implement `src/nodes/complexity_assessor.py`:
  - Check Tier 1 exact patterns first
  - Check Tier 3 triggers (ambiguity keywords + historical conditions)
  - Default Tier 2

**Validation:** Run on 500 sample complaints, manually verify 50 tier + domain assignments. Iterate rules.

### Day 4: Context Assembler

**Tasks:**
- Implement `src/nodes/context_assembler.py`:
  ```python
  async def assemble_context(complaint: Complaint) -> ContextPackage:
      results = await asyncio.gather(
          get_flat_history(complaint.flat_id),
          get_adjacent_history(complaint.flat_id),
          get_building_pattern(complaint.tower, complaint.category),
      )
      return ContextPackage(*results)
  ```
- Each function is a pure DB query with async execution
- Serialize context into a format suitable for LLM prompt inclusion

**Validation:** For 10 real complaints with known cross-flat references, verify the adjacent_history correctly returns the referenced flats' complaints.

### Day 5: Hypothesis Library — Water/Plumbing Domain

**Tasks:**
- Create `src/config/hypothesis_library.yaml` with water_plumbing domain fully specified
- Write system prompts in `src/agents/prompts/`:
  - `pipe_failure.md`
  - `structural_seepage.md`
  - `environmental.md`
  - `hvac_condensate.md`
- Implement `src/agents/hypothesis_agent.py`:
  ```python
  class HypothesisAgent:
      def __init__(self, hypothesis_config, llm_client):
          self.config = hypothesis_config
          self.llm = llm_client
      
      async def evaluate(self, complaint, context) -> HypothesisResult:
          filtered_context = filter_evidence(context, self.config.evidence_filter)
          prompt = load_prompt(self.config.prompt_template)
          response = await self.llm.complete(
              system=prompt,
              user=format_input(complaint, filtered_context),
              response_format=HypothesisResult
          )
          return response
  ```
- Implement `src/nodes/spawn_hypotheses.py` — reads library config, spawns agents in parallel via asyncio.gather

**Validation:** Take 5 ambiguous water/leak complaints from the sample dataset. Run through hypothesis agents. Manually evaluate if scores are reasonable.

### Day 6: Pattern State + Arbiter

**Tasks:**
- Set up Redis locally (Docker: `redis:7`)
- Implement `src/memory/pattern_state.py`:
  - On each complaint: update sliding window aggregates
  - Cluster detection using DBSCAN over spatial-temporal features
  - Expose query: `get_active_clusters(building_id, floor, domain)`
- Implement `src/agents/pattern_interpreter.py` with prompt in `src/agents/prompts/pattern_interpreter.md`
- Implement `src/agents/arbiter.py` with prompt in `src/agents/prompts/arbiter.md`

**Arbiter prompt emphasis:**
- Input all hypothesis scores + pattern interpretation + cost weights
- Apply asymmetric loss: higher-cost hypothesis needs less confidence to be selected
- May recommend multi-action decisions (e.g., plumber today + structural tomorrow)
- Must produce reasoning trace

### Day 7: LangGraph Orchestration

**Tasks:**
- Implement `src/pipeline/resolv_graph.py` per the topology in ARCHITECTURE.md
- Define `ResolvState` TypedDict
- Wire all nodes and conditional edges
- Test end-to-end flow with sample complaints from each tier

**Validation:** Trace execution on 3 complaints (one per tier). Verify:
- Tier 1 exits after rule_route (no LLM calls)
- Tier 2 hits single_reasoning (1 LLM call)
- Tier 3 spawns hypotheses → interprets patterns → arbiter (3-5 LLM calls)

### Day 8: Electrical Domain + FastAPI

**Tasks:**
- Replicate hypothesis library pattern for `electrical` domain (4 hypotheses)
- Write all prompts for electrical hypotheses
- Build `src/main.py` FastAPI app:
  ```
  POST /complaints           # Submit new complaint, returns routing decision + trace
  GET  /complaints/{id}      # Get complaint by ID with full reasoning trace
  GET  /complaints/stats     # Processing stats, tier distribution
  GET  /clusters/active      # Currently active complaint clusters
  ```

### Day 9: Evaluation Framework

**Tasks:**
- Create `eval/run_sample_evaluation.py`:
  - Iterate all 15,864 complaints
  - Run each through pipeline
  - Log: tier, domain, hypothesis scores, final routing, latency, cost
  - Compare final routing to human-assigned category (this is a rough baseline — human classification is inconsistent)
- Create `eval/generate_report.py`:
  - Overall tier distribution vs target (35/40/25)
  - Average cost per complaint
  - Latency p50/p95/p99 per tier
  - Accuracy on 398 seepage-keyword complaints
  - Accuracy on 674 cross-flat complaints
  - Total LLM cost for full run

**Expected output:** Markdown report with all metrics + CSV of detailed per-complaint results.

### Day 10: Polish + Demo

**Tasks:**
- Fix any issues surfaced by evaluation
- Build simple web UI for demo (single HTML + Tailwind page, no framework):
  - Input a complaint text + flat ID
  - Show tier assignment, domain classification
  - Show each hypothesis agent's output in separate cards
  - Show pattern signals
  - Show arbiter decision with full reasoning trace
- Record 3-minute demo video
- Update README with setup instructions

## Tech Decisions

**LLM Provider:** Start with Anthropic (Claude Haiku for Tier 2, Sonnet for Tier 3, Opus for Arbiter). Groq as fallback/cost-optimization for Tier 2 after evaluation.

**Why not local models:** For MVP speed. Local Llama models require GPU infrastructure. Can migrate later for cost optimization.

**Why PostgreSQL not MongoDB:** Structured queries, strong indexing, proper joins for adjacency lookups. JSONB column handles flexibility.

**Why Redis:** In-memory speed for pattern state is critical. Sliding windows are trivial with Redis sorted sets.

**Async everywhere:** Python asyncio throughout. No blocking calls in the pipeline.

## What NOT to build in Phase 1

- **Multi-tenant architecture** — that's Phase 3
- **Integration with real MyGate / NoBrokerHood APIs** — stub these
- **Real vendor dispatch** — stub the execution layer
- **Learning / calibration loop** — that's Phase 2
- **React dashboard** — single HTML page is enough for demo
- **Auth / user management** — not needed for MVP
- **Cloud deployment** — local Docker is enough, Railway/Render later

## Definition of Done for Phase 1

- [ ] Full sample complaint set loaded into PostgreSQL
- [ ] LangGraph pipeline runs end-to-end for all tiers
- [ ] 2 domains (water_plumbing, electrical) with full hypothesis libraries
- [ ] Evaluation report generated showing tier distribution, cost, latency, accuracy on ambiguous cases
- [ ] Demo UI works — can submit a complaint and see full reasoning trace
- [ ] FastAPI endpoints functional
- [ ] Code committed to GitHub with README setup instructions

## Known Risks & Mitigations

**Risk:** Hypothesis agent prompts may produce inconsistent scoring across runs.
**Mitigation:** Use structured output (Anthropic tool use or OpenAI response_format). Set temperature=0. Run evaluation 3 times on a sample to measure variance.

**Risk:** DBSCAN clustering on small data may not produce meaningful clusters.
**Mitigation:** The sample has tens of thousands of complaints over multiple years — plenty for clustering. Tune eps and min_samples parameters per site.

**Risk:** Human-assigned category is unreliable ground truth.
**Mitigation:** Manual labeling of 500 sample complaints by domain expert (you) for Phase 2 calibration. For Phase 1 eval, use human category as rough baseline.

**Risk:** LLM API costs during full eval run (15,864 complaints).
**Mitigation:** Estimate: ~₹5-12 × 15,864 = ₹80K-190K for full eval. Run eval on 2,000-sample first, extrapolate. Only full run before demo.

## Start Here

When Claude Code opens this project, the first step is:

```bash
# Read context
cat README.md
cat ARCHITECTURE.md
cat HYPOTHESIS_LIBRARY.md
cat PHASE1_BUILD.md

# Verify data
ls data/
# Should see: data/*.xlsx (local only; gitignored)
```

Then start with Day 1-2: data foundation. Don't jump ahead.
