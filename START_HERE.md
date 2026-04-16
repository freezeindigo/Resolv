# START_HERE — First Message to Claude Code

Copy-paste the following as your first message when you open Claude Code in the resolv/ directory.

---

## First message to paste:

```
I'm building Resolv.AI — an agentic decision intelligence system for facility management 
complaint routing. Before writing any code, please read these files in order and then 
summarize your understanding:

1. README.md — high-level context and design principles
2. ARCHITECTURE.md — detailed system architecture
3. HYPOTHESIS_LIBRARY.md — domain-to-hypothesis agent mappings
4. PHASE1_BUILD.md — the specific 10-day build plan we're following

After reading, tell me:
- What problem Resolv is solving and why current solutions don't work
- The core architectural principle that separates "agents" from "pipeline nodes"  
- Why the hypothesis library is dynamic (not fixed at 3 agents)
- The exact tiered processing flow (Tier 1/2/3)
- What Phase 1 builds and what it explicitly does NOT build

Do not write any code yet. I want to verify your understanding is correct before we start.
Once you confirm, we'll begin with Day 1-2 of PHASE1_BUILD.md — the data foundation.

The Godrej complaint data is at data/godrej_complaints.xlsx (15,864 rows). Don't touch it 
yet — we'll load it in Day 1-2.
```

---

## After Claude Code responds with its understanding:

**If the summary is correct:** Reply with `Good. Let's start Day 1-2. Set up the project structure and write the data loading script.`

**If anything is wrong or missing:** Correct it specifically. Example: `Your summary missed that hypothesis agents run in parallel with isolated evidence filters, which is the key reason they're agents and not one prompt. Please re-read section 4.3 of ARCHITECTURE.md and re-summarize.`

---

## Working Pattern

For each day in PHASE1_BUILD.md, use this pattern:

1. **Start:** `Let's start Day X. Read PHASE1_BUILD.md section for Day X and tell me the specific tasks.`

2. **Review plan:** Claude Code will list tasks. You approve or adjust.

3. **Execute:** `Proceed with task 1.` (then task 2, etc.)

4. **Verify:** After each task, ask Claude Code to show you:
   - What file was created/modified
   - A quick test that it works
   - Any assumptions it made

5. **End of day:** `End of Day X. Commit everything with a clear message, then update PHASE1_BUILD.md to mark Day X as complete.`

---

## Guardrails for Claude Code

Repeat these whenever needed:

**When Claude Code tries to add a new "agent":**
> Does this require LLM reasoning that cannot be collapsed into deterministic logic, or is it a database query / API call / logging step? If it's the latter, it's a pipeline node, not an agent.

**When Claude Code tries to add complexity:**
> Check PHASE1_BUILD.md "What NOT to build in Phase 1." If this isn't in the deliverables for Phase 1, don't build it now.

**When Claude Code writes a prompt:**
> Put the prompt in a separate file under src/agents/prompts/. Prompts are first-class artifacts, not inline strings.

**When unsure about architecture:**
> Refer back to ARCHITECTURE.md. If it's not documented there, document it before implementing.

---

## Managing Claude Code Context

Claude Code has a context window — if the session gets long, it will lose earlier context. To manage this:

1. **Commit frequently.** After each working feature, git commit. If you lose context, you can always `git log` and Claude Code can re-orient.

2. **Keep PHASE1_BUILD.md updated.** Mark days as complete. Claude Code can re-read it anytime.

3. **If Claude Code seems confused:** Say `Re-read README.md and PHASE1_BUILD.md and tell me where we are.`

4. **For complex debugging:** Start fresh sessions for complex problems. Paste the specific error + relevant file contents.

---

## When to Come Back to Me (Claude.ai)

Use Claude Code for:
- Writing code
- Running tests
- Debugging
- File management

Come back to me (Claude.ai in this conversation) for:
- Architecture decisions ("should I add X or Y?")
- Prompt engineering (the actual system prompts for hypothesis agents)
- Understanding tradeoffs
- Preparing for AJVC conversations
- Reviewing Claude Code's output for architectural drift

---

## Setup Checklist Before Starting

- [ ] Create a GitHub repo: `resolv`
- [ ] Clone locally
- [ ] Copy this resolv_project folder contents into the repo root
- [ ] Place `godrej_complaints.xlsx` in `data/` folder
- [ ] Install Claude Code: `npm install -g @anthropic-ai/claude-code`
- [ ] Run `claude` in the resolv/ directory
- [ ] Paste the first message above

Go.
