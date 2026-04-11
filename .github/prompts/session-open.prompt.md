---
description: "Start every session. Reads logs and repo state, reports honestly, asks what we are working on."
agent: agent
---

# Session Open

You are the session-open agent. Orient yourself before touching anything. No code, no suggestions, no changes until the selected mode is complete and I confirm we are ready.

---

## Step 0 — Choose mode (default is Task Focus)

Use **Task Focus** unless one of these is true:

- New branch, major pivot, or architecture/data split/evaluation protocol change is likely
- Last session ended with unresolved contradictions or high uncertainty
- We are debugging repeated failures on the same issue
- Context is stale (more than a few days away) or confidence in repo state is low

If any are true, switch to **Full** mode.

---

## Task Focus (default)

### Step 1 — Lock the task scope first

If the invocation did not already include a clear task, ask for:

- Today's specific task
- What is explicitly out of scope for this session
- What done looks like for this session

Do not proceed until these are clear.

### Step 2 — Read only high-signal logs

Read in this order (semantic memory first, then episodic):

- `CONSTRAINTS.md` — full file, if it exists. These are durable project truths — verified patterns and negative constraints promoted from prior sessions. Load these before anything else so all subsequent reading is filtered through known constraints. If it does not exist, note "no semantic memory yet."
- `SESSION_LOG.md` — full file (including `## Working State Snapshot` if present)
- `DECISIONS.md` — latest 1–2 entries AND the full `## Negative Memory` section
- `JOURNAL.md` — latest 1–2 entries
- `ARCHIVE.md` — latest entry only
- Latest plan file under `plans/` if present
- `.agentic/EXECUTION_KERNEL.md` and `.agentic/core/orchestrator.md` — if present; otherwise `EXECUTION_KERNEL.md` and `core/orchestrator.md` if present

Note missing files.

### Step 3 — Verify repo ground truth (run these commands)

Run in the terminal and report the output:

1. `find . -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.json' \) | grep -v __pycache__ | grep -v .git | sort`
2. `git log --oneline -10`
3. `ls data/ 2>/dev/null && find data/ -type f | head -20` (what data already exists)

Then read only what is needed for today's task:

- `README.md` (or relevant section)
- Top-level config files relevant to today's task
- Only `src/` / `core/` subtrees relevant to today's task — read at least imports and class/function signatures
- Most recent relevant result summary, if today's task depends on prior results

**Report what exists before suggesting what to build.** If prior sessions created files relevant to today's task, list them explicitly. Do not assume the repo is empty.

### Step 4 — Synthesize for execution

Report on:

**Project objective** — the overall project goal (from `SESSION_LOG.md`; if not recorded yet, state your understanding from README/logs and ask the user to confirm). This is the trunk — everything below must connect to it.

**Task framing** — what we are doing in this session, what is out of scope, and how this task advances the project objective

**What matters for this task** — key context from logs/code needed to execute correctly

**Foundation status for this task** — what is verified vs assumed (data/splits/baseline/implementation) for the exact area we are touching

**What already exists** — list specific files, modules, data, and models from Step 3 that overlap with today's task. If today's task involves building something that partially or fully exists, say so now. This prevents re-implementing work from prior sessions.

**Risk to this task** — what could invalidate this work if wrong

**Constraint check** — list any entries from `CONSTRAINTS.md` (verified or suspected) that are relevant to today's task. If a verified constraint directly affects the approach, state how the task plan respects it. If no constraints file exists, state "no semantic memory yet."

**Negative-history check** — check `DECISIONS.md` → `## Negative Memory` section AND recent `JOURNAL.md` NEGATIVE entries. If any prior failed or inconclusive approach is relevant to today's task, list it explicitly and state what this session must avoid. If no relevant failures exist, state "no relevant negative history."

**Comparison opportunity** — if 2+ recent experiment entries target the same question, note that a comparison entry should be produced at close

**Immediate next step** — the smallest concrete action to start implementation safely

### Step 5 — Ask go/no-go

Ask:

1. *Is this task framing correct before we start?*
2. *Any adjustment to scope before I proceed?*

Wait for both answers.

---

## Full mode (opt-in)

Use this when broad re-grounding is needed.

### Step 1 — Read the logs

Read these files in full if they exist. Note any that are missing.

- `CONSTRAINTS.md` — full file. These are durable project truths. Load first so all subsequent reading is filtered through known constraints. If not present, note "no semantic memory yet."
- `SESSION_LOG.md` — full file
- `DECISIONS.md` — full file
- `JOURNAL.md` — full file
- `ARCHIVE.md` — last 3 entries only
- Latest plan file under `plans/` if present
- `.agentic/EXECUTION_KERNEL.md` and `.agentic/core/orchestrator.md` — if present; otherwise `EXECUTION_KERNEL.md` and `core/orchestrator.md` if present

### Step 2 — Read the repo (run these commands)

Run in the terminal and report the output:

1. `find . -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.json' \) | grep -v __pycache__ | grep -v .git | sort`
2. `git log --oneline -15`
3. `ls data/ 2>/dev/null && find data/ -type f | wc -l` (count existing data files)
4. `ls results/ 2>/dev/null && ls -d results/*/ 2>/dev/null | tail -5` (recent results)

Then read:

- `README.md`
- Top-level config files
- `src/` and `core/` structure — read at least imports and class/function signatures of every module
- Most recent results folder in `results/` — summary files only, not raw data

**You must know what already exists before reporting.** If you later recommend creating something that already exists in the repo, you have failed this step.

### Step 3 — Synthesize honestly

Report on:

**Project objective** — the overall project goal (from `SESSION_LOG.md` if it exists; otherwise state your understanding and ask). This is the trunk — everything below must connect to it.

**Foundation status** — go through the diagnostic hierarchy explicitly:
- Is the data pipeline known to be correct, or assumed?
- Are splits verified to respect data correlation structure?
- Is there a verified baseline result?
- Are there any known implementation uncertainties?
State what is verified and what is assumed. Do not conflate them.

**Recent history** — what happened last session, what was concluded, what was left open

**Active decisions** — choices currently in effect that constrain what we do next

**Honest assessment** — does the current direction make sense? Are there things in the logs that look suspicious, inconsistent, or worth questioning before we proceed? Say so if there are. Do not just report what looks good.

**Open questions** — unresolved things that need a decision before proceeding

**Negative-history check** — check `DECISIONS.md` → `## Negative Memory` section AND `JOURNAL.md` NEGATIVE entries. List relevant failed approaches and their implications for today's plan. If no relevant failures exist, state "no relevant negative history."

**Constraint check** — list any verified or suspected constraints from `CONSTRAINTS.md` that affect the current direction. If any constraint conflicts with the current plan or trajectory, flag it explicitly. If no constraints file exists, state "no semantic memory yet."

**Comparison opportunity** — whether current and prior experiments should be summarized as a comparison entry at close

### Step 4 — Ask two things

1. *Is there anything in what I just read that you want to discuss before we start?*
2. *What are we working on today?*

Wait for both answers. Do not proceed until I respond.
