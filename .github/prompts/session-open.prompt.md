---
description: "Start every session. Reads logs and repo state, reports honestly, asks what we are working on."
agent: agent
---

# Session Open

> **How to use:**
> - Default (recommended): `@session-open.md` (Task Focus mode)
> - Heavy reorientation: `@session-open.md full: <reason>` (Full mode)

---

Orient yourself before touching anything. No code, no suggestions, no changes until the selected mode is complete and I confirm we are ready.

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

Read:

- `SESSION_LOG.md` — full file
- `DECISIONS.md` — latest 1–2 entries
- `JOURNAL.md` — latest 1–2 entries
- `ARCHIVE.md` — latest entry only

Note missing files.

### Step 3 — Read only relevant repo context

Directory listing excluding `data/`, `outputs/`, `results/`, `.git/`. Then read only what is needed for today's task:

- `README.md` (or relevant section)
- Top-level config files relevant to today's task
- Only `src/` / `core/` subtrees relevant to today's task
- Most recent relevant result summary, if today's task depends on prior results

### Step 4 — Synthesize for execution

Report on:

**Task framing** — what we are doing in this session and what is out of scope

**What matters for this task** — key context from logs/code needed to execute correctly

**Foundation status for this task** — what is verified vs assumed (data/splits/baseline/implementation) for the exact area we are touching

**Risk to this task** — what could invalidate this work if wrong

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

- `SESSION_LOG.md` — full file
- `DECISIONS.md` — full file
- `JOURNAL.md` — full file
- `ARCHIVE.md` — last 3 entries only

### Step 2 — Read the repo

Directory listing excluding `data/`, `outputs/`, `results/`, `.git/`. Then read:

- `README.md`
- Top-level config files
- `src/` and `core/` structure
- Most recent results folder in `results/` — summary files only, not raw data

### Step 3 — Synthesize honestly

Report on:

**Project** — one sentence: what this is and what it is genuinely trying to achieve

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

### Step 4 — Ask two things

1. *Is there anything in what I just read that you want to discuss before we start?*
2. *What are we working on today?*

Wait for both answers. Do not proceed until I respond.
