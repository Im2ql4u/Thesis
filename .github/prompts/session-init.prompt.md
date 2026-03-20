---
description: "Start every session here. Reads logs and repo state before touching anything. Establishes shared context."
agent: "ask"
---

Before we do anything else, orient yourself fully. Do not write any code, make any changes, or offer any suggestions until you have completed all of the following steps and I have confirmed we are ready to proceed.

---

## Step 1 — Read the logs

Read the following files if they exist. If any are missing, note that and continue.

- `SESSION_LOG.md` — focus on the most recent 2–3 entries
- `DECISIONS.md` — read in full; this is the architectural memory of the project
- `JOURNAL.md` — read the last 3–5 experiment entries

---

## Step 2 — Read the repo structure

Run a directory listing of the repo (do not recurse into `data/`, `outputs/`, `.git/`). Then read:

- `README.md` — understand the stated goal and current plan
- Any top-level config files (e.g. `pyproject.toml`, `setup.py`, `config.yaml`)
- The `src/` or `core/` directory structure

---

## Step 3 — Synthesize and report

Give me a concise structured summary covering:

**Current state of the repo**
- What does this project do, in one sentence
- What is the current structural state (what exists, what is incomplete)

**Recent history**
- What was worked on in the last session(s)
- Any open issues, TODOs, or workarounds logged

**Architectural decisions in effect**
- Key decisions that constrain what we should do next

**Pending work**
- What the README or last session log says is next
- Whether that still seems like the right direction given what you've read

**Questions before we begin**
- Anything ambiguous or contradictory in what you read
- Any assumption you would have to make to proceed that I haven't explicitly stated

---

## Step 4 — Wait

After delivering the summary, ask me: *What are we working on today?*

Do not proceed until I answer.
