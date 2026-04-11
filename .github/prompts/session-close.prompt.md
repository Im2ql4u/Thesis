---
description: "End every session. Reflection questions, digest, archive, log reset. Run before closing VS Code."
agent: agent
---

# Session Close

You are the session-close agent. Close the session properly.

---

## Step 0 — Git status check (always, before anything else)

Run `git status` and `git log --oneline -5` in the terminal. Evaluate:

- **Uncommitted changes exist?** List the changed files. For each file, assess:
  - Is this new working code that passed tests this session? → Stage and commit: `git add -p && git commit -m "<type>(<scope>): <what>"`
  - Is this a work-in-progress that is NOT tested or passing? → Commit with WIP prefix: `git add -p && git commit -m "WIP(<scope>): <what — state and why incomplete>"`
  - Is this generated output, data, or temporary files? → Do NOT commit. Add to `.gitignore` if missing.
- **Unpushed commits exist?** After committing, run `git log --oneline origin/$(git branch --show-current)..HEAD`. If commits exist that haven't been pushed, push them: `git push`.
- **No changes at all?** Note this in the close report — either nothing was built, or commits were made but not tracked.

Report the git status clearly in chat before proceeding to close mode selection:
```
Git status:
- Committed this session: <n> commits
- Pushed: yes / no (pushed now / already up to date)
- Uncommitted files handled: <list or "none">
```

---

## Step 0.5 — Choose close mode (default is Quick Close)

Choose mode by evaluating the session, not by habit.

Use **Full Close** if any are true:

- An experiment was run
- A genuine architectural or methodological decision was made
- A workaround was introduced or changed
- There is unresolved uncertainty that could mislead next session

If none are true, use **Quick Close**.

If uncertain, choose Full Close.

---

## Quick Close (default)

### Step 1 — Ask two reflection questions

Ask and wait for answers:

1. What is the single most important thing to carry into the next session?
2. What is still uncertain or risky right now?

### Step 2 — Append a compact ARCHIVE entry

Append:

```
## [YYYY-MM-DD] — <session title>

### Technical summary
- Goal: <one line>
- Accomplished: <one line>
- Not done / blocked: <one line>
- Recommended next action: <one line>

### Human reflection
**Carry forward:** <answer 1>
**Still uncertain:** <answer 2>

### Session metrics
- Steps completed: <n of m planned>
- Material deviations: <count>
- Evaluation gates triggered: <count + verdict>
- Unresolved uncertainties: <count>

---
```

### Step 3 — Update SESSION_LOG.md for fast restart

Replace with only:

```
# Session Log

Last session: [YYYY-MM-DD] — <title>
See ARCHIVE.md for full history.

## Next session
**Project objective:** <overall project goal — carry forward from prior session or update if changed>
**Active plan file:** <plans/YYYY-MM-DD_<descriptor>.md or none>
**Recommended starting point:** <from summary>
**Open questions:** <from summary>
**Unverified assumptions:** <anything assumed but not checked>
**Active workarounds:** <list TODOs currently in codebase>
**Foundation status:** <brief: what is verified, what is assumed>
**Context freshness:** fresh / stale / unknown
**Contradiction flags:** none / <short note if current conclusions conflict with prior logs>

## Session metrics (latest)
**Steps completed:** <n of m planned>
**Material deviations:** <count>
**Evaluation gates triggered:** <count + verdict>
**Unresolved uncertainties:** <count>
```

### Step 4 — Report in chat

Say in chat:

```
Session closed (Quick Close).

**Archive entry written:** [YYYY-MM-DD] — <title>
**SESSION_LOG.md:** reset for next session

Key carry-forward:
- <most important next action>
- <most important uncertainty>
```

### Step 5 — Memory consolidation (lightweight)

Run the **lightweight variant** of the memory-consolidation skill:
1. Scan this session's outputs for candidate patterns (repeated findings, confirmed failures, stable decisions)
2. If any candidate crosses the promotion threshold (3+ occurrences across sessions), update `CONSTRAINTS.md`
3. Produce the Working State Snapshot and append to `SESSION_LOG.md`

If `CONSTRAINTS.md` does not exist yet and this session produced a finding worth tracking, create it from the template.

---

## Full Close (opt-in)

Use this for sessions with experiments, major decisions, or high uncertainty.

### Step 1 — Reflection questions first

Before writing anything to the logs, ask the human these questions and wait for answers:

1. What was the most important thing you understood this session that you did not understand before?
2. Is there anything that was written or done this session that you do not fully understand yet?
3. Looking at what was built — what would a skeptic say about it? What are its weakest points?
4. What would you do differently if starting this session again?

Record their answers verbatim in the archive entry. These are as important as the technical summary — they are a record of understanding, not just activity.

---

### Step 2 — Write the session digest

Now write the compressed digest. Answer only these questions — not a transcript:

- What was the central goal?
- What was actually accomplished? (concrete, specific)
- What was attempted but did not work, and why?
- What decisions were made, even implicitly?
- What workarounds are currently in place?
- What is unverified or assumed that should be verified?
- What does a skeptic say about today's results?
- What is the single most important thing the next session needs to know?
- Recommended next action

Target: 20–30 lines. If longer, you are summarizing events not conclusions.

---

### Step 3 — Append to ARCHIVE.md

```
## [YYYY-MM-DD] — <session title>

### Technical summary
<digest from step 2>

### Human reflection
**Understood this session:** <their answer>
**Still unclear:** <their answer>
**Skeptic's view:** <their answer>
**Would do differently:** <their answer>

---
```

If ARCHIVE.md exceeds 10 entries, compress the oldest 5 into a `## Older History` block answering: what was this project doing, what was tried, what was concluded, what is the current state. Remove the individual entries after compressing.

---

### Step 4 — Update DECISIONS.md

Only if a genuine architectural or methodological decision was made:

```
### [YYYY-MM-DD] — <title>
**Decision:** <what was chosen>
**Alternatives considered:** <what else was on the table>
**Reasoning:** <why this>
**Constraints introduced:** <what this makes harder>
**Confidence:** high / medium / low
```

Zero entries per session is fine. Do not pad this file.

### Step 4.5 — Update Negative Memory in DECISIONS.md

If any approach was tried and failed, was abandoned, or produced inconclusive results this session, append an entry to the `## Negative Memory` section of `DECISIONS.md`:

```
### [YYYY-MM-DD] — FAILED: <what was tried>
**What:** <the approach or pattern that was attempted>
**Why it failed:** <root cause or best understanding>
**Evidence:** <concrete output, error, or metric that proves failure>
**What to do instead:** <known better alternative, or "unknown — needs investigation">
**Severity:** dead-end | needs-rethink | minor-setback
```

Do not force entries. If nothing failed, skip this step. If a failure was already logged in a prior session's negative memory and this session hit the same root cause, note the recurrence — if it is the third occurrence, promote it to a permanent decision/constraint entry.

---

### Step 5 — Update JOURNAL.md

Only if an experiment was run:

```
### [YYYY-MM-DD] — <experiment title>
**Motivation:** <what question were we answering>
**Method:** <what was done>
**Results:** <numbers, with units>
**What the numbers actually mean:** <interpretation>
**What we cannot explain:** <anything anomalous or uncertain>
**Caveats:** <what might be wrong with this interpretation>
**What a skeptic would say:** <honest critique of the result>
**Output reference:** results/YYYY-MM-DD_<n>/
**Next question:** <what this makes us want to investigate>
```

If the experiment failed, was blocked, or was inconclusive, use the NEGATIVE format in `JOURNAL.md` instead of forcing a positive-result narrative.

If 2+ experiment entries address the same question, add a comparison entry using the `## Comparison` template from `JOURNAL.md`.

If JOURNAL.md exceeds 8 entries, compress the oldest 4 into `## Earlier Experiments`.

---

### Step 6 — Reset SESSION_LOG.md

Replace with only:

```
# Session Log

Last session: [YYYY-MM-DD] — <title>
See ARCHIVE.md for full history.

## Next session
**Project objective:** <overall project goal — carry forward from prior session or update if changed>
**Active plan file:** <plans/YYYY-MM-DD_<descriptor>.md or none>
**Recommended starting point:** <from digest>
**Open questions:** <from digest>
**Unverified assumptions:** <anything assumed but not checked>
**Active workarounds:** <list TODOs currently in codebase>
**Foundation status:** <brief: what is verified, what is assumed>
**Context freshness:** fresh / stale / unknown
**Contradiction flags:** none / <short note if current conclusions conflict with prior logs>

## Session metrics (latest)
**Steps completed:** <n of m planned>
**Material deviations:** <count>
**Evaluation gates triggered:** <count + verdict>
**Unresolved uncertainties:** <count>
```

---

### Step 6.5 — Memory consolidation (full)

Run the **full** memory-consolidation skill:
1. Inventory current memory state (CONSTRAINTS.md, DECISIONS.md, JOURNAL.md, ARCHIVE.md)
2. Extract candidate patterns — repeated findings, stable decisions, confirmed negative patterns
3. Promote qualifying patterns to `CONSTRAINTS.md` (create from template if it doesn't exist)
4. Compress episodic memory if JOURNAL.md > 8 entries or ARCHIVE.md > 10 entries
5. Prune Negative Memory entries that have been resolved or promoted to CONSTRAINTS.md
6. Produce Working State Snapshot and append to SESSION_LOG.md

Report memory consolidation results as part of the Step 7 chat report.

---

### Step 7 — Report in chat

After writing all logs, say in chat:

```
Session closed. Here is what was recorded:

**Archive entry written:** [YYYY-MM-DD] — <title>
**DECISIONS.md:** <n entries added, or "no new entries">
**JOURNAL.md:** <n entries added, or "no new entries">
**CONSTRAINTS.md:** <n promoted / n suspected / no changes>
**SESSION_LOG.md:** reset for next session + working state snapshot

Key things carried forward:
- <most important open question>
- <active workarounds>
- <what is still unverified>

Session metrics:
- Steps completed: <n of m planned>
- Material deviations: <count>
- Evaluation gates triggered: <count + verdict>
- Unresolved uncertainties: <count>
```
