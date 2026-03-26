---
description: "End every session. Reflection questions, digest, archive, log reset. Run before closing VS Code."
agent: agent
---

# Session Close

> **How to use:**
> - Default (recommended): `@session-close.md` (Quick Close)
> - Full end-of-session ritual: `@session-close.md full` (Full Close)

---

Close the session properly.

---

## Step 0 — Choose close mode (default is Quick Close)

Use **Quick Close** by default.

Switch to **Full Close** if any are true:

- An experiment was run
- A genuine architectural or methodological decision was made
- A workaround was introduced or changed
- There is unresolved uncertainty that could mislead next session

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

---
```

### Step 3 — Update SESSION_LOG.md for fast restart

Replace with only:

```
# Session Log

Last session: [YYYY-MM-DD] — <title>
See ARCHIVE.md for full history.

## Next session
**Recommended starting point:** <from summary>
**Open questions:** <from summary>
**Unverified assumptions:** <anything assumed but not checked>
**Active workarounds:** <list TODOs currently in codebase>
**Foundation status:** <brief: what is verified, what is assumed>
**Context freshness:** fresh / stale / unknown
**Contradiction flags:** none / <short note if current conclusions conflict with prior logs>
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

If JOURNAL.md exceeds 8 entries, compress the oldest 4 into `## Earlier Experiments`.

---

### Step 6 — Reset SESSION_LOG.md

Replace with only:

```
# Session Log

Last session: [YYYY-MM-DD] — <title>
See ARCHIVE.md for full history.

## Next session
**Recommended starting point:** <from digest>
**Open questions:** <from digest>
**Unverified assumptions:** <anything assumed but not checked>
**Active workarounds:** <list TODOs currently in codebase>
**Foundation status:** <brief: what is verified, what is assumed>
**Context freshness:** fresh / stale / unknown
**Contradiction flags:** none / <short note if current conclusions conflict with prior logs>
```

---

### Step 7 — Report in chat

After writing all logs, say in chat:

```
Session closed. Here is what was recorded:

**Archive entry written:** [YYYY-MM-DD] — <title>
**DECISIONS.md:** <n entries added, or "no new entries">
**JOURNAL.md:** <n entries added, or "no new entries">
**SESSION_LOG.md:** reset for next session

Key things carried forward:
- <most important open question>
- <active workarounds>
- <what is still unverified>
```
