---
description: "Full codebase explanation mode. Reads the entire repo and explains how it works — structure, data flow, design decisions, results, and implications. Use to onboard yourself after a break, or to build deep understanding before modifying something."
agent: "ask"
---

Enter codebase explanation mode. Your goal is to give me a complete, technically deep understanding of this project — how it is built, why it is built that way, how the pieces connect, and what the results mean.

Do not explain things at a surface level. I want to understand this well enough to modify any part of it without breaking the rest.

---

## Step 1 — Read everything relevant

Before explaining anything, read:
- `README.md`
- `DECISIONS.md`
- `JOURNAL.md` (last several entries)
- The full `src/` and `core/` directory structure
- Key implementation files (identify these yourself based on the directory structure)
- The most recent results in `results/` or `outputs/`
- Any config files

---

## Step 2 — Project overview

Explain:
- What this project does, precisely — not just the high-level goal but the specific computational task
- What the inputs are (data format, shape, source, preprocessing)
- What the outputs are (format, meaning, how they are evaluated)
- What the central technical challenge is

---

## Step 3 — Architecture walkthrough

Walk through the codebase as if you are giving a technical onboarding to a senior engineer who has never seen it:

- What is in `src/` and `core/` — what each module's *purpose* is, not just what it contains
- The data flow from raw input to final output, step by step
- Where the core algorithm or model lives and how it works
- How components depend on each other — what breaks if you change X
- Any non-obvious design decisions that show up in the code structure

Use concrete references to actual file and function names throughout.

---

## Step 4 — Design decisions

For each significant architectural or methodological decision:
- What was chosen
- Why (as best you can infer from code, comments, and `DECISIONS.md`)
- What the alternative was and why it was not used (if inferrable)
- What this decision makes easy and what it makes hard

---

## Step 5 — Results and what they mean

Walk through the most recent experimental results:
- What was measured and how
- What the numbers actually say
- What is strong, what is uncertain, what is unexplained
- How the results connect back to the stated goal

---

## Step 6 — Known issues and open questions

From `SESSION_LOG.md`, `JOURNAL.md`, TODOs in code, and your own reading:
- What is incomplete or partially implemented
- What has been flagged as a workaround that needs a proper solution
- What open questions the project has not answered yet

---

## Step 7 — Check my understanding

After walking through everything, ask me 3–5 questions about the codebase. These should:
- Target the parts most likely to be misunderstood
- Check that I understand the dependencies between components
- Test whether I understand *why* things are done the way they are, not just *what* they do

If I answer incorrectly, correct me precisely and explain why. If I answer well, confirm it and build on it.

The goal is that after this session, I can navigate and modify this codebase confidently.
