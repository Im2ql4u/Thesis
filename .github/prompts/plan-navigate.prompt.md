---
description: "Navigates an existing plan against current repo state and results. For when a plan already exists and you need to know where you are, whether it still holds, and what to do next."
agent: "ask"
---

A plan already exists. Your job is not to execute it blindly — it is to read the current state of the repo and results, compare that against the plan, and tell me honestly where we are and whether the plan still makes sense.

---

## Step 1 — Read the plan

Read `README.md` or whatever planning document exists. Identify:
- The stated goal
- The approach / methodology
- The phases or milestones
- Any explicit success criteria

---

## Step 2 — Read the current state

Read:
- `SESSION_LOG.md` and `JOURNAL.md` for what has actually been done
- The `src/` and `core/` structure for what has actually been built
- Any results in `results/` or `outputs/` — look at the most recent outputs, not just the filenames

---

## Step 3 — Gap analysis

Compare plan against reality honestly:

**What has been completed** (per plan)
**What is partially done** — built but not validated, or validated partially
**What has drifted** — where the implementation differs from the plan, even subtly
**What is blocked** — dependencies, missing data, unresolved decisions
**What is stale** — parts of the plan that no longer make sense given what we now know

Be specific. Vague status reports are useless.

---

## Step 4 — Result and signal review

Look at the most recent experimental results. Ask:

- Do these results support the direction the plan is heading?
- Is there any signal in the results that suggests the approach should change?
- Are there anomalies or unexpected patterns that have not been investigated?
- Has anything been validated that the plan assumed would be hard? Anything assumed easy that turned out hard?

---

## Step 5 — Plan validity

Give me an honest assessment:

- Does the original plan still make sense, or has something changed?
- Are there assumptions in the plan that the current results have undermined?
- Is there a more direct path to the stated goal than what the plan describes?

If the plan needs to change, say so and say why. Do not soften this.

---

## Step 6 — Recommended next action

Given everything above, what should we do next?

- State the single most important thing to work on
- State why it is the most important (not just the most obvious next step)
- State what completing it will tell us that we do not currently know

Then wait for my confirmation.
