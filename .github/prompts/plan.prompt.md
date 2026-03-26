---
description: "Elaborated engineering plan. Run after brainstorm/diagnose/review has established direction. Produces a full specification: architecture, training design, evaluation protocol, step-by-step with verifications, risk register. Use Copilot agent mode."
agent: agent
---

${input:context:Briefly describe what was decided in the preceding brainstorm or diagnose session, and what this plan needs to cover.}

# Plan

> **How to use:** `@plan.md` after a brainstorm, diagnose, or review session has already established what we are doing and why. Invoke in **Plan Mode** (`Shift+Tab` in Composer) so Cursor researches the codebase before committing to steps. This prompt produces an elaborated engineering specification — not questions, not re-examination, but a committed, detailed plan ready to implement.

---

## Precondition

The thinking has already been done. A direction has been chosen, the problem has been framed, the diagnosis is understood. This phase does not re-examine whether the direction is right — it plans how to execute it rigorously and completely.

If direction is not yet clear: use `@brainstorm.md`, `@diagnose.md`, or `@experts/framing.md` first.

---

## Step 1 — Read the full project context

Before writing anything, read:

- `README.md` — stated goal and current approach
- `DECISIONS.md` — active decisions that constrain what we do
- `JOURNAL.md` — what has been tried, what was found, what failed
- `SESSION_LOG.md` — most recent conclusions and what is pending
- The full `src/` and `core/` structure — what exists, what can be reused, what must be built
- Relevant results in `results/` — current state of evidence

Every step must be consistent with what already exists and what has already been decided.

---

## Step 2 — State the plan objective precisely

One paragraph answering:

- What specifically will be built or changed by the end of this plan?
- What question will be answerable that is not currently answerable?
- What does a successful outcome look like — concretely, in terms of code and examinable results?
- What does a failed outcome look like — how will we know if this plan did not work?

This is the success criterion. Every step must serve it.

---

## Step 3 — Foundation verification steps

Before new work, list what must be verified true for the plan to stand on solid ground. These are checks on existing work, not new tasks.

For each unverified assumption beneath the current task:

- **Data** — pipeline produces correct output on a known input? Splits valid and correlation-respecting?
- **Implementation** — existing code computes what it claims? Silent errors possible?
- **Baseline** — verified baseline result exists to compare against?

If any are unverified, they become the first steps in the plan — before any new modelling work. Building on an unverified foundation is how weeks of work become meaningless.

---

## Step 4 — Architecture and approach specification

State explicitly, with reasoning for each:

- What architecture will be used and why — referencing the specific problem structure, not general reputation
- What the model is specifically being asked to learn — the scoped residual task after any decomposition, not the full problem
- What the inputs are precisely: shape, dtype, normalization applied, any derived features and how they are computed
- What the outputs are precisely: shape, meaning, units, how they will be evaluated
- What physical or domain constraints apply and exactly how they will be enforced (hard constraint in architecture, soft penalty in loss, or not enforced and why)
- What the baseline is and exactly how it will be computed — the same data, same splits, same metric

If any of these are still open, flag them. They must be resolved before implementation begins.

---

## Step 5 — Training design specification

State explicitly, with reasoning:

- **Loss function** — what it measures, why it is appropriate. If multi-term: exact weighting strategy, how balance will be monitored during training, what imbalance looks like and how to fix it
- **Optimizer** — which one and why. Initial learning rate with reasoning.
- **Learning rate schedule** — warmup duration if any, decay strategy, rationale
- **Gradient clipping** — threshold and why
- **Batch size** — with reasoning about the tradeoff for this problem
- **Pretraining or curriculum stages** — in order, with what each stage trains and when to move to the next
- **Training monitoring** — beyond loss: what gradient norms, activation statistics, or loss component ratios to track and what values indicate a problem
- **Stopping criterion** — specific, not "until convergence"

Nothing here should be a default without a reason.

---

## Step 6 — Validation and evaluation specification

State explicitly:

- How train/val/test splits are constructed — the method and why it respects the correlation structure of this specific data
- What metrics will be reported, in what units, and why they measure what actually matters for this problem
- What the baseline achieves on these metrics — the floor everything is compared against
- What subsets will be evaluated beyond the aggregate: worst cases, rare events, specific spatial regions, specific time periods
- How many seeds will be run — variance must be reported, not a single run
- When the held-out test set will be touched — once, at the end, after all decisions are made

---

## Step 7 — Step-by-step implementation plan

Break the work into the smallest steps that are each independently verifiable. No step should be evaluable only in hindsight.

For each step:

```
### Step N — <title>

**What this does:** <one sentence — the purpose, not the method>
**What it produces:** <exactly what will exist when this step is done>
**Verification:** <the exact check that confirms this step is correct —
  a unit test with known answer, an analytical sanity check, a plot,
  a comparison to a known value. Not "looks reasonable" — specific.>
**Files touched:** <src/ or core/ files created or modified>
**Risk:** <what is most likely to go wrong here and what the symptom would be>
```

Order steps so each is verified before the next begins. If a later step cannot be tested without the whole pipeline, break it down further.

---

## Step 8 — Results examination plan

For each training run or significant computation in the plan, specify in advance:

- What outputs will be produced and where they will be stored
- What will be examined in those outputs and how
- What a good result looks like — specifically, not vaguely
- What a suspicious result looks like — and what the investigation would be
- What a clearly failed result looks like — and what it would imply about the plan

This is written before the run, not after. Having an expectation before seeing results is what makes the examination honest. A result that matches expectations is informative. A result that surprises requires explanation.

---

## Step 9 — Known risks

For each major risk to the plan:

- **What it is**
- **Which step it would affect**
- **How it would manifest** — what would we observe?
- **Early warning sign** — what would we see before it becomes a problem?
- **Mitigation** — what to do if it manifests

Cover at minimum: silent failures, known architectural failure modes, data assumption risks, training instability risks.

---

## Step 10 — What this plan does not address

State explicitly what is out of scope and why. Prevents scope creep and makes clear what the next plan must cover.

---

## Step 11 — Wait for confirmation

Present the complete plan.

Ask: *"Does this match your understanding of what we're building? Is anything here wrong, underspecified, or missing?"*

Do not begin implementation until confirmed.
