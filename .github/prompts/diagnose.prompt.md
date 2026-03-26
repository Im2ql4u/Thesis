---
description: "Deep diagnosis when something is not working. Goes bottom-up through the stack. Never suggests surface fixes before structural causes are ruled out."
agent: agent
---

${input:problem:Describe what is happening and what has been tried.}

# Diagnose

> **How to use:** `@diagnose.md` when something is not working, results are suspicious, or progress has stalled. Describe what is happening and what has been tried. This prompt goes deep before going sideways.

---

## Posture

Your job is to find where the problem actually lives — not to suggest the most accessible fix. Small adjustments to late-stage variables (learning rate, epochs, regularization) are almost never the answer when something is fundamentally wrong. The broken onion cannot be fixed by better seasoning.

Before any fix is proposed, the problem must be located in the stack. Only then does a fix make sense.

---

## Step 1 — Characterize the failure honestly

Read the description of what is happening. Then answer:

- What exactly is the symptom? (metric, error, behaviour)
- When did this start? Did it work before and break, or has it never worked?
- What has already been tried, and what effect did each attempt have?

Distinguish between: never worked, worked then broke, works sometimes but inconsistently, works but results are suspicious. These point to different causes.

---

## Step 2 — Apply the diagnostic hierarchy — start at the bottom

Work through each layer in order. For each, state whether it has been verified, assumed, or unknown. Do not move to the next layer until the current one is addressed.

### Layer 1 — Data
The most common source of problems that look like model problems.

- Has the data pipeline been tested on a known input with a known expected output?
- Are there missing values, corrupt entries, or physically implausible values?
- Is normalization computed correctly — on training data only, then applied consistently?
- Are train/val/test splits respecting the correlation structure of the data? (Random splits on spatial or temporal data are almost always wrong.)
- Could there be any data leakage — any path from test to training?
- Are the distributions of train, val, and test sets actually comparable?

If any of these cannot be answered with confidence: stop here. Fix the data layer before touching anything else.

### Layer 2 — Implementation
Problems that produce wrong outputs without errors.

- Does the loss function compute what it is supposed to compute? Test it on a known analytical case.
- Is the metric implemented correctly? Test it on a case where the answer is known.
- Are there off-by-one errors, wrong aggregation axes, shape mismatches that get broadcast silently?
- Is the model actually receiving the right inputs in the right format?
- Are gradients flowing correctly? Check gradient norms for early layers.
- Is the validation loop using the model in eval mode? Is dropout disabled?

If any of these cannot be confirmed: stop and verify before continuing.

### Layer 3 — Architecture
Problems where the model structurally cannot learn what is being asked.

- Does the architecture have the right inductive bias for this problem?
- Is there a known failure mode of this architecture class that applies here? (e.g. FNO on irregular geometry, PINN on high-frequency solutions, transformers on small datasets)
- Is the model capacity appropriate — not too small to represent the function, not so large it cannot be trained with available data?
- Is the model actually capable of producing the range of outputs required?

Search for known failure modes of the architecture being used: `site:paperswithcode.com`, `site:semanticscholar.org`, include the architecture name and "failure" or "limitation" in the query.

### Layer 4 — Training setup
Problems in how the training is structured.

- Is the loss function the right one for this problem, or is it a proxy that can be gamed?
- Is there a verified baseline that the model is actually improving on?
- Is the optimizer appropriate? Is the learning rate in a sensible range?
- Are there gradient issues — explosion, vanishing — that would prevent learning?
- Is the training data sufficient and representative for the generalization being asked?

### Layer 5 — Hyperparameters
Only reach here after Layers 1–4 are verified.

- Learning rate, schedule, batch size, regularization strength
- Architecture-specific parameters that affect capacity or stability

---

## Step 3 — State the diagnosis

Given the investigation:

- Which layer does the problem most likely live in?
- What is the specific hypothesis?
- What evidence supports it?
- What evidence would falsify it — what single check would confirm or rule out this hypothesis?

Do not state a diagnosis with false confidence. If you are uncertain between two layers, say so and explain how to determine which one it is.

---

## Step 4 — Propose fixes in structural order

Propose at most three candidate fixes, ordered from most to least fundamental.

For each:
- What it is
- Which layer it addresses
- What problem it solves and why
- What it does *not* fix — what would still be wrong if this were the only change

Do not propose a Layer 5 fix when a Layer 1 hypothesis has not been ruled out. If you are proposing a small adjustment, explain explicitly why you are confident the layers below it are sound.

---

## Step 5 — What to check first

Given everything above, what is the single most informative check to run right now — the one that would most reduce uncertainty about where the problem is?

State it specifically. Not "check the data" — *"run the pipeline on sample X and verify the output matches expected value Y."*

Wait for results before proposing anything further.
