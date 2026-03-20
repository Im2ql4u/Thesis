---
description: "Deep problem interrogation before any planning or code. Challenges the problem framing, not just the solution. Use at the start of any non-trivial task."
agent: "ask"
---

Before any planning, before any code, I want you to interrogate this problem. The goal is to make sure we are solving the right thing, not just solving the stated thing well.

**Problem or task:**
${input:problem:Describe what you want to build or solve}

---

## Step 1 — Restate with scrutiny

Restate the problem in your own words. Then ask:

- Is this the actual problem, or a symptom of a deeper one?
- Is there a simpler version of this problem that would be sufficient?
- What would success actually look like — concretely, not abstractly?

If the restatement reveals ambiguity, ask me to resolve it before continuing.

---

## Step 2 — Assumptions register

List every assumption you are making that I have not explicitly stated. Categorize them:

- **Data assumptions** — format, size, completeness, distribution
- **System assumptions** — environment, dependencies, performance requirements
- **Intent assumptions** — what I actually want vs. what I said
- **Constraint assumptions** — time, complexity, maintainability

For each, rate your confidence: *likely correct / uncertain / could easily be wrong*. Ask me to confirm or correct the uncertain ones before proceeding.

---

## Step 3 — At least two approaches

Propose at least two meaningfully different ways to approach this. Not variations on one solution — genuinely different strategies.

For each:
- What it is in one sentence
- Its core strength
- Its core weakness
- What it requires of the codebase (new dependencies? restructuring? clean-slate?)

Include at least one approach that is simpler than what I probably have in mind.

---

## Step 4 — Where it will likely go wrong

Identify:
- The step most likely to fail silently (producing wrong results without error)
- The step most likely to fail loudly (crash, obvious error)
- Any place where the math or logic is non-trivial and easy to get subtly wrong
- Any place where validation will be hard — where it will be easy to convince ourselves it works when it doesn't

---

## Step 5 — Propose a plan

Break the chosen approach into the smallest sensible steps. Each step must be:
- Something I can understand without running the code
- Something I can verify independently
- Small enough that a mistake is easy to isolate

Mark any step that touches existing code in `src/` or `core/` — those need extra care.

---

## Step 6 — Wait

Present everything above, then wait for my confirmation before writing any code.

If I push you to just start, remind me that catching a wrong framing now is worth more than speed.
