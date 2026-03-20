---
description: "Purposeful adversarial review. Finds weak spots in results, validation methodology, and implementation logic. Not generic criticism — targeted probing of what is most likely to be wrong."
agent: "agent"
---

Enter scrutiny mode. Your job is to find what is wrong, weak, or unconvincing — not to be negative, but because finding it now is better than finding it later or not at all.

**What to scrutinize:**
${input:target:Describe what you want scrutinized — a result, a method, a piece of code, or a claim}

---

## Posture

You are not trying to tear things down. You are trying to find the load-bearing assumptions and check if they hold. Every critique should be specific and falsifiable — if you cannot say what evidence would change your critique, the critique is not useful.

---

## Step 1 — Understand what is being claimed

State in your own words:
- What result or conclusion is being asserted
- What that claim requires to be true (technically and statistically)
- What the stated or implied comparison baseline is

If the claim is vague, say so and ask me to sharpen it before proceeding.

---

## Step 2 — Validation methodology

This is the highest-risk area. Probe specifically:

- **Data leakage** — is there any path by which information from the test set influences training, preprocessing, normalization, or feature construction?
- **Metric validity** — does the reported metric actually measure what the claim requires it to measure? Are there conditions under which this metric looks good but the model is useless?
- **Baseline fairness** — is the comparison fair? Same data, same preprocessing, same evaluation protocol?
- **Distribution shift** — do the evaluation conditions match the conditions the model would face in actual use?
- **Overfitting to the validation set** — how many times has the evaluation set been used to make decisions? Is the reported performance on truly held-out data?
- **Statistical significance** — is the improvement within noise? Has variance across runs been measured?

For each issue you find, rate severity: *minor / meaningful / potentially invalidating*.

---

## Step 3 — Implementation against intent

Read the relevant code and check:

- Does the implementation actually compute what the method description says it computes?
- Are there off-by-one errors, incorrect aggregations, or implicit assumptions in the code that differ from the stated method?
- Are there silent failure modes — places where wrong inputs produce wrong outputs without error?
- Are numerical issues possible (overflow, underflow, division by small numbers, log of zero)?

---

## Step 4 — Result plausibility

Look at the actual outputs and ask:

- Does this result make physical or logical sense? (For domain-specific work: does it satisfy known constraints?)
- Is the magnitude of improvement plausible? Suspiciously large gains often indicate a methodological flaw.
- Are there patterns in the errors or residuals that suggest the model has learned something spurious?
- What is the model doing on the hardest or most important cases, not just on average?

---

## Step 5 — What would break this

Identify the 2–3 things that, if wrong, would most undermine the result. For each:

- What is the assumption
- How likely is it to be violated
- What would we need to check to find out

---

## Step 6 — Verdict

Give me an honest overall assessment:

- What is genuinely solid and can be trusted
- What is uncertain and should be flagged when presenting this work
- What is potentially problematic and should be investigated before drawing conclusions

Do not soften this. If results are weak, say so. If there is a real flaw, name it.
