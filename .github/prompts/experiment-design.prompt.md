---
description: "Design a scientific ML experiment properly before running it. Forces clarity on hypothesis, controls, and success criteria before touching code."
agent: "ask"
---

I want to run an experiment. Help me design it properly before I touch any code.

**What I want to find out:**
${input:question:What is the scientific question you are trying to answer?}

**My current hypothesis:**
${input:hypothesis:What do you expect to happen, and why?}

**What I am planning to do:**
${input:plan:Describe your current plan for the experiment}

Before I run anything, I want you to:

1. **Restate the hypothesis** as a falsifiable claim. What exact result would prove it wrong?

2. **Identify confounds.** What else could cause the result I expect to see, other than my hypothesis being correct? How can I control for them?

3. **Define success and failure clearly.** What specific metrics, thresholds, or patterns would constitute a clear result — positive or negative? "It performs better" is not sufficient.

4. **Identify the minimum experiment.** What is the smallest experiment that would give me a meaningful signal? Can I test the hypothesis on a simpler problem or smaller dataset first?

5. **Flag risks.** What is most likely to go wrong — computationally, numerically, or experimentally? Where should I add sanity checks before running the full experiment?

6. **Propose controls and baselines.** What should I compare against? What is the simplest baseline that would make a null result meaningful?

I want to run experiments that teach me something, not just experiments that produce numbers.
