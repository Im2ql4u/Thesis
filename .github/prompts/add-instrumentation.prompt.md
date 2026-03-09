---
description: "Add structured logging and instrumentation to existing code so I can follow what it is actually doing at runtime. For maintaining insight into complex ML training loops, simulations, or pipelines."
agent: "ask"
---

Add instrumentation to the following code so I can follow what it is doing at runtime. I want to understand it deeply, not just see that it runs.

**Requirements for the logging:**
- Before each non-trivial computational block, log what is about to happen and why — the *intent*, not just the action.
- After key operations, log the shape, range, and a statistical summary (mean, std, min, max) of the most important tensors or arrays. Flag anything that looks numerically suspicious (NaNs, infs, values near zero when they should not be, exploding values).
- At the start and end of each epoch / major iteration, log a structured summary: loss components, gradient norms, parameter norms, learning rate, and anything domain-specific.
- Use a consistent log format that I can grep or filter. Prefix each log line with the name of the function and a step counter.
- Add assertions where a reasonable person would want the program to crash rather than silently continue (e.g. shapes, value ranges, physical constraints).
- Do not add logs that just repeat what is obvious from the code.

**What I am most uncertain or suspicious about in this code:**
${input:concerns:What do you want to watch most carefully?}

Code to instrument:
```python
${input:code:Paste the code here}
```

After adding instrumentation, tell me: what are the three most informative log lines you added, and what failure mode would each one catch?
