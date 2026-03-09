---
description: "Structured debugging that forces me to form a hypothesis first. Prevents outsourcing reasoning to the model."
agent: "ask"
---

I have a bug. Before you give me any answer, I will state my hypothesis.

**My hypothesis:** ${input:hypothesis:State what you think the cause is and why}

**The error or unexpected behavior:**
```
${input:error:Paste the error message or describe the behavior}
```

**Relevant code:**
```
${input:code:Paste the relevant code}
```

Now:
1. Is my hypothesis correct? If not, explain specifically where my reasoning went wrong.
2. Walk me through the execution state step by step leading to this error. I want to understand it, not just fix it.
3. Give the fix, but explain *why* it works in terms of what was actually wrong — not just what changed.
4. What would have helped me catch this earlier? (test, assertion, logging, type hint, etc.)
