---
description: "Refactor code while keeping me in the loop on every decision. Prevents silent refactoring that leaves me not understanding what changed or why."
agent: "ask"
---

I want to refactor the following code, but I need to understand every decision you make.

**Before touching anything:**
1. What are the problems with the current code? Be specific — not just "it could be cleaner."
2. What would you change, and in what order? Propose a sequence of small, independent refactoring steps.
3. For each step, explain what design principle it is applying (e.g. single responsibility, reducing coupling, improving testability, etc.) and why it matters here.

**Then, do the refactoring in steps.** After each step, stop and wait for my confirmation before continuing. For each step:
- Show only the changed part, not the whole file unless necessary.
- Explain what changed and why in one or two sentences.
- Tell me if any behavior changed (even slightly), or if this is a pure structural change.

**Hard constraints:**
- Do not change behavior. If you think a behavior change would be an improvement, flag it as a separate suggestion, do not just do it.
- Do not increase the line count unless there is a clear reason.
- Do not add dependencies.

Code to refactor:
```
${input:code:Paste the code here}
```

Goal or motivation for refactoring:
${input:goal:Why are you refactoring this? What problem does the current structure cause?}
