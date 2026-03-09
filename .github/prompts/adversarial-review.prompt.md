---
description: "Ask the model to critically review code it just wrote, or code I paste. Forces honest critique rather than self-congratulation."
agent: "ask"
---

Review the following code as a senior scientist/engineer who did **not** write it and has no ego investment in defending it.

Be honest and direct. I want to know:

1. **What is wrong with this?** Include both correctness issues and design issues. Do not soften criticism.
2. **What would a careful reviewer flag in a code review?**
3. **What is the most likely way this breaks** — in edge cases, at scale, with unexpected input, or under numerical stress (if applicable)?
4. **Is there a simpler approach that achieves the same thing?** If yes, why was it not chosen?
5. **What assumptions are baked in** that I may not have noticed?
6. **What would I need to understand deeply** to be confident this is correct?

Do not tell me what the code does. I know what it does. Tell me where it is wrong, fragile, or unnecessarily complex.

Code to review:
```
${input:code:Paste the code here}
```
