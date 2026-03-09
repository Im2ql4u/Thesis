---
description: "Request the simplest possible implementation of something — skeleton only, no error handling, no abstraction, no helpers. For learning and understanding core logic."
agent: "ask"
---

Implement the following, but with these strict constraints:

- **Write the most naive, unoptimized version that could possibly work.**
- No error handling unless it is part of the core logic.
- No abstraction, no helper functions, no classes unless strictly necessary.
- No comments explaining what the code does syntactically — I can read code.
- Do add a short comment before each logical block explaining **what problem it is solving at that moment**, not what it does line by line.
- The implementation should be as short as possible. If the core logic can fit in 20–40 lines, it should.

After the implementation, answer:
1. What is the one thing a person would need to understand to have written this themselves?
2. Where in this code is it most likely I would make a mistake if maintaining it six months from now?
3. What would a more complete version add, and why did you leave it out?

What to implement: ${input:task:Describe what to implement}
