---
description: "Architecture or method decision for physics-informed ML or scientific computing. Forces tradeoff analysis before committing to an approach."
agent: "ask"
---

I need to decide between approaches for a scientific computing / physics-informed ML problem. Help me think through this rigorously.

**The problem:**
${input:problem:Describe the physical or mathematical problem you are solving}

**The approaches I am considering:**
${input:approaches:List the approaches, even if vaguely — e.g. PINN vs neural operator, CTNN vs BF, FEM vs data-driven, etc.}

For each approach I am considering, and any others you think are relevant:

1. **What are the theoretical assumptions?** What must be true about the problem for this approach to work well?
2. **Where does it fail?** What problem characteristics make this approach break down or perform poorly?
3. **What is the computational cost** relative to the others at inference and training time?
4. **What does the literature say?** Are there benchmark problems or papers that directly compare these approaches on similar problems?
5. **Steelman the alternative.** For each approach, give the strongest case for why a careful practitioner might prefer it over the others.

After the analysis, give me your honest recommendation — but flag your uncertainty. If the right answer depends on something I have not told you, tell me what that is.

I want to make this decision based on understanding, not just on which approach sounds more sophisticated.
