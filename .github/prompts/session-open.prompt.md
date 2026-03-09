---
description: "Start a new session by providing full project context as a brain dump. Follows antirez's method: include known bad approaches, good hunches, invariants, and goals."
agent: "ask"
---

I am starting a new session on my project. Here is my full context:

**What the project is:**
A reusable, general-purpose toolkit for representation learning in scientific ML. The core thesis is that structured mathematical representations — Fourier bases, wavelets, spherical harmonics, Helmholtz decomposition, and related frameworks — are underexplored and underutilized in modern ML, and that choosing representations informed by physics or domain structure gives meaningful advantages over purely data-driven approaches. This is a long-term engineering project, not academic research. The goal is a well-understood, well-structured repo I can use and extend across many scientific problems.

**Current state:**
There are roughly 80 existing notebooks covering a wide range of experiments and ideas across multiple phases. However, I do not have a clear mental model of the repo — I do not know what works, what the architecture looks like end to end, or what the key results actually are. The notebooks are exploration history, not a clean codebase. The project needs to be understood before it can be extended.

**What I am trying to do today:**
${input:goal:Describe the specific goal for this session}

**My current thinking — approaches I think might work:**
${input:good_hunches:Describe what you think might be the right direction, even if not fully formed}

**Approaches I suspect are wrong, and why:**
Treating the existing notebooks as the codebase and building on top of them without first understanding them. The risk is compounding confusion on top of confusion. At some point a clean rewrite or restructure, informed by what the notebooks actually discovered, is probably more valuable than continuing to extend them.

**Invariants that must not be violated:**
- I must understand what I build. Code or architecture I cannot explain is a liability, not an asset.
- The toolkit should be general and reusable across problems, not overfit to any single experiment or domain.
- Representations should be chosen for principled reasons — mathematical, physical, or empirical — not arbitrarily.

**What I do not understand well enough yet:**
What the existing 80 notebooks actually contain, which results are reliable, what the current architecture looks like as a whole, and where the real boundaries of this approach are. I am starting from a place of honest uncertainty about the state of my own project.

---

Given this context, before we start:
1. Confirm your understanding of what I am trying to do.
2. Flag any contradictions or gaps in my context that I should address.
3. Ask any clarifying questions that would meaningfully change how you approach this.
4. Then wait for me to confirm before we proceed.