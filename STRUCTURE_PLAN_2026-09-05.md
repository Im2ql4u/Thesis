# Chapters 6–8: merged structural plan (2026-09-05)

Supersedes the first version of this file. Merges my structural read with a second
external assessment, after verifying that assessment's factual claims against the source.
Companion to `AUDIT_TRIAGE_2026-09-05.md`, which covered correctness.

---

## First: two claims in the second assessment that do not hold

**It claims Chapter 6 carries an "older, cruder" formulation of the SR story** — "natural
gradients are beneficial when Fisher is well-conditioned, catastrophically wrong when it is
not" — contradicting Chapter 7's mature version. **Not true.** `results.tex:1296-1302`
reads:

> Ill-conditioning alone does not decide whether SR helps; what matters is whether the
> metric is *also* statistically resolvable. This gives a three-regime picture (developed in
> Chapter 7, Q2) … The failures above are the third regime, not the second.

That is the mature formulation, and it explicitly defers to Chapter 7. Commit `6c9b6bf`
("propagate the chapter-7 formulations backward") already did this job.

**It claims the Discussion says the shell reconstruction shows structure is "unambiguously
geometric … rather than a sampling artifact,"** contradicting Chapter 6's more careful
decomposability scoping. The phrase does not occur anywhere in the thesis, and §8.2 already
carries the careful version verbatim.

Both assessments so far have been partly reading a document that no longer exists. The
consequence for the plan is concrete: **there is no hunt for mutated claims to do.** The
duplication is real; the drift between retellings has already been fixed.

---

## Where the second assessment improves on my plan

### 1. Move the collocation programme out of Chapter 6 entirely

My first plan tried to save §6.4 in place by adding a fourth question to the chapter
introduction. That was papering over the problem. §6.4 is seven pages that answer none of
Chapter 6's stated questions, and Chapter 7 §§7.3–7.4 already owns the collocation
mechanism. Splitting one programme across two chapters is exactly what makes the reader
hold it in two places.

Consolidating gives three chapters with genuinely different jobs:

| | Its one job |
|---|---|
| **Ch 6** | What we obtained: trustworthy states, the physics they reveal, and how they represent it. |
| **Ch 7** | Why we obtained it, and where it fails: architecture, conditioning, sampling paradigm, and the MCMC-free programme end to end. |
| **Ch 8** | What it means, and what would have to be true for it to be wrong. |

**What did we obtain → why → what does it mean.** That is a better spine than either of my
four-triad diagnosis or the current arrangement.

### 2. Swap §6.2 and §6.3 — physics before representations

I had not proposed this. It is right, and there is decisive evidence for it that neither
assessment cited:

**§6.2 currently forward-references §6.3 three times.** Lines 251, 513 and 553 all invoke
"the physical crossover in Γ = V_int/T" — a quantity defined and measured in §6.3
(`sec:wigner-energy`, line 697). The representation section leans on a physical crossover
the reader has not yet met.

Reorder to **energies → Wigner physics → representations** and three things happen at once:

- the forward reference becomes a backward one, and the φ↔ψ transition at ω≈0.1 lands as a
  *finding* ("the network reorganises exactly where the physics does") rather than as a
  coincidence asserted in advance;
- the gauge caveat stops having to be imported from a chapter that does not exist yet —
  representations now sit directly against Chapter 7's controlled comparison;
- the chapter closes on the handoff that writes itself: *this anatomy describes what a
  successful solution looks like; it does not explain why message passing produces it.*

### 3. Retire the Q1/Q1a/Q1b/Q2/Q3 headings

On the page the hierarchy reads as a technical report and keeps reminding the reader they
are inside a framework rather than an argument. Descriptive headings instead:

- Message passing reorganises the variational geometry
  - The correlator: compression onto physical directions
  - The backflow: preventing collective collapse
- When natural gradients help, and when they hurt
- The sampling measure sets the reachable regime
- A Wigner-informed proposal, and the wall beyond it

### 4. Chapter 7's title over-promises "one lens"

§7.4 — Wigner-ring proposal, Dirichlet angular gaps, adaptive refitting, the
Slater–Jastrow cancellation — is physics-informed sampling, not tangent-kernel analysis.
The slogan has outgrown the science. Retitle to something like **"Why the method works:
geometry, optimisation, and sampling."** The tangent geometry stays the intellectual centre
without pretending every experiment is viewed through it.

### 5. The operative principle: transitions, not summaries

This is the best line in the second assessment and it is what makes every cut below safe:

> You often solve coherence by summarising. But summarising repeatedly creates bulk. You
> can get the same cohesion with one sentence at the boundary.

Every synthesis block that gets deleted is replaced by one transition sentence. That is the
rule for the whole pass.

---

## Where I would modify it

### A. Keep the collocation *energy table* in Chapter 6

The second assessment moves the entire programme. But "an accurate many-body wavefunction
can be trained with no Markov chain in the gradient" is one of the abstract's three headline
questions — it is a **result**, not a mechanism. Demoting Table 6.8 into a "why the method
works" chapter demotes the thesis's third contribution.

**Fix:** §6.1 presents *both training routes side by side* in one benchmark table — SR–VMC
and collocation, against the same references. The reader learns in one place how accurate
the method is, by either route. Everything else about collocation — reliability campaign,
negative results, gradient-quality chain, the frontier — moves to Chapter 7, which is where
"why it works and where it fails" belongs.

### B. Do not drop the research-question labels entirely

A UiO examiner may well check that the thesis answers its own stated research questions.
Descriptive headings, yes — but name the research question in the first sentence of each
section, and add an explicit RQ→section mapping to the Conclusion. Readable headings,
traceable answers.

### C. §6.3 is a catalogue, and neither assessment examined its table load

Seven pages, four tables, eight figures. Six diagnostics × four particle numbers × five
confinements is a matrix, and the prose currently walks it cell by cell, so the reader
cannot tell which numbers carry the claim and which are corroboration.

Build the section on the two that carry it — **shell-topology fractions** and the **g(r)
reconstruction**. The Lindemann tables, quantile tables, FWHM and per-N radial tables become
one appendix table the text points at once. This is the largest single cut available in
Chapter 6 and it removes no result.

### D. The appendices have to follow the move

Appendices C (the collocation–backflow catch-22, 6 pp) and D (post-catch-22 history, 2 pp)
are both collocation. If the programme consolidates into Chapter 7 they become Chapter 7's
appendices, and D.2's N=20 numbers should fold into Chapter 7's collocation results rather
than living alone in an appendix that the Discussion then mis-cites.

---

## The merged architecture

| | now | after | contents |
|---|---|---|---|
| **Ch 6** *The state and the physics* | 21 pp | **~13** | 6.1 Energies, both training routes (4) · 6.2 The Wigner-molecule crossover (5) · 6.3 What the network represents (4) |
| **Ch 7** *Why the method works: geometry, optimisation, sampling* | 12 pp | **~17** | short opening (1) · 7.1 Message passing reorganises the geometry (6) · 7.2 Conditioning: when natural gradients help (3) · 7.3 Training without a Markov chain: reliability and its boundary (5) · 7.4 The Wigner-informed proposal and the remaining wall (2) |
| **Ch 8** *Discussion* | 11 pp | **~6** | re-chartered: limits of the evidence, failure mechanisms, literature, what it predicts |
| **Ch 9** | 3 pp | 3 pp | + explicit RQ→section mapping |
| | **47** | **~39** | |

Chapter 7 absorbs seven pages from Chapter 6 and still ends up smaller than the two of them
were, because the collocation programme is stated once instead of three times. §7.5's
synthesis dissolves into Chapter 8's opening — the two currently do the same job.

---

## Order of work

1. **§8.1–§8.3 triage.** Move the four keeper paragraphs into Chapter 6 (fixed-node
   argument under the energy table; spin caveat and decomposability scoping into the Wigner
   section); cut the rest. Mostly deletion, biggest single legibility gain.
2. **Move §6.4 into Chapter 7**, keeping only the energy table in §6.1. Fold §8.4's
   mechanism subsections in at the same time so collocation is assembled once.
3. **Swap §6.2 and §6.3**, and rewrite the three section-boundary transitions. The forward
   references to Γ resolve themselves.
4. **Cut Chapter 7's opening** from four framing devices to three paragraphs; keep a
   compressed headline paragraph, drop "One idea, one lens," "The three questions" and
   "Discipline" as separate blocks.
5. **Retitle Chapter 7 and its sections**; add the RQ mapping to Chapter 9.
6. **§6.3 catalogue → argument**; secondary tables to the appendix.
7. **`\paragraph{}` thinning pass.** Chapter 6 currently carries 33 paragraph headers plus
   19 subsections — 52 headed blocks in 21 pages. Test: a `\paragraph{}` earns its heading
   only if the sentence after it could not have followed the previous paragraph.
8. **Typesetting sweep last**, once pagination is stable: check the §7.2 heading break and
   the long §7.4 running head.

---

## What I would still not change

- **Chapter 6's evidence.** The cuts above are to scaffolding and to secondary diagnostic
  tables, not to results.
- **The collocation negative results.** "What worked and what did not" is the most
  distinctive writing in the thesis. In Chapter 7 it should get *more* prominence, not less.
- **The preface and the epigraphs.** Seven is one or two past where I would stop, but no
  examiner will mark it.
