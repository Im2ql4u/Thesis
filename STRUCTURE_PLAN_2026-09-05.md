# Chapters 6–8: structural read and cutting plan (2026-09-05)

Written after a full read of Chapters 6, 7 and 8 against the compiled page budget.
Companion to `AUDIT_TRIAGE_2026-09-05.md`, which covered correctness. This one is about
whether the thing can be read.

---

## The page budget, which is where the diagnosis starts

| | pages | |
|---|---|---|
| **Ch 6** Quantum dots | **21** | §6.1 energies 3 · §6.2 representations 4 · §6.3 Wigner 7 · §6.4 collocation 7 |
| **Ch 7** Tangent kernel | **12** | §7.1 Q1 6 · §7.2 Q2 **<1** · §7.3 Q3 1 · §7.4 frontier 2 · §7.5 synthesis 2 |
| **Ch 8** Discussion | **11** | §8.1 energies 2 · §8.2 Wigner 1 · §8.3 representations 2 · §8.4 collocation 6 |
| **Ch 9** Conclusions | **3** | |

Two numbers explain most of what feels wrong.

**Collocation occupies 24 pages across four locations** — §6.4 (7), §7.3–7.4 (3), §8.4 (6),
Appendices C+D (8) — while **Q2, one of the three questions the chapter is built around,
occupies less than one page.** The programme with the weakest results in the thesis has the
largest footprint, and the reader meets it four separate times.

**Chapter 8 contains 11 pages of prose with zero tables and zero figures**, discussing
tables and figures that are twenty pages earlier. That is the "leak between results and
discussion" — not a leak so much as a second pass over the same material with the evidence
out of reach.

---

## The core structural problem: three incompatible tables of contents

The reader is handed three different partitions of the same material, in sequence:

- **Abstract:** three questions — how far can a compact net go; what does message passing
  buy; can it be trained without a Markov chain.
- **Ch 6 intro:** a *different* three — how accurate is the method; what does the network
  learn; can it serve as an instrument. (And then §6.4, collocation, answers none of them.)
- **Ch 7 intro:** a *third* three — Q1 architecture, Q2 optimiser, Q3 paradigm. Plus "one
  idea, one lens", plus a "Discipline" paragraph, plus a "Headline findings" block.
- **Ch 8 intro:** four threads — energies, Wigner, representations, collocation. Which are
  §6.1, §6.3, §6.2, §6.4 in a different order.

That last point is the sharpest thing I can tell you. **Chapter 8 is Chapter 6's table of
contents with commentary attached.** §6.1→§8.1, §6.3→§8.2, §6.2→§8.3, §6.4→§8.4, in that
correspondence. It reads as restatement because structurally it *is* a restatement. A
discussion organised by the results chapter's own outline has, by construction, nothing to
say that the results chapter could not have said in place.

**The thesis already owns one good spine: the abstract's.** It maps cleanly —

| Abstract question | Where it is answered |
|---|---|
| How far can a compact, well-conditioned network go? | §6.1 energies, §6.3 Wigner physics |
| What does letting particles talk to each other buy? | §7.1 (Q1a + Q1b) |
| Can it be trained without a Markov chain? | §6.4 results, §7.3–7.4 mechanism |

Every later triad is a re-derivation of that one. If Ch 6 and Ch 7 both used the abstract's
questions as their organising frame instead of inventing their own, the "a lot going on and
not tied together" feeling would largely resolve without cutting a word. The threads are
already there; they are just relabelled at every chapter boundary.

---

## Recommendations, in order of value

### 1. Re-charter Chapter 8. Do not delete it — give it the job nothing else does.

Keep the chapter (a UiO examiner expects one), but stop organising it by the results. Its
new charter: **what would have to be true for this to be wrong, and what follows if it is
right.** That is:

- the boundaries of the evidence (where the benchmark stops; what gauge-like means for
  what can be claimed; the fixed-spin-sector caveat),
- the *mechanisms* behind the failures — why REINFORCE succeeds where residual losses
  fail, why Langevin refinement fails, the three-regime optimiser boundary, the catch-22,
- comparison with the MCMC-based literature,
- what the picture predicts for larger systems.

None of that is restatement. It comes to **5–6 pages instead of 11.**

What moves out, and where:

| Currently in | Goes to | Why |
|---|---|---|
| §8.1 fixed-node/variationality argument (~1 para) | directly under Table 6.2 | It is the reading instruction for that table. It is genuinely new and currently orphaned two chapters away. |
| §8.1 walk-through of Tables 6.1–6.2 | **cut** | Pure restatement. |
| §8.2 total-spin caveat | §6.3 opening | This is a real limitation on every structural claim in §6.3 and it should be read *before* them, not after. |
| §8.2 "decomposability, not sampling correctness" scoping | §6.3, next to the reconstruction result | Same reason. It is the best paragraph in §8.2. |
| §8.2 remainder | **cut** | Re-narrates §6.3 with the same numbers. |
| §8.3 mechanism/qualification paragraphs | keep in Ch 8 | Genuinely interpretive. |
| §8.3 walk-through of §6.2's tables | **cut** | Restatement. |
| §8.4 mechanisms (REINFORCE, Langevin, optimiser boundary) | §6.4 "What worked and what did not" | That section is already the right home and is the best-written part of Ch 6. |
| §8.4 comparison with MCMC results, implications | Ch 9 | It is a conclusion, not a discussion. |

**Saving: ~5–6 pages, and one fewer place the reader has to look for anything.**

### 2. Drop Chapter 7 from three questions to two.

Q2 gets less than a page against Q1's six. Either it is a headline question or it is not,
and at that length it reads as an aside wearing a section heading. But its finding — SR is
indistinguishable from Adam under |Ψ|² sampling and decisive under collocation — is not an
aside at all: it is the *hinge* between Q1 and Q3, because it says the conditioning is set
by the sampling measure rather than by the optimiser.

Merge Q2 and Q3 into one section: **"Conditioning: the sampling measure sets what the
optimiser can do."** The chapter then has two axes — **architecture** and **conditioning** —
which is what it actually argues. Two is also easier to hold than three, and it removes one
of the four competing triads.

While there: the chapter opens with *four* framing devices before any evidence — "One idea,
one lens", "The three questions", "Discipline", "Headline findings". That is roughly a page
of scaffolding. Keep "Headline findings" (it earns its place; a reader should be able to
stop after it) and fold the other three into the opening two paragraphs.

### 3. Honour the promise in the Ch 6 intro, or move §6.4.

The intro says "three questions organise the chapter" and then a fourth, seven-page section
appears that answers none of them. Cheapest fix: add the fourth question to the intro —
*and can it be trained without a Markov chain?* — which also brings Ch 6 into line with the
abstract's spine. One sentence, and the chapter's stated structure becomes true.

### 4. §6.3 is a catalogue. Make it an argument.

Seven pages, four tables, eight figures, no subsection longer than a page. Six diagnostics
× four particle numbers × five confinements is a matrix, and the prose currently walks the
matrix cell by cell. The reader cannot tell which numbers carry the claim and which are
corroboration.

Pick the two that actually carry it — **shell-topology fractions** and the **g(r)
reconstruction** — and build the section on them. The rest (Lindemann tables, quantile
tables, FWHM, the per-N radial tables) become one appendix table that the text points at
once. This is where the "cramped" feeling is worst and where the largest cut is available
without losing a single result.

### 5. The typographic cause of "hard on the eyes".

Ch 6 carries **33 `\paragraph{}` headers plus 19 subsections — 52 headed blocks in 21
pages**, roughly one heading every two-and-a-half paragraphs. Ch 8 carries 31 blocks in 11
pages. Headed blocks chop the page into cells and remove the connective tissue between
findings; used at this density they stop signposting and start fragmenting.

Ch 7 uses them well, because each one advances an argument rather than labelling a
measurement. Rule of thumb for the pass: a `\paragraph{}` earns its heading if the sentence
after it could not have followed the previous paragraph. Most of Ch 6's would fail that
test; deleting the header and keeping the text usually reads better immediately.

### 6. The Ch 6 / Ch 7 boundary needs one honest sentence.

§6.2 ("What the networks learn") and §7.1 ("Q1") are about the same two objects. The
advertised split is "what it is" vs "why", but that is not the real one. The real one is:

> §6.2 is the **anatomy of one trained solution**; §7.1 is a **controlled comparison
> between two architectures**. The first tells you what this state looks like; the second
> tells you which of its features are properties of the architecture rather than of the
> training run.

Say exactly that at the top of §6.2 and the leak closes. It also explains why the gauge
caveat has to appear in §6.2 before Chapter 7 exists — currently that import is unmotivated
and is a large part of why the caveat feels over-repeated.

---

## What I would not change

- **Chapter 6's length is not the problem.** 21 pages of primary results in a 120-page
  thesis is right. The problem is that 11 more pages restate them and 6 of those restate
  them a third time.
- **The collocation negative results.** §6.4's "What worked and what did not" is the most
  distinctive writing in the thesis and the section an examiner is most likely to remember.
  It should get *longer* relative to everything else around it, not shorter.
- **The epigraphs and the preface.** Seven epigraphs is one or two past where I would stop,
  but this is taste and no examiner will mark it.

---

## Order of work

1. **§8.1–§8.3 triage** (half a day). Move the four keeper paragraphs into Ch 6, cut the
   rest. This is the single biggest legibility gain and it is mostly deletion.
2. **§8.4 mechanisms into §6.4** (half a day). Then Ch 8 is down to its new charter.
3. **Rewrite the Ch 8 opening** to state the new charter. Ten minutes, and it is what makes
   the chapter feel like a chapter rather than an echo.
4. **Merge Q2 into Q3** and trim the Ch 7 scaffolding (half a day).
5. **§6.3 catalogue → argument** (one day; the largest single cut).
6. **`\paragraph{}` thinning pass** over Ch 6 and Ch 8 (half a day).
7. **Add the fourth question to the Ch 6 intro** and align all chapter framings to the
   abstract's three. One hour, done last, once the pieces are where they will stay.

Expected outcome: roughly **10 pages shorter**, one spine instead of four, and every result
stated once — in the place where its table is.
