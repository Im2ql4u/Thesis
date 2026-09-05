# The de-narration pass: unified plan

The macrostructure is settled and is not touched by this. What follows is a sentence-level
pass whose primary operation is **deletion**.

---

## What the diagnosis actually localises

Everything in the external review checks out against the source. More usefully, the
problem has a sharp boundary:

- **All thirteen flagged headings were introduced by my own commits this session**
  (`4589f78`, `b341289`, `14a2c4f`). None is pre-existing.
- The tic counts I measured (143 instances of five mannerisms, thinned to 82) are almost
  entirely in prose written during the restructure.
- The chapters that read best — Appendix A, Methods §5, the CTNN section §2.4 — are the
  ones I barely touched.

So this is not a diffuse problem with the thesis. It is a specific problem with the
connective tissue I added, and I know exactly which passages those are. That makes the
pass bounded rather than a general "make it more academic" rewrite, which would damage the
good parts.

**One miss I should own:** I claimed in commit `2e37bbd` to have removed the generic
optimisation folklore from Chapter 2. One instance survives at
[theory.tex:491](Thesis/theory.tex#L491) — "Paradoxically, this noise often *helps*: it
prevents the algorithm from getting trapped in poor local minima" — and §§2.2.1–2.2.3 are
still the untouched textbook exposition, which is why Chapter 2 now oscillates between
overwritten and generic.

---

## The operational rule, with evidence for it

**Delete rather than rewrite.** I have direct evidence that rewriting fails here. Last
pass I turned

> Three unrelated puzzles, on the face of it. They have one answer, and it is geometric.

into

> These look like three unrelated puzzles, and they share one geometric answer.

That is the same rhetorical move at lower volume. The narrator is still standing between
the reader and the science. Correspondingly, my "not X but Y" count went 15 → 12 → 12
across two passes: rewriting preserved the construction while changing its clothes.

The classification to apply per sentence is the review's, and it is better than my
syntactic one:

**fact · definition · evidence · inference · limitation · causal transition ·
metacommentary**

Delete the last. Keep limitations — the distinction is between *stating a caution* (keep)
and *commenting on how cautious the writing is being* (delete):

| delete | keep |
|---|---|
| "The reader should carry forward the definitions and not expect the tables." | "We compute neither here." |
| "Two things matter in them, and neither is the raw energy." | "Below ω = 0.1 no DMC reference exists for N ≥ 6." |
| "the first needs its scope stated precisely" | "Because the sector histograms and g(r) use the same samples, the reconstruction tests decomposability rather than sampling correctness." |

---

## The one point that is scientific, not stylistic

The review's strongest observation, which I under-weighted in my own: **antithesis
manufactures false binaries.**

> "not the sampler, the ansatz"

is rhetorically satisfying and stronger than the evidence. The exact claim is that the
collapse persists under sampler reinitialisation and therefore *cannot be attributed to
sampler initialisation alone*. In my review I defended all twelve surviving "not X but Y"
constructions as carrying argument. Some do — "ill-conditioning is not the criterion,
estimability is" names two genuinely exclusive alternatives. Others compress a conditional
finding into an exclusive one. This needs a per-instance judgement, and the test is whether
the binary is real, not whether the sentence is good.

---

## Phases

### Phase 0 — Headings first (mechanical, ~30 min)

Do this before the prose, not after. Thirteen renames lower the register of the whole
document at once, and calm headings make the remaining performance audible in a way it
currently is not.

| now | to |
|---|---|
| What this leaves us needing | Requirements on the trial state |
| What learning means here | The variational learning objective |
| What survives into the rest of the thesis | Tangent-space diagnostics |
| What this is, and what it is not | Limits of component-level interpretation |
| Whitening, and where it pays | Natural-gradient preconditioning |
| What it achieves | Collocation performance |
| What worked and what did not | Ablations and failed variants |
| A gauge caveat, and what survives it | Gauge dependence and robust observables |
| What the energies establish | Scope of the energy benchmarks |
| How far the internal numbers can be pushed | Interpretation of internal diagnostics |
| Where MCMC-free training stops, and why | Boundaries of Markov-chain-free training |
| What the picture predicts | Implications and extensions |
| The ledger | Evidential status of the findings |
| A closing thought | Concluding perspective |

Also: Chapter 7's title, *Why the method works*, is declarative in the same way.
**Architecture, optimisation, and sampling geometry.**

Keep *Networks that can be differentiated twice* — it names the actual constraint and is
informative rather than rhetorical.

### Phase 1 — Delete the roadmaps and metadiscourse (highest yield)

Every "what follows" paragraph, every advance statement of a chapter's job, every "the
reader should". Specifically:

- Ch 1 opening: keep the Hamiltonian and the three difficulties; delete "and this chapter
  is about those three" and the entire section-by-section roadmap.
- Ch 2 §2.1: delete "saying so at the outset avoids a misunderstanding…" and the whole
  "So the chapter that follows…" roadmap paragraph.
- Ch 2 §2.6.4 and §2.7 both exist to map Theory onto the later thesis. Keep one.
- Ch 3 opening: delete "The previous chapter ended with four facts… This chapter builds
  the state those facts allow… Very little of what follows is a free choice." Open on the
  constraint itself.
- Ch 6 opening: keep the answer/instrument line *only if* the three-question roadmap goes.
- Ch 7: delete the road-map paragraph entirely.
- Ch 8: delete the opening two sentences and "Four things remain to be said." Begin on the
  synthesis.

### Phase 2 — Stop announcing the unity

It is currently declared five times: Introduction ("These are not four separate stories"),
Ch 7 ("three unrelated puzzles… one answer"), Ch 8 ("turned out to be one geometry"),
Ch 9 ("They are one geometry"), and Ch 2 §2.7's dictionary paragraph.

State it fully **once**, in the Discussion. Elsewhere make the connection locally and
causally — *the empirical QGT depends on the sampling measure, so loss of overlap at weak
confinement degrades both the gradient variance and the reliability of SR* — which
demonstrates the unity instead of asserting it.

### Phase 3 — Remove advance answers from Results

Ch 7 §7.1 currently gives the answer before the comparison. Ch 6 §6.1 says "the result to
notice is…" before the table. Reorder to **comparison → observed difference →
interpretation**. This is the change that converts the reader from audience back into
participant.

### Phase 4 — The Introduction

The four research questions currently contain their own answers, which removes the
movement from the entire second half. Turn them back into questions; the contributions
list immediately following already carries the answers, so nothing is lost and the
ordering becomes honest. Then cut the diagnostics inventory by half and compress the
roadmap to three sentences.

### Phase 5 — Verdict words

Present counts: *exactly* 24, *decisive* 11, *clean* 11, *genuine* 10, *precisely* 9,
*establishes* 8, *intrinsic* 6. These grade the author's own evidence. "This establishes a
clean separation of concerns" describes near-unity CKA between two feature
representations; *clean separation of concerns* is the interpretation, not the
measurement. Target: down by roughly two-thirds, each survivor justified by the evidence
actually being exact.

### Phase 6 — The Abstract (two problems, one rewrite)

It is simultaneously over-persuasive (four separate declarations of unity, "it turns out",
"Underneath both is a single mechanism") and **factually stale**: no `Var(E_L)`, one wall
where the thesis now argues two, and no loop. Both have to be fixed in the same pass or
the edit is wasted.

### Phase 7 — Methods and Appendices

Methods carries 16 sentences over 45 words and 11 over 60 — more than any other chapter —
and the worst is 93 words with a three-item enumeration nested inside a colon inside a
parenthesis ([method.tex:555](Thesis/method.tex#L555)). Break the nesting; that is the
clarity problem, not length as such. Passive density there is 23% against Results' 12%.

Appendices: delete "The central point for learning is:" (B), "Bottom line." (B), the
"worked negative result — the kind that redirects a research programme" framing (C), and
"hero runs" (D). Also §§2.2.1–2.2.3, which are the surviving generic patch, compress to
two paragraphs.

### Phase 8 — Sentence-level sweep

Only now, and only on what survives. Replace rhetorical transitions with causal ones:

> ~~So much for the state seen from outside. How much of that did the network need to hold
> internally?~~
>
> The increasing real-space complexity does not produce a corresponding increase in the
> dimensionality of the learned correlator representation.

One sentence closes the previous section and opens the next, and the question arises from
the result rather than from the narrator.

---

## What survives

The review's budget is right: **two or three departures in the whole document.** My
nominations:

- "Representability is cheap; trainability is the difficulty." *(Introduction — conceptually
  central and genuinely economical.)*
- "The order is there; the density hides it." *(§6.2 — concise, physical, and it explains
  why the diagnostics exist.)*
- "Ill-conditioning is not the criterion. Estimability is." *(Ch 7/9 — condenses a real
  result, and the binary is a true one.)*

Everything else goes to plain, including lines I defended last time: "Falling quiet and
falling over are not the same thing", "A variational wavefunction is two things at once",
"the shape is a loop".

The Preface keeps its personality. It is the one place where a first-person voice is
legitimate, and its problem is not that it is personal but that its habits leaked into the
body. Two small cuts there: "That is worth remembering, and easy to forget" (the sentence
before it already says it), and possibly "That interest is most of what encouragement
actually consists of."

---

## Where I differ from the review

**On deleting punchlines that carry content.** It wants "Neither statement is about quantum
dots" cut. That sentence makes a real generalisability claim. The fix is to lose the punch
and keep the content: *"Neither statement is specific to quantum dots."* The test should be
**does deleting this lose information**, not **is this a punchline**.

**On the Conclusion's motifs.** It says keep the loop or the record, not both, and drop the
ledger. I would keep the **ledger content** — the three-tier separation of established /
internally consistent / indicated is the most useful thing in the chapter — under the
boring heading *Evidential status of the findings*, and drop the word "ledger" and the
metaphor around it. That keeps the substance and loses the framing device.

**On Chapter 6's opening line.** It says keep "A variational wavefunction is two things at
once" if the chapter is calmed. Under a two-to-three departure budget it does not make the
cut, and the sentence is doing less work than the two I nominated.

---

## Stopping condition

Without a target I will either stop early, as I did last time, or over-grind into a new
mechanical habit. Checkable:

| | now | target |
|---|---|---|
| metadiscourse instances | ~25 | ≤ 3 |
| verdict words (exactly/clean/decisive/genuine/precisely) | 65 | ≤ 22 |
| enumerative announcements | 9 | ≤ 4 |
| body word count | — | −10% to −15% |
| "special" sentences | ~15 | 3 |
| Methods sentences > 60 words | 11 | ≤ 2 |
| passive density, Theory + Methods | 24%, 23% | ≤ 16% |

The build must stay clean at every checkpoint, and no scientific content may be removed —
only the narrator.
