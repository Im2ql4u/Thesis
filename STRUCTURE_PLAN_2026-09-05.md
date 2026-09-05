# Chapters 6–9: what is done, and what is left (2026-09-05, third revision)

Supersedes the previous two versions. The macrostructure is settled and should not be
rearranged again. What remains is reduction and polish.

---

## Verification first

Three claims from the latest external read did not survive checking. Recording them so
they are not chased:

- **The rotated page 84 is a build artifact.** `/Rotate 90` appears in no committed PDF
  and vanishes on any clean rebuild (`latexmk -C` then a full run). It comes from an
  interrupted incremental build, not from the source. No package in the document sets
  page attributes.
- Two earlier reads claimed a crude SR formulation in Chapter 6 and a contradictory
  $g(r)$ scoping sentence. Neither exists; both were fixed before those reads.

Everything else in that read checked out, including one item that matters more than the
rest.

---

## Done in this pass

| | |
|---|---|
| **Ch 8's over-unification** | It claimed "one mechanism explains every failure," directly contradicting Ch 7's own §7.4.5, which shows the wall below $\omega=0.0035$ is a Slater–Jastrow balance problem and *explicitly not the sampler*. Rewritten as **two walls**: statistical, which a better proposal moves, and representational, which it cannot. This was my error, introduced when Ch 8 was compressed, and it is the more interesting claim in its corrected form. |
| **Ch 7 spoiler in §6.3** | The backflow subsection announced the conventional-vs-copresheaf result a chapter before the controlled comparison that earns it. Cut to a description plus a forward pointer. |
| **Six-point Wigner synthesis** | Compressed to two paragraphs. The numbered checklist restated pages the reader had just finished. |
| **Total-spin caveat** | Moved from immediately after §6.2's opening — where it stalled the argument at the moment it took off — to the synthesis, where it qualifies the conclusion. |
| **"What follows, in short"** | Cut. It was the old *Headline findings* in better prose, and it named every result before §7.1 began. Replaced with a five-line road map. |
| **Conclusion** | Rebuilt. See below. |
| **Three defects** | A stray `\paragraph{Localization and energy partitioning.}` left by the splice; a sentence damaged by an earlier edit in §6.3.1; a cross-reference pointing at Ch 6 for a table that now lives in Ch 7. |
| **Tone** | "not by somebody else's node" → neutral phrasing; "The honest statement is…" → just the statement. |

### The conclusion now has a job

It was four "On the *X* side" blocks mirroring the old chapter order — the last remaining
replay of the results, three pages of it, now that Ch 8 is genuinely interpretive. Rebuilt
around two things nothing else in the thesis does:

**The loop.** Solve the state → the state teaches the physics → the physics builds the
sampler → the sampler reaches states the original method could not. Ch 6 finds the shell
geometry; Ch 7 §7.4 turns it into a proposal and gets to $\omega=0.0035$. That circle is
the thesis's most distinctive structural claim and it was stated nowhere. It now opens
the chapter and gives it its shape.

**The ledger.** Three tiers, kept apart: *established against something outside this
thesis* (only $N\le12$ at $\omega\ge0.1$, plus $N{=}2$ to $\omega=0.01$); *internally
consistent in ways that could have failed* (the deep-Wigner picture, the architecture
result — with the reasons the agreement is not cheap); *indicated and no more* (the
size-scaling, seeded only at $N{=}6$). The old opening promised exactly this separation
and then narrated findings instead. The preface's "wrong on the record" obliges it.

The four research questions are answered in a paragraph each, which absorbs the RQ
mapping that had been bolted on. The chapter closes back on the preface, grounded in its
actual account of the importance-sampling bug.

**Current shape:** Ch 6 = 15 pp, Ch 7 = 19 pp, Ch 8 = 5 pp, Ch 9 = 4 pp. Whole thesis 116
pages, clean build.

---

## What is left, in order

### 1. §7.3.3 "What worked and what did not" — the last lab-notebook stretch

Eight findings in positive/negative list form, and the densest unbroken prose in the
thesis. The content should stay; the form should not. Convert the enumeration to a
compact table:

| Attempt | Outcome | Mechanism | Consequence |

and keep prose only for the three that develop the argument — **Langevin**, **CG-SR**,
and the **$N{=}20$ reversal**. Those three are the ones the Discussion later builds on;
the rest (finite differences, Pfaffian, hard cusp gating, warm starts) are inventory and
read better as rows. Saves roughly a page and makes the negative-results programme
scannable, which is how a reader actually uses it.

### 2. `\paragraph{}` thinning in Ch 7

Ch 6 is now at 15 headers (was 33) and Ch 8 at 3 (was 23). Ch 7 sits at 33 after
absorbing collocation. Test for each: *could the sentence after this heading have followed
the previous paragraph?* If yes, delete the heading and keep the text. Concentrated in
§7.3.3–§7.3.4.

### 3. §6.2's diagnostic catalogue

Still 6 pp with 4 tables and 8 figures, prose walking a $6\times4\times5$ matrix. Build
the section on the two diagnostics that carry the claim — shell-topology fractions and
the $g(r)$ reconstruction — and move the Lindemann, quantile, FWHM and per-$N$ radial
tables to an appendix that the text points at once. Largest single cut still available;
removes no result. Left until now deliberately, because it is the most content-affecting
edit and is easier to judge against a stable chapter.

### 4. A read-through for voice

The register is close now. The remaining risk is that the vivid lines cluster. They
should be rare enough to land: *"A variational wavefunction is two things at once."*
*"The network that produced it is not."* *"Falling quiet and falling over are not the same
thing."* If a page has two, one of them is probably decoration. Also worth a targeted
sweep: sentences that tell the reader what the previous sentence meant.

### 5. Typesetting, last

Only once the pagination has stopped moving. Check the §7.2 heading break, the long
§7.4 running head, and widow/orphan lines around the tables in §7.3.

---

## Settled — do not revisit

- The three-chapter division of labour: what we obtained → why it works → what it means,
  with the conclusion doing the ledger and the loop.
- Collocation living in Ch 7, with its energy table in Ch 6 beside the SR–VMC table.
- Wigner physics before representations.
- Descriptive section headings, with the research questions traceable through Ch 9.
- Ch 6's evidence. The cuts above are to scaffolding and secondary tables, not results.
- §7.3's negative results, which stay prominent. They are the most distinctive material
  in the thesis and the part an examiner is most likely to remember.
