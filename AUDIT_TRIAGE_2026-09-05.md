# Triage of the external audit (2026-09-05)

I checked every claim in the audit against the LaTeX source, the generated tables, and
the run data in `results/`. This document sorts the 40 items by **what an examiner will
actually press on**, separates real defects from matters of taste, records the items the
audit got wrong, and adds one item it missed that is more serious than anything on its list.

Verdict on the audit: **roughly 30 of 40 are real.** The numerical/cross-reference cluster
(items 9–29) is almost entirely correct and mechanically fixable. The register cluster
(1–8) is real in kind but inflated in degree. Four items are wrong or misread. The two it
flags as hardest (30, 31) are correct and are indeed the ones that matter — but there is a
worse one underneath them.

---

## Tier 0 — Found during verification; not in the audit

### T0.1 The kinetic→Coulomb ablation claim is contradicted by its own data file

`Thesis/results_kernel.tex:274-278` (§7.1.2, and repeated in the headline block at line 93,
which feeds the abstract) states:

> Ablating the backflow (Δx = 0, **sampler re-optimised**) and splitting the recovered
> energy into kinetic and Coulomb parts gives two seed-stable facts. (i) A
> **kinetic→Coulomb crossover**: at ω=1 the backflow buys kinetic energy; at ω=0.01 the
> gain is entirely Coulomb (ΔV_Coul ≈ +0.017, ΔT ≈ 0).

The experiment behind Fig. 7.4 is `scripts/run_message_ablation.py`. It draws **one** sample
set from the full model (`x = s.sample(...)`, line 94) and evaluates all four arms on those
same configurations. Consequently, in `results/analysis/2026-07-02_message_ablation/ablation.csv`:

- `V_trap` and `V_coul` are **bit-identical across all four arms** within every checkpoint.
  They cannot change; only `T` can.
- At N=6, ω=0.01, seed 0: `msg+bf` T = 0.0334989, E = 0.6903537; `msg+nobf` T = 0.0336120,
  E = 0.6904668. So **ΔV_Coul = 0 exactly** and **ΔT = ΔE = +0.000113 Ha**.
- 0.000113 / 0.69035 = **+0.0164%**. The "+0.017" in the text is the ablation cost **in
  percent**, relabelled as a Hartree value and attributed to the Coulomb channel.

So three things are wrong at once:

1. **"sampler re-optimised" is false.** It is a fixed-common-probe ablation on the full
   model's samples. (That is the *right* design for a clean comparison — it just has to be
   described accurately.)
2. **The kinetic→Coulomb crossover is not measurable in this experiment.** By construction
   every arm difference is 100% kinetic, at ω=1 and at ω=0.01 alike.
3. **"the gain there is Coulomb (not kinetic) correlation"** (line 93, and its echo in the
   abstract's mechanism sentence) has no supporting data anywhere in `results/`. I searched
   the whole tree: `ablation.csv`/`summary.json` are the only Coulomb-resolved ablation
   artefacts that exist.

**Fix.** Delete claim (i) and the "sampler re-optimised" parenthetical. Replace with the
statement the data does support, which is still worth making: *on a common probe drawn from
the full model, deleting the CTNN backflow costs 5.6% of the energy at ω=1 and 0.017% at
ω=0.01; the entire measured effect is kinetic-energy (weak-form) reduction, because the
probe measure is held fixed.* If you want the kinetic/Coulomb split claim, it needs a new
run in which each arm is re-relaxed and re-sampled — that is a day of compute, and I would
not attempt it this week. Cut the claim instead.

**Severity: highest.** It is a mechanistic claim in the headline findings block, it
propagates to the abstract, and the contradicting file is in the repository.

### T0.2 The same numbers force an honest reframing of Q1b

Fig. 7.4 says deleting the CTNN backflow at ω=0.01 costs **+0.017%**. Table 7.1 says a
separately-trained **conventional** backflow costs **+0.58%** at the same point. The audit
(item 18) flags this as an unreconciled tension and guesses wrongly at the cause. The
resolution is clean, and you should state it:

- Fig. 7.4 is an **instantaneous** ablation of a converged CTNN model.
- Table 7.1 is a **retrained** comparison of two architectures.
- Therefore: at the crystal the CTNN backflow is nearly **inert** (removing it is free),
  while the conventional backflow is not merely inert but **actively harmful** — it
  collapses onto one collective mode and drags the nodes with it.

That is a defensible and interesting finding, but it is *not* the story currently told
("message passing keeps the backflow working at the crystal"). What message passing buys in
the backflow at ω=0.01 is **the absence of a pathology**, not a positive energetic
contribution. An examiner who puts Fig. 7.4 next to Table 7.1 will arrive here in about
ninety seconds. Get there first, in your own words, in §7.1.2.

---

## Tier 1 — Substantive; an examiner will press on these

### T1.1 (audit 30) Var(E_L) is load-bearing and has no reported value

Confirmed. `\Var` appears at `results_kernel.tex` lines 53, 161, 431, 432, 441, 575 — six
mentions, **zero numbers**, no table, no figure. Line 575 (§7.5) reads "The single
cross-cutting discriminator, robust where dimension and rank are not, is Var(E_L), lower for
the message-passing ansatz at every ω." Line 161 asserts the CTNN "carr[ies] the lower
Var(E_L)" without a value.

This is the worst *structural* problem in the thesis: you build the epistemology on three
invariants, declare one of them the master discriminator precisely because the others are
gauge-like, and then never report it.

**Fix, in order of preference:**
1. **Compute it.** Var(E_L) is a by-product of any VMC evaluation and you have the
   checkpoints. A four-row table (CTNN vs DeepSet × ω) placed in §7.1.1 fixes items 30, 31
   and 32 at once and is the highest-value half-day in the whole revision.
2. If the numbers do not in fact come out lower for the CTNN at every ω, **say what they
   are** and weaken line 575 to match. That is still a far better position than the
   current one.
3. Only if (1) is impossible: strike "the single cross-cutting discriminator" from §7.5 and
   let energy + ablation carry the chapter. Do not leave the sentence as it stands.

### T1.2 (audit 31) State overlap is declared an invariant and never used for Q1

Confirmed. S² appears only at N=2 (>0.997) and in the Q3 collocation-vs-VMC test. Of the
three declared invariants, Q1 rests on two. Either compute S² between a trained CTNN and a
trained DeepSet at matched energy (cheap — same estimator you already use in §7.3), or
narrow the §7.5 sentence to "energy and ablation for Q1; overlap for Q3."

### T1.3 (audit 16) Backflow rank is "~N" in Ch. 7 and "≈2N" in Ch. 6

Confirmed and unambiguous. `results.tex:242` says rank ≈ 2N. `results_kernel.tex` says
"full rank ~N" at lines 90, 265, 270, 272 while quoting 10/21/36 at N = 6/12/20. The
displacement field lives in 2N dimensions; 2N = 12/24/40; observed 8.8–9.8 / 18.2–20.8 /
33.8–36.9. So the honest statement is **"close to the 2N maximum, reduced by the
centre-of-mass projection"** — you enforce zero-mean displacement (`method.tex`, the
`\eqref{eq:bf-com}` mapping), which removes 2 of the 2N directions, giving a ceiling of
2N−2 = 10/22/38. That matches the data well and *explains* it. Currently the −2 is never
mentioned anywhere. One sentence in §7.1.2 turns an apparent error into a correct
mechanistic detail.

### T1.4 (audit 32) The abstract's Q1b headline is a gauge-like quantity

"the message-passing backflow holds full rank" is exactly the class of quantity §7.5
declares gauge-like and says the conclusions do not rest on. The energy evidence (Table 7.1)
stands on its own, so the *finding* survives — the *presentation* contradicts your own
caveat. **Fix:** lead the abstract sentence with the energy (conventional degrades to +0.5%
at the crystal, CTNN holds ~0.02%) and give rank as the structural signature that explains
it. Same edit in the Ch. 7 headline block and the conclusion.

### T1.5 (audit 19, 20) ESS is "degenerate" and "usable" at the same value

Real, and there are two separate problems tangled together.

**(a) Undeclared denominators.** §7.3 quotes "0.11% of the batch (about 5 of 4096)";
§6.4.4 quotes "ESS collapses to 1–6 (out of the same 65k proposal)"; §6.4.3 quotes "5–15
per epoch (out of 4096 kept points)"; Table 7.3 quotes 1.5 of 4096; §8.4.3 quotes "5–50".
These are not all the same normalisation. **Fix:** pick one (ESS as a count out of the 4096
*kept* points), state it once in §5.x, and convert every quoted figure.

**(b) A genuine and unaddressed tension.** §7.3 says at N=6, ω=0.01 the estimator is
"degenerate — a hard limit" and S² → 0 under Adam. Table 6.9 shows Adam collocation at that
exact point reaching +0.24 to +0.29% on all three seeds. Both are true. The reconciliation
you have available:
- §7.3 analyses the **static origin-centred** proposal; Table 6.9's robust recipe uses
  **16× oversampling plus a GMM refitted every 30 epochs**. Different samplers. Table 6.9's
  *baseline* rows at ω=0.001 (+3.9 to +7.2%) are exactly the degeneracy §7.3 describes.
- More interesting, and worth stating plainly: at ω=0.01 the energy is within 0.3% while
  the state overlap has gone to ~0. **Energy alone does not certify the state.** That is a
  finding, and it supports your whole invariants argument. (Add the honest caveat that S²
  is itself a ratio estimator and degrades when the two states barely overlap — so "≈0"
  should be read as "unresolvable", not "orthogonal".)

One paragraph in §7.3 handles both. This item converts from a weakness into a result.

### T1.6 (audit 7, 33) "the physical energy-scaling law"

Partly real. The classical Wigner exponent ω^(2/3) *is* an independent theoretical
constraint, and your fitted 0.689 sits close to it — that is a legitimate consistency
check. But: the prefactor 0.690 is anchored at ω = 0.01 where no external reference exists,
`results.tex:211` calls it "derived in §6.3.3" (it is **fitted**, not derived), and §8.4/§9
call it "the physical energy-scaling law."

**Fix (three words, three places):** "the fitted Wigner scaling curve, whose exponent 0.689
is close to the classical 2/3". Ch. 7's preamble already says the deep-Wigner errors are
"internal consistency, not absolute accuracy" — the rest of the thesis just needs to use
the same language.

### T1.7 (audit 40) The unused theory, and a promise the results never keep

Confirmed. §1.5.3 "Role in this thesis" promises "high-accuracy FCI feasible for moderate M"
as a benchmark for N = 2, 6. **No FCI number appears anywhere in the thesis.** HF is
likewise never used — your determinant is built from harmonic-oscillator orbitals.

Do **not** delete §§1.3–1.5; second quantization and FCI are expected background in a UiO
computational-physics thesis and their absence would be noticed. **Do** rewrite §1.5.3
(four sentences) to say what is true: these methods define the language of correlation
energy and set the accuracy scale the thesis is measured against; we do not compute them
here, and our external reference is DMC. Removing an unkept promise is worth more than the
section costs.

---

## Tier 2 — Numerical and cross-reference defects; mechanical, and all real

Every one of these is verified. None requires new science. Budget one to two days.

| # | Location | Defect | Fix |
|---|---|---|---|
| 9 | abstract; §6.1.1 summary; Fig. 6.1 caption; §8.1 | Four incompatible accuracy bands. Table 6.2's PINN+BF for N∈{6,12}, ω≥0.1 is **0.0083–0.0485%**. Text says "10⁻²–10⁻¹%", "0.01–0.08%", "2.5×10⁻²–5×10⁻²%", "0.026–0.048%". None matches; three exclude the best point. | Quote **0.008–0.05%** in all four places. |
| 10 | §8.1 | "the energies connect smoothly to the DMC-validated points at ω=0.01" — in a paragraph about N=2, 6, 12. Only N=2 has DMC at ω=0.01. | "…to the ω=0.01 points (externally validated only at N=2)". |
| 11 | Table 6.9 caption | "(% above DMC)" for a grid whose ω≤0.01 cells at N=6 are scored against your own PINN+CTNN energy. | Caption: "% above the reference of Table 6.8 — DMC where one exists, our converged variational energy below ω=0.1." |
| 12 | abstract | "Energies match diffusion Monte Carlo … for N∈{6,12,20}". Table 6.2 is explicit that N=20 is Haas, not DMC. | "…match DMC for N∈{6,12} and the tabulated reference of [HaasHFQD] at N=20". (See also the memory note on this citation's provenance — the `.bib` note is now correct; the abstract is the last place still saying DMC.) |
| 13 | Table 6.8 caption | The %err column scores **Campaign (best)** for N=6,12 and **Multi-stage** for N=2, silently. Verified arithmetic on all 13 rows. | One clause in the caption. |
| 14, 26 | §6.4.2, §6.4.3 | Prose quotes **Multi-stage** numbers while the table's %err scores **Campaign**. N=12: prose +0.018/+0.028/+0.122 vs table +0.009/+0.024/+0.102. N=6, ω=1.0: prose "+0.009%" vs table "+0.013". Three mismatched pairs on one page. | Quote the table, or name the column each time. |
| 15 | §6.4.3 | "~32k proposal evaluations per gradient step — **modest compared to** the ~3×10³ MCMC samples per step". 32k is 10× 3k. | The defensible version: proposal draws need only an amplitude evaluation and no chain equilibration, so ~32k proposals cost less wall-clock than 3×10³ correlated MCMC samples with local energies. State that, or drop the comparison. |
| 21 | §8.4.3 | "the systematic pattern in Table 6.8: … to poor (+5.68%) at (N=20, ω=0.1)". Table 6.8 **has no N=20 rows**, and Appendix D.2 says **+5.53%**. | Cite Appendix D.2; use +5.53%. |
| 22 | §6.1, §6.4.2, §7.1.1 | Three unreconciled parameter regimes: production (54k / 137k / 236k), collocation (25k / 49k), Ch. 7 ladder (20k–164k). These are genuinely different configurations — nothing says so. | One sentence in §6.1 naming the three regimes. |
| 23 | Table 3.1 | §3.6 says the table exists "so that the architecture claims of Chapter 7 can be read as matched-capacity comparisons" — and lists widths and layers with **no parameter counts**. | Add a parameter-count column. Also note explicitly that the Ch. 7 ladder deliberately goes *past* matched capacity (the 164k DeepSet), which strengthens the result rather than weakening it. |
| 24 | §7.1.1 | "every DeepSet needs d_eff ≈ 2.9–3.6" and "3.25 ± 0.03 (DeepSet)" presented as one measurement. First is across the capacity ladder, second across seeds at fixed capacity. | Two clauses. |
| 25 | §6.4.2 | Pfaffian E = 20.261, 20.245 vs 20.15932 are **+0.50%** and **+0.43%**, not "both ~0.4%". | Quote both. |
| 28 | §8.1 | The audit's claim here is **wrong** (see below), but "always slightly higher, **as required by variationality**" is contradicted by your own next paragraph, which correctly argues that sitting below fixed-node DMC is legitimate. | Keep "always slightly higher"; drop "as required by variationality". |
| 29 | §9 | "all three systems (N = 2, 6, 12) cross over smoothly" — drops N=20, whose three-shell result the abstract highlights. | Add N=20. |

---

## Tier 3 — Citations and structure; cheap, and visibly improves the thesis

### T3.1 The citation gap is worse than the audit says — and cheaper to fix

The audit claims Raissi et al. (2019) "is not in the bibliography." **It is in
`references.bib` — it is simply never `\cite`d**, so it never prints. I diffed cited keys
against bib entries: **47 of 99 entries are cited; 52 sit unused.** Among the unused:

- `raissi2019pinns` — and §2.6, the section that *defines* PINNs in a thesis titled
  *Physics-Informed…*, cites **nothing** except `De_Ryck_2024` at the very end.
- **`zaheer2018deepsets`** — "DeepSet" appears 18 times in Ch. 7 as a central comparison
  object and is never cited. This is the most conspicuous omission on the list.
- **`CarleoTroyer2017-ScienceNQS`** — the founding NQS paper, uncited in an NQS thesis.
- `HolzmannCeperley2003-Backflow`, `LangeEtAl2024-NQSReview`, `PfauEtAl2024-ExcitedStatesNQMC`,
  `KeebleEtAl2023-SpinlessFermionsNQS`, `kharazmi2019vpinn`, `krishnapriyan2021failure`,
  `wang2021gradpath`, `mishra2020forward`, `deryck2022kolmogorov`.

**Fix: half a day.** The references already exist; you are adding `\cite` commands. Highest
priority: Raissi + Lagaris in §2.6, Zaheer at the first "DeepSet", Carleo–Troyer in the
introduction, Holzmann–Ceperley at the first backflow mention.

Genuinely missing from the `.bib` and worth adding: **Jastrow (1955)**, **Feynman–Cohen
(1956)**, **Barron (1993)** (three "Barron/spectral" discussions cite only [37] —
`appendix.tex:179`, `appendix.tex:454`, `theory.tex:1343`), and the scalable-SR literature
(**minSR / Chen & Heyl; Rende et al. 2024**) — you run matrix-free CG-SR on a ~10⁵-parameter
model, which is precisely the problem that literature solves. An examiner in this field will
know it is missing.

Note: the two appendix sections at lines 179 and 454 have the **same title**
("Barron/spectral perspective"). Give them distinguishing titles.

### T3.2 Bibliography rendering defects (audit 39 — right symptom, wrong diagnosis)

- `Chin1990-QuadraticDMC` **has** volume 42 and page 6991 in the `.bib`. They are dropped
  because the entry is `@misc` with `howpublished`. **Change to `@article`.**
- `FlyvbjergPetersen1989-Blocking` renders a trailing `\newblock American Institute of
  Physics.` — stray `publisher` field on an `@article`. Delete it.
- `KimEtAl2023-UltracoldFermiNQS` renders "Preprint; details as provided in source list" in
  the printed bibliography. The paper is published — fill in the reference. (Four other
  entries carry the same stub note but are uncited, so they do not print; clean them anyway.)

### T3.3 (audit 27) §1.7 and §5.1 are duplicate method sections that have drifted

Confirmed, and worse than described. `theory.tex:539-640` and `method.tex:663-760` present
the **same equations** for g(r), shell detection, Φ_m and λ_φ, with:

- shell threshold **τ ∈ [2.0, 3.0]** vs **τ ∈ [2.5, 3.5] (default 3.0)**;
- trap-unit notation **r̃** vs **y**;
- radial probability **P(r) ∝ r·g(r)** vs **P(r) ∝ r^(d−1)·g(r)**.

Two conflicting definitions of a detector threshold in one document is the kind of thing
that costs a grade band. **Fix: delete the equations from §1.7**, leave two paragraphs of
physical motivation for why shell topology and bond-orientational order diagnose a Wigner
molecule, and cross-reference §5.1 as the definition. This is also the single biggest page
saving available.

---

## Tier 4 — Matter of taste; real in kind, overstated in degree

I counted these on the flattened source. The audit inflates most of them.

| Claim | Audit | Actual | My read |
|---|---|---|---|
| Gauge caveat repeated | 12× | **11 passages** (intro ×2, §6.2 ×3, §7 ×4, §8 ×4, §5.6, §9) | **Real.** It reads defensive. Keep the full statement in §7.5, one in §6.2's preamble, one line in §9; make the rest cross-references. Half a day, and the thesis reads more confident, not less careful. |
| "buys" | 7 | **10** | Four are inside §7.1.2 where it is the chapter's technical framing. Vary the two in the abstract and conclusion; leave the rest. |
| "load-bearing" | 4 | **4** | Change two. |
| "seen from two sides" | 3 | **2** | Fine as is. |
| "itself a result" | 5 | **3** | Cut one — the §7.5 heading is the one worth keeping. |
| Epigraphs | 4 | **7** (preface, intro, chs. 1–3, 8, 9) | The audit *under*counted. Seven is past the line for a physics thesis. I would keep the preface epigraph and drop or halve the rest — but this is genuinely your call, and no examiner will fail you for it. |

**Self-announcing sentences (audit 3).** Of the six, I would cut two: "That last sentence
is not a metaphor" (§7.1.3) and "That last clause carries the weight" (§8.1). Both tell the
reader how to read a sentence that reads fine unaided. "The energies are the least
interesting result in this thesis and the most necessary one" is a good sentence — keep it.
"below that no benchmark is available, and we say so" (abstract) reads as defensive; state
the boundary without the flourish.

**Restatement (audits 4, 5).** Real in magnitude: Ch. 8 is 11 pages (89–100) and §8.1
re-walks Tables 6.1–6.2, §8.3 re-narrates §6.2 and §7.1. A separate Discussion chapter is
the UiO norm, so do not restructure — but **cut §8.1's table walk-through** and keep only
the interpretive paragraphs (the fixed-node/variationality argument is genuinely good and
appears nowhere else). Same treatment for §8.3. That should reclaim 3–4 pages and remove
most of the redundancy complaint without touching the architecture of the thesis.

**Retracted earlier draft (audit 6).** Two places (§7.1.1, §7.5), not the sprawl implied.
The §7.5 mention is doing real work — it is your evidence for *why* component quantities
are gauge-like. Keep that one; cut the §7.1.1 restatement to a clause.

---

## Items the audit gets wrong — do not act on these

- **17** ("Table 7.2 does not support the claim"). It does. The claim is that the
  conventional backflow *collapses toward Wigner* while the CTNN does not, and Table 7.2
  shows exactly that (conv → 1.0 at ω=0.01 at all three N; CTNN 9.8/20.8/35.2). The text
  already says "at large ω both architectures use ~N modes" — it never claims CTNN is
  higher everywhere. **But** the audit noticed something worth pre-empting: the
  conventional backflow is nominally *higher* rank at ω=0.1 in all three cells. Add one
  sentence saying so and that rank at ω≥0.1 is not the discriminator.
- **28** ("a comparison is reversed"). Misread. The sentence is scoped to N∈{6,12}, and all
  six PINN+BF entries there are positive (+0.0083 to +0.0485). The N=2 cases the audit
  points at are outside the sentence's scope. Only the "as required by variationality"
  clause needs fixing (see Tier 2).
- **35** ("Raissi is not in the bibliography"). It is in `references.bib`; it is uncited.
  The fix is one `\cite`, not a new entry. Same for Lagaris — though that one does need
  adding.
- **39** ("Chin 1990 has no volume or pages"). It has both; the `@misc` entry type discards
  them. Change the entry type.

---

## Suggested order of work

**Day 1 — the two that change what the thesis claims.**
T0.1 (delete the kinetic→Coulomb claim and the false method description) and T0.2 (reframe
Q1b honestly). Then start the Var(E_L) evaluation running (T1.1) — it is the long pole.

**Day 2 — the rest of Tier 1.** T1.3 rank/2N−2, T1.4 abstract reframing, T1.5 the ESS
paragraph, T1.6 scaling-law wording, T1.7 §1.5.3, and T1.2 once Var(E_L) lands.

**Days 3–4 — Tier 2 in one sweep.** Do the accuracy band (item 9) first and propagate it;
then the Table 6.8/6.9 column and caption work as a block, since 13/14/26 are one problem.

**Day 5 — Tier 3.** Citations (half a day, mostly `\cite` insertions), `.bib` entry types,
and the §1.7 excision.

**Day 6 — Tier 4, and only what you agree with.** Gauge-caveat thinning and the §8.1/§8.3
cuts give the best return; the rest is preference.

**Day 7 — full read-through against this list, then recompile and check the PDF.**

The audit's closing judgement is right: everything except items 30 and 31 is mechanical.
It missed that T0.1 is neither mechanical nor optional — but it is a deletion, which makes
it the fastest fix on the list, not the slowest.
