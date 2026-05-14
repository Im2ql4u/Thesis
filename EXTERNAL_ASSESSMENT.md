# External Assessor Report
## Thesis: "On the State of Many-Body Quantum Mechanics, and on PINNs"

**Assessor role:** External examiner (simulated)
**Grading framework:** UiO/MNT Master's Thesis Grading Guidelines (A–F scale)
**Date of assessment:** 2026-05-14
**Basis:** Full reading of all LaTeX source files, bibliography, appendices, and cross-reference with grading rubric PDF

---

## Preliminary Note

This report is unsparing by design. The request was for an honest, detailed, and ruthless assessment. Softening the language would be a disservice. Every weakness identified here is paired with a specific location in the thesis so it can be found, evaluated, and addressed. Strengths are also documented specifically — this is not a hatchet job, it is an audit.

---

## Summary Verdict

**Recommended grade: B (Very Good) — with reservations that must be resolved before final submission**

This is a genuinely strong thesis with a real scientific contribution at its core. The MCMC-free collocation result at N=6 is numerically significant; the Wigner-molecule diagnostics constitute the most rigorous structural analysis of these systems I have seen in the variational neural-network literature; the REINFORCE/SR regime mapping is clearly argued and well-evidenced. The appendix on bi-Laplacian conditioning is publication-quality mathematics.

What prevents an A is not one large failure but an accumulation of fixable problems: a placeholder citation that is used in live results text, an incomplete N=20 study that is presented as a contribution, multiple duplicated sections, several hundred lines of commented-out LaTeX that were never cleaned up, and a title that does not represent the work. None of these are fatal. All of them are visible to an external reader and signal that the manuscript has not received a final editorial pass. A thesis at the B/A borderline that has not had that pass will not cross to A.

If the placeholder citation is resolved, N=20 is either completed or honestly scoped out of the contributions list, and the duplications are collapsed, the grade argument for A becomes much stronger on the merits of the science alone.

---

## Rubric-by-Rubric Assessment

### 1. Work

**Grade: B**

The volume and depth of work is substantial. The thesis covers:
- A full ansatz implementation chain from Slater determinants through Jastrow, pairwise branch, and optional backflow
- An MCMC-free collocation training pipeline (genuinely novel for this class of problem)
- Systematic comparison across N∈{2,6,12} and five confinement strengths ω∈{0.001,0.01,0.1,1.0,10.0}
- Physical validation well beyond energetics: shell topology, bond-orientational order, pair distribution reconstruction, Lindemann ratios
- Representation analysis with PCA, effective rank, and CKA similarity across training regimes
- A mathematically rigorous appendix on conditioning of the Laplacian and Coulomb singularity

The negative result for the Langevin refinement strategy (+153% energy error at N=20, ω=0.1) is honestly reported and carefully analyzed. That is good scientific practice.

**Weakness:** N=20 is explicitly incomplete. The results chapter states "Training is ongoing and further improvement is expected." This is not a minor caveat — N=20 is listed as part of the N∈{2,6,12,20} experimental grid described in the abstract and introduction, and backflow at N=20 is listed in the Conclusion as an open question (§Conclusions, "Open questions" paragraph). A system that is listed in the contributions and described in methods but not completed in results is a structural problem. If N=20 cannot be completed before submission, it must be removed from the stated scope, not waved away with "ongoing."

---

### 2. Analysis and Discussion

**Grade: A–/B+**

The discussion chapter is the strongest part of the thesis in terms of analytical depth. The regime decomposition — collocation working well at moderate N and high ω, degrading at low ω and large N due to ESS collapse — is not just stated but mechanistically explained through the exponential degradation of ESS with Nd. The SR-to-Adam handoff is explained through Fisher information geometry, and the failure mode on the SR side (rank deficiency under low ESS) and the Adam side (slower convergence when Fisher is reliable) are both described.

The physical validation analysis is notably thorough. The decomposition of g(r) into II/IO/OO shell contributions, the cosine similarity metric (0.989–1.000), and the angular Lindemann ratio distinguish this from standard energy-table papers. The claim that these are "the first topology-resolved, bond-orientational and stiffness metrics" for parabolic quantum dots across the Fermi-liquid-to-Wigner crossover is significant if it stands.

**Weakness 1:** No external DMC reference exists at ω∈{0.001, 0.01} for N∈{6,12}. The thesis's own method produces energies there, but there is no independent validation. This is acknowledged in the text but the acknowledgement is quiet. An assessor reading this will note that the regime where MCMC-free training is expected to struggle most (low ω, high N) is also the regime with no external check.

**Weakness 2:** The placement of the representation analysis section before the Wigner-molecule results in the Results chapter creates a sequencing problem. The representations are supposed to encode the physics — but the reader does not yet know whether the physics is correct when they read the representation analysis. The ordering argument runs: if Wigner physics is established first, then the representation analysis reads as "here is what that correct physics looks like in latent space," which is much more compelling than "here are latent-space structures that we will later claim correspond to physics."

**Weakness 3:** The VMC/DMC section in Theory (§2.6) ends with a "Conclusion" subsection (§2.6.4). This is a category error — a conclusion is a chapter-level structure, not a subsection of a technical methods section. It breaks the reading flow and implies the chapter is over when it is not.

---

### 3. Critical Reflection

**Grade: A–**

This is genuinely one of the thesis's strengths and distinguishes it sharply from papers that only report what worked.

The Langevin negative result is presented without hedging: non-equilibrium samples invalidate importance weights, the error is 153%, the strategy is formally incorrect. This is the kind of honest reporting that takes some courage to include and that an assessor should credit.

The ESS scaling wall is named clearly: "No amount of oversampling prevents eventual ESS collapse; the fundamental scaling limit of the MCMC-free approach is set by the dimensionality of the configuration space." This is a clear-eyed acknowledgement of the method's own ceiling.

The open questions in the Conclusion are honest and specific: the ESS scaling problem, the incomplete N=20 backflow, the lack of a principled SR/Adam switching criterion. These are real open questions, not vague "future work" gestures.

**Minor weakness:** The preface (which is beautifully written) contains the most honest reflection on failure and the importance-sampling bug. That material should not live only in the preface — the methodological lesson from that bug is relevant to the methods chapter and could strengthen the argument for why the final design choices were made. Currently the preface is a philosophical meditation that does not connect to the technical arguments.

---

### 4. Own Contribution / Achievement of Goals

**Grade: B**

**Clear contributions:**
- The MCMC-free collocation approach for neural VMC (no Markov chain during training, importance resampling with Gaussian mixture proposal) — this is the main methodological novelty
- The REINFORCE/SR regime mapping as a function of (N,ω) — useful and novel
- The topology-resolved Wigner diagnostics applied to the trained neural states — a real advance in physical validation
- The representation analysis (PCA + CKA + effective rank) distinguishing correlator from backflow function spaces — novel and insightful

**Incomplete contributions:**
- N=20 results: listed in scope, mentioned in contributions, not completed
- Backflow at N=20: explicitly deferred to "open questions"

**Unclear contributions:**
- The claim that these are the "first" topology-resolved diagnostics for parabolic dots across the crossover (made in the abstract) is a strong priority claim. The thesis does not cite any prior work that attempted similar diagnostics and fell short. If there is no prior work, the claim is defensible but should be stated as "to the author's knowledge, no prior neural VMC study of these systems has applied topology-resolved diagnostics." If prior work exists, the distinction must be stated precisely.

**Missing contribution:**
- The thesis would benefit enormously from a concrete time/computational-cost comparison between MCMC-free and MCMC-based training. The claim that collocation training is faster/cheaper is implicit throughout but is never directly quantified. Wall-clock time or iteration count to convergence, side-by-side, would make the computational argument concrete.

---

### 5. Scientific Grounding

**Grade: B+ with a critical caveat**

The literature coverage is generally solid. FermiNet, PauliNet, DeepWF, and the relevant QMC literature are cited. The Deep Sets architecture is correctly attributed. The Kato cusp conditions are cited appropriately.

**CRITICAL FAILURE: The HaasHFQD citation**

In `Thesis/references.bib`:

```bibtex
@mastersthesis{HaasHFQD,
  author = {Daniel Haas},
  title  = {Title to be confirmed},
  school = {University to be confirmed},
  year   = {2024},
  note   = {Hartree--Fock results for 2D parabolic quantum dots}
}
```

This placeholder citation is **actively used in the results chapter text** to support the statement that "HF overestimates the ground-state energy by a correlation energy that grows with both N and the coupling strength Γ∝1/√ω~\cite{HaasHFQD}." The reader is being asked to accept a quantitative claim on the authority of a reference that has no title, no institution, and no verified existence. This is not a formatting oversight — it is an incomplete scientific grounding for a factual claim in the results.

If this thesis is submitted as-is, an assessor who checks the bibliography will find this immediately. It must be fixed before submission: obtain the full citation details, confirm the thesis exists and contains the stated HF data, and update the entry completely.

**Secondary grounding issues:**

- Second Quantization is given substantial space in the Theory chapter (§2.2 or equivalent) but is essentially unused in the computational approach, which works in first quantization throughout. The theoretical investment is not cashed out.
- The Fock-Darwin orbital basis is invoked for the Slater determinant construction but its derivation and properties are described at high level. The specific orbital ordering conventions used in the implementation are not stated, which matters for verifying the antisymmetry is implemented correctly.
- The claim that pure REINFORCE (β=0) is "strictly superior" to hybrid objectives is made in the Conclusion but the supporting ablation data in Results should be referenced explicitly at that point. An assessor following the chain of evidence should be able to trace the claim to a specific table or figure.

---

### 6. Theoretical Insight

**Grade: A–**

The theoretical development in the appendix is the strongest formal mathematics in the thesis. The bi-Laplacian conditioning analysis (κ ~ (k_max/k_min)^4) derived from Fourier analysis, and the analogous Coulomb analysis, provide a mathematically rigorous justification for the design choices (REINFORCE, not direct backprop; analytic cusp, not learned). This is the kind of appendix that could be pulled out and submitted as a technical note.

The natural-gradient / stochastic reconfiguration connection is stated clearly and with appropriate precision. The relationship between SR and the quantum geometric tensor is correctly characterized.

The function-space separation argument (correlator on low-dimensional manifold r_eff ≲ 3 vs. backflow with intrinsic dimension an order of magnitude larger) is a genuine theoretical insight that has direct practical implications for memory allocation. This is the thesis at its most analytically creative.

**Weaknesses:**

**Duplication in Theory chapter:** There is a subsection "Hilbert Spaces and Function Representations" (§2.1.1) and then a later subsection "Hilbert Spaces" (§2.1.3) that covers substantially overlapping material. These two subsections need to be merged or one must be removed. As it stands, the reader covers the same conceptual ground twice in the same chapter, which dilutes the pedagogical clarity.

**Model Systems section is empty:** §2.4 "Model Systems" appears as a heading with no content — the body is commented out in the LaTeX source. This is a drafting artifact that was never cleaned up. Either the section exists and has content, or it does not exist. A section heading with no body text is an error in a submitted thesis.

**Structural diagnostics duplication:** The section on Wigner-molecule diagnostics (shell topology, bond-orientational order, Lindemann ratios) appears in both the Theory chapter (§2.7) and the Methods chapter (Diagnostics section). This is not an intentional cross-reference — it is duplicate content. The theory chapter should contain the physics motivation and definitions; the methods chapter should contain the implementation decisions. Currently both chapters contain definitions and motivation.

---

### 7. Description of Goals

**Grade: B–**

The three-layer framework is clear and well-motivated. The introduction states the research questions precisely: what must a practitioner decide when applying PINNs to the quantum many-body Schrödinger equation, and what do those decisions cost? This is a well-posed question.

**Weakness 1: The title is wrong for this thesis.**

"On the State of Many-Body Quantum Mechanics, and on PINNs" reads like a survey paper, not an original research thesis with specific contributions. A reader encountering this title has no idea that the thesis contains: (a) a novel MCMC-free collocation training pipeline; (b) the first topology-resolved Wigner diagnostics for parabolic quantum dots; (c) a formal conditioning analysis of the bi-Laplacian. The title undersells the work substantially. A more accurate title might be: "Physics-Informed Neural Wavefunctions for Two-Dimensional Quantum Dots: Ansatz Design, MCMC-Free Training, and Wigner Diagnostics" — or any title that actually names the system, the method, and the contribution.

**Weakness 2: The N=20 goal is stated but not achieved.**

The abstract, introduction, and methods all include N=20 in the stated experimental scope (N∈{2,6,12,20}). The results chapter treats N=20 as incomplete and explicitly defers further training. The stated goal and the achieved goal are not the same. This creates a gap between what the thesis promises and what it delivers that a careful assessor will note.

**Weakness 3: The "Parts" structure.**

The document uses "Part I Introduction," "Part II Theory," etc. This is unusual for a master's thesis and typically seen in PhD dissertations or edited volumes. Parts imply a level of structural separation that is not warranted here — the chapters flow naturally and do not require part-level organization. The Parts add bureaucratic structure without aiding navigation.

---

### 8. Structure, Language, and Form

**Grade: B–**

The language is mostly strong. The introduction and conclusion are well-written; the preface is the most elegantly written section. Technical language is used correctly and consistently for the most part.

**Formal errors:**

**Duplicate package imports in main.tex:**
- `\usepackage{graphicx}` appears at line 2 and again at line 105
- `\usepackage{subcaption}` appears at line 5 and again at line 104
- `\usepackage{multirow}` appears at line 4 and again at line 12

These are minor but signal that the preamble was assembled without care.

**Spelling errors:**
- `method.tex:136` — "nesseccity" (should be "necessity")
- `theory.tex:68` — "explaination" (should be "explanation")

Two spelling errors in a document of this length is not egregious, but in a submitted thesis they signal an insufficient proofread.

**Commented-out LaTeX code:**

The theory chapter contains extensive commented-out LaTeX blocks — on the order of hundreds of lines (lines 141–298 are a large block of commented Rayleigh-Ritz material, but there are others). This is draft-stage residue that should not appear in a submitted document. The compiled PDF will look fine, but any reader of the source (including a supervisor who edits the LaTeX) will see ghost material. A submitted thesis should not contain commented-out content.

**Table formatting:**

In the collocation results table (Table labelled `tab:collocation`), the "Campaign (best)" column appears to be missing values for the N=12 rows — the rows show Multi-stage methodology and percentage error but the Campaign column is either blank or misaligned. This requires inspection to determine whether it is a rendering artifact or missing data, but it reads as an error in the submitted form.

**Acknowledgements:**

One paragraph, one person (supervisor Prof. Morten Hjorth-Jensen). There is no acknowledgement of computational resources, which for a thesis involving systematic multi-N, multi-ω neural network training runs is a significant omission. If any compute cluster or cloud resource was used, its acknowledgement is typically expected. If a research group, colleagues, or co-investigators contributed ideas or data (including the HaasHFQD HF data), they should be acknowledged even if not cited as co-authors.

**Code references in academic text:**

At least one footnote in the methods chapter references a specific source code class/method (`BackflowNet.phi`). This is appropriate in a technical report or software documentation but breaks the academic register of a thesis. Implementation details should be described in terms of the algorithm, not the codebase structure.

---

### 9. Level of Skill

**Grade: A–**

The demonstrated skill level is the highest-scoring dimension of this thesis. The student has:
- Implemented a complete neural VMC pipeline from scratch, including antisymmetrization, pairwise architecture, analytic cusp, backflow, and importance resampling
- Designed and run a systematic ablation study across gradient estimators (β sweep) and optimizers (SR vs. Adam)
- Applied rigorous structural diagnostics from condensed matter physics (bond-orientational order, Lindemann ratios) to neural variational states
- Performed representation analysis using PCA, effective rank, and CKA — tools from ML theory applied to physics outputs
- Derived formal conditioning bounds in the appendix
- Compared against three independent reference methods (HF, FCI, DMC) across a grid of (N, ω)

This level of breadth and technical execution is well above average for a master's thesis. The integration of physics, ML, and numerical analysis is handled with sophistication. The fact that the ESS scaling wall is not only identified but explained geometrically (mismatch between Gaussian proposal and Wigner-molecule geometry of |Ψ|²) demonstrates physical intuition, not just computational competence.

---

## Red Flags Summary (Ordered by Severity)

### Critical — Must Fix Before Submission

**1. Placeholder citation used in live results text (references.bib)**

The `HaasHFQD` entry has `title = {Title to be confirmed}` and `school = {University to be confirmed}`. It is cited in the results chapter to support a factual claim about HF energetics. This will be immediately visible to any assessor who checks the bibliography. A thesis cannot be submitted with an unresolved placeholder citation that is cited in the body text. This is not a stylistic concern — it is a scientific grounding failure.

**Action required:** Obtain full citation details for the Haas thesis (title, institution, year), verify that it contains the HF data used as benchmark, and update the bibliography entry completely.

**2. N=20 scope claim not matched by results**

The experimental scope stated in the abstract, introduction, and methods includes N=20. The results section acknowledges that N=20 training is incomplete ("Training is ongoing and further improvement is expected"). The open questions section in the Conclusion lists N=20 backflow as future work. This is a three-way contradiction that an assessor will read as: the student announced work they had not yet done.

**Action required:** Either complete the N=20 experiments before submission, or remove N=20 from the stated scope and contributions, and update the abstract, introduction, methods framing, and results accordingly. Do not submit with this gap intact.

---

### Serious — Should Fix Before Submission

**3. "Model Systems" section is empty (theory.tex §2.4)**

A section heading with no body content is a structural error. The section presumably existed at draft stage and was commented out. Either restore the content or remove the heading.

**4. Structural diagnostics section duplicated (theory.tex §2.7 and method.tex Diagnostics chapter)**

Substantial content on Wigner diagnostics appears in both chapters. This is not cross-referencing — it is duplication. Decide where this content lives (methods, as implementation; or theory, as physics background) and remove or condense the other occurrence.

**5. Duplicate Hilbert Spaces subsections (theory.tex §2.1.1 and §2.1.3)**

Two subsections covering substantially overlapping material within the same chapter. One should be absorbed into the other or the material reorganized so the two subsections are clearly distinguished.

**6. Commented-out LaTeX throughout theory.tex**

Hundreds of lines of commented-out draft material remain in the submitted source. This is invisible in the PDF but is visible to any reader of the source and signals an incomplete cleanup. Remove all commented-out content before submission.

**7. Title does not represent the work**

"On the State of Many-Body Quantum Mechanics, and on PINNs" is a survey title for a research thesis with specific, novel contributions. A reader cannot determine from this title what system was studied, what method was developed, or what was found. The title should be revised to accurately represent the work.

---

### Moderate — Fix If Time Permits

**8. Representation analysis before Wigner results (Results chapter ordering)**

The representation analysis section appears before the Wigner-molecule validation section. The reader is asked to interpret latent-space structures before knowing whether the physics those structures are supposed to encode is correct. Reversing the ordering would make the argument cleaner.

**9. No wall-clock or iteration-count comparison between MCMC-free and MCMC training**

The MCMC-free approach is presented as a methodological contribution but its computational cost relative to standard VMC is never directly quantified. A table or plot showing convergence speed (iterations to convergence, or time to a given energy tolerance) for MCMC vs. MCMC-free would substantiate the claim that the new approach is practical.

**10. No external validation at ω∈{0.001, 0.01}**

The two most extreme confinement regimes have no DMC reference energies. The text acknowledges this but does not quantify how much uncertainty this introduces into the conclusions about low-ω accuracy. If generating DMC benchmarks at these points is feasible, it would substantially strengthen the physical validation claims.

**11. "Conclusion" subsection inside VMC/DMC theory section (theory.tex §2.6.4)**

A subsection called "Conclusion" inside a theory chapter section is a structural oddity. If this material is needed, it should be integrated into the section prose or moved to the chapter's conclusion.

**12. Duplicate package imports in main.tex**

`graphicx`, `subcaption`, and `multirow` are each imported twice. Clean up the preamble.

**13. Sparse acknowledgements**

A single paragraph acknowledging only the supervisor, with no mention of computational resources or the source of benchmark data (HaasHFQD), is unusually thin. At minimum, computational resources used for training should be acknowledged.

---

### Minor — Polish Before Submission

**14. Spelling errors**
- `method.tex:136` — "nesseccity" → "necessity"
- `theory.tex:68` — "explaination" → "explanation"

**15. Code reference in academic text**

The footnote referencing `BackflowNet.phi` by source class name should be rewritten in terms of the algorithm. Appendix code listings are the appropriate place for implementation-level detail.

**16. The "Parts" document structure**

The use of Part-level structure (Part I, Part II, etc.) is unusual for a master's thesis and adds bureaucratic overhead without navigational benefit. Consider removing the Parts and using chapter-level organization only.

**17. Table formatting in collocation table**

The "Campaign (best)" column appears to have missing or misaligned values for N=12 rows. Verify and fix.

**18. Second Quantization theoretical investment not cashed out**

A substantial portion of the theory chapter develops second quantization formalism that is not used in the computational approach. Either connect this to the method (e.g., as a bridge to the FCI comparison) or reduce its scope.

---

## Strengths Worth Acknowledging Explicitly

These are not present for balance — they are substantive and an assessor should credit them.

**1. The preface is exceptional.** Philosophically sophisticated reflection on the role of failure in computational physics, written with real prose quality. This is rare in a master's thesis.

**2. The MCMC-free collocation result is scientifically significant.** +0.009% relative error at (N=6, ω=1.0) without any MCMC step is a real achievement. The ESS analysis that establishes its limits is equally valuable — it is rarer to document a method's ceiling than to trumpet its floor.

**3. The REINFORCE/direct-backprop comparison is well-argued.** The formal conditioning argument in the appendix provides mathematical grounding for a design choice that is often made informally. The β ablation (showing β=0 is strictly superior) is clean evidence.

**4. The Wigner-molecule diagnostics are the most thorough I have seen for this class of variational neural states.** The II/IO/OO decomposition of g(r), the cosine similarity metric, the angular Lindemann ratios — these go well beyond energy tables and constitute a meaningful advance in how neural VMC results can be physically validated.

**5. The representation analysis (PCA + CKA + effective rank) is novel and useful.** The finding that the correlator lives on a manifold with r_eff ≲ 3 while the backflow field has an order-of-magnitude higher intrinsic dimension is a structural insight with direct engineering implications (memory allocation, architecture selection). This is the kind of finding that gets cited.

**6. The appendix conditioning analysis is publication quality.** The bi-Laplacian scaling (κ ~ (k_max/k_min)^4) derived from Fourier analysis, and the analogous Coulomb analysis, are rigorous and could stand alone as a technical paper.

**7. Honest reporting of negative results throughout.** The Langevin failure (+153% error, explicitly stated as formally incorrect), the ESS collapse at large N/low ω, the SR-to-Adam handoff — none of these are buried or hedged. This is scientific integrity in practice.

---

## Overall Assessment

This thesis has the scientific content of an A paper and the manuscript preparation of a B paper. The gap is not in what was discovered or how it was analyzed — it is in whether the document accurately represents what was done and can withstand a line-by-line reading by an external examiner.

The two critical issues (HaasHFQD citation and N=20 scope claim) must be resolved before submission and could individually cause an otherwise strong thesis to be graded down. The structural issues (duplicate sections, commented-out content, empty section heading) are fixable in a few hours of editing. The missing computational cost comparison and the absence of low-ω DMC benchmarks are limitations that can be acknowledged more explicitly rather than resolved.

If the critical issues are resolved and the serious structural problems are addressed, this work merits an A. The science supports it. The question is whether the student has time and willingness to do the final editorial work that A-grade manuscripts require.

---

*Assessment prepared for internal use. Not for external distribution without review by the student and supervisor.*
