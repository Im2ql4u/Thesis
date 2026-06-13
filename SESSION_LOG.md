# Session Log

Last session: [2026-04-15] — N=20 ShellFlow collapse diagnosis and stabilized relaunch

## Next session
**Recommended starting point:** Inspect the live `v2` N=20 stabilized diagnostics in `outputs/2026-04-15_shellflow_n20_jastrow_diag_v2/` and decide whether ESS remains healthy past the first tens of epochs before making any further architecture or schedule changes.
**Open questions:** Does the stabilized relaunch maintain non-collapsed ESS after early epochs, and if so does `2shell` or `3shell` materially improve energy/error? Is the remaining failure now mostly Layer 1 sampling or Layer 3 architecture mismatch?
**Unverified assumptions:** Assumed that weighted raw-candidate ShellFlow refits plus tempered/clipped resampling are sufficient to prevent the previous proposal-drift failure mode throughout the full run, not just at epoch 0.
**Active workarounds:** ShellFlow refits now skip under catastrophic resampling collapse instead of trying to adapt through it; this is an explicit protective workaround until the N=20 low-omega sampling regime is better understood.
**Foundation status:** Verified: initial N=20 `v1` failure was diagnosed as a refit-on-collapsed-resamples problem; code patched in `src/functions/Neural_Networks.py` and `src/run_weak_form.py`; focused ShellFlow tests pass; stabilized `v2` relaunch started with epoch-0 ESS around 2k-3.3k instead of 1.
**Context freshness:** fresh
**Contradiction flags:** yes — ESS recovery at epoch 0 is a major improvement, but energies are still grossly wrong, so restored sampling alone may not be enough to make the recipe scientifically viable.

See ARCHIVE.md for full history.

---
## Session 2026-05-25 — Oral-exam notes clarity pass + slide trims

**Tasks completed:**
1. `oral_exam_slides.tex`: per user request, dropped norm-based bounds and implicit bias from the A--Q1 (generalization) frame — it now carries only double descent, the NTK statement, and the NTK$\leftrightarrow$SR link. Reworked A--Q4 (optimization) to drop the NTK linearization (it now lives on A--Q1) and instead present non-convex landscape + saddle escape, implicit regularization, Adam, and natural gradient/SR. Rebuilt; only the harmless title-page vbox warning remains.
2. `oral_exam_notes.md`: large clarity pass answering a list of specific examiner-style questions, section by section:
   - Slide 5: added "Is the CTNN just a GNN?" — yes, it is a one-round message-passing GNN (copresheaf-style, node+edge feature spaces with learned node$\leftrightarrow$edge transport maps, confirmed against `src/PINN.py:CTNNBackflowNet`); the label only records the two-headed (scalar correlator + vector backflow) use and the near-identity ODE-flow parametrization.
   - Slide 6: defined every symbol in the risk decomposition ($\mathcal F$, $R(\cdot)$, $f_S$, $f_{\mathcal F}$, $f^\star$) and tied approximation$\to$bias, estimation$\to$variance.
   - Slide 7: rewrote the NTK explanation from scratch — what $u_t$ is, how $\dot u_t$ relates to $\dot\theta$ via the chain rule, the kernel as kernel/matrix/operator, the $JJ^\top$ (NTK) vs $J^\top J$ (SR/Fisher) identification, and what $P$ is in Paper B.
   - Slide 8: explained the constant $c$ in $\Omega(2^{cd})$.
   - Slide 9: defined $g$, $\inf_\theta$, $L^2(\mu)$/$\mu$, the constant $c$, and the precise meaning of "dimension hidden in $C_g$" (curse relocated from exponent to constant).
   - Slide 10: reframed to lead with saddle escape / implicit regularization / Adam / natural gradient; demoted NTK to a pointer back to Slide 7.
   - Slide 11: defined Sobolev rate, the generic constants, $\sup_\theta$, and $\mathrm{Lip}(\Theta)$.
   - Slide 12: explained $\#(\varepsilon)=O(\kappa\ln 1/\varepsilon)$ in words, why we diagonalize in Fourier, what $L$ is and how it is the PDE stiffness, what "conditioning" concretely changes (two handles: shrink $L^*L$ frequency range vs reshape $TT^*$), and corrected/clarified the "$L=\mathrm{Id},\mathbf A=I$ supervised special case" (it means no operator-stiffness factor, not that supervised training is easy).
   - Slide 18: rewrote the three-body sensitivity ratio as a concrete step-by-step ("hold $r_{ij}$ fixed, wiggle a third electron, see if the cell output wiggles"); linked the $r(\omega)$ numbers to the ablation and kinetic-only signatures.
3. Rebuilt `oral_exam_notes.pdf` via `scripts/md2tex_notes.py` (now 29 pages, clean compile, no leftover placeholders).

**Changed files:** oral_exam_slides.tex, oral_exam_slides.pdf, oral_exam_notes.md, oral_exam_notes.tex, oral_exam_notes.pdf.

**Status:** Both PDFs build cleanly. Notes now answer the flagged conceptual questions inline.

---
## Session 2026-05-23 — Oral exam slides + notes full restructure

**Tasks completed:**
1. Rewrote `oral_exam_slides.tex` end-to-end per the new outline in `thoughts.md`:
   - Reordered: terminology + MBSE solvers (HF, CCSD, FCI, DMC, VMC, NQS) and our ansatz / CTNN moved to the front, *before* Paper A, so the talk opens on physics the examiners already own.
   - Paper A: four-question structure preserved but each frame compressed; A-Q2 now explicitly carries the article's $\|x-y\|$ example (1 layer = exp in $d$, multi-layer = poly); A-Q3 explicitly defines smoothness order $s$ and ties FCI/HF/CCSD scaling into the curse discussion; A-Q4 names natural-gradient, Adam, SGD and explicitly identifies SR with the quantum-Fisher NTK.
   - Paper B: Q1/Q2/Q3 collapsed into a single slide (universal-approximation in Sobolev; zero-variance + heavy-VMC certification handles stability and quadrature for us); B-Q4 kept as the centerpiece with $L^*L \circ TT^*$ and the bi-Laplacian $k^4$ example.
   - Added a dedicated "catch-22" frame: 17 parameter-space interventions vs.\ VMC+SR; the only fix was removing $\nabla_\theta$ through the Laplacian path (REINFORCE-only) while keeping the kinetic term in the loss.
   - Results split into two frames: SR+VMC across $N \in \{2,6,12,20\}$ with the CTNN cell ablation; collocation/PINN results with explicit "tricks that mattered" and the SR↔Adam regime boundary at $\omega \le 0.01$.
   - Inputs / attribution and intrinsic-dimensionality slides retained but trimmed; new explicit three-body slide carrying the codimension-in-pairwise observation as the single original contribution.
2. Added 7 backup slides for material that is interesting at an abstract level but not in the main talk:
   - NTK proof sketch + linearization (paired with SR identification);
   - Barron proof sketch with the "neurons are MC samples in frequency space" line;
   - Design choices → preconditioners summary table + REINFORCE gradient;
   - 2D cusp + log-divergent $(1/r)^2$ spike;
   - Force-aligned backflow across the Wigner crossover (sign flip);
   - Sharp PINN–CTNN coupling transition at $\omega \approx 0.1$;
   - $N{=}20$ Jastrow-beats-backflow as an expressivity-vs-budget tradeoff.
3. Fixed every overflowing frame; final build has only the harmless title-page vbox warning (intrinsic to metropolis+seahorse; present in the original).
4. Completely rewrote `oral_exam_notes.md` (1100+ lines) as a study + recital document, per slide, with the deep explanations the user asked for:
   - NTK in full: chain-rule derivation, Jacot frozen-kernel limit, what "linear" implies (kernel regression, convex, spectrum-governed), spectral bias.
   - Norm-based bounds explained (notes only, not on slides).
   - Explicit NTK ↔ SR identification: $S =$ quantum Fisher $=$ empirical NTK of $\log|\Psi|$ in the sampling measure.
   - Width vs depth example $\|x-y\|$ (1-layer exponential, multi-layer polynomial) made precise; what *width* is for per Hanin–Sellke.
   - Barron proof step-by-step (inverse Fourier → expectation against $\mu_g$ → MC over $n$ → Bienaymé → single-neuron approx of $\Gamma$); smoothness order $s$ defined.
   - Natural-gradient vs Adam vs SGD comparison, including how batching's noise has an implicit-regularization role.
   - B–Q4 mechanics in depth: $\lambda$, $\kappa$, $L^*L$, $TT^*$, why $\Delta^2 \to k^4$, why backflow is specifically problematic, the catch-22 result.
   - Three-body sensitivity ratio derivation and codimension-in-pairwise interpretation; what we could run now (per-pair / per-particle aggregated message PCA) if asked to extend.

**Inputs vs.\ outputs:** `thoughts.md` was the design brief. All slide content stays within the cited papers' content (Paper A, Paper B) and our own thesis results — no external citations added.

**Changed files:**
- `oral_exam_slides.tex` (full rewrite)
- `oral_exam_slides.pdf` (rebuilt)
- `oral_exam_notes.md` (full rewrite)
- This session log; `DECISIONS.md`; `JOURNAL.md` entry below.

**Status:** Slides build cleanly. Notes are reorganized per the new outline with all four deep-explanation categories.

---
## Session 2026-05-17 — Architecture diagnostics integration

**Tasks completed:**
1. Generated thesis-ready figures from pre-computed npz data (`scripts/plot_arch_thesis.py`):
   - `fig_arch_attribution.pdf` — 3-panel Jastrow channel attribution by ω
   - `fig_arch_bfgeo.pdf` — backflow direction and spin-resolved displacement
   - `fig_arch_force_alignment.pdf` — force alignment sign reversal across Wigner crossover (centerpiece result)
   - `fig_arch_sensitivity.pdf` — safe-feature sensitivity vs attribution (for appendix)
   - Three-body and ablation data → tables, not figures
2. Copied 4 keeper figures to `results/figures/results/`
3. `method.tex`: Added `\section{The cell view: both networks as a unified message-passing system}` with three-body table (2.03×/2.93×/1.05× across ω); added safe-core channel suppression note in pair branch; added REINFORCE gradient-norm qualifier (2.4× lower than FD-Colloc)
4. `results.tex`: Added architecture overview paragraph; added `\subsection{Message-passing is essential}` with ablation table (22–30% energy cost, 100% kinetic); added attribution figure with spin-channel shift story; added BF geometry figure; added `\subsection{Force-aligned backflow across the Wigner crossover}` with force_alignment figure and full narrative of the sign reversal
5. `appendix.tex`: Added `\section{Safe-feature sensitivity analysis}` with sensitivity figure documenting global channel suppression mechanism
6. `discussion.tex`: Added force-alignment result reference in energetic-role-of-backflow paragraph
7. PDF compiles cleanly: 96 pages, no undefined references

**Key new results integrated:**
- Force alignment sign reversal: trap-aligned at ω≥0.1 (cos≈+0.97), full sign flip at ω=0.001 (cos_trap≈−0.83, cos_Coul≈+0.83) — lattice-correction mode at Wigner crossover
- Message-passing cell accounts for 22–30% energy at moderate ω, all kinetic; 100% ΔE is kinetic (potentials unchanged)
- Three-body sensitivity collapses at ω=0.001 (1.05×) from 2.93× at ω=0.1 — physical interpretation: Wigner lattice is pairwise-dominated

**Changed files:** method.tex, results.tex, appendix.tex, discussion.tex + figures copied to results/figures/results/

**Status:** PDF compiles cleanly. All new diagnostic results from architecture_diagnostics runs are integrated.

---
## Session 2026-05-11 — Thesis restructuring: full five-layer integration

**Tasks completed:**
1. Full structural audit of all thesis .tex files — identified TODOs, stale text, missing citations, broken structure
2. Rewrote `method.tex` Optimization chapter as "Training the Wavefunction" organized around Layers I, III, IV — removed all code parameter names, added PINN/Schrödinger challenges section, condensed sampling and SR sections to principled prose
3. Trimmed Analysis chapter in `method.tex` — folded two unused subsections (linear probes, near-field gradient) into brief paragraph
4. Restructured `results.tex` — chapter intro rewritten to five-layer framing; energy section renamed to Layer II; training methodology subsection removed (now in methods chapter); "what worked/didn't" replaced with proper Layer IV and Layer V sections; N=2 collocation absence noted
5. Merged `app:catch22` and `app:postcatch22` into single chapter in `appendix.tex` with section hierarchy properly demoted; moved experimental record (positive/negative findings) to new appendix subsection
6. Fixed three red TODO table references in `discussion.tex` — replaced with inline prose using known numbers
7. Fixed `[Ref.]` placeholder in results.tex energy table → `\cite{Pederiva2000-QD-DMC}`
8. Fixed `eq:is-weights` duplicate label
9. Repaired all broken cross-references after restructuring
10. PDF builds cleanly: 99 pages (down from 103), no undefined references

**Changed files:** method.tex, results.tex, discussion.tex, appendix.tex, theory.tex (orphan paragraph), conclusion.tex (checkpoint filename removed)

**Status:** PDF compiles cleanly. Major structural work complete.

---
## Session 2026-05-12 — Three-layer restructuring + citation fixes

**Tasks completed:**
1. Planned and approved full restructuring from five-layer to three-layer framework (Layer 1: Ansatz, Layer 2: Training paradigm, Layer 3: Optimization)
2. Added Deep Sets section (§ in ML chapter of theory.tex) with Zaheer et al. citation and explanation of why particle-only pooling is insufficient for correlated systems
3. Removed CCSD chapter from theory.tex — replaced with FCI scaling argument (exponential scaling motivates neural VMC)
4. Rewrote method.tex ch1 opening as "The Ansatz" — framed as a decision story around four simultaneous constraints (antisymmetry, equivariance, physical laws, gradient stability)
5. Strengthened pair branch transition as a motivated decision ("not optional")
6. Fixed `[kilde]` → `\cite{zaheer2018deepsets}` with user-provided exact arXiv entry
7. Rewrote introduction.tex three-layer paragraphs replacing old five-layer content
8. Restructured conclusion.tex around three layers
9. Updated preface, discussion opening, theory part-level intro
10. Renamed results chapter to "Results: Accuracy, Scaling, and Physical Content"
11. Added comparative benchmark text in results.tex (HF vs FCI vs DMC context, no CCSD)
12. Added placeholder `\cite{HaasHFQD}` for Daniel Haas master's thesis (HF reference) — **details to be confirmed by user**
13. Wrote `scripts/run_jastrow_diagnostics.py` for CTNN Jastrow-only representation analysis
14. Fixed all layer numbering (I/II/III/IV/V → 1/2/3) across all files
15. PDF builds cleanly: 100 pages, no undefined references

**Pending from this session (now resolved):**
- HaasHFQD bib entry filled with actual details (see session 2026-05-14)

---
## Session 2026-05-14 — External assessment + full editorial pass

**Tasks completed:**
1. Wrote `EXTERNAL_ASSESSMENT.md` — full external-examiner audit against UiO/MNT grading rubric. Recommended grade: B with clear path to A if blockers resolved.
2. Fixed `HaasHFQD` citation: title = "Deep Learning Methods for Quantum Many-body Systems: A Study on Neural Quantum States", school = University of Oslo, year = 2024, month = sep.
3. Fixed thesis title: new title names system, method, and contribution explicitly.
4. Removed `\part{}` structure from `main.tex` (flat chapter sequence).
5. Removed duplicate package imports: graphicx×3→1, multirow×2→1, subcaption×2→1, makecell×2→1, usetikzlibrary×2→1.
6. theory.tex structural surgery:
   - Deleted duplicate "Hilbert Spaces and Function Representations" subsection (lines 24–43 original)
   - Renamed surviving subsection to "Hilbert Spaces and Dirac Notation"
   - Fixed spelling: "explaination" → "explanation"
   - Deleted empty `\section{Model Systems}` (body was commented out)
   - Deleted generic `\subsection{Conclusion}` inside VMC/DMC section; replaced with forward-linking sentence
   - Removed ~200 lines of commented-out draft content across the file
   - Added `\label{sec:fci}` to FCI section (was unlabelled)
   - Added Second Quantization bridge paragraph connecting to FCI and first-quantized ansatz
7. Diagnostics duplication resolved: Theory section trimmed to physics definitions + forward ref; Methods chapter opens with backward ref. Sanity-anchor paragraphs (experimental numbers) removed from Theory chapter.
8. method.tex: "nesseccity" → "necessity"; BackflowNet.phi footnote rewritten to algorithmic prose; "Mapping to implementation" subsection rewritten without class/attribute names; diagnostics chapter backward cross-reference added.
9. results.tex: Wigner-molecule section moved before representation analysis; N=20 paragraph reframed as "scaling boundary" (removed "Training is ongoing"); collocation table N=12 and N=20 rows padded with `---` in Campaign column with explanatory footnote.
10. Philosophical coherence: framing paragraphs added to Introduction, Theory opening, Methods opening, Results opening, and Conclusion closing. Conclusion now explicitly closes the preface's variational-principle epistemology ("the map is the contribution").
11. acknowledgement.tex: added sentence crediting Haas thesis as HF benchmark source.
12. Wrote `RERUN_REQUIRED.md` documenting computational gaps (N=12 campaign data, low-ω DMC benchmarks).
13. PDF compiles cleanly: 92 pages, no undefined references.

**Changed files:** references.bib, main.tex, theory.tex, method.tex, results.tex, conclusion.tex, introduction.tex, acknowledgement.tex + new files: RERUN_REQUIRED.md, EXTERNAL_ASSESSMENT.md

**Status:** All assessment blockers resolved. PDF compiles cleanly. Grade argument for A is now on the science, not the manuscript.
