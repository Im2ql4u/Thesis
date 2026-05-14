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
