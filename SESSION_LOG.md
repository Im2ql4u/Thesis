# Session Log

Last session: [2026-05-16] — Architecture diagnostics + thesis table fix

## What was done this session
1. **Thesis table fix** (commit `542beda`): Fixed malformed N=12 rows in `tab:collocation` (was 6 columns, needed 7). Populated Campaign (best) column from model_quality_inventory (12-13 existing runs per ω). Added matching `---` cells for N=20 with caption note. Script: `scripts/extract_n12_campaign_best.py`.

2. **Architecture diagnostics** (commit `95193f5`): Produced four figures in `results/figures/architecture_diagnostics/architecture_diagnostics.pdf` validating thesis design claims. Key numbers:
   - Spin attribution: 0.03 (ω=1.0) → 1.05 (ω=0.001) — Wigner crystal physics visible in input utilisation
   - Effective rank: 2.3/24 node, 1.4/24 edge — network uses ~2D manifold out of 24 dims
   - REINFORCE vs FD-Colloc gradient norms: 173 vs 413 (2.4× amplification from FD second-derivative path)

3. **Failed experiments**: ShellAware BF cascade for N=20 (all checkpoints lost to OOM during final eval); N=12 shellaware bootstrap (91% error after 600 epochs from random init). These showed the shellaware Jastrow does not converge reliably from scratch in these settings.

## Next session
**Recommended starting point:** Integrate the four diagnostic figures into the thesis. Write the missing CTNN architecture schematic (TODO, results.tex line 428). The figures go in: Fig A → methods gating subsection, Fig B/C → results §4.1, Fig D → results §4.2 collocation training.
**Open questions:** Does Figure A's per-channel attribution show the expected suppression pattern (r² channel relatively more important near r→0)? The numerical values were printed but the visual needs human inspection.
**Active workarounds:** "Near-coalescence" gradient comparison uses chunk-mean r_min rather than per-sample r_min — adequate for showing the overall trend but not the extreme-r_min regime.
**Foundation status:** thesis table is correct; diagnostic figures are committed; existing collocation energy results stand unchanged.
**Context freshness:** current

See ARCHIVE.md for full history.
