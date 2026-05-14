# Experiments That Need to Be Rerun or Completed

This file documents the computational gaps that exist at the time of the editorial pass
(2026-05-14). It is written before any prose edits so the state is on record.

---

## 1. N=12 Campaign-column data (HIGH PRIORITY)

Table `tab:collocation` in `Thesis/results.tex` has a column mismatch for N=12.

The N=6 rows have 7 columns: N | ω | DMC | Multi-stage | Campaign (best) | %err | Arch
The N=12 rows have 6 columns: N | ω | DMC | Multi-stage | %err | Arch

The "Campaign (best)" column is absent for N=12.

**To resolve, choose one:**

a) If 30-run reliability campaign data exists for N=12: extract the best campaign-run
   energy and uncertainty for each (N=12, ω) row and populate the missing column.

b) If no campaign was run for N=12: the table fix is editorial — add a `---` cell in the
   Campaign column for each N=12 row, and add a table footnote:

   > "No standalone reliability campaign was conducted for $N{=}12$; the Multi-stage
   > entry is the best available collocation energy."

Check `results/` or experiment logs for N=12 campaign-run outputs before choosing (b).

---

## 2. N=20 low-ω confinement points

The collocation table covers ω ∈ {1.0, 0.5, 0.1} for N=20 but not ω ∈ {0.01, 0.001}.

**Assessment:** No rerun is strictly required. The three available ω values are sufficient
to establish the scaling-wall argument. The prose fix (removing "Training is ongoing") handles
this editorially without new experiments. If low-ω N=20 runs are attempted later, add rows
to the table and update the N=20 discussion paragraph accordingly.

---

## 3. DMC benchmarks at ω=0.001 and ω=0.01 for N≥6

No external DMC reference exists at the two most extreme confinement values for N=6 and N=12.
This is a **literature gap**, not a code gap — these DMC energies have not been published in
the cited references for these parameter combinations.

**Options:**

a) Run DMC at those points if computational resources allow. This would substantially
   strengthen the validation claim at extreme confinement.

b) (Sufficient for submission) Acknowledge the gap explicitly in table footnotes and
   in the discussion:
   > "No DMC benchmark is available at $\omega < 0.01$ for $N > 2$; the collocation
   > energies represent the best available reference at those confinement strengths."

The discussion chapter (§Discussion of energy results) should be updated to include
this sentence if option (b) is chosen.

---

## 4. Backflow at N=20 (open question — no action required before submission)

The conclusion explicitly lists N=20 backflow as an open question. No experiment needs to
be run before submission. The prose only needs to be internally consistent: if this is an
open question, it must not appear anywhere else as a completed result.

Status: correctly scoped as future work.

---

## Summary table

| Item | Required before submission? | Action |
|---|---|---|
| N=12 Campaign column | Yes (table is malformed) | Populate or mark `---` with footnote |
| N=20 low-ω | No | Prose fix only |
| DMC at ω<0.01 | No | Prose acknowledgement |
| Backflow at N=20 | No | Already scoped as open question |
