# Archive

Compressed session history. Maintained by session-close. When this file exceeds 10 entries, the oldest 5 are compressed into ## Older History.

Read the last 2–3 entries during session open. Read the full file only when reconstructing history.

---

## Format

```
## [YYYY-MM-DD] — <session title>

### Technical summary
[what was done, what was concluded, what is open]

### Human reflection
**Understood this session:** ...
**Still unclear:** ...
**Skeptic's view:** ...
**Would do differently:** ...

---
```

---

## [2026-03-26] — Low-Omega Campaign Failure and Retune Missteps

### Technical summary
- Central goal: Recover low-omega training quality, enforce REINFORCE-only policy where requested, and push beyond N=2 while keeping omega-first transfer logic.
- Accomplished: Ran/checked multiple campaigns (`v12b`, `v13`, `v14`) and confirmed two-stage low-omega run behavior; launched higher-N low-omega campaign (`v15`) with staged `omega=0.01` warmup/polish then `omega=0.001` transfer.
- Attempted but did not work: N=2 at `omega=0.001` remained far from target (~+90% vs configured reference) despite repeated retunes; intermediate runs consumed time without solving the core regime failure.
- Decisions made: enforced "no SR for low omega" in executed campaigns; narrowed low-omega focus to explicit staged runs; prioritized getting `omega=0.01` near 0.1% before transferring to `omega=0.001` at higher N.
- Workarounds in place: repeated manual retune/relaunch loops and checkpoint-chain continuation instead of resolving deeper regime mismatch; errors at some low-omega points are still sensitive to reference-table limitations for unsupported omega values.
- Unverified/assumed: assumption that better `omega=0.01` convergence at higher N will transfer cleanly to `omega=0.001` has not been validated yet.
- Skeptic view: campaign management improved, but there is still no convincing mechanism-level explanation for why `omega=0.001` stalls at high error after multiple restarts.
- Single most important carry-forward: treat `omega=0.001` failure as unresolved core problem and debug by foundation layers (data/sampling -> implementation -> training), not by ad hoc parameter churn.
- Recommended next action: finish `v15` stage-1 diagnostics quickly; if `omega=0.01` does not approach ~0.1% promptly, run targeted ablations (proposal overlap/ESS, weight distribution, reference consistency) before another long transfer run.

### Human reflection
**Understood this session:** "i understand everything, just now why it isnt working as i want"
**Still unclear:** "why it isnt working as i want"
**Skeptic's view:** "all the errors obviously but hidden present"
**Would do differently:** "next session we are to continue working on fixing all the errors"

---

## [2026-03-28] — Adaptive Sigma Fix and 8h N12 Rescue Campaign (Unresolved)

### Technical summary
- Central goal: resolve persistent low-omega transfer failure at `omega=0.001`, then complete corrected transfer runs for N=2/6/12 under bounded wall-clock execution.
- Accomplished: audited overlap/ESS behavior, confirmed adaptive proposal-width logic was not active in runtime path, then implemented and wired adaptive `sigma_fs` into [src/run_weak_form.py](src/run_weak_form.py).
- Accomplished: ran targeted diagnostics showing strong ESS improvement at N=2 transfer regime after adaptation, indicating sampling viability was partially restored in that easier case.
- Accomplished: launched multi-GPU tmux campaigns, including an explicit 8-hour bounded N=12 rescue run (`v17`) across profiles A/B/C/D on available GPUs.
- Attempted but did not work: corrected transfer for N=2/6/12 did not reach target quality; N=12 transfer remained skip-dominated with repeated `ESS < min_ess` reverts.
- Attempted but did not work: 8-hour campaign produced bridge checkpoints for A/C/D but no successful transfer completion; profile B did not complete bridge.
- Why it failed (current best explanation): Layer 1 + Layer 4 issue persists at ultra-low omega for higher N, where proposal overlap and ESS gating interaction still blocks effective optimization progress.
- Decisions made: prioritize diagnostics-first workflow over additional blind retunes; use hard wall-clock campaign limits; run low-omega rescue in tmux across all currently free GPUs.
- Workarounds in place: profile-sweep rescue scripts with varied ESS floors and oversampling; worker end-status inferred from stage logs when wrapper summary files are missing.
- Unverified/assumed: assumption that adaptive widening alone can unlock N=12 `omega=0.001` transfer remains unverified; missing/NaN DMC handling around some low-omega contexts still requires explicit verification.
- Skeptic view: evidence supports a local sampling improvement, but no reproducible end-to-end win at the target hard regime; current method may still be structurally brittle for N=12 low-omega transfer.
- Single most important carry-forward: treat N=12 `omega=0.001` as unresolved and re-open from foundation checks (reference validity, overlap statistics, gating policy) before another long run.
- Recommended next action: run one short, instrumented N=12 transfer ablation sweep focused on ESS-floor/oversample/gating diagnostics with explicit success criteria before launching any new long campaign.

### Human reflection
**Understood this session:** "nothing worked, that is the question answers"
**Still unclear:** "nothing worked, that is the question answers"
**Skeptic's view:** "nothing worked, that is the question answers"
**Would do differently:** "nothing worked, that is the question answers"

---
