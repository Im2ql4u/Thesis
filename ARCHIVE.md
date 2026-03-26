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
