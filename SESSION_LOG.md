# Session Log

Last session: [2026-04-06] — Higher-N Phase 1 execution (N=12 smoke + N=20 post-bugfix diagnostics)

## Next session
**Recommended starting point:** Decide whether to proceed with Phase 2 in [plans/2026-04-06_higher-N-scaling.md](plans/2026-04-06_higher-N-scaling.md): N=12 full DiagFisher vs Adam campaign (6 runs, 4000 epochs each).
**Open questions:** Is N=20 still sampling-limited post-bugfix (ESS mean 4.8 at omega=1.0), and should N=20 be paused for a Layer-1 sampling plan while N=12 continues?
**Unverified assumptions:** Assumed warm-start N=20 checkpoints trained pre-bugfix remain usable under corrected sampling; assumed 100-200 epoch diagnostics are sufficient to gate later long runs.
**Active workarounds:** None in code. Operationally, N=20 is gated by ESS evidence before investing long-run compute.
**Foundation status:** Verified: Phase 1 launcher, N=12 DiagFisher smoke, N=20 post-bugfix diagnostics, and summary artifact in outputs/higher_n/phase1/phase1_summary.txt.
**Context freshness:** fresh
**Contradiction flags:** yes — post-bugfix N=20 still shows poor ESS and large VMC error, so optimizer transfer alone may not solve high-N instability.

See ARCHIVE.md for full history.
