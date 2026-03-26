# Session Log

Last session: [2026-03-26] — Low-Omega Campaign Failure and Retune Missteps
See ARCHIVE.md for full history.

## Next session
**Recommended starting point:** Continue from `outputs/2026-03-26_1403_campaign_v15_n6_lowomega_2stage_reinforce_only/` and evaluate whether N=6 `omega=0.01` converges toward ~0.1% before attempting to interpret `omega=0.001` transfer.
**Open questions:** Why does low-omega transfer remain near +90% at `omega=0.001` despite stable `omega=0.01` performance and repeated retunes?
**Unverified assumptions:** Assumed that better higher-N `omega=0.01` checkpoints will transfer materially better to `omega=0.001`; assumed current sampling/reward settings are not the dominant failure mode.
**Active workarounds:** REINFORCE-only low-omega policy; staged warmup/polish then transfer scripts (`scripts/campaign_v14_lowomega_2stage_reinforce_only_tmux.sh`, `scripts/campaign_v15_n6_lowomega_2stage_reinforce_only_tmux.sh`) used as tactical recovery without resolving root mechanism.
**Foundation status:** Verified: training pipelines run to completion and references exist for N=2/N=6 at `omega=0.001` and `0.01`; Assumed: optimization failure is mostly tuning-related rather than remaining data/sampling implementation pathology.
**Context freshness:** fresh
**Contradiction flags:** yes — recent decision to disable SR for low omega conflicts with 2026-03-24 entry that re-enabled SR globally after sampling bugfix.
