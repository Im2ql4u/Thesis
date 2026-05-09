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
