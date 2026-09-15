# Wall time per gradient step on ml7 (RTX 2080 Ti, float64, shared GPUs)

Source: `scripts/bench_walltime.py`, `scripts/bench_score_matrix.py`; logs in this folder,
JSON in the sibling `2026-09-14_walltime_cuda_*` and `2026-09-14_score_matrix_cuda` folders.
One ansatz (CTNN V-cycle Jastrow + CTNN backflow, 108 485 parameters), omega = 1, untrained.
VMC: 3072 walkers, 30 Metropolis sweeps/step. Collocation: 16 x 4096 proposals -> 4096 points.
N = 20: 1024 walkers and 1024 collocation points (memory on the shared cards). SR on 1024
samples. All eight GPUs were in use by other users at 67-100 % utilisation throughout.

Seconds per step (sample + local energy + update). "SR loop" builds the score matrix with one
backward pass per sample (the thesis's code); "SR vmap" with torch.func (identical O, rel. diff
<= 1e-14). Entries marked * are composed from the Adam route's sample/energy and the separately
timed score-matrix + solve cost; at N = 6 the composition matches the measured SR step to 1 %.

| N  | VMC+Adam | colloc+Adam | VMC+SR loop | colloc+SR loop | VMC+SR vmap | colloc+SR vmap |
|----|----------|-------------|-------------|----------------|-------------|----------------|
| 2  | 5.1      | 4.2         | 26.4        | 24.3           | 7.6*        | 7.2*           |
| 6  | 16.3     | 17.6        | 62.1        | 63.4           | 18.6*       | 19.9*          |
| 12 | 64.7     | (pending)   | 184.9*      | -              | 66.3*       | -              |
| 20 | 75.4     | 67.0        | 387.0*      | 378.6*         | 78.6*       | 70.2*          |

Score matrix, B = 1024: loop 19.1 / 45.6 / 121.0 / 312.2 s, vmap 1.96 / 1.83 / 2.37 / 3.80 s
at N = 2 / 6 / 12 / 20 (10-82x); Gram + solve 1.15 s at every N.

Per retained sample, sampling costs 1.8 ms (VMC) vs 0.85 ms (collocation) at N = 6 and
16.9 vs 8.7 ms at N = 20. The exact-Laplacian local energy is 62-84 % of an Adam step at N >= 6.
