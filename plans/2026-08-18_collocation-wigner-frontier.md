# Q3 collocation frontier: breaking the deep-Wigner wall without MCMC

## The question
Can a many-body wavefunction be trained to DMC quality WITHOUT MCMC during training, using only
importance-sampled collocation from an analytic proposal? The thesis showed YES for N=6 (all omega, to
+0.24%) and N=12 (omega>=0.1), but hits a wall: N=12 at omega<=0.01 fails (ESS->1), and N=20 is stuck at
+1.45%. This document records why, and the plan to push past it MCMC-free.

## Root cause (confirmed)
Every proposal in the thesis pipeline is an ORIGIN-centered Gaussian mixture (run_weak_form.py:310,
`torch.randn * s`, widths sigma_f in {0.8,1.3,2.0}*ell). But at low omega the electrons crystallise into
a Wigner molecule on a SHELL far from the origin, so |Psi|^2 has no mass where the proposal puts it.
Classical N=6 = (1,5): 1 centre + 5 on a ring at 1.334*omega^(-2/3), i.e. 1.96 / 2.87 / 4.22 oscillator
lengths at omega = 0.1 / 0.01 / 0.001. No non-origin proposal was ever tried; the one coverage-fix
attempt (Langevin refinement) failed because short non-equilibrium dynamics bias the REINFORCE gradient.

## What works: the Wigner-ring proposal (validated)
A ring proposal -- radial Gaussian x angle for the ring electrons + a centre Gaussian -- lifts ESS on
TRUSTED states, no training needed:
  origin-Gaussian (thesis) : ESS = 1.5 / 4096   (already collapsed at omega=0.01)
  ring (1,5), hand-tuned   : ESS = 22.5
  ring (1,5), fitted to data: ESS = 47.5         (15-32x lift; fitting doubles it)
Pure (1,5) beats a (1,5)+(0,6) mixture at N=6 (the state is (1,5); mixing dilutes). This alone extends
collocation's reach at MODERATE Wigner (omega=0.01-0.1) where the thesis proposal collapses.

## The deep-Wigner wall (confirmed, quantified): ANGULAR CRYSTALLISATION
Nearest-neighbour angular gap std of the 5 ring electrons, well-converged states:
  omega = 1.0 / 0.1 / 0.01 / 0.001  ->  std = 40.8 / 33.3 / 25.5 / 6.7 deg  (72 = perfect pentagon)
The molecule goes from a "ring liquid" (uniform angle) to an angular CRYSTAL (electrons pinned at 72 deg,
gaps <20deg never occur). A uniform-angle ring almost never lands on the crystalline config at
omega=0.001 -> ESS -> 1. This EXPLAINS the thesis wall and WHY the ring works at 0.01 (liquid) but not
0.001 (crystal).

## Why factorised proposals struggle at omega=0.001
The target is a highly-correlated ~12-DOF distribution: radial pinning x angular crystal x e-e
avoidance. Importance sampling in high dimensions needs the proposal to match the CORRELATIONS, not just
the marginals; a factorised proposal's joint coverage is the product of per-DOF coverages and collapses.

## The plan (MCMC-free, "done properly")
1. **Dirichlet-gap crystalline proposal (angular fix, correct & tractable).** Model the 5 angular gaps
   (which sum to 2pi) as gaps/2pi ~ Dirichlet(alpha,...,alpha): exact density, exact sampling, and one
   parameter alpha interpolates liquid (small) <-> crystal (large). Fit alpha to the measured gap
   variance (alpha ~ 92 for std 6.7deg). This is the consistent version of the first (buggy) crystalline
   test, whose sampler and density did not match. TEST ESS first.
2. If the Dirichlet-gap fix lifts ESS but residual correlations (radial-angular, e-e) still limit it,
   go **autoregressive**: sample electrons sequentially, each conditioned on the placed ones (captures
   all correlations; density = product of conditionals, tractable). Symmetrise over placement order.
3. Only if needed, a learned normalising-flow proposal.
4. Training recipe once ESS is healthy: cascade in omega (small steps), re-fitting the proposal
   (r0, sigma_r, alpha) to the current |Psi|^2 at each step (adaptive), REINFORCE + MAD clipping, Adam
   (CG-SR is unstable at ultra-low omega). Always heavy-VMC-evaluate the final energy.

## Discipline
Test ESS on trusted states BEFORE any training (cheap, decisive). Keep sampler and density CONSISTENT
(the first crystalline test failed this). No MCMC anywhere.
