"""Kernel-picture analysis program for neural quantum states.

Generalizable (any N, any omega) tooling to analyse trained NQS wavefunctions
through the tangent-kernel picture. See plans/2026-06-13_kernel-analysis-program.md.

Modules
-------
reference   : exact / reference wavefunctions (2-electron parabolic dot, any omega)
system      : build a generalizable ansatz + sampler + log|Psi| closure
diagnostics : O / S / K kernel builder, spectra, SR-vs-plain alignment, GS-quality,
              learned two-body correlation extraction
"""
