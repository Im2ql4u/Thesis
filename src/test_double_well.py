#!/usr/bin/env python
"""Test script for DoubleWell module."""

import sys

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

# Test imports
print("Testing DoubleWell module...")

try:
    from functions.DoubleWell import (
        double_well_coulomb_interaction,
        double_well_potential,
        sample_double_well,
    )

    print("✓ DoubleWell imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test with mock params
import torch

params = {
    "omega": 1.0,
    "device": "cpu",
    "torch_dtype": torch.float64,
    "n_particles": 2,
    "d": 2,
}

# Register params globally
import utils

utils._runtime_params = params

# Test sampling
print("\nTesting sampling...")
try:
    x = sample_double_well(10, 2, 2, well_separation=4.0, params=params)
    print(f"✓ Sampling works: x.shape = {x.shape}")
    print(f"  Particle 0 mean x: {x[:, 0, 0].mean():.3f} (should be ~-2.0)")
    print(f"  Particle 1 mean x: {x[:, 1, 0].mean():.3f} (should be ~+2.0)")
except Exception as e:
    print(f"✗ Sampling error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test potential
print("\nTesting potential...")
try:
    V = double_well_potential(x, well_separation=4.0, params=params)
    print(f"✓ Potential works: V.shape = {V.shape}, V.mean = {V.mean():.4f}")
except Exception as e:
    print(f"✗ Potential error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test Coulomb interaction
print("\nTesting Coulomb interaction...")
try:
    V_coul = double_well_coulomb_interaction(x, params=params)
    print(f"✓ Coulomb works: V_coul.shape = {V_coul.shape}, V_coul.mean = {V_coul.mean():.4f}")
except Exception as e:
    print(f"✗ Coulomb error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test different separations
print("\nTesting energy behavior at different separations...")
separations = [0.0, 2.0, 4.0, 8.0]
for sep in separations:
    x_test = sample_double_well(100, 2, 2, well_separation=sep, params=params)
    V_trap = double_well_potential(x_test, well_separation=sep, params=params)
    V_coul = double_well_coulomb_interaction(x_test, params=params)
    V_total = V_trap + V_coul

    print(
        f"  d={sep:.1f}: V_trap={V_trap.mean():.3f}, V_coul={V_coul.mean():.3f}, V_total={V_total.mean():.3f}"
    )

print("\n✓ All tests passed!")
