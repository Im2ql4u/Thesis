---
applyTo: "tests/**"
---

# Tests — ML testing standards

- Known-answer tests: verify modules compute the correct value on simple inputs, not just "does it run."
- Use deterministic seeds in tests for reproducibility.
- Tolerance-based assertions for floating point: `assert abs(actual - expected) < 1e-6`.
- Integration tests: verify full pipeline from data loading through model output.
- Smoke tests: 2–5 batches on tiny data, verify loss is finite and decreasing.
- Never test against hardcoded absolute paths. Use `tmp_path` fixtures.
- Test config loading: verify configs parse, contain required fields, and have valid values.
- All tests runnable with `pytest -v`. No manual setup required.
