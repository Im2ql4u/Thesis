---
applyTo: "config/**,**/*.yaml,**/*.yml"
---

# Config — experiment configuration standards

- Every tunable parameter must live in a config file, not in source code.
- Configs must be valid YAML. Validate on load with a schema or explicit checks.
- Include comments documenting the purpose and valid range of each parameter.
- Use descriptive keys: `learning_rate`, not `lr`. `batch_size`, not `bs`.
- Group related parameters under sections: `model`, `training`, `data`, `evaluation`.
- Never store absolute paths. Use relative paths or environment variables.
- Include a `seed` field for reproducibility.
- Config files are immutable per run. Copy config to results directory at run start.
