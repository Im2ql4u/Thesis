# Tool Interfaces

You must use these interfaces when invoking reusable agent capabilities.

## Design Rules

- Inputs and outputs are explicit.
- Output is structured and easy to parse in prompts.
- Tools report uncertainty and failure reasons directly.
- Tools do not make product decisions; experts/modes do.

## Dispatch Convention

These interfaces are behavior contracts, not direct API endpoints.

They must be mapped to editor-native capabilities (search, file edit, task/test run, diff inspection, etc.) while preserving the same input/output semantics.

Whenever a tool contract is invoked in prompt-driven execution, report it in this compact form:

```
Tool: <name>
Input: <key fields>
Action: <what was executed via editor-native tools>
Output: <structured result using this interface schema>
```

If an expected output field is unavailable, set it to `unknown` and state why.

## specialist_output

Every expert must emit this normalized output block at the end of their response, in addition to their domain-specific analysis. This is the contract the fusion step uses to merge multi-expert outputs.

```
Specialist: <expert name>
Claims: <numbered list of specific, falsifiable claims>
Evidence: <concrete outputs, metrics, or checks supporting each claim>
Confidence: high | medium | low
Risk: low | medium | high
Recommendation: <single concrete action>
Open uncertainty: <what is not yet confirmed>
Notes: <optional — bounded free-form context, max 3 sentences>
```

Rules:
- Confidence must be justified by evidence, not self-assessed from fluency.
- If evidence is indirect or single-source, confidence cannot be "high."
- If the expert cannot fill a required field, it must state why rather than omit it.
- The Notes field is the only place for free-form prose. All other fields are structured.

## navigate

Input:
- query: string
- scope: entire_codebase | current_file | tests_only | docs_only
- max_results: integer

Output:
- results: list of {file, line, snippet, relevance}
- notes: optional assumptions or search limitations

## edit_atomic

Input:
- target_file: string
- intent: string
- change_unit: string
- acceptance_check: string

Output:
- applied: boolean
- diff_summary: string
- changed_lines_estimate: integer
- follow_up_check: string
- failure_reason: optional string

## test_quick

Input:
- command: string
- timeout_seconds: integer
- expected_signal: optional string

Output:
- passed: boolean
- key_output: string
- duration_seconds: number
- failure_signature: optional string

## verify_intent

Input:
- stated_intent: string
- observed_diff: string
- observed_behavior: string

Output:
- alignment: high | medium | low
- mismatches: list
- recommendation: accept | iterate | escalate

## evaluate_risk

Input:
- claim: string
- baseline_context: string
- evidence_summary: string

Output:
- risk_level: low | medium | high
- claim_validity: supported | weakly_supported | unsupported
- uncertainty: low | medium | high
- recommendation: ship | iterate | rollback

## codebase_impact

Input:
- changed_files: list
- dependency_context: string

Output:
- boundaries_touched: list
- debt_risk: low | medium | high
- safe_sequence: list
- recommendation: commit_now | split_change | refactor_first

## prioritize_next

Input:
- candidate_actions: list
- constraints: string

Output:
- ranked_actions: list of {action, impact, effort, confidence, risk}
- top_recommendation: string
- defer_list: list

## reproduce

Input:
- commit_ref: string
- config_ref: string
- environment_notes: string

Output:
- reproducible: boolean
- missing_requirements: list
- state_health: healthy | degraded | unknown
- recommendation: proceed | repair_environment | stop

## synthesize

Input:
- signals: list of specialist_output blocks (or structured findings from modes)
- constraints_context: string (contents of CONSTRAINTS.md verified entries, if available)
- negative_memory_context: string (relevant entries from DECISIONS.md Negative Memory, if available)

Output:
- merged_understanding: string (max 3 sentences)
- resolution_log: list of {tension, resolution, evidence_basis} (or "unresolved — needs <check>")
- recommendation: string (single concrete action)
- confidence: high | medium | low
- confidence_basis: string
- risk: low | medium | high
- what_could_make_this_wrong: string
- quality_signals:
  - robustness: string
  - interpretability: string
  - coherence: string
  - reuse_potential: string
