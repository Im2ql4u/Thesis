# Orchestrator Contract

You must follow this routing and state model for direct asks, prompt-driven workflows, and expert escalation.

## Input Types

- Build/change request
- Debug/failure request
- Validate/review request
- Plan-generation request
- Planning/prioritization request
- Operations/reproducibility request

## Routing Policy

1. Classify request intent.
2. Select primary mode:
- Build/change -> implement
- Debug/failure -> diagnose
- Validate/claim -> review (validate)
- Plan-generation -> plan (no code changes)
- Planning/ordering -> prioritization expert
- Repro/resume/run-health -> operations expert
3. Apply Execution Kernel loop.
4. Invoke gates and experts only when trigger conditions are met.

Mode-lock rule:
- If the user asks for plans/options only, do not execute implementation in the same turn.
- Require explicit user transition signal before switching from plan to implement.

## State Model

Track these fields continuously:
- task_goal
- active_hypothesis
- requested_mode
- last_action
- last_evidence
- current_risk
- next_action
- escalation_reason (optional)

State is append-only per cycle so interrupted work can resume without losing reasoning context.

## Core Cycle

1. Plan
- Define next atomic unit.
- Define exact evidence check.

2. Execute
- Run selected tool operation(s).
- Capture structured output.

3. Evaluate
- Compare output to expected result.
- Check atomicity and clarity gates.

4. Route
- Continue in same mode, or
- Retry with smaller unit, or
- Escalate to expert.

Workspace safety route:
- If a request includes "clean", "reset", or "remove", route through cleanup safety policy first.
- Produce a path-level approval table and await explicit approval before any removal/reset action.

## Expert Invocation Triggers

Budget: at most 2 experts per cycle. Default path is zero experts. If a third expert appears necessary, stop and escalate to the user.

Each trigger below requires a concrete observable — not a subjective judgment. If the observable is not met, do not invoke the expert.

Evaluation expert:
- A result claim is about to be accepted AND no executable test validates it.
- A metric delta is being interpreted AND no baseline value exists for comparison.
- Confidence is stated as high but evidence is indirect or single-source.

Codebase expert:
- Diff touches files in 3+ distinct directories or modules.
- A new dependency is introduced or an existing module's public interface changes.
- A workaround is being committed that was originally flagged as temporary.

Prioritization expert:
- 3+ candidate next actions exist and none has a clear evidence advantage.
- Estimated effort for the top candidate exceeds 1 session and alternatives exist.

Operations expert:
- Training run ETA exceeds 30 minutes.
- A checkpoint or resume path is being relied on but has not been tested.
- Environment or dependency versions differ from the last successful run.

Synthesis expert (does not count against the 2-expert budget):
- Two or more experts were invoked in the same cycle and their outputs need merging.
- Diagnose and brainstorm (or any two modes) produced different framings of the same problem.
- Evidence from implementation contradicts the plan hypothesis.
- Review found issues that span multiple expert domains.

## Fusion Step

When two or more experts are invoked in the same cycle, their outputs must be merged before routing continues. Do not pick one expert's output and ignore the other. Do not pass both outputs as separate blobs into the next action.

### Synthesis expert routing

When two or more expert outputs need merging, route through the **synthesis expert** (`@experts/synthesis`) instead of applying mechanical fusion. The synthesis expert performs evidence-weighted reasoning to reconcile agreements, tensions, and contradictions — not just field-level comparison.

Invoke the synthesis expert with all expert outputs pasted as input. The synthesis expert:
1. Maps the signals (what each expert claims and why)
2. Classifies relationships (agreement, complementary, tension, contradiction)
3. Resolves conflicts by evidence quality, conservatism, and project constraints
4. Checks the fused recommendation against `CONSTRAINTS.md` and Negative Memory
5. Emits one fused recommendation with confidence, risk, and quality signals

The synthesis expert does NOT count against the 2-expert-per-cycle budget — it operates on expert outputs, not alongside them.

If the synthesis expert cannot resolve a contradiction (evidence is insufficient), it will recommend a specific disambiguating check as the next action rather than forcing a resolution.

### Fallback fusion procedure (when synthesis expert is unavailable)

If the synthesis expert cannot be invoked (e.g., tool limitations), apply the mechanical fusion procedure:

1. Extract the structured fields from each expert output (claim, evidence, confidence, risk, recommendation, uncertainty).
2. For each field, compare across experts:
   - **Agreement:** both experts align on the field value → use the shared value.
   - **Complementary:** experts address different aspects → combine into one merged field.
   - **Conflict:** experts disagree on the same aspect → resolve using the tie-break rules below.
3. Emit one fused output using the standard specialist output shape (see `tools/INTERFACES.md` → `specialist_output`).

### Conflict tie-break rules (apply in order)

1. **Evidence weight:** the expert with more concrete, verifiable evidence wins. Self-reported confidence without evidence does not count.
2. **Conservatism:** if evidence quality is tied, the more conservative recommendation wins (rollback > iterate > ship; refactor_first > split_change > commit_now).
3. **Escalation:** if conservatism is also tied, the fusion step does not resolve — escalate to the user with both positions stated clearly.

### Fused output shape

```
Fused state:
- merged_claims: <combined claims from all experts>
- supporting_evidence: <strongest evidence for each claim>
- overall_confidence: <lowest confidence across experts, unless evidence justifies raising it>
- overall_risk: <highest risk across experts>
- recommendation: <single action — resolved by tie-break if conflicting>
- open_uncertainty: <union of all expert uncertainties>
- conflict_log: <any disagreements and how they were resolved, or "none">
```

The fused output is the sole input to the routing decision. Individual expert outputs do not feed forward independently.

## Output Contract

Every completed cycle returns:
- status: done | iterate | blocked | escalated
- summary: one-sentence result
- evidence: command/check and result
- changed_artifacts: list
- decision_rationale: short explanation
- uncertainty: explicit remaining unknowns
- safety_note: destructive risk status and approval basis
