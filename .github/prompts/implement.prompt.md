---
description: "Full implementation cycle — build, run, examine results honestly, report. Does not stop when code runs. Stops when results are understood. Requires confirmed plan."
agent: agent
---

${input:task:Describe the task to implement. A confirmed plan should already exist.}

# Implement

> **How to use:** `@implement.md` after a plan has been confirmed. This prompt drives the full implementation cycle: build, run, examine results honestly, iterate. It does not stop when the code runs — it stops when the results are understood.

---

## Precondition

A plan exists and has been confirmed. The objective, architecture, training design, and evaluation protocol are specified. If they are not, use `@plan.md` first.

---

## Phase 1 — Before writing any code

### Read the plan and existing codebase

Read the confirmed plan in full. Then read:
- `DECISIONS.md` — active constraints
- All relevant files in `src/` and `core/` — what exists, what can be reused, what must be extended vs. built new

Search the codebase for anything that does what you need before writing new code. Reuse and extend. Do not duplicate.

### Explain what you are about to build

Before writing any non-trivial module or function, state in plain language:

- What this component does — its purpose, not its implementation
- How it fits into the overall plan
- What it assumes about its inputs
- What its output will be used for

Then ask: *"Does that match your understanding of what this should do?"*

Wait for confirmation before writing. If the answer reveals a mismatch, resolve it first. This step costs thirty seconds and can save hours.

### Foundation check

Before building anything new, verify that the foundation beneath it is solid:

- Run a quick check on the data pipeline: does it produce the expected output on a known input?
- Confirm the baseline is implemented and its result is recorded
- Confirm existing components that will be called by the new code are working correctly

If any of these fail, fix them before proceeding. Do not build on an unverified foundation.

---

## Phase 2 — Implementation standards

### Structure

- Reusable logic in `src/` or `core/`. Scripts in `scripts/` call into those — they do not contain logic.
- One script per well-defined task, named for what it does: `train_residual_correction.py`, not `run_v3.py`
- Config in files (`config/` directory, YAML). No hardcoded parameters anywhere. Every run must be reproducible from a commit hash + config file alone.
- Results written to `results/YYYY-MM-DD_<descriptor>/`. One run, one folder. Never overwrite.

### Code quality

- Functions do one thing. If it needs a comment to explain what it does, rename or split it.
- Type hints on all function signatures.
- Named constants with units where relevant: `GRID_RESOLUTION_M = 1000`, `LR_INIT = 1e-3`, `MAX_EPOCHS = 200`
- No silent failures. If something can go wrong, it should raise an exception with a message that says what happened, where, and what the caller should check.
- Numerical stability: after every loss computation and every gradient step, check for NaN/Inf. Do not continue silently if either is detected.

```python
if torch.isnan(loss):
    raise RuntimeError(
        f"Loss is NaN at epoch {epoch}, step {step}. "
        f"Check: input normalization, loss function implementation, "
        f"learning rate ({optimizer.param_groups[0]['lr']:.2e})"
    )
```

- Workarounds get a `# TODO: [proper fix]` comment and are logged.

### Dependencies

Do not introduce a new library without flagging it, explaining why the existing stack does not suffice, and waiting for acknowledgment.

---

## Phase 3 — At decision points and workarounds — stop

### Decision points

When facing a genuine choice between two reasonable options:

1. Stop
2. State what the decision is
3. Present both options with their tradeoffs — not just which you prefer
4. State your recommendation and the specific reason
5. Ask what the human thinks

Do not silently choose. The human must understand why the code is the way it is, not just that it exists.

### Workarounds

When a proper solution requires more work than the current step allows:

1. Stop
2. Explain what was found and why the proper solution is not being used now
3. State what the proper solution would be
4. Ask: proceed with the workaround, or fix the underlying problem first?

Do not log silently and continue.

---

## Phase 4 — GPU and remote execution

Check GPU availability explicitly before any training:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  {props.name} | {props.total_memory / 1e9:.1f} GB")
else:
    # If GPU was required by the plan, do not proceed silently
    raise RuntimeError(
        "GPU not available. This training job requires GPU. "
        "Check CUDA installation or run on a GPU machine."
    )
```

For any run exceeding a few minutes on a remote machine, use `tmux`:

```bash
tmux new-session -d -s run_name \
  'python scripts/train_xyz.py --config config/xyz.yaml \
   2>&1 | tee logs/train_xyz_$(date +%Y%m%d_%H%M).log'

# To monitor:
tmux attach -t run_name

# To detach and leave running:
# Ctrl+B then D
```

Log GPU utilization and memory at regular intervals during training. Use `torch.cuda.memory_allocated()` or a periodic `nvidia-smi` call in the training loop.

---

## Phase 5 — Git during implementation

```bash
# Before starting
git status
git log --oneline -5

# For a new hypothesis or approach
git checkout -b hypothesis/<short-description>

# Commit after each logical unit — not at the end of everything
git add -p  # stage selectively
git commit -m "feat(scope): what was added and why

Context: what decision or problem this reflects"
```

Do not commit until the unit runs without error. Use `WIP:` prefix and explanation if committing broken state is necessary.

---

## Phase 6 — Testing

For every new module in `src/` or `core/`:

- At least one test against a known-answer case. Not "does it run" — does it compute the right thing.
- For mathematical functions: test against an analytical result or a known limiting case.
- For data processing functions: test shapes, dtypes, value ranges, and at least one specific known value.
- Tests in `tests/`, runnable with `pytest -v`.

Write the test before or immediately after writing the function. Not at the end.

---

## Phase 7 — Run and examine results

This is where most prompts stop. This one does not.

### Run the code

Execute the plan. Monitor actively:

- Loss curves: train and validation together
- Gradient norms per layer (or at minimum the global norm)
- Loss component magnitudes if multi-term loss
- GPU memory and utilization
- Any NaN/Inf warnings

If anything anomalous appears during the run — unexpected loss spikes, NaN, unusually fast convergence — stop, note it, investigate before continuing. Do not run to completion and then examine.

### Examine the outputs

After the run completes, examine actual outputs. Do not skip this. Reading log files and reporting numbers is not examining outputs.

Look at:
- Sample predictions vs. ground truth on a handful of representative cases
- Residuals or errors — are they random, or do they have structure? Structure in the errors means the model has not learned something it should have.
- Performance on the worst cases in the validation set, not just the average
- Whether the output range and distribution are physically plausible
- Whether the improvement over baseline is consistent across the dataset, or concentrated in a subset

---

## Phase 8 — Honest results report

This is mandatory. It is produced in chat, not only in the logs.

```
## Results — [YYYY-MM-DD HH:MM]
**Script:** scripts/<n>.py  |  **Config:** config/<n>.yaml  |  **Commit:** <hash>
**Device:** <GPU + memory>  |  **Duration:** <time>

---

### What was run
<One sentence: what this experiment was testing>

### Raw results
<Metrics with units. No interpretation here — just the numbers.>

Metric         | This model | Baseline | Δ
---------------|------------|----------|----
<metric>       | <value>    | <value>  | <value>

Seeds run: <n>  |  Variance: <std or range across seeds>

### What these results mean
<Interpretation: what does this tell us about the model and the problem.
Not "the model performs well" — what does it actually tell us.>

### What these results do NOT tell us
<What cannot be concluded from this experiment alone.
What alternative explanations exist for this outcome.>

### What is unexplained
<Anything in the results that is surprising, inconsistent, or not understood.
These are the most important things in this section.>

### What a skeptic would say
<Honest critique: what would someone trying to find problems with this result say?
What are the weakest points? What is the most likely methodological concern?>

### Issues encountered during the run
<Everything that went wrong, required a workaround, or was unexpected.>

### Active workarounds
<Any TODOs introduced, with their TODO references.>

### Output location
results/<dated-folder>/

### Recommended next action
<What this result implies we should investigate or change — not what confirms the plan,
but what the result genuinely points toward.>
```

---

## Phase 9 — Mid-implementation understanding check

After completing a significant component — before moving to the next — ask:

*"Here is what was just written: [two-sentence description]. What would you expect it to produce on [simple specific input]?"*

If the answer is wrong, correct it and explain why before continuing. If the answer is right, build on it. The code and the understanding should grow together.

---

## Phase 10 — Before declaring done

Five gates. All five.

1. Code runs and produces sensible output on a known input
2. NaN/Inf checked in all outputs
3. Committed with a meaningful message
4. Honest results report produced (Phase 8) if a run was performed
5. Chat-facing summary produced:

```
**What was done**
- <file>: <one-line purpose>
[one line per file created or modified]

**Decisions made**
- <decision>: <why — one sentence>

**Workarounds in place**
- <workaround>: <TODO ref and proper fix>

**What I am uncertain about**
<Do not leave this blank. If everything seems fine, say what you would check next
to increase confidence that it actually is.>

**One question for you**
<A specific question about what was just built that you should be able to answer
if you have been following along. Not "any questions?" — something concrete,
answerable, that tests understanding of what was built and why.>
```
