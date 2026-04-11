---
name: results-analysis
description: 'Systematically analyze experiment results after training. Use when a training run completes, when comparing multiple experiments, when preparing results for sharing, or when results are confusing or unexpected. Covers metrics, error patterns, comparisons, and honest assessment.'
---

# Results Analysis

## When to Use
- After training run completes
- Comparing multiple experiments
- Preparing results for a report or discussion
- Results are surprising, inconsistent, or hard to interpret

## Procedure

### 1. Collect raw results
- Parse training logs for final metrics
- Load best checkpoint metrics (not just last epoch)
- Record exact commit hash and config used
- Note training duration, GPU used, any interruptions

### 2. Compute metrics properly
- Report on HELD-OUT test set, never on training data
- Include confidence intervals or variance across seeds (minimum 3 seeds for any claim)
- Report ALL metrics specified in the plan, not just the best-looking ones
- If metrics contradict each other, flag this explicitly

### 3. Compare against baselines
```
Metric         | This run | Baseline | Δ      | Significant?
---------------|----------|----------|--------|-------------
<metric>       | <value>  | <value>  | <diff> | <yes/no/untested>
```
- Use appropriate statistical tests (paired t-test, bootstrap CI)
- "Better" means statistically significant improvement, not just a larger number

### 4. Analyze error patterns
- Look at WORST predictions, not just averages
- Check if errors are random or structured (systematic bias, specific subgroups, edge cases)
- If structured errors exist, hypothesize the cause
- Compare error distributions between this run and baseline

### 5. Honest assessment
Answer each question in one sentence:
- What did we learn that we didn't know before?
- What does a skeptic's best counterargument look like?
- What DON'T these results tell us?
- Would we be comfortable sharing these results externally?
- What is the single most important follow-up experiment?

### 6. Produce report
Use the standard results template from the implement prompt. Do not skip any section. The "What is unexplained" section is the most important — this is where new insights come from.

## Acceptance Criteria
- [ ] All planned metrics computed on held-out data
- [ ] Comparison table against baseline produced
- [ ] Error analysis performed (not just aggregate metrics)
- [ ] Honest assessment completed (all 5 questions answered)
- [ ] Report written with no sections skipped
