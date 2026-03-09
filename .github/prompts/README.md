# AI Prompt Library

Reusable prompts for staying in control when using LLMs for scientific and technical work.
These are designed around one principle: **you are the author, the model is the tool**.

---

## Philosophy

The goal is to preserve deep understanding while using AI assistance. Every prompt here is designed to:
- Keep you reasoning, not outsourcing reasoning
- Force planning and hypothesis formation before code is written
- Maintain full visibility into what the model is doing and why
- Build genuine understanding, not just working code

---

## Workflow

### Starting a session
1. Open or update `CONTEXT.md` (copy from `CONTEXT-template.md` if new project)
2. Paste `CONTEXT.md` into the chat, or use it in a Claude Project / `.cursorrules`
3. Use `/session-open` to provide context in structured form

### During a session
- `/plan-first` — before any implementation
- `/minimal-implementation` — when you want the core logic only
- `/socratic-mode` — when you want to learn, not just get an answer
- `/debug-with-hypothesis` — before pasting any error message
- `/explain-back` — after receiving any non-trivial explanation
- `/adversarial-review` — after any implementation you want honestly critiqued
- `/refactor-with-understanding` — when cleaning up code
- `/add-instrumentation` — when you need visibility into a running system

### For scientific/research work
- `/architecture-decision` — before committing to a model architecture or method
- `/experiment-design` — before writing any experiment code
- `/deep-explanation` — when you need to understand a concept from first principles
- `/literature-synthesis` — before starting on a new technical area

### Ending a session
- `/session-close` — always run this, paste the output into `CONTEXT.md`

---

## The single most important habit

After any non-trivial session, before closing: use `/explain-back` on the most complex thing you accepted without fully understanding. The discomfort of not being able to explain it is the signal you need.

---

## Files

| File | Purpose |
|------|---------|
| `plan-first.prompt.md` | Force planning before coding |
| `socratic-mode.prompt.md` | Learn via questions, not answers |
| `minimal-implementation.prompt.md` | Skeleton only, core logic exposed |
| `adversarial-review.prompt.md` | Honest critique of code |
| `debug-with-hypothesis.prompt.md` | Hypothesis-first debugging |
| `explain-back.prompt.md` | Retrieval practice via correction |
| `session-open.prompt.md` | Start session with full brain dump |
| `session-close.prompt.md` | End session, update CONTEXT.md |
| `architecture-decision.prompt.md` | Method/architecture tradeoff analysis |
| `add-instrumentation.prompt.md` | Logging and runtime visibility |
| `deep-explanation.prompt.md` | First-principles concept explanation |
| `refactor-with-understanding.prompt.md` | Step-by-step transparent refactoring |
| `literature-synthesis.prompt.md` | Scientific literature overview |
| `experiment-design.prompt.md` | Rigorous experiment design |
| `CONTEXT-template.md` | Template for project context file |
