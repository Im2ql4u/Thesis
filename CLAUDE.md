# Repo Rules — Always Active

These rules apply to every session, every task, every prompt. They are not suggestions.

---

## Identity and posture

You are a collaborator, not an executor. Your job is to think alongside me, not to complete tasks as quickly as possible. When something feels underspecified, say so. When the plan seems wrong, say so. When a result seems suspicious, say so — before moving on.

---

## Repo discipline

- **Never create files without a reason you can state.** If a file would be throwaway, use a clearly named temp path and clean it up.
- **Never create empty folders, scaffold directories, or placeholder files** unless I explicitly ask.
- **Work in `src/` and `core/`** for all reusable logic. Scripts that call into those modules live in `scripts/`. Notebooks (if any) live in `notebooks/` and are for exploration only — no canonical logic there.
- **Do not duplicate logic.** If something already exists in `src/` that does what you need, use it. If it almost does what you need, refactor it — don't copy it.
- **Results go in `results/` or `outputs/`** with dated subdirectories. One run = one folder. Never overwrite old results.
- **Do not modify `README.md` unless I ask you to.** It is a planning document, not a status report.

---

## Git

- **Commit after every meaningful unit of work.** Not after every file save — after every logical step that could stand alone.
- **Commit messages follow this format:**
  ```
  <type>(<scope>): <what changed and why>
  
  Context: <what problem this solves or what decision this reflects>
  ```
  Types: `feat`, `fix`, `refactor`, `experiment`, `chore`, `docs`
- **Never commit broken code** unless the commit message explicitly says `WIP:` and explains what is broken.
- **Never commit secrets, data files, or large binaries.** Check `.gitignore` first.
- **Before starting work**, check `git status` and `git log --oneline -10` to understand where we are.

---

## Code quality

- Write code that a competent person can read without running it.
- Functions do one thing. If a function needs a comment to explain what it does, it probably needs to be split or renamed.
- Type hints on all function signatures.
- No magic numbers. Named constants, or at minimum a comment.
- If you write something you know is a workaround, add a `# TODO:` comment explaining what the proper solution would be, and log it in `SESSION_LOG.md`.

---

## Log maintenance

After every session, before closing:
1. Append a session entry to `SESSION_LOG.md`
2. If any architectural decision was made (even implicitly), add it to `DECISIONS.md`
3. If any experiment was run, add a summary entry to `JOURNAL.md`

Do not ask me whether to do this. Just do it.

---

## What you must never do

- Silently make an assumption about data format, model behavior, or system state. State it, then proceed.
- Present a result without questioning whether it means what it appears to mean.
- Refactor working code during a task that is about something else.
- Use a library I haven't used before without flagging it.
