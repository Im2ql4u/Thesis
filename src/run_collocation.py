"""Thin entrypoint for weak-form collocation training.

Keeps CLI behavior in one place (`run_weak_form.main`) while giving
orchestrators a stable, intent-revealing runner path.
"""

from run_weak_form import main


if __name__ == "__main__":
    main()