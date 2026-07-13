# Contributing Guidelines

SimYuj is a deterministic, event-driven quantum network simulator. Contributions
should keep the core library small, reusable, and predictable.

This guide covers the workflow for bugs, documentation, tests, maintenance, and
small library improvements. Keep changes focused, preserve deterministic
simulation behavior, and avoid mixing unrelated work in one PR.

---

## 1. Code Style & Type Hints

* All code **must use Python type hints**.
* Aim for clean, readable functions with descriptive names.
* Use **docstrings** for all public functions and classes.
* Follow the project's formatting and linting rules (see below).

---

## 2. Branching & Pull Requests

* **Do NOT push directly to `main`.**

* Create a feature branch for every change.
* Branch names should include the related GitHub issue number.

Format:

    <type>/<issue-number>-<short-description>

Examples:

    feature/12-event-data-model
    bugfix/34-timeline-ordering
    refactor/21-signal-history

* Including the issue number improves traceability between branches, pull requests, and discussions.

* All changes must come through a **Pull Request (PR)**.

* Every PR requires **at least one reviewer**.

* Keep PRs **small and focused**; avoid mixing unrelated changes.

---

## 3. Testing Requirements

* Behavior changes and bug fixes should include targeted tests.
* Tests must be **deterministic** — use fixed RNG seeds when randomness is involved.
* All tests must pass before merging.

Run tests using:

```
pytest
```

---

## 4. Code Quality Tools

We use automated tools to keep the codebase consistent:

* **black** — code formatting
* **flake8** — linting
* **isort** — sorting imports
* **mypy** — static type checking

Run these before committing:

```
black src tests examples docs/source --check
isort src tests examples docs/source --check-only
flake8 src tests examples docs/source
mypy src/simyuj
pytest
python -m sphinx -b html docs/source docs/build/html
```

Or install the pre-commit hooks:

```
pre-commit install
```

These hooks will automatically run checks before allowing a commit.

---

## 5. Commit Message Format

Follow this conventional structure:

```
<type>: short message

Optionally, reference a GitHub issue:

[Optional longer explanation]
```

**Types include:**

* `feat` — new feature
* `fix` — bug fix
* `test` — adding or updating tests
* `docs` — documentation changes
* `refactor` — restructuring code without changing behavior
* `chore` — maintenance tasks

**Examples:**

```
feat: implement event comparator
fix: correct timeline schedule validation
docs: add architecture overview for timeline engine
```

---

## 6. Design Principles

* **Prefer clarity over cleverness.**
* Keep components **loosely coupled** and well defined.
* Make **deterministic behavior** the default.
* Document assumptions and non-trivial logic in code.
* Avoid premature optimization; focus on correctness first.
* Favor pure functions when possible; keep side effects controlled.

---

## 7. Getting Help

If you're unsure about a design or implementation:

* Open a GitHub Discussion / Issue
* Create a draft PR early and request feedback

We encourage communication early so problems are solved before becoming large.

---

Thanks for contributing!
