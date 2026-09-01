# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Numerical implementation and calibration of a model of optimal public finance
under automation shocks, developed alongside a paper on the same topic. The
model itself is not yet implemented — this repo is currently scaffolding.

## Commands

- Install dependencies: `uv sync`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_import.py::test_package_imports`
- Add a runtime dependency: `uv add <package>`
- Add a dev-only dependency: `uv add --dev <package>`
- Run a script inside the project environment: `uv run python <script>.py`
- Launch Jupyter for exploration: `uv run jupyter lab`

## Structure

- `src/tai_public_finance/` — the installable package; model code will live here.
- `tests/` — pytest suite.
- `research-context/` — maintained summary of the research handoff pipeline; see below.
- `runs/` — immutable records of material computational executions.
- Dependencies are managed with `uv` and locked in `uv.lock` — use `uv add`/`uv remove`
  rather than hand-editing `pyproject.toml`'s dependency lists, so the lockfile stays
  in sync.

Architecture beyond this isn't established yet — the model's state variables,
solution method, and calibration targets will follow the paper's structure as
it gets implemented.

## Research handoff from Codex

A sibling, non-git-tracked workspace (`../TAI public finnace codex`) is where
Nathan and Codex do the theory, derivations, and formal specification work for
this project. Treat it as read-only reference material — read freely, never
edit or delete anything there.

Work arrives here as an **approved computational specification (CS###)** from
that workspace's `computation/` registry, not as a paraphrased chat request:
"the implementer receives an approved CS, not a chat transcript." See
[research-context/](research-context/) for a maintained, periodically
refreshed summary of that pipeline — `handoff-contract.md` for the object
model and lifecycle, `portfolio-snapshot.md` for current status. As of the
last sync, nothing is approved yet: do not start implementing a specific
spec's model equations until its status reaches `approved`.

## Computational specification workflow

Once a specification is approved and implementation begins:

- **Branch:** `cs<NNN>/<slug>` for spec work (e.g.
  `cs001/riccati-sylvester-lyapunov-solve`). Everything else uses
  `chore/<slug>`. ID-first is the sanctioned exception for machine
  identifiers (branch refs, package names) in the codex workspace's naming
  convention — prose still leads with the name, e.g. "LQ anchor and impulse
  responses (CS001)".
- **Worktree:** every task gets one, always — no exceptions for "small" or
  chore work. The primary checkout (this directory, no path suffix) stays
  on `main` only; nothing gets implemented there directly:
  `git worktree add "../TAI public finnace claude-cs<NNN>-<slug>" -b cs<NNN>/<slug>`
  (substitute `chore/<slug>` for non-spec work)
- **Branch base:** off `main` (or the latest commit merged into it), not off
  another unmerged branch, unless stacking is a deliberate, stated choice.
- **Code layout:** a shared `src/tai_public_finance/primitives/` (canonical
  parameter/calibration object) plus one package per spec, e.g.
  `src/tai_public_finance/cs001_lq_anchor/`, internally separated into
  equation construction, solver, an *independent* diagnostics/residual
  evaluator, and reporting/CLI, per that spec's own implementation-handoff
  section. Create a spec's package only once it's approved — not before.
- **Runs:** see [runs/README.md](runs/README.md).
- **GitHub:** one issue per spec, titled `Implement <Title> (CS###)`, opened
  when a spec reaches `approved`, labelled `computation`. PRs link the issue
  and name the spec + version/fingerprint implemented.

## Multi-session coordination

Nathan runs several concurrent Claude Code sessions against this repo as a
matter of course — check [STATUS.md](STATUS.md) and `git worktree list`
before assuming you have it to yourself, and update your own STATUS.md row
when you start and finish a task. Before touching a branch someone else's
row claims, use `ListAgents` and message them directly rather than guessing
— it resolves ambiguity in one round trip and has repeatedly beaten every
other way of figuring out who's doing what.
