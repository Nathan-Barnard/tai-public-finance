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
- Dependencies are managed with `uv` and locked in `uv.lock` — use `uv add`/`uv remove`
  rather than hand-editing `pyproject.toml`'s dependency lists, so the lockfile stays
  in sync.

Architecture beyond this isn't established yet — the model's state variables,
solution method, and calibration targets will follow the paper's structure as
it gets implemented.
