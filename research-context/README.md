# Research context

A bounded, periodically-refreshed snapshot of the parts of the sibling Codex
research workspace (`../TAI public finnace codex`, read-only, not
git-tracked) that matter for implementation here. This is a summary, not a
copy — the codex workspace is canonical; when in doubt, re-read it directly.

- [`handoff-contract.md`](handoff-contract.md) — the object model, lifecycle,
  and rules governing how work reaches this repo. Stable; changes rarely.
- [`portfolio-snapshot.md`](portfolio-snapshot.md) — current status of each
  computational problem and specification. Volatile; goes stale quickly.

**Last synced:** 2026-09-01, against `computation/roadmap.md`,
`computation/computation_registry.yaml`, `computation/README.md`,
`computation/AGENTS.md`, and `PROJECT_NAMING.md` in the codex workspace.

## Refreshing this

1. Re-read `computation/roadmap.md` and the `problems`, `specifications`, and
   `implementation_repository` blocks of
   `computation/computation_registry.yaml` in the codex workspace. Re-read
   `computation/README.md`, `computation/AGENTS.md`, and `PROJECT_NAMING.md`
   only if something here looks structurally out of date — not every sync.
2. Update `portfolio-snapshot.md`'s table and the binding-status line.
3. Update `handoff-contract.md` only if the object model or lifecycle itself
   changed upstream.
4. Update the "Last synced" date and source-file list above.

Asking Claude Code to "sync research-context from codex" is a sufficient
trigger for this — it's a bounded, well-defined task. If drift becomes a
problem later (more contributors, longer gaps between syncs), a staleness
check on the "Last synced" date would be the first thing worth adding —
deliberately not built now.
