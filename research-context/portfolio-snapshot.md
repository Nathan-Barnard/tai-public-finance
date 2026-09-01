# Portfolio snapshot

Mirrors `computation/roadmap.md` and `computation/computation_registry.yaml`
in the codex workspace as of the sync date in [README.md](README.md). Codex
is canonical; this is a snapshot — refresh it there, don't hand-edit around
a discrepancy here.

## Implementation repository binding

As of last sync: **unbound**.

```yaml
binding_status: unbound
url: null
local_path: null
default_implementer: claude_code
compatible_implementers: [claude_code, codex]
```

(from `computation_registry.yaml`). This repo —
`https://github.com/Nathan-Barnard/tai-public-finance`, local path
`/Users/nathanbarnard/Documents/TAI public finnace claude` — is the intended
binding target, but the registry hasn't been updated to point at it yet.
That edit happens on the codex side, not here.

## Computational problems and specifications

| Problem | Spec | Title | Status | Lane | Notes |
|---|---|---|---|---|---|
| CP001 | CS001 | LQ anchor, matrix equations, and impulse responses | Active / draft | `L1_interactive` | Simplest, most shovel-ready; numpy/scipy only |
| CP002 | CS002 | Nonlinear deterministic Ramsey transition | Proposed / draft | `L1_interactive` | Waits on CS001 passing |
| CP003 | CS003 | Small-risk portfolio and welfare correction | Proposed / draft | `L2_local_batch` | Waits on CS002 |
| CP004 | CS004 | Interior-first global five-state Ramsey PDE feasibility | Active / draft | `L2_local_batch` → Colab | **Current priority.** Neural-residual solve, provisionally direct PyTorch |
| CP005 | CS005 | Marked-Poisson commitment and fiscal-capacity computation | Proposed / draft | `L1_interactive` | Separate branch — do not route through the Brownian LQ stack |

All five specifications are `draft`. None are `approved`. Do not begin
implementing a spec's model equations until it reaches `approved`.

## Accepted decisions

- **CD001** — separate exploratory calibrations with a matched PDE-LQ
  bridge: CS001 and CS004 each use their own illustrative primitive vector
  for standalone smoke tests; no discrepancy between their results may be
  attributed to approximation method until both are re-run on one identical,
  shared primitive vector (the "bridge" run). This is why
  `src/tai_public_finance/primitives/` (see [CLAUDE.md](../CLAUDE.md)) has
  to be a genuinely shared module once implementation starts, not a
  convenience.

## Refresh

See [README.md](README.md#refreshing-this).
