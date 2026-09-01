# Codex → Claude Code handoff contract

Condensed from `computation/README.md` and `computation/AGENTS.md` in the
codex workspace — see them directly for full detail. This file is about the
*mechanism*, not the current state; for what's actually active right now see
[portfolio-snapshot.md](portfolio-snapshot.md).

## Object model

| Object | ID | Purpose |
|---|---|---|
| Computational problem | `CP###` | Economic question, required result tier, dependencies, success test |
| Candidate approach | `CA###` | Algorithm/library option, reuse evidence, expected scaling |
| Computational specification | `CS###` | Versioned, implementation-ready contract |
| Benchmark | `CB###` | Fair comparison of approaches on the same case/budget |
| Decision | `CD###` | Why an approach or specification was selected, deferred, or superseded |
| Run | `RUN-...` | Immutable record of one execution |

Stable IDs are references, not names or claims of quality. A draft
specification is not approved; a completed run is not automatically a
verified result.

## Specification lifecycle

`draft → review_ready (fingerprinted) → approved → implemented → superseded`

**"The implementer receives an approved CS, not a chat transcript."** Nothing
before `approved` is a real implementation contract — draft specs may still
change in ways that would invalidate work built against them. After a spec
is fingerprinted at `review_ready` or later, a fingerprint-relevant change
creates a new successor rather than editing in place; old runs stay as stale
evidence rather than being overwritten.

An implementation handoff must name: the approved spec plus its
version/fingerprint, repository/base commit, owned files and non-goals,
exact validation commands and acceptance thresholds, the chosen approach and
rejected alternatives, resource lane and stop conditions, and the run-record
location.

## Resource lanes

| Lane | Typical use | Target | Default machine |
|---|---|---:|---|
| `L0_smoke` | imports, dimensions, tiny meshes | ≤60s | MacBook Air |
| `L1_interactive` | one baseline calibration, coarse IRFs | ≤10min | MacBook Air |
| `L2_local_batch` | parameter sweeps, mesh refinement, robustness | ≤2h | MacBook Air |
| `L3_colab` | accelerated or memory-heavier benchmark | ≤6h/session | Google Colab |
| `L4_overnight` | convergence grids, many seeds | ≤14h | Local or Colab |
| `L5_paid_compute` | confirmed workload that benefits from rented hardware | explicit cost + go/no-go | External |

Wall-time targets are budgets for feedback, not scientific tolerances.
`L5_paid_compute` needs an estimated cash cost, an admin/setup estimate, a
cheaper baseline, a stopping rule, and Nathan's explicit approval.

## Ownership boundary

The codex workspace owns the problem definition, specification, evidence
contract, and readable status. This repository owns executable code, tests,
dependency locks, and large outputs. Once this repo is bound as the
implementation repository, canonical `RUN-*.yaml` records live here (see
[../runs/README.md](../runs/README.md)) — codex's `computation/runs/` keeps
only an index/link back, not a duplicate.

## Naming

Human-facing text (chat, docs, issues, PRs) always writes `Name (ID)`, e.g.
"LQ anchor and impulse responses (CP001)" — never a bare ID first. Machine
identifiers (branch names, package names, command arguments) may be
ID-first; see [CLAUDE.md](../CLAUDE.md) for this repo's branch and package
conventions.
