# Status

Live "who's doing what" board for the concurrent Claude Code sessions that
work on this repo. Update your own row when you start and when you finish;
don't edit anyone else's. This is a convenience, not a lock — it doesn't
prevent collisions by itself. Use `ListAgents` and message the owning
session directly before touching a branch someone else's row claims.

| Session | Location | Branch | Task | Status |
|---|---|---|---|---|
| tai-public-finnace-claude-9d | primary checkout | — | CS001: LQ anchor, matrix equations, IRFs | done, merged (PR #2) |
| tai-public-finnace-claude-67 | primary checkout | — | Computation-project conventions, Codex handoff docs, CI | done, merged (PR #1) |
| tai-public-finnace-claude-59 | `../claude-cs004-five-state-ramsey-pde-feasibility` | `cs004/five-state-ramsey-pde-feasibility` | CS004 Block 0 (repo binding, fingerprints) | active |
| tai-public-finnace-claude-c6 | `../claude-emp001-uk-quasi-empirical-pilot` | `emp001/uk-quasi-empirical-pilot` | EMP001: UK quasi-empirical scenario pilot | active |
| tai-public-finnace-claude-54 | primary checkout | `main` | Repo setup, literature, multi-session coordination | active |
| tai-public-finnace-claude-27 | `../claude-cs001-joint-shock-atlas` | `cs001/joint-shock-atlas` | CS001 overnight joint productivity-automation shock atlas (new `shock_atlas.py`/`shock_atlas_cli.py`/`test_shock_atlas.py`, one new `outputs/cs001-joint-shock-atlas-*` dir, one run record; touches no shared module) | done 2026-09-02 00:05 UTC: run `RUN-20260901T230703Z-CS001-ATLAS-ee48affe-01`, branch pushed (3 commits, new files only), PR not opened pending Nathan; see `outputs/cs001-joint-shock-atlas-20260901T230702Z/MORNING_REPORT.md` |
| tai-public-finnace-claude-b7 | `../claude-cs002-deterministic-bvp-d0-d1` | `cs002/deterministic-bvp-d0-d1` | CS002 D0–D1 exploratory nonlinear deterministic transition prototype (draft-spec exception per CS002 v0.2 + CP002/CP003 work plan; new `cs002_nonlinear_transition/` package, tests, `configs/cs002/`, one `outputs/cs002-d0-d1-frozen-transition-*` dir, one run record; touches no shared module; local commits only, no push/PR on this branch) | active, started 2026-09-02 |

*Seeded 2026-09-01 from a live `ListAgents` + status-check round; keep it
current from here rather than trusting this snapshot.*
