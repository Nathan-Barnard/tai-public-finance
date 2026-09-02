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
| cs002-d2-review-repairs-aa69f0-6e | `.claude/worktrees/cs002-d2-review-repairs-aa69f0` (harness-assigned worktree dir; dispatch asked for a sibling `../claude-cs002-d2-review-repairs` but a worktree-isolated session can't `git worktree add` outside its own dir — same precedent as `tai-public-finnace-claude-b7` below. The branch name itself *did* work: `git checkout -b cs002/d2-review-repairs 3aa85b6` succeeded, so the branch matches dispatch exactly even though the directory doesn't) | `cs002/d2-review-repairs`, base `3aa85b6` (exact, verified) | CS002 D2 review repairs, dispatched off `outputs/cs002-d2-mean-reversion-20260902T190139Z` + `RUN-20260902T190139Z-CS002-D2-0b721f21-01`: (1) `terminal.py`'s LQ terminal map omits the K_bar physical-unit multiplier entirely (silently a no-op at K_bar=1, invisible in every run so far — confirmed by reading the code); (2) `experiment_d2.py`'s `_horizon_mesh_comparisons` compares only the 4 raw BVP variables under one pooled max()-derived tolerance instead of the full reported-path set with per-path effect-scaled tolerances (confirmed). Scope: `cs002_nonlinear_transition/` package + its tests only; one new D2-repair output dir + run record; supersedes `RUN-20260902T190139Z-CS002-D2-0b721f21-01` (left untouched, not edited). **Collision note 2026-09-02**: session `cs002-d2-repair-cdb1a2-56` was independently dispatched the identical task/branch/base into a separate worktree (`.claude/worktrees/cs002-d2-repair-cdb1a2`) after I'd already started; contacted directly via SendMessage and asked to stand down, awaiting confirmation. | active, implementation in progress |

*Seeded 2026-09-01 from a live `ListAgents` + status-check round; keep it
current from here rather than trusting this snapshot. This worktree's copy
predates the `b7`/`0e`/`d9` rows added later on `main` (this branch forked
before those landed and never merged back) — reconcile at merge time.*
