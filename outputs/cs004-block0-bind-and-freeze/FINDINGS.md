# CS004 Block 0: bind and freeze (2026-09-01)

**Run:** `RUN-20260901T201958Z-CS004-3eeff03f-01` (see `../../runs/`).
**Scope:** repository binding, authoritative source manifest, economic/numerical
identity, and reproducibility conventions for the interior five-state Ramsey PDE
feasibility problem (CP004 / CS004 v0.5, `draft`). Permitted under CS004's own
"exploratory prototype exception" (`computation/README.md`) — CS004 is not
`approved`, so no model equation, solver, evaluator, or network code was written.
**Outcome:** repository unambiguously bound; source records fingerprinted; economic
and numerical identity transcribed without modification; reproducibility
conventions frozen where CS004 fixes them, explicitly marked unresolved where it
deliberately leaves them to be learned.

## A. Repository binding

| Item | Value |
|---|---|
| Absolute repository root (primary checkout) | `/Users/nathanbarnard/Documents/TAI public finnace claude` |
| **This block's working directory (isolated worktree)** | `/Users/nathanbarnard/Documents/TAI public finnace claude-cs004-five-state-ramsey-pde-feasibility` |
| Remote | `https://github.com/Nathan-Barnard/tai-public-finance.git` (fetch + push, `origin`) |
| Branch | `cs004/five-state-ramsey-pde-feasibility` (new; branched from `a6df37f`, the tip of `chore/computation-workflow-setup`, **not** from `main` and **not** from `cs001/...`) |
| HEAD after this block's first commit | `3eeff03f6872a8d0bb9711d5b369700950c60f2a` |
| Working tree at HEAD | clean (verified after each commit) |
| Applicable repository instructions | `/Users/nathanbarnard/Documents/TAI public finnace claude/CLAUDE.md` (no `AGENTS.md` or equivalent exists anywhere in the repo — confirmed by `find . -iname AGENTS.md`) |
| Implementation language | Python 3.13 (`requires-python = ">=3.13"`, `.python-version` = `3.13`; matches the already-installed `3.13.5`) |
| Provisional framework | Direct PyTorch (CS004 v0.5, "Provisional framework and curvature-preserving architecture") — **not yet added** to `pyproject.toml`/`uv.lock`; see "Unresolved inputs" below |
| Package manager | `uv` 0.12.5 |

### Why a new worktree, and why branched off `a6df37f`

- `main` (`c28ff1c`) predates the computational-workflow scaffolding entirely —
  no `CLAUDE.md` conventions section, no `research-context/`, no `runs/`, no CI.
  Branching CS004 off `main` would lose all of that.
- `chore/computation-workflow-setup` (`a6df37f`, open PR
  [#1](https://github.com/Nathan-Barnard/tai-public-finance/pull/1)) is exactly
  `main` plus that shared scaffolding, with no CS001-specific code. This is the
  correct common ancestor for a second spec.
- `cs001/lq-anchor-matrix-equations-irfs` (open PR
  [#2](https://github.com/Nathan-Barnard/tai-public-finance/pull/2)) is
  `a6df37f` plus 8 CS001-specific commits, still being actively amended by other
  sessions during this block (a peer session pushed a new CS001 commit while this
  block was in progress). Branching CS004 from its tip would couple CS004's
  history to CS001's unreviewed, still-moving PR.
- CLAUDE.md's own worktree guidance names this exact situation — "a PyTorch-based
  spec alongside a numpy/scipy one" — as the case for `git worktree add`. CS001 is
  numpy/scipy-only; CS004's provisional framework is PyTorch. A worktree gives
  CS004 its own `.venv` and working directory without touching the primary
  checkout, which several other sessions were concurrently using for CS001 and a
  new EMP001 branch-off during this block (confirmed via `ListAgents` and two
  unprompted cross-session status messages; both peers acknowledged no collision
  once informed of the isolation — see this session's transcript).

### Verified commands (this worktree, this block)

| Command | Result |
|---|---|
| `uv sync` | succeeded; resolved the existing lock (`numpy`, `scipy`, `matplotlib`, `pyyaml` + dev group) into a fresh `.venv` for this worktree |
| `uv run pytest` | `1 passed` (only `tests/test_import.py` exists at this branch's base — CS001's test suite lives only on the `cs001` branch, not merged here) |
| `uv run python -c "..."` (platform/package versions) | Python 3.13.5, macOS-15.6.1-arm64, numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.1, pyyaml 6.0.3 — identical to the primary checkout |
| `git worktree add ... -b cs004/... a6df37f` | succeeded; primary checkout's own branch/status verified unaffected immediately after |
| CI (`.github/workflows/tests.yml`) | `uv sync` + `uv run pytest` on `ubuntu-latest`, triggered on push/PR to `main` — same two commands as above, so this block's local verification exercises the identical commands CI will run |

### From ignored/local files or external environments

- `.venv/` (gitignored, per-worktree) — recreated above via `uv sync`; nothing
  hand-configured.
- No `.env`/secrets file exists or is required for Block 0 or the currently
  drafted equation-to-function map.
- Nothing from Kaggle/Colab/paid-GPU accounts is needed yet (first use is Block 6
  in CS004's own block table).

## B. Authoritative source manifest

Full path/version/status/mtime/SHA-256 for all seven records this freeze is
traceable to: [`source_manifest.json`](source_manifest.json). No record beyond
the seven named in the handoff was needed to resolve an ambiguity.

The freeze is pinned to **CS004 v0.5** specifically
(`sha256:aa5f379c198d133295dcdb706fb80d7e10a158a7be6320251c1f6960197f68af`). If
that file's hash changes on a future read, every frozen item below must be
re-checked before Block 1 proceeds — this manifest does not auto-refresh.

## C. Economic and numerical identity (frozen, not rederived)

Transcribed from CS004 v0.5 and its linked records without modification.

- **Model:** maintained Brownian Version 5.1, `q_D=1`, one external risky claim
  (not domestic public equity), private domestic capital ownership, worker-only
  Ramsey welfare objective, `G=0`.
- **State vector:** `(z, x, k, n, tau)` — log productivity, latent automation
  state, `log(K/L)`, `N/K`, tax rate. Level model coordinates throughout the
  source records; no rescaled/normalized coordinate system is defined anywhere
  in the read material (see "coordinate scaling," unresolved, below).
- **Controls:** `(chi, theta, nu)` — normalized worker consumption `c/K`, risky
  position `s/K`, tax speed `d(tau)/dt`.
- **Branch:** smooth full-specialisation interior branch only. The exact
  three-regime global firm block is explicitly out of scope for this
  calculation; a candidate path leaving the branch is a falsification condition,
  not something to reflect or clip.
- **Required derivatives** (`value_jet`, CS004 equation-to-function map item 4):
  `v, v_z, v_x, v_k, v_n, v_tau, v_zz, v_xx, v_zn, v_xn, v_nn`. Verbatim: "No
  other second derivatives enter the interior PDE." This is consistent with
  `CURRENT_PROJECT_CONTEXT.md`'s statement that only `z` and `x` carry Brownian
  shocks (orthogonal OU diffusions) while `k`, `n`, `tau` evolve by locally
  deterministic laws of motion given the controls — the diffusion is degenerate
  by construction, not an approximation choice.
- **Policy recovery** (equation-to-function map item 5): `chi = 1/v_n`,
  `nu = (kappa_tau/y)(v_tau/v_n)`, `theta = -(beta_I*v_n + D_v)/(beta_I*v_nn)`
  with `D_v = sigma_z*lambda_I^P*v_zn + sigma_x*lambda_I^A*v_xn`.
- **Independent residual paths:** the optimized HJB residual and the
  unoptimized Hamiltonian-before-maximization must be coded as two genuinely
  separate functions sharing one primitive evaluator — CS004 explicitly
  forbids defining either by calling the other with recovered/candidate
  controls substituted in, "far enough to expose transcription errors."
- **Precision policy:** FP64 is authoritative throughout. FP32/mixed precision
  is permitted only as an optional warm-start/profiling lane that "must
  reproduce FP64 evaluator checks" — never a substitute for the FP64 result.

### Three fiscal-boundary objects — kept distinct (CS004 v0.5, "Three fiscal-boundary objects")

| Object | Definition | Status |
|---|---|---|
| **Economic solvency frontier** | `n >= n_sol(z,x,k,tau)`, schematic — the maximal-debt boundary compatible with a viable continuation | Not solved anywhere in CS004; its global location is explicitly out of scope for this specification |
| **Conservative reference floor** | `n_ref(tau) = -0.12*tau*(1-tau)`, minimum `-0.03` at `tau=1/2` | A compact stopped-region diagnostic (R39), viable under **one** explicit deleveraging policy until `(z,x,k)` first exits the declared compact region — not the solvency frontier, not a proof the training box is viable |
| **Artificial guardrails / box faces** | State-box truncations and wide control limits (transfer upper bound, portfolio/debt caps, tax alarm band, tax-speed cap) | Pure numerical scaffolding with no economic-boundary meaning unless a candidate depends on or hits them; must never be assigned a boundary value, reflection, or economic interpretation |

Both configuration files frozen in this block (`configs/cs004/*.json`) keep
these three objects in physically separate JSON blocks (`reference_floor` vs.
`guardrails`) precisely so they cannot be conflated downstream. The broad
config's `n`-buffer is training support only; it is explicitly *not* claimed
viable (per the source note, sampled points below the reference floor are
expected and permitted for residual fitting, not for reporting).

### Two configuration families (CD001)

| | Broad standalone PDE feasibility | Matched PDE–LQ bridge |
|---|---|---|
| File | [`configs/cs004/broad_pde_feasibility_v1.json`](../../configs/cs004/broad_pde_feasibility_v1.json) | [`configs/cs004/matched_pde_lq_bridge_v1.json`](../../configs/cs004/matched_pde_lq_bridge_v1.json) |
| SHA-256 | `b060925f4327a1464dc12320aba0271d3411db1ae3d05e42406d5cc11a8296cd` | `35aedb34db1c64ea478257358011fe47ac4865cd2d39740b7ce4de765f07bbaf` |
| Primitive vector | Its own (`rho=0.04, delta=0.06, alpha_bar=0.40, kappa_z=kappa_x=0.08, ...`) | The **exact** CS001 LQ vector, byte-identical in content (embedded snapshot — see below) |
| Purpose | Solver-feasibility / stress configuration | Isolates method (PDE vs. LQ) from calibration differences |
| Must never be substituted for the other | — both configs say so explicitly in their own `purpose`/`empirical_status` fields — | |

**Known gap, not silently patched:** the bridge config cannot yet hold a live
relative-path reference to `configs/primitives/lq_farhi_annual_v1.json` /
`configs/cs001/lq_farhi_smoke.json`, because this branch (based on the
pre-CS001 scaffolding commit) doesn't contain those files — they exist only on
the still-open `cs001` branch. The bridge config instead embeds a
provenance-stamped snapshot (exact values, source path, source commit
`89f6a939e5aa11527400bb473916c1a0f3ecf8d6`, and source blob SHA-256) and flags
itself for replacement with a real reference once CS001 merges. This is a
deliberate, visible duplication with a recorded fix, not an unnoticed one.

## D. Reproducibility conventions

| Convention | Status |
|---|---|
| **Coordinate scaling → model coordinates** | **Unresolved.** No rescaling convention exists in any read record. CS004 only states the requirement on whatever is eventually chosen: "every reported derivative must be converted back to these model coordinates before entering the residual." Block 1/2 must pick a scheme and test its inverse map explicitly. |
| **Parameter/domain input locations** | Frozen this block: `configs/cs004/broad_pde_feasibility_v1.json` and `configs/cs004/matched_pde_lq_bridge_v1.json`, following the existing `configs/<spec>/<name>.json` convention from CS001. |
| **Deterministic fingerprinting procedure** | **Partially frozen, one gap flagged.** CS001 already has a working, reusable procedure — `canonical_json` + `sha256_of` in `src/tai_public_finance/primitives/parameters.py` (semantic, key-sorted JSON hash; visible in every CS001 `RUN-*.yaml`'s `input_fingerprints` block). That module does not exist on this branch yet (it was added by CS001's own implementation commit, not the shared scaffolding commit this branch is based on — confirmed by `git ls-tree` against both). This block's own fingerprints (above, and in `source_manifest.json`) therefore use plain `shasum -a 256` over raw file bytes instead. **Action for Block 1:** once `src/tai_public_finance/primitives/` is available on this branch (by merge/rebase after CS001 lands), re-fingerprint the two CS004 config files with the canonical-JSON procedure for consistency with CS001's convention, or explicitly decide raw-file hashing is sufficient and say so — don't let the two specs silently diverge on this. |
| **Precision and device policy** | Frozen: FP64 authoritative; CPU is the authoritative Mac smoke baseline; CUDA only after CPU-parity checks pass, on Kaggle (preferred) or Colab. **Gap in the source records, flagged, not resolved by assumption:** none of the seven records mention Apple's local Metal/MPS backend at all — the stated path is literally CPU (Mac) → CUDA (cloud), with no discussion of whether local MPS is used, skipped deliberately, or simply wasn't considered on a Mac-only local machine. Worth a one-line decision before Block 3, not worth blocking on now. |
| **Random-seed convention** | **Unresolved.** No seed value, RNG library (`torch` vs. `numpy`), or determinism statement exists for CS004 anywhere in the read records — unlike CS001 (dense deterministic linear algebra, no draws needed). CS004's training loop will need one once it exists; out of scope for Block 0. |
| **Checkpoint directory/naming** | Proposed this block (not previously specified): `outputs/<run-name>/checkpoints/checkpoint_best.{pt}` and `checkpoint_latest.{pt}`, gitignored (`.gitignore` updated this block: `outputs/*/checkpoints/`), referenced by content hash + size from the run's `RUN-*.yaml` record — this is runs/README.md's existing stated policy for "large arrays, checkpoints, and figures," applied to CS004 rather than invented fresh. |
| **Rolling best/latest policy** | Proposed: overwrite `checkpoint_best.pt` only when validation residual improves, always overwrite `checkpoint_latest.pt`; no per-epoch history kept locally (disk cap, below). CS004 Block 4 names "best and latest checkpoints" as a requirement; the overwrite-in-place mechanics are this block's proposal to meet that under the disk constraint, not text lifted from the spec. |
| **Maximum local disk allocation** | Proposed, explicitly provisional: **2 GB** total across all `outputs/cs004-*/` run artifacts (checkpoints + small structured outputs), separate from whatever PyTorch itself will cost to install. Measured free disk **6.0 GiB** at the end of this block (`df -h /`; the codex roadmap's snapshot from earlier the same day recorded 7.6 GB — it has already dropped, partly from this block's own second `.venv`, 358 MB). This number needs Nathan's confirmation or adjustment before Block 3/4, not just this block's say-so — flagged, not silently adopted as final. |
| **Run-record / diagnostic-output destination** | Frozen (matches existing convention exactly, no change needed): `runs/RUN-<timestamp>-CS004-<sha>-NN.yaml` (index, git-tracked) + `outputs/cs004-<run-name>/` (bundle: report/summary/FINDINGS/figures, large items gitignored per the policy above). |
| **Train/validation/collar/ordinary-path/stress-path separation** | Frozen as a *requirement*, not yet implemented: CS004 explicitly requires **four disjoint evaluation sets** (central interior points; a predeclared outer collar including faces/edges; closed-loop paths from ordinary central initial states; stress paths near each economically relevant face) plus **separate training and validation draws** — six pools in total, and "training or grid nodes cannot double as the only evaluation set." Exact sampling procedure, counts, and distributions are Block 1/4 work, deliberately not fixed here. |
| **Local/Kaggle/Colab/paid-GPU launch interfaces** | Frozen as a *requirement*: all four must invoke one shared package/CLI, never divergent notebooks (`computation/README.md`, "Reuse-before-build rule"). Concrete form proposed, not created: a `src/tai_public_finance/cs004_five_state_pde/` package with a CLI entrypoint, mirroring `cs001_lq_anchor`'s `python -m tai_public_finance.cs001_lq_anchor.cli --config ... --output-dir ... --run-id ...` pattern exactly. **Not created in this block** — CLAUDE.md is explicit that a spec's package is created "only once it's approved," and CS004 is `draft`. Creating even an empty package now would cross that line for no Block-0 benefit. |

## Unresolved inputs that block Block 1 (named concretely, not vaguely)

1. **`eta_sol`** (positive reference-floor reporting margin) — CS004 leaves this
   an unfixed symbol. Block 1/5 must choose a number and justify it against
   measured slack; not invented here per the task's own instruction not to guess
   thresholds the spec deliberately defers.
2. **Bridge domain for `z`, `n`, `tau`** — CD001/CS004 only narrow `x` and `k`
   for the matched bridge; the other three states' bridge-specific bounds are
   not stated anywhere in the read records. Marked `unresolved` in
   `matched_pde_lq_bridge_v1.json` rather than guessed.
3. **Smaller feasible `epsilon`** for the bridge — CS004 requires "at least one
   smaller feasible value" before interpreting LQ/small-risk convergence but
   never names one.
4. **`src/tai_public_finance/primitives/` availability on this branch** — Block
   1's primitive/domain screen (`automation()`, `domestic_primitives()`,
   `international_pricing()`) is exactly what that shared package already
   implements on the `cs001` branch. Block 1 should reuse it, not reimplement
   it, once it's merged/rebased in — implementer's choice on timing, but it
   should be a deliberate decision, not a surprise mid-Block-1.
5. **Local MPS vs. CPU-only** on the Mac before any cloud step (see precision/
   device row above) — a one-line decision, not currently blocking, but should
   be settled before Block 3's cost microprofile.
6. **Disk cap number** (2 GB proposed) — needs Nathan's confirmation given how
   little headroom remains (6.0 GiB free).

None of these block *this* block's completion — CS004's own Block 0 table entry
requires exactly "one reproducible launch description," which this is.

## Confirmation: no solver/evaluator/network/training work started

Checked explicitly: no file under `src/tai_public_finance/` was created or
modified this block (the package remains the bare pre-CS001 skeleton —
`__init__.py`, `py.typed` — verified by `find src -type f` before and after).
No `automation()`/`domestic_primitives()`/`international_pricing()`/
`value_jet()`/residual/policy-recovery function exists anywhere in this
worktree. No PyTorch dependency was added. Only configuration (JSON), a
`.gitignore` extension, and documentation/run-records were written.

## Files created or changed (this worktree, both commits)

| File | Commit |
|---|---|
| `.gitignore` (extended) | `3eeff03` |
| `configs/cs004/broad_pde_feasibility_v1.json` | `3eeff03` |
| `configs/cs004/matched_pde_lq_bridge_v1.json` | `3eeff03` |
| `outputs/cs004-block0-bind-and-freeze/source_manifest.json` | this commit |
| `outputs/cs004-block0-bind-and-freeze/FINDINGS.md` (this file) | this commit |
| `runs/RUN-20260901T201958Z-CS004-3eeff03f-01.yaml` | this commit |
| `runs/README.md` (index entry added) | this commit |

## Elapsed time and next-block estimate

This block's total elapsed implementation time (research-record reading,
repository/branch/worktree investigation, cross-session coordination, config
authoring, verification, write-up) ran toward the upper end of the declared
30–60 minute range, driven mostly by the repository-topology investigation
(three open branches, two of them actively being modified by peer sessions
during this block) rather than by the freeze content itself, which was
mechanical once the topology was settled. Machine runtime was negligible
(`uv sync` + `uv run pytest`, well under `L0`'s 60-second budget).

**Block 1 (primitive/domain screen) estimate:** CS004's own table gives 30–60
minutes elapsed for evaluating the entire central box, buffer, faces, edges,
and corners against `L0<=60s` per bundle. That estimate looks realistic *if*
the shared `primitives/` package is reused rather than reimplemented (see
unresolved item 4) — reimplementing `automation()`/`domestic_primitives()`/
`international_pricing()` from scratch against the CS004 vector, rather than
extending the existing tested module, would plausibly push Block 1 past 60
minutes given CS001's own primitive/production module took nontrivial
implementation+test time as part of a larger block. Recommend resolving item 4
as the first concrete action of Block 1, before writing any evaluation code.
