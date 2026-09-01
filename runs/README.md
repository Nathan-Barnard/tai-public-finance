# Runs

Immutable records of material executions for this repo's computational
work, mirroring the naming convention used in the codex workspace's
`computation/runs/`:

```text
RUN-YYYYMMDDTHHMMSSZ-CS###-<git-short-sha>-NN.yaml
```

One file per execution that informs a model choice, result, figure,
calibration, benchmark, cost estimate, or publication claim. Smoke runs used
only while coding don't need one — CI logs are enough.

Once the codex-side registry binds this repository as the implementation
target, this becomes the canonical location for run records; codex's
`computation/runs/` keeps only an index/link back here — don't duplicate the
full record in both places.

Keep large arrays, checkpoints, and figures out of git — reference them by
an external durable location, content hash, and size instead. Never edit a
completed record to match a later specification; create a new one and, if
needed, a short note linking the original and its replacement.

- [`RUN-20260901T184527Z-CS001-89f6a939-01.yaml`](RUN-20260901T184527Z-CS001-89f6a939-01.yaml) — CS001 Stage 1 (deterministic/first-order 4x4 LQ system) and Stage 2A (leading small-risk portfolio/welfare) baseline on the illustrative Farhi-based smoke calibration; passed all 35 acceptance checks. Full bundle (matrices, tidy IRFs, figures, diagnostics) under `../outputs/cs001-lq-anchor-baseline/`; findings summarized in that directory's `FINDINGS.md`. `report.json`, `irfs.csv`, and `figures/` are not committed (see `.gitignore`) — regenerate with the command in this record, or verify a copy against the sha256 hashes recorded in `artifacts` below.
