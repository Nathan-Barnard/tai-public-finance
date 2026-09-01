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

No runs yet — nothing is implemented.
