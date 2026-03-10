# Autonomy Golf Checklist

Use this when the project already has autonomy golf installed and you just need to maintain it correctly.

If you need to compare your local setup against the canonical source, use [Entrpi/autonomy-golf](https://github.com/Entrpi/autonomy-golf).

## Before You Change Anything

1. Read [../CHANGELOG.md](../CHANGELOG.md) to see the current template and latest entry shape.
2. Read [autonomy-golf-agent.md](autonomy-golf-agent.md) if you need the fuller integration rules.
3. Confirm which subsystem the change belongs to.

## When You Make A Change

1. Write or update the change itself.
2. Decide which top-level provenance tier or tiers actually apply.
3. Keep directly derivative same-tier details nested under the main bullet.
4. Keep `Grounding` separate from provenance.
5. Be ready to explain the change's meaning, motivation, and purpose before you touch the changelog header math.

## When You Update The Changelog

1. Use a Linux-kernel-style header: `subsystem: summary`.
2. Keep the changelog on a deliberate lag-by-one commit-ID model:
   - while work is still in flight, keep it under `Latest` as `### New commit — subsystem: summary`
   - do not guess or prefill a commit hash
   - do not amend a commit just to stamp its own hash into the changelog; that changes the hash again and leaves the entry stale
   - only stamp a real date and commit ID from a subsequent commit, then move that older entry into committed history
   - if more work starts after that commit, open a fresh `New commit` entry for the next change
3. Add the bounded `score`.
4. Add `complexity` only when it differs from `score`.
   - compute it as the sum of top-level provenance weights, plus `+1` for each nested sub-bullet under provenance items scored `3` or higher
   - do not count `Meaning:`, `Motivation:`, or `Purpose:` narrative lines toward complexity
5. Explain the change's meaning, motivation, and intended purpose.
   - a short nested `Meaning:`, `Motivation:`, `Purpose:` trio is the default pattern when the entry would otherwise read like a task list
6. Record:
   - files changed
   - checks run
   - measured effects, if any
7. If you changed scored provenance bullets, recompute any derived changelog values that depend on them:
   - the header `score`
   - the header `complexity`
   - any embedded parser-summary tables or quoted rollup values

## When You Validate

1. Prefer the strongest practical grounding the change deserves.
2. Record weaker grounding honestly if stronger validation is not practical.
3. Treat test coverage as one meaningful grounding dimension when automated tests are the right validation surface.
4. If you touched `tools/changelog_scores.py` or `tools/render_autonomy_badge.py`, run targeted tool validation first:

   ```bash
   python3 -m py_compile tools/changelog_scores.py tools/render_autonomy_badge.py
   ```

## When You Refresh Outputs

1. Run:

   ```bash
   python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify
   ```

2. Then run:

   ```bash
   python3 tools/render_autonomy_badge.py
   ```

That refreshes both generated outputs:

- `docs/autonomy-golf-badge.svg`
- the README snapshot block

Do not hand-edit the README snapshot block. It is generated output owned by `tools/render_autonomy_badge.py`.

## Definition Of Done

- changelog parses cleanly
- badge and README snapshot match the changelog
- provenance is conservative
- grounding is honest
