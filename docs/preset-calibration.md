# Preset Calibration Research

This note captures the current research direction for turning the recent manual eval-policy work into a reusable preset calibration system.

If the goal is new-machine bring-up rather than refining one known preset / hardware pair, also see `docs/platform-calibration.md` and `calibrate.py`. That layer sits above the lower-level calibration subcommands described here and turns them into one orchestrated bring-up report with a candidate new default for the autoresearch stage on that hardware, resumable phase artifacts, confidence-bearing reference comparisons, and a promotion bundle for the resulting default/calibration artifacts. The same run also writes the candidate default into a stable local platform-default cache so later systems can reuse the calibrated point for that device.

The same calibration machinery also now serves as the revalidation path after meaningful autoresearch changes. If the model, optimizer, attention path, or eval implementation changes enough to alter runtime or eval signatures, the old calibration should be treated as suspect even on the same hardware.
The important nuance is that signature drift is not the preferred trigger. Agent judgment should normally fire first. If the findings suggest the change could generalize across preset shapes or hardware classes, rerun calibration proactively even before a signature mismatch forces the issue.

## Goal

We want a repeatable way to determine, for a given preset and hardware target:

- the best canonical eval batch for each eval sequence length
- the best cheap / reference / full eval rung timings
- the error of those reduced rungs against a full upstream-shaped baseline
- a default rung policy that changes with run length instead of being hard-coded globally

The intent is similar to the existing checkpoint policy system: measure first, then codify a selector.

## What We Just Learned

The current eval work established a few stable properties on the reference M5 machine:

- Long-context BPB is effectively batch-invariant once the batch is in a sane range.
- The locally efficient long-context batch rule is `batch = max(1, 4096 // seq_len)`.
- Full upstream eval should stay sequential. At full budget, slicing produced the same BPB as sequential prefix traversal while adding wall time.
- Reduced-budget canonical eval benefits from stratified sampling across the upstream horizon rather than a single deterministic prefix.
- The best cheap / reference / full rung choice is preset-sensitive, not global.

That last point is the main justification for a calibration harness.

## Implemented First Pass

The repo now has the first three pieces of the intended shape:

1. `autoresearch_mlx/eval_policy.py`
2. `tools/calibrate_eval_policy.py`
3. runtime integration in `autoresearch_mlx/train.py`
4. generated JSON artifacts for raw calibration runs

### `autoresearch_mlx/eval_policy.py`

This is now the checked-in policy layer, analogous to `autoresearch_mlx/checkpoint_policy.py`.

It defines:

- `EvalCalibration`
- `EvalRung`
- `EvalRecommendation`
- `AutoEvalDecision`

It currently stores:

- preset key
- hardware key
- exact preset identity
- canonical eval seq len
- canonical eval batch
- cheap rung timing / error
- reference rung timing / error
- full rung timing
- provenance string for the calibration source

The checked-in calibrations currently cover:

- `m5-tiny`
- `m5-small`
- `m5-balanced`
- `m5-xlarge`

all on the current reference machine:

- `apple-m5-32gb-10gpu`

`m5-large` is now a shipped bridge preset, but it does not yet have a checked-in eval ladder row. It therefore uses the normal explicit fallback path until it is calibrated.

The runtime selector now consumes these values by default for shipped preset shapes when the user has not explicitly overridden canonical eval settings. Mutated preset shapes still fall back to the default canonical settings until they are calibrated, and exact-hardware matching is required before a checked-in row is trusted at runtime.
The rows are now also interpreted against the current eval-semantics and runtime-shape signatures, so the selector can distinguish "new hardware" from "same hardware, but this code has changed enough to require revalidation."

### `tools/calibrate_eval_policy.py`

This is now the offline harness for repeating the workflow mechanically.

Implemented modes:

- `train-grid`
  - short training runs over a constrained operating-point grid
  - supports `device_batch_size`, `total_batch_size`, `seq_len`, and `window_pattern`
  - intended to expose throughput / memory tradeoffs for a preset on a given machine without turning calibration into an architecture search
- `eval-batch`
  - eval batch sweep on an existing checkpoint
  - intended to find fast, metric-stable eval batches
- `eval-rungs`
  - cheap / reference / full eval ladder on an existing or newly minted checkpoint
  - intended to produce the error-vs-overhead table for selector design
- `telemetry-summary`
  - summarize passive runtime evidence for a preset / hardware / policy row
  - intended to show whether a checked-in calibration is still only a seed or has broader stable coverage

The tool can reuse an existing checkpoint or mint a fresh short checkpoint for rung calibration.

### `autoresearch_mlx/train.py`

The trainer now uses the checked-in eval tradeoff table by default:

- shipped preset shape + no explicit canonical override
  - auto-select cheap / reference / full from `eval_policy.py`
- mutated preset shape
  - keep the default canonical settings until that shape is calibrated
- explicit canonical override
  - treat the run as manual and do not apply the selector

This keeps the runtime policy conservative: only measured preset / hardware rows drive automatic eval behavior.

The trainer now also exposes selector provenance directly in its run config and final summary:

- selected rung
- detected hardware key
- calibration status
- calibration key
- seed and effective calibration confidence
- freshness
- repeat count
- telemetry count
- commit/day spread
- last-seen date
- observed rungs
- stable rungs
- any limit reason that capped a more aggressive rung
- calibration measurement date
- calibration eval/runtime signatures
- current eval/runtime signatures
- policy version

That output is there to make semantic drift visible instead of implicit.

Ordinary eligible runs also append passive eval telemetry to `~/.cache/autoresearch/eval_policy_telemetry.jsonl`. That ledger is not the checked-in policy table; it is the evidence stream used to track freshness, repeat coverage, commit/day spread, rung timing stability, and whether a shipped row still deserves to auto-select more aggressive rungs.

In the current implementation, a rung counts as timing-stable once it has at least `3` eligible runs and its eval-seconds relative MAD is at most `5%`. Confidence promotion is driven by stable rung coverage, not just raw run count.

## Why The Scope Grew

The initial eval-policy work underscoped the real problem.

We do not only need to pick among cheap / reference / full eval rungs. We also need a mechanical way to discover:

- train-side device batch sweet spots
- eval-side batch sweet spots
- model-shape and hardware capability limits

That broader shape is why the first automation tool includes both training and eval modes instead of only an eval selector.

## Recommended Calibration Workflow

For a new preset / hardware combination:

1. Train a checkpoint at a fixed short budget, currently `120s`.
2. Sweep eval batch at the target canonical sequence length if the batch rule is not already trusted.
3. Measure:
   - cheap: `2048 / 262144 / batch_rule`, sliced
   - reference: `2048 / 1572864 / batch_rule`, sliced
   - full: `2048 / 20971520 / batch_rule`, sequential
4. Store:
   - `eval_seconds`
   - `val_bpb`
   - `abs_error_vs_full`
   - `speedup_vs_full`
   - overhead at representative training budgets, such as `5m` and `8h`

This gives enough information to choose eval defaults without pretending the selector is universally portable.

For a fuller preset calibration, add a short training grid before step `2`:

- vary `device_batch_size x total_batch_size`
- optionally vary `seq_len` and a small candidate set of `window_pattern` values
- keep depth / width fixed so the sweep remains a calibration problem instead of a new model search
- record `steady_state_tok_per_sec`, `peak_vram_mb`, and `grad_accum_steps`
- choose the best supported local operating point before doing eval calibration

## Initial Policy Shape

Based on the current preset table, the selector should choose from a fixed ladder, not interpolate arbitrary token counts:

- `cheap`
- `reference`
- `full`

The first plausible runtime rule is:

- choose `full` when projected eval overhead is small enough to be routine
- otherwise choose `reference` when it is accurate enough for the run length
- otherwise choose `cheap`

The exact thresholds should come from checked-in measured policy data, not from code comments or ad hoc judgment. That is why the policy module exists separately from the calibration tool.

## Why This Should Be Preset-Sensitive

The current measurements already show materially different economics:

- `m5-tiny`: `reference` is a very plausible `5m` rung
- `m5-small`: `reference` is the cleanest middle-rung case
- `m5-balanced`: `reference` looks more like a long-run rung than a short-run default
- `m5-xlarge`: same story, with even more expensive full upstream eval

A single global eval selector would hide exactly the information the calibrations are supposed to expose.

## Integration Points

The remaining integration points are:

- `autoresearch_mlx/constants.py`
  - keep only basic defaults, not the full policy logic
- `README.md`
  - document the rung system as measured policy rather than static constants
- `CHANGELOG.md`
  - record new calibration tables when a preset / hardware profile is added or revised

## Practical Command Shape

The intended operating-point workflow now looks like:

1. `train-grid`
   - identify the best local training shape for the preset on the target hardware
2. `eval-batch`
   - verify the fast canonical eval batch at the chosen eval sequence length
3. `eval-rungs`
   - measure cheap / reference / full against a fixed checkpoint
4. `telemetry-summary`
   - inspect what passive runtime evidence has accumulated after ordinary runs

The important constraint is to keep the training grid small and interpretable. A good calibration grid is something like:

- `device_batch_size x total_batch_size`
- one or two `seq_len` candidates
- one or two `window_pattern` candidates

and not an open-ended product over every model hyperparameter.

## Open Questions

- Should the hardware key be manual (`m5-32gb-10gpu`) or partially auto-detected?
- Should calibrations be stored in Python, JSON, or both?
- Should the training checkpoint budget for calibration stay fixed at `120s`, or vary by preset size?
- Should the runtime selector expose explicit modes like `cheap`, `reference`, `full`, and `auto`?
- Should the operating-point grid grow to include a constrained `depth` ladder for future hardware classes, or should that remain outside calibration entirely?

## Current Recommendation

Keep the runtime policy simple for now:

- keep the selector ladder discrete
- use measured preset-specific data
- require exact hardware-key matches for automatic selection
- treat the checked-in rows as low-confidence seed calibrations until they have broader repeat coverage
- let opportunistic telemetry from ordinary eligible runs update freshness and repeat evidence, but not silently rewrite the checked-in policy rows
- aggregate confidence from stable coverage, not just raw counts:
  - repeated stable telemetry on one rung can promote a row to `telemetry-repeated-single-hardware`
  - multiple stable rungs across multiple days/commits can promote further to `telemetry-cross-rung-single-hardware` or `telemetry-cross-session-stable`
- let confidence and freshness gate how aggressive the runtime selector may be:
  - seed rows may auto-pick `cheap` or `reference`
  - `full` should require broader stable cross-rung evidence
  - stale-age rows should fall back visibly rather than being trusted indefinitely
- keep full upstream sequential
- keep reduced rungs sliced across the upstream horizon
- continue using the calibration tool to fill out more preset / hardware rows before broadening the automatic selector beyond the shipped preset shapes
