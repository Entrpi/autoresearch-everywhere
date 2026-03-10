# Preset Calibration Research

This note captures the current research direction for turning the recent manual eval-policy work into a reusable preset calibration system.

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

The repo now has the first two pieces of the intended shape:

1. `autoresearch_mlx/eval_policy.py`
2. `tools/calibrate_eval_policy.py`
3. generated JSON artifacts for raw calibration runs

What is still missing is runtime integration: `train_mlx.py` does not yet call into `eval_policy.py` to choose cheap / reference / full automatically.

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

- `m5-fast`
- `m5-balanced`
- `m5-large`
- `m5-xlarge`

all on the current reference machine:

- `apple-m5-32gb-10gpu`

The runtime selector still does not consume these values. For now, the module is the checked-in policy source, not yet the active trainer authority.

### `tools/calibrate_eval_policy.py`

This is now the offline harness for repeating the workflow mechanically.

Implemented modes:

- `train-batch`
  - short training runs over a device-batch sweep
  - intended to expose throughput / memory tradeoffs for a preset on a given machine
- `eval-batch`
  - eval batch sweep on an existing checkpoint
  - intended to find fast, metric-stable eval batches
- `eval-rungs`
  - cheap / reference / full eval ladder on an existing or newly minted checkpoint
  - intended to produce the error-vs-overhead table for selector design

The tool can reuse an existing checkpoint or mint a fresh short checkpoint for rung calibration.

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

For a fuller preset calibration, add a short training sweep before step `2`:

- vary `device_batch_size`
- keep the preset shape fixed
- record `steady_state_tok_per_sec` and `peak_vram_mb`
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

- `m5-fast`: `reference` is a very plausible `5m` rung
- `m5-balanced`: `reference` is the cleanest middle-rung case
- `m5-large`: `reference` looks more like a long-run rung than a short-run default
- `m5-xlarge`: same story, with even more expensive full upstream eval

A single global eval selector would hide exactly the information the calibrations are supposed to expose.

## Integration Points

The likely runtime integration points are still:

- `train_mlx.py`
  - replace direct canonical constants with a call into `eval_policy.py`
- `autoresearch_mlx/constants.py`
  - keep only basic defaults, not the full policy logic
- `README.md`
  - document the rung system as measured policy rather than static constants
- `CHANGELOG.md`
  - record new calibration tables when a preset / hardware profile is added or revised

## Open Questions

- Should the hardware key be manual (`m5-32gb-10gpu`) or partially auto-detected?
- Should calibrations be stored in Python, JSON, or both?
- Should the training checkpoint budget for calibration stay fixed at `120s`, or vary by preset size?
- Should the runtime selector expose explicit modes like `cheap`, `reference`, `full`, and `auto`?

## Current Recommendation

Keep the runtime policy simple for now:

- keep the selector ladder discrete
- use measured preset-specific data
- keep full upstream sequential
- keep reduced rungs sliced across the upstream horizon
- continue using the new calibration tool to fill out more preset / hardware rows before wiring the selector into `train_mlx.py`
