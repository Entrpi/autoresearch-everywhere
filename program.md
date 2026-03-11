# autoresearch

This is the generic agent entrypoint for the repo.

Use the top-level entrypoints by default:

- `uv run prepare.py`
- `uv run calibrate.py`
- `uv run train.py`

Use engine-specific files only when the task is explicitly backend-specific or you need to inspect implementation details.

## Setup

To start a new run:

1. Agree on a run tag and create a fresh branch `autoresearch/<tag>` from the current default branch.
2. Read the top-level context:
   - `README.md`
   - `CHANGELOG.md`
   - `program.md`
3. If the task is MLX-first or Apple-Silicon-first, also read:
   - `docs/program-mlx.md`
   - `docs/platform-calibration.md`
   - `docs/preset-calibration.md`
4. If the task is CUDA-specific, inspect:
   - `autoresearch_cuda/config.py`
   - `autoresearch_cuda/runtime.py`
   - `autoresearch_cuda/prepare.py`
   - `autoresearch_cuda/train.py`

## Default Workflow

There are now two normal entry stories.

### New or unfamiliar hardware

1. `uv sync`
2. `uv run prepare.py`
3. `uv run calibrate.py --mode fast`
4. Optionally `uv run calibrate.py --mode full --output-dir <same-dir>`
5. Use the emitted candidate default as the starting point for experiments

This is the preferred front door when the machine is not already well understood.

### Known hardware / already calibrated path

1. `uv sync`
2. `uv run prepare.py`
3. `uv run train.py --smoke`
4. Start experiments with `uv run train.py`

## Engine Selection

The top-level entrypoints dispatch by engine.

- On Apple Silicon, the default is `mlx`.
- On a CUDA machine, the default is `cuda` when CUDA is detected.
- If auto-detection is ambiguous, pass `--engine mlx` or `--engine cuda`.

Examples:

```bash
uv run prepare.py
uv run train.py --preset m5-balanced
uv run train.py --engine cuda --preset upstream
uv run calibrate.py --mode fast
```

Use `--list-engines` on `train.py` or `prepare.py` to see the currently installed engine names.

## Mutation Scope

The repo is no longer single-file in the old upstream sense.

What to mutate depends on the backend and the task:

- MLX training/runtime work:
  - `autoresearch_mlx/train.py`
  - `autoresearch_mlx/model.py`
  - `autoresearch_mlx/optim.py`
- MLX data/eval work:
  - `autoresearch_mlx/prepare.py`
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/eval_policy.py`
  - `autoresearch_mlx/eval_telemetry.py`
- Shared platform/bring-up work:
  - `calibrate.py`
  - `tools/calibrate_platform.py`
  - `autoresearch_platform/`
- CUDA runtime/bring-up work:
  - `autoresearch_cuda/`

## Revalidation

Do not rely only on static signatures to decide whether to rerun calibration.

Use judgment first. If a finding looks likely to generalize across preset shapes or hardware classes, rerun `calibrate.py` proactively. Typical triggers include:

- architecture changes such as SwiGLU or attention rewrites
- runtime-path changes such as compilation, batching, cache behavior, or checkpointing
- evaluator changes that alter sequence length, slicing, token budgets, or canonical/proxy semantics
- optimizer changes that alter accumulation, state shape, or memory pressure

The signatures are the conservative backstop, not the primary decision-maker.

## Notes

- `docs/program-mlx.md` is now the MLX-specific supplement, not the generic front door.
- `autoresearch_mlx/train.py` and `autoresearch_mlx/prepare.py` remain available as direct low-level entrypoints, but the repo-level interface is now the top-level `train.py` and `prepare.py`.
