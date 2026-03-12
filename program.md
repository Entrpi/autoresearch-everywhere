# autoresearch-everywhere

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
5. If the task is backend-specific kernel work, also read:
   - `docs/kernel-lab.md`
   - the backend lab implementation under `autoresearch_*/lab*.py`

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
- Shared kernel-lab work:
  - `kernel-lab.py`
  - `autoresearch_lab/`
- MLX kernel-lab work:
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_profile.py`
  - `autoresearch_mlx/lab_trace.py`
  - `autoresearch_mlx/lab_workspace.py`
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
- For kernel-lab work, treat `profile` / `extract` / `orchestrate` as the heuristic layer, the ledger in `results/kernel_lab/ledger.jsonl` as the memory of what has already been proved, and the trace commands as the truth layer. On MLX that means `capture` plus human `review-trace`; on CUDA that means `capture`, `trace-profile`, and `auto-review`, with GUI Nsight inspection as the escalation path rather than the first step. Use `evidence` and `promotion-check` to make the current state explicit. On CUDA, `orchestrate` should now be fed from a trace-profile artifact rather than a heuristic workspace catalog, because the backend story is currently trace-first, not workspace-first. Microbench wins are enough to justify trying a target; use a `.gputrace` before making stronger Apple Silicon performance claims when practical, feed that capture back into `orchestrate --trace-metadata ...`, record whether the trace looked strong or weak, and prefer promotion-ready workspaces over opening redundant fresh ones. If a target has a direct trainer-side hook, use `integration-ab` for smoke and `integration-suite` for real promotion evidence. Beyond smoke tests, omit `--preset` so the lab uses the calibrated platform default for the current device; only pin a preset manually when you are intentionally testing a different operating point. Treat `integration-ab` and `integration-suite` as repeated evidence gathering, not yes/no checks: promotion should follow repeated balanced A/B runs with stable direction, and ideally survive the next stronger preset too.
