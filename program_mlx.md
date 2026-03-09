# autoresearch-mlx

This is the MLX version of the autoresearch experiment.

## Setup

To set up a new MLX experiment, work with the user to:

1. Agree on a run tag based on today's date.
2. Create a fresh branch `autoresearch/<tag>` from the current main branch.
3. Read the in-scope files:
   - `README.md`
   - `prepare_mlx.py`
   - `train_mlx.py`
   - `autoresearch_mlx/data.py`
   - `autoresearch_mlx/model.py`
   - `autoresearch_mlx/optim.py`
4. Verify that `~/.cache/autoresearch/` contains data shards, `tokenizer.pkl`, `token_bytes.npy`, and the `token_cache/` directory. If not, tell the human to run `uv run prepare_mlx.py`.
   Optional: if `prepacked_cache/` is present, the runtime will use it automatically for matching sequence lengths.
5. Initialize `results.tsv` with the header row if it does not exist.
6. Confirm setup and start experimenting.

## Experimentation

Each experiment runs on Apple Silicon using MLX. The training script runs for a fixed 5-minute wall-clock training budget. Launch it with:

```bash
uv run train_mlx.py
```

The default preset in `train_mlx.py` is `m5-balanced`. Useful alternatives:

```bash
uv run train_mlx.py --preset m5-fast
uv run train_mlx.py --preset m5-large
uv run train_mlx.py --preset m5-xlarge
uv run train_mlx.py --preset upstream
```

Use `uv run train_mlx.py --no-prepacked-cache` when you explicitly want to benchmark or debug the live packing path instead of the optional prepacked row caches.
Use `uv run train_mlx.py --benchmark-skip-eval` when you want a warmup-aware comparison that separates startup cost from steady-state throughput. By default, the trainer auto-detects the warmup cutoff statistically from step-time stabilization and reports `warmup_done_step`. Use `--benchmark-warmup-steps <N>` only when you need a fixed override for an ablation or apples-to-apples replay.

`uv run prepare_mlx.py` now builds the shipped prepacked row caches by default so `m5-fast`, `m5-balanced`, `m5-large`, and `m5-xlarge` all have a prepared fast path. Use `uv run prepare_mlx.py --skip-prepacked-cache` only when you intentionally want the live packing fallback.

For runs above 5 minutes, `train_mlx.py` now enables resumable checkpoints by default using the repo's conservative checkpoint-frequency selector. Use `--checkpoint-path` to choose the checkpoint directory while keeping that selector, `--checkpoint-interval` to pin the cadence, or `--no-checkpoint` to disable it.

What you CAN do:
- Modify `train_mlx.py`.
- Modify `autoresearch_mlx/model.py`.
- Modify `autoresearch_mlx/optim.py`.

What you CANNOT do:
- Modify `prepare_mlx.py`.
- Modify `autoresearch_mlx/data.py`.
- Add new dependencies unless the human explicitly asks for that.
- Change the BPB metric or the fixed time budget.

The goal is the same as upstream: minimize canonical `val_bpb`. Treat `proxy_val_bpb` as a fast local signal, not the promotion metric.

## Output format

When the script finishes it prints:

```text
---
val_bpb:          <float>
proxy_val_bpb:    <float>
training_seconds: <float>
total_seconds:    <float>
peak_vram_mb:     <float>
mfu_percent:      <float>
train_tflops:     <float>
loader_percent:   <float>
grad_percent:     <float>
accum_percent:    <float>
optimizer_percent: <float>
other_step_percent: <float>
checkpoint_percent: <float>
eval_percent:     <float>
util_window_steps: <int>
util_window:      <string>
checkpoint_count: <int>
session_tokens_M: <float>
session_steps:    <int>
benchmark_warmup_mode: <auto|fixed>
benchmark_warmup_steps: <int>
warmup_done_step: <int>
benchmark_warmup_step_seconds: <float>
benchmark_warmup_wall_seconds: <float>
steady_state_training_seconds: <float>
steady_state_tokens_M: <float>
steady_state_steps: <int>
steady_state_tok_per_sec: <float>
cumulative_training_seconds: <float>
cumulative_checkpoint_seconds: <float>
cumulative_checkpoint_count: <int>
total_tokens_M:   <float>
num_steps:        <int>
num_params_M:     <float>
depth:            <int>
proxy_eval_tokens: <int>
canonical_seq_len: <int>
canonical_tokens: <int>
canonical_batch:  <int>
```

`training_seconds`, `total_seconds`, `checkpoint_percent`, `eval_percent`, and `checkpoint_count` are current-invocation metrics, so a resumed run no longer mixes cumulative training time with per-invocation wall-clock time. `mfu_percent` is retained as a backward-compatible alias for measured step compute-share utilization on the MLX path and remains resume-aware because its step telemetry is restored from checkpoints. The more informative new fields are `train_tflops`, the explicit loader/grad/optimizer/checkpoint/eval percentages, the separate `session_*` / `cumulative_*` counters, and the `benchmark_*` / `steady_state_*` fields when you invoke the warmup-aware benchmark mode. `benchmark_warmup_mode` tells you whether the warmup cutoff came from the default statistical detector or an explicit override, and `warmup_done_step` records the first global step treated as steady-state.

## Logging results

Use a TSV file with these columns:

```text
commit	val_bpb	memory_gb	status	description
```

The status is one of `keep`, `discard`, or `crash`. `keep` and `discard` are based on canonical `val_bpb`.

Also keep `CHANGELOG.md` current for meaningful changes. Each changelog entry should clearly separate:

- `Human-driven (5)`: what the human identified and tightly specified.
- `Human-directed, AI-shaped (4)`: what the human directed but the agent concretely designed.
- `AI-identified within brief, human-shaped (3)`: what the agent surfaced inside a broad human-scoped workstream, and the human materially reshaped before implementation.
- `AI-identified within brief, human-approved (2)`: what the agent surfaced inside a broad human-scoped workstream and the human approved with little reshaping.
- `Self-initiated, human-approved (1)`: what the agent initiated outside explicit human direction but still got approved before landing.
- `Fully autonomous (0)`: what the agent initiated without explicit human direction or approval.
- `Grounding`: the files changed, the checks run, and any measured effects.

Omit empty provenance sections instead of adding `None in this entry.`
Show an autonomy golf score in each commit header. Score each provenance bullet as `Human-driven = 5`, `Human-directed, AI-shaped = 4`, `AI-identified within brief, human-shaped = 3`, `AI-identified within brief, human-approved = 2`, `Self-initiated, human-approved = 1`, and `Fully autonomous = 0`; `Grounding` is not scored.
Treat top-level provenance bullets as the scored units. If a point is directly derivative of a main bullet and stays at the same autonomy level, record it as a nested sub-bullet so it remains visible without adding score.
Use `python3 tools/changelog_scores.py --group-by day --format csv` when you want a plotting-friendly daily autonomy summary, or `--group-by entry --verify` to sanity-check header totals against the parsed bullets.
When provenance is ambiguous, prefer `Human-directed, AI-shaped` over `AI-identified within brief, human-shaped`, prefer `AI-identified within brief, human-shaped` over `AI-identified within brief, human-approved`, prefer `AI-identified within brief, human-approved` over `Self-initiated, human-approved`, and prefer `Self-initiated, human-approved` over `Fully autonomous`.
When grounding is ambiguous, prefer the stronger practical check if it is feasible in this environment. For performance or optimization claims, try to produce a matched A/B or fixed-budget benchmark instead of relying on smoke tests alone. For stability claims, prefer an end-to-end run that actually exercises the changed path. If only weaker grounding is practical, state that limitation explicitly in both the changelog and the user-facing report.
Benchmarks should run long enough to produce a high-signal result on the specific change item. Pick a run shape where the affected path happens enough times to matter. For example, checkpoint-overhead changes should usually be tested with enough runtime to produce many checkpoint saves, not just one or two, and scaling claims should prefer a heavier preset if the smaller ones do not expose the bottleneck clearly.
On this hardware, the default canonical matched benchmark window for optimization grounding is `60s`, not `30s`. Treat shorter runs as smoke checks or constrained fallbacks, and say so explicitly when you use them.
Do not describe an optimization as established unless the grounding matches the claim. Correctness-only checks can support “works” or “does not regress obvious behavior,” but not “is faster” or “scales better.”
Do not land fully human-authored code changes on this branch. If a change must be authored entirely by a human, do that work in a fork.

## Loop

1. Inspect the current branch and commit.
2. Try one idea.
3. Commit the change.
4. Run `uv run train_mlx.py > run.log 2>&1`.
5. Extract the results from `run.log`.
6. If the run crashes, inspect the traceback, fix obvious bugs, and retry a small number of times.
7. Record the result in `results.tsv`.
8. Update `CHANGELOG.md` if the run led to a meaningful code or workflow change.
9. Keep only improvements on canonical `val_bpb`.

Do not stop and ask whether to continue once the loop starts unless the human explicitly interrupts you.
