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
total_tokens_M:   <float>
num_steps:        <int>
num_params_M:     <float>
depth:            <int>
proxy_eval_tokens: <int>
canonical_seq_len: <int>
canonical_tokens: <int>
canonical_batch:  <int>
```

## Logging results

Use a TSV file with these columns:

```text
commit	val_bpb	memory_gb	status	description
```

The status is one of `keep`, `discard`, or `crash`. `keep` and `discard` are based on canonical `val_bpb`.

Also keep `CHANGELOG.md` current for meaningful changes. Each changelog entry should clearly separate:

- `Human-driven`: what the human identified and tightly specified.
- `Human-directed, AI-shaped`: what the human directed but the agent concretely designed.
- `AI-identified within brief, human-shaped`: what the agent surfaced inside a broad human-scoped workstream, and the human materially reshaped before implementation.
- `AI-identified within brief, human-approved`: what the agent surfaced inside a broad human-scoped workstream and the human approved with little reshaping.
- `Self-initiated, human-approved`: what the agent initiated outside explicit human direction but still got approved before landing.
- `Fully autonomous`: what the agent initiated without explicit human direction or approval.
- `Grounding`: the files changed, the checks run, and any measured effects.

When provenance is ambiguous, prefer `Human-directed, AI-shaped` over `AI-identified within brief, human-shaped`, prefer `AI-identified within brief, human-shaped` over `AI-identified within brief, human-approved`, prefer `AI-identified within brief, human-approved` over `Self-initiated, human-approved`, and prefer `Self-initiated, human-approved` over `Fully autonomous`.
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
