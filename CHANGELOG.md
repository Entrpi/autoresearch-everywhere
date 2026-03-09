# Changelog

This changelog is intended to be useful for research, not just release bookkeeping.
Each entry records:

- `Human-driven (5)`: the human identified the change and specified it tightly enough that the agent mostly executed.
- `Human-directed, AI-shaped (4)`: the human set the direction or requirement, but the agent designed the concrete mechanism, structure, or validation plan.
- `AI-identified within brief, human-shaped (3)`: inside a broad human-scoped workstream, the agent surfaced the opportunity, and the human materially shaped the exact target, scope, or framing before implementation.
- `AI-identified within brief, human-approved (2)`: inside a broad human-scoped workstream, the agent surfaced the opportunity and the human approved it with little additional shaping.
- `Self-initiated, human-approved (1)`: the agent initiated the change outside explicit human direction in the thread, but still got human approval before landing it.
- `Fully autonomous (0)`: changes or experiments the agent initiated without explicit human direction or approval in the thread.
- `Grounding`: the files changed, the checks run, and any measured effects.

If an entry has no measurements yet, it should say so explicitly.
Entries should omit empty provenance sections rather than spelling out `None in this entry.`
Each commit entry should also show an autonomy golf score in the header. Score each provenance bullet as `Human-driven = 5`, `Human-directed, AI-shaped = 4`, `AI-identified within brief, human-shaped = 3`, `AI-identified within brief, human-approved = 2`, `Self-initiated, human-approved = 1`, and `Fully autonomous = 0`. `Grounding` does not contribute to the score.
Top-level provenance bullets are the scored units. If a point is directly derivative of a main bullet and stays at the same autonomy level, record it as a nested sub-bullet so it remains visible without adding score.
When provenance is ambiguous, prefer `Human-directed, AI-shaped` over `AI-identified within brief, human-shaped`, prefer `AI-identified within brief, human-shaped` over `AI-identified within brief, human-approved`, prefer `AI-identified within brief, human-approved` over `Self-initiated, human-approved`, and prefer `Self-initiated, human-approved` over `Fully autonomous`.
This changelog should bias toward under-claiming rather than over-claiming successful autonomy. When specific provenance attributions are corrected, prefer the more conservative tiering if there is real ambiguity. The long-term goal remains to push as much work as possible into the `Fully autonomous` category over time.
This branch does not admit fully human-authored code changes. If a change must be authored entirely by a human, it belongs in a fork rather than this branch's mainline history.
Grounding should also be conservative. When a claim is about performance, stability, or behavioral improvement, prefer the strongest practical evidence over the quickest smoke pass, and record the actual strength of that evidence rather than the intended standard.
For benchmarked changes, "strong enough" means long enough and heavy enough to produce a high-signal result on the changed behavior. Choose a run shape where the affected path executes enough times to matter. For example, checkpoint-overhead claims should usually be grounded with a run that produces many checkpoint saves rather than only one or two, and scaling claims should prefer a model/preset large enough for the bottleneck to show up clearly.
On this hardware, the default canonical matched benchmark window for optimization grounding is `60s`, not `30s`. Use shorter runs for smoke checks or when the changed path cannot practically support a longer benchmark, and say so explicitly when you do.

## Unreleased

### New commit — Benchmark resume-ready latency — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced a stronger grounding need for checkpoint optimization: raw `mx.load` timings are not enough, because they understate what a resumed run actually pays before it becomes productive again.
  - Added a dedicated resume-ready benchmark that measures cold runtime build, checkpoint restore, and the first completed optimizer step after resume.
  - Added repeat support and aggregate summaries so checkpoint policy work can use medians instead of hanging on a single noisy resumed run.

**Grounding**

- Files:
  - `tools/profile_resume_ready.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile tools/profile_resume_ready.py`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-large --resume-steps 2 --repeats 3 --json-out results/analysis/m5_large_resume_ready.json`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-xlarge --resume-steps 2 --repeats 3 --json-out results/analysis/m5_xlarge_resume_ready.json`
- Measurements:
  - Resume-ready summary (`3` trials each, prepacked path, `5` seed train steps before save):

    | Preset | median resume-ready (s) | median runtime build (s) | median restore (s) | median first step wall (s) | median steady step wall (s) | median resume tax vs steady wall (s) |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | `m5-large` | `0.641` | `0.039` | `0.016` | `0.581` | `0.568` | `0.073` |
    | `m5-xlarge` | `0.683` | `0.060` | `0.020` | `0.549` | `0.538` | `0.168` |

  - Interpretation:
    - The user-facing resume-ready delay is not dominated by checkpoint metadata or loader restore. It is mostly runtime rebuild plus the first resumed optimizer step.
    - `m5-large` is reasonably tight across the three trials; the median resume-ready delay is about `0.64s`, with about `73ms` above a steady resumed step.
    - `m5-xlarge` is centered closer to about `0.68s`, and one of the three trials again had a much slower first resumed step. That makes the median much more trustworthy than the mean for policy work at this shape.
    - This is the right benchmark to use for future checkpoint-mode comparisons, because it measures "time until the resumed run is productive again" directly instead of inferring from lazy file-load timings.

## Committed History

### March 9, 2026 — `26e4c64` — Auto-detect benchmark warmup cutoff — score `4`

**Human-directed, AI-shaped (4)**

- Requested that the benchmark warmup cutoff be determined statistically by default instead of relying on a manually specified fixed step count, and that the resulting cutoff be tracked explicitly as `warmup_done_step`.
  - Switched the benchmark accounting to record per-step session timings and determine the warmup boundary after the run.
  - Preserved `--benchmark-warmup-steps` as an explicit fixed override for ablations and replay.
  - Added `benchmark_warmup_mode` and `warmup_done_step` to the training summary so the cutoff is explicit in the reported metrics.

**Grounding**

- Files:
  - `train_mlx.py`
  - `program_mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-unreleased --verify`
  - `./.venv/bin/python train_mlx.py --preset m5-fast --time-budget 5 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python train_mlx.py --preset m5-fast --time-budget 5 --benchmark-warmup-steps 62 --benchmark-skip-eval --no-checkpoint`
- Measurements:
  - Detection rule:
    - The auto cutoff compares early-step windows against a trailing reference window taken from the end of the run instead of guessing a fixed warmup length ahead of time.
    - The reference center is the trailing-window median, and the tolerance is the larger of a `3%` relative band or `3 x` the trailing-window median absolute deviation.
    - The chosen cutoff is the first step prefix after which a short stability window stays within that tolerance, so `warmup_done_step` marks the first step whose timing looks statistically indistinguishable from the trailing steady-state band.
  - Auto-vs-fixed `m5-fast` comparison (`5s`, eval skipped, no checkpoint):

    | Mode | `benchmark_warmup_steps` | `warmup_done_step` | warmup step seconds | warmup wall seconds | steady-state steps | steady-state training seconds | steady-state `tok_per_sec` |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | `auto` | `37` | `37` | `1.258` | `1.291` | `316` | `3.749` | `43153.8` |
    | `fixed` | `62` | `62` | `1.012` | `1.046` | `333` | `3.990` | `42734.9` |

  - Interpretation:
    - The default statistical detector produces a concrete benchmark cutoff without requiring a manually chosen step count.
    - The detector is responsive to real startup shape, so it should not be expected to pick the same cutoff across materially different warmup profiles.
    - On the final `m5-fast` rerun, the auto and fixed modes still landed in the same steady-state performance band, which is the main practical requirement: the new default is more ergonomic without obscuring the benchmark semantics.

### March 9, 2026 — `9330302` — Profile checkpoint save/load path — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced checkpoint persistence as the next optimization target now that runtime instrumentation, the prepacked fast path, and the live-packing fallback have all been tightened.
  - Added a dedicated save/load profiler so checkpoint work can be aimed at the dominant cost center instead of guessing from end-to-end wall time alone.

**Grounding**

- Files:
  - `tools/profile_checkpoint_path.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile tools/profile_checkpoint_path.py`
  - `./.venv/bin/python tools/profile_checkpoint_path.py --preset m5-large --train-steps 5`
  - `./.venv/bin/python tools/profile_checkpoint_path.py --preset m5-xlarge --train-steps 5`
- Measurements:
  - Save-path breakdown on the default prepared prepacked path:

    | Preset | save total (s) | model write (s) | optimizer write (s) | loader write (s) | metadata write (s) | model bytes | optimizer bytes | loader bytes |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | `m5-large` | `0.254` | `0.224` | `0.028` | `0.0017` | `0.0008` | `80.2M` | `118.1M` | `22` |
    | `m5-xlarge` | `0.473` | `0.422` | `0.050` | `0.0004` | `0.0002` | `159.4M` | `218.3M` | `22` |

  - Interpretation:
    - On the shipped prepacked path, checkpoint cost is overwhelmingly dominated by writing model and optimizer safetensors. Loader serialization and metadata are effectively free by comparison.
    - That means the next meaningful checkpoint optimization is not Python-side cleanup inside the current save path; it would need to change semantics or scheduling, such as lighter resume tiers or asynchronous/background save behavior.
    - The current restore-side numbers from this tool are exploratory only. `mx.load` appears lazy enough that raw file-load timings understate "resume ready" cost, so restore optimization should not be driven from those numbers yet without a stronger resume-readiness benchmark.

### March 9, 2026 — `9a79473` — Optimize live-packed fallback buffer — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced the live-packing fallback as the next optimization target after the shipped preset fast path and warmup-aware benchmark tooling were in place.
  - Replaced the linear best-fit scan with a length-indexed FIFO packing buffer for the live-packed training path.
  - Reused the same packing structure in prepacked-cache construction so the packing behavior stays consistent across both paths.

**Grounding**

- Files:
  - `autoresearch_mlx/data.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/data.py train_mlx.py tools/profile_loader_path.py`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-balanced --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python /tmp/autoresearch_livepack_before/tools/profile_loader_path.py --preset m5-balanced --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-large --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python /tmp/autoresearch_livepack_before/tools/profile_loader_path.py --preset m5-large --steps 80 --warmup-steps 5 --no-prepacked-cache`
- Measurements:
  - Matched fallback-only before/after profile (`80` measured steps after `5` warmup steps):

    | Preset | version | `tok_per_sec` | total mean (ms) | loader mean (ms) | grad mean (ms) | optimizer mean (ms) |
    | --- | --- | ---: | ---: | ---: | ---: | ---: |
    | `m5-balanced` | before | `33306.25` | `61.49` | `0.963` | `45.85` | `13.47` |
    | `m5-balanced` | after | `34334.42` | `59.65` | `0.174` | `45.14` | `13.28` |
    | `m5-large` | before | `14456.33` | `283.34` | `1.328` | `222.57` | `52.66` |
    | `m5-large` | after | `14522.86` | `282.04` | `0.245` | `222.67` | `52.40` |

  - Interpretation:
    - The new packing buffer removes most of the Python-side fallback cost on both tested presets: loader-call time dropped by about `82%` on `m5-balanced` and `m5-large`.
    - The end-to-end win is real but modest because these steps are still compute-dominated: about `+3.1% tok/s` on `m5-balanced` and `+0.5% tok/s` on `m5-large`.
    - This is worth keeping as a fallback-path cleanup, but the grounded effect is much smaller than the raw loader-time drop might suggest.

### March 9, 2026 — `8c6ed4f` — Add warmup-aware MLX benchmark tooling — score `4`

**Human-directed, AI-shaped (4)**

- Requested that we understand the apparent `m5-xlarge` prepacked regression rather than guess at it.
  - Added a focused loader-path profiling harness that separates `next(loader)`, forced `mx.eval(x, y)`, gradient computation, and optimizer update timing.
  - Added warmup-aware benchmark reporting to `train_mlx.py`, including startup/warmup seconds and steady-state tokens-per-second, with an option to skip eval noise during benchmarking.
  - Used it to test whether the observed xlarge regression was actually in the prepacked loader path or was just run-level noise.

**Grounding**

- Files:
  - `train_mlx.py`
  - `tools/profile_loader_path.py`
  - `program_mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py tools/profile_loader_path.py`
  - `./.venv/bin/python train_mlx.py --smoke --benchmark-warmup-steps 1 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 60 --benchmark-warmup-steps 5 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 60 --benchmark-warmup-steps 5 --benchmark-skip-eval --no-checkpoint --no-prepacked-cache`
  - repeated `ABAB` run using the same two commands above
  - `python3 -m py_compile tools/profile_loader_path.py`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 80 --warmup-steps 5`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 20 --warmup-steps 5`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 20 --warmup-steps 5 --no-prepacked-cache`
- Measurements:
  - `80` measured xlarge steps after `5` warmup steps:

    | Train path | `tok_per_sec` | total mean / median (ms) | loader mean / median (ms) | grad mean / median (ms) | optimizer mean / median (ms) |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | prepacked cache | `7637.86` | `536.28 / 534.48` | `0.46 / 0.08` | `447.99 / 446.38` | `84.05 / 83.95` |
    | token cache / live packing | `7359.01` | `556.60 / 549.47` | `1.66 / 0.90` | `463.84 / 456.97` | `87.54 / 86.47` |

  - `20` measured xlarge steps after `5` warmup steps:

    | Train path | steady `tok_per_sec` | steady total mean / median (ms) | warmup total mean / median (ms) |
    | --- | ---: | ---: | ---: |
    | prepacked cache | `7742.33` | `529.04 / 527.70` | `619.00 / 541.36` |
    | token cache / live packing | `7736.51` | `529.44 / 527.13` | `554.78 / 539.72` |

  - Interpretation:
    - There is no evidence here of a structural steady-state xlarge regression caused by the prepacked loader path.
    - The only consistent loader-path effect is that prepacked reduces `loader_call` time. Forced `mx.eval(x, y)` materialization stays effectively zero on both paths, so the earlier regression is not explained by unified-memory handoff showing up outside the loader bucket.
    - The earlier `60s` end-to-end xlarge regression now looks more like run-level variance or warmup/compile noise than a genuine fast-path loss.
  - Warmup-aware `ABAB` end-to-end benchmark (`60s`, `benchmark_warmup_steps=5`, eval skipped):

    | Leg | Train path | `session_steps` | `session_tokens_M` | warmup step seconds | steady-state `tok_per_sec` | `train_tflops` | `loader_percent` |
    | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | `A1` | prepacked cache | `116` | `0.475` | `3.229` | `7974.2` | `2.206` | `0.04` |
    | `B1` | token cache / live packing | `115` | `0.471` | `2.686` | `7858.0` | `2.176` | `0.27` |
    | `A2` | prepacked cache | `116` | `0.475` | `3.101` | `7938.6` | `2.197` | `0.06` |
    | `B2` | token cache / live packing | `115` | `0.471` | `2.738` | `7809.5` | `2.163` | `0.38` |

  - Interpreting the warmup-aware `ABAB` run:
    - Prepacked pays an extra startup cost of about `0.45s` on average across the first `5` warmup steps.
    - After warmup, prepacked runs about `1.6%` faster in steady-state (`7956 tok/s` average vs `7834 tok/s` average).
    - That implies a rough xlarge crossover at about `56` steady-state steps, or about `29s`, before the prepacked path amortizes its slower warmup and comes out ahead overall.
    - Under the repo's current `60s` benchmark window, the xlarge prepacked path should now be treated as a small net win, not a regression.

### March 9, 2026 — `d7f4d23` — Make prepacked caches default for shipped presets — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced making the shipped M5 presets use prepacked caches as the normal prepared-state fast path instead of treating them as an extra opt-in, and the user refreshed that priority.
  - Switched `prepare_mlx.py` to build the shipped prepacked cache coverage by default.
  - Added an explicit opt-out path for intentionally leaving the live packing fallback in place.
  - Tightened the loader logging so missing prepacked coverage is visible instead of silently falling through to token-cache/live packing.

**Grounding**

- Files:
  - `prepare_mlx.py`
  - `autoresearch_mlx/data.py`
  - `README.md`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile prepare_mlx.py autoresearch_mlx/data.py`
  - `./.venv/bin/python prepare_mlx.py --num-shards 1`
  - `./.venv/bin/python train_mlx.py --preset m5-large --time-budget 0.2 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 0.2 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-large --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-large --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
  - `./.venv/bin/python train_mlx.py --preset m5-balanced --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-balanced --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
  - `./.venv/bin/python train_mlx.py --preset m5-fast --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-fast --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
- Measurements:
  - Default `prepare_mlx.py --num-shards 1` behavior on the existing cache directory detected the raw data, tokenizer, and token caches, then built the missing shipped prepacked coverage automatically:
    - `train seq_len=1024`
    - `train seq_len=2048`
    - `val seq_len=1024`
    - `val seq_len=2048`
  - The shipped preset coverage now hits the prepacked train path without extra preparation flags:
    - `m5-fast`: `Data loader (train): using prepacked cache.`
    - `m5-balanced`: `Data loader (train): using prepacked cache.`
    - `m5-large`: `Data loader (train): using prepacked cache.`
    - `m5-xlarge`: `Data loader (train): using prepacked cache.`
  - `60s` A/B comparison (`512` proxy/canonical eval tokens):

    | Preset | Train path | `val_bpb` | `proxy_val_bpb` | `train_tflops` | `loader_percent` | `session_steps` | `session_tokens_M` |
    | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | `m5-fast` | prepacked cache | `1.956952` | `1.920689` | `0.624` | `0.44` | `7741` | `3.963` |
    | `m5-fast` | token cache / live packing | `1.975841` | `1.901032` | `0.594` | `3.58` | `7369` | `3.773` |
    | `m5-balanced` | prepacked cache | `1.730654` | `1.739594` | `1.384` | `0.18` | `1072` | `2.195` |
    | `m5-balanced` | token cache / live packing | `1.731765` | `1.707002` | `1.306` | `1.23` | `1014` | `2.077` |
    | `m5-large` | prepacked cache | `1.911791` | `1.929047` | `1.692` | `0.13` | `224` | `0.918` |
    | `m5-large` | token cache / live packing | `1.906361` | `1.934655` | `1.607` | `0.51` | `213` | `0.872` |
    | `m5-xlarge` | prepacked cache | `2.169967` | `2.132617` | `2.152` | `0.05` | `114` | `0.467` |
    | `m5-xlarge` | token cache / live packing | `2.140543` | `2.115287` | `2.233` | `0.36` | `118` | `0.483` |

  - Interpreting the matched runs:
    - `m5-fast` shows a clear prepacked win: `7741` vs `7369` steps, `3.963M` vs `3.773M` session tokens (`+5.0%`), and `loader_percent` dropped from `3.58` to `0.44`.
    - `m5-balanced` also shows a clear prepacked win: `1072` vs `1014` steps, `2.195M` vs `2.077M` session tokens (`+5.7%`), and `loader_percent` dropped from `1.23` to `0.18`.
    - `m5-large` shows a real fast-path win from prepacking: `224` vs `213` steps in the same `60s`, `0.918M` vs `0.872M` session tokens (`+5.3%`), and `loader_percent` dropped from `0.51` to `0.13`.
    - `m5-xlarge` showed lower `loader_percent` but a small single-run loss on total tokens; later dedicated profiling suggests that negative result was likely run-level variance rather than a structural steady-state fast-path regression.

### March 9, 2026 — `b2b08df` — Add robust MLX utilization instrumentation — score `4`

**Human-directed, AI-shaped (4)**

- Requested that optimization work focus on robust utilization instrumentation rather than another raw speed change.
  - Replaced the fake per-step `mfu` readout with measured step compute-share utilization and estimated training TFLOPs.
  - Added persistent step-telemetry counters so resumed runs keep coherent step-level utilization summaries.
  - Added explicit loader/grad/accumulate/optimizer/checkpoint/eval breakdowns to the final run summary while retaining `mfu_percent` as a backward-compatible alias.
  - Tightened resumed-run reporting so current-invocation timing and checkpoint percentages no longer mix with cumulative training progress.
    - `training_seconds`, `total_seconds`, `checkpoint_percent`, `eval_percent`, and `checkpoint_count` are now current-invocation metrics.
    - `cumulative_training_seconds`, `cumulative_checkpoint_seconds`, and `cumulative_checkpoint_count` are printed separately.
    - utilization remains resume-aware because the step-telemetry window is restored from checkpoints.

**Grounding**

- Files:
  - `train_mlx.py`
  - `autoresearch_mlx/checkpoints.py`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py autoresearch_mlx/checkpoints.py`
  - `./.venv/bin/python train_mlx.py --smoke`
  - `./.venv/bin/python train_mlx.py --preset m5-balanced --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-fast --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-large --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python train_mlx.py --smoke --checkpoint-path /tmp/autoresearch_util_resume_smoke --checkpoint-interval 0.5`
  - `./.venv/bin/python train_mlx.py --resume-from /tmp/autoresearch_util_resume_smoke --time-budget 1.5`
  - `./.venv/bin/python train_mlx.py --smoke --checkpoint-path /tmp/autoresearch_resume_metrics --checkpoint-interval 0.5`
  - `./.venv/bin/python train_mlx.py --resume-from /tmp/autoresearch_resume_metrics --time-budget 1.5`
- Measurements:
  - `20s` profile comparison runs (`512` proxy/canonical eval tokens):

    | Preset | `val_bpb` | `proxy_val_bpb` | `mfu_percent` | `train_tflops` | `loader_percent` | `grad_percent` | `accum_percent` | `optimizer_percent` | `other_step_percent` | `checkpoint_percent` | `eval_percent` | `peak_vram_mb` | `util_window_steps` | Train loader |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
    | `m5-fast` | `2.018503` | `1.946999` | `99.50` | `0.706` | `0.45` | `66.24` | `5.97` | `27.30` | `0.05` | `0.00` | `0.11` | `146.5` | `2912` | prepacked cache |
    | `m5-balanced` | `1.929286` | `1.908801` | `99.83` | `1.359` | `0.16` | `75.85` | `1.77` | `22.21` | `0.01` | `0.00` | `0.17` | `949.9` | `349` | prepacked cache |
    | `m5-large` | `2.184426` | `2.200882` | `99.28` | `1.716` | `0.71` | `78.84` | `2.22` | `18.23` | `0.01` | `0.00` | `0.42` | `1944.3` | `74` | token cache / live packing |
    | `m5-xlarge` | `2.246450` | `2.284754` | `99.86` | `2.207` | `0.13` | `83.54` | `0.67` | `15.64` | `0.00` | `0.00` | `0.78` | `4294.2` | `38` | token cache / live packing |

  - smoke run:
    - `mfu_percent=99.32`
    - `train_tflops=0.564`
    - `loader_percent=0.63`
    - `optimizer_percent=26.50`
  - smoke checkpoint/resume path:
    - checkpointed smoke ended with `checkpoint_count=2` and `checkpoint_percent=2.24`
    - resumed smoke ended with `mfu_percent=99.43`, `train_tflops=0.591`, `util_window_steps=175`, and `checkpoint_count=1`
  - resume-metrics smoke path:
    - checkpointed smoke ended with `training_seconds=1.0`, `total_seconds=1.1`, `checkpoint_count=2`, `cumulative_training_seconds=1.0`, `cumulative_checkpoint_seconds=0.020`, and `cumulative_checkpoint_count=2`
    - resumed smoke ended with `training_seconds=0.5`, `total_seconds=0.5`, `checkpoint_count=1`, `cumulative_training_seconds=1.5`, `cumulative_checkpoint_seconds=0.027`, and `cumulative_checkpoint_count=3`
- Confirmed behavior:
  - per-step progress lines now report `util` and `tflops` instead of a fake `mfu`
  - final summaries now expose the step-time split and end-of-run checkpoint/eval shares
  - resumed runs restore the saved step-telemetry state and continue the utilization window instead of resetting it to zero
  - resumed runs no longer print a cumulative `training_seconds` beside a per-invocation `total_seconds`
  - across the shipped M5 presets, utilization stays near saturation while `train_tflops` rises with model size and the larger presets still reveal the token-cache/live-pack fallback in their train-loader path
  - no performance-improvement claim is attached to this change; the grounding here is instrumentation correctness and observability

### March 9, 2026 — `62f2f9c` — Auto-enable checkpoint cadence for longer runs — score `4`

**Human-directed, AI-shaped (4)**

- Requested that the checkpoint-frequency selector be wired into `train_mlx.py` as a default-on behavior for runs longer than 5 minutes.
  - Added a shared checkpoint-policy module and used it from the trainer so long runs now auto-select a checkpoint cadence from the measured save-cost calibrations.
  - Made `--checkpoint-path` keep the default cadence selector unless `--checkpoint-interval` is explicitly pinned.
  - Added `--no-checkpoint` as the explicit escape hatch for disabling the automatic long-run behavior.

**Grounding**

- Files:
  - `train_mlx.py`
  - `README.md`
  - `program_mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py autoresearch_mlx/checkpoint_policy.py tools/checkpoint_tradeoff.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-unreleased --verify`
  - interrupted startup probe: `.venv/bin/python train_mlx.py --preset m5-balanced --time-budget 301 --eval-tokens 512 --canonical-eval-tokens 512`
  - interrupted startup probe: `.venv/bin/python train_mlx.py --preset m5-balanced --time-budget 301 --checkpoint-path /tmp/autoresearch-policy-existing-path --eval-tokens 512 --canonical-eval-tokens 512`
  - interrupted startup probe: `.venv/bin/python train_mlx.py --preset m5-balanced --time-budget 301 --no-checkpoint --eval-tokens 512 --canonical-eval-tokens 512`
  - interrupted behavior probe: `.venv/bin/python train_mlx.py --smoke --time-budget 10 --checkpoint-interval 0.5`
- Confirmed behavior:
  - `time_budget > 300s` with no checkpoint flags now auto-selects `checkpoint_interval=120.0` and an automatic checkpoint directory for `m5-balanced`.
  - `time_budget > 300s` with an explicit `--checkpoint-path` but no interval now keeps the provided path and still auto-selects `checkpoint_interval=120.0`.
  - `--no-checkpoint` suppresses both the automatic path and cadence selection.
  - an explicit `--checkpoint-interval` without `--checkpoint-path` now auto-selects the checkpoint directory and reached a real checkpoint-save attempt during the smoke probe.
  - no performance claim is attached to this change; the grounding here is behavioral rather than benchmark-driven.

### March 9, 2026 — `1c67475` — Add checkpoint interval tradeoff tooling — score `4`

**Human-directed, AI-shaped (4)**

- Requested a checkpoint-frequency tradeoff plot with expected resume-needed frequency on one axis and an optimal target derived from measured checkpoint overhead.
  - Later requested that the practical rule be generalized from discrete hourly/dayly-style thresholds into a logical interval scan anchored at `hourly <= 0.1%` save-only overhead.

**Grounding**

- Files:
  - `.gitignore`
  - `autoresearch_mlx/constants.py`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `tools/checkpoint_tradeoff.py`
- Validation:
  - `python3 -m py_compile tools/checkpoint_tradeoff.py`
  - `env MPLCONFIGDIR=/Users/ent/Codex/autoresearch/.mplconfig .venv/bin/python tools/checkpoint_tradeoff.py`
- Measurements:
  - Generated:
    - `results/analysis/checkpoint_tradeoff.png`
    - `results/analysis/checkpoint_tradeoff.csv`
    - `results/analysis/checkpoint_tradeoff.md`
    - `results/analysis/checkpoint_tradeoff.json`
  - Scenario table:

    | Profile | Robustness | Events/day | Mean hours between resumes | Optimal interval (min) | Expected waste (%) | Save cost (ms) |
    | --- | --- | ---: | ---: | ---: | ---: | ---: |
    | Exact full-state resume (m5-large calibrated) | exact step-boundary full-state resume | 0.25 | 96.00 | 3.66 | 0.064 | 70 |
    | Exact full-state resume (m5-large calibrated) | exact step-boundary full-state resume | 1.00 | 24.00 | 1.83 | 0.127 | 70 |
    | Exact full-state resume (m5-large calibrated) | exact step-boundary full-state resume | 2.00 | 12.00 | 1.29 | 0.180 | 70 |
    | Exact full-state resume (m5-large calibrated) | exact step-boundary full-state resume | 4.00 | 6.00 | 0.92 | 0.255 | 70 |
    | Exact full-state resume (m5-large calibrated) | exact step-boundary full-state resume | 8.00 | 3.00 | 0.65 | 0.360 | 70 |
    | Exact full-state resume (m5-large calibrated) | exact step-boundary full-state resume | 24.00 | 1.00 | 0.37 | 0.624 | 70 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume | 0.25 | 96.00 | 4.59 | 0.080 | 110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume | 1.00 | 24.00 | 2.30 | 0.160 | 110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume | 2.00 | 12.00 | 1.62 | 0.226 | 110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume | 4.00 | 6.00 | 1.15 | 0.319 | 110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume | 8.00 | 3.00 | 0.81 | 0.451 | 110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume | 24.00 | 1.00 | 0.47 | 0.782 | 110 |
  - Human-factors interval scan:

    | Profile | Interval | Save-only overhead (%) | Allowed overhead (%) | Pass |
    | --- | ---: | ---: | ---: | --- |
    | Exact full-state resume (m5-large calibrated) | 1m | 0.1167 | 0.100 | False |
    | Exact full-state resume (m5-large calibrated) | 2m | 0.0583 | 0.100 | True |
    | Exact full-state resume (m5-xlarge calibrated) | 1m | 0.1833 | 0.100 | False |
    | Exact full-state resume (m5-xlarge calibrated) | 2m | 0.0917 | 0.100 | True |
  - Human-factors recommendations:

    | Profile | Recommended max interval | Reason |
    | --- | ---: | --- |
    | Exact full-state resume (m5-large calibrated) | 2m | 2m is the shortest friendly interval under the fixed 0.100% save-only overhead cap (1m fail); hourly anchor overhead is 0.0019%. |
    | Exact full-state resume (m5-xlarge calibrated) | 2m | 2m is the shortest friendly interval under the fixed 0.100% save-only overhead cap (1m fail); hourly anchor overhead is 0.0031%. |
  - Tradeoff image: [results/analysis/checkpoint_tradeoff.png](results/analysis/checkpoint_tradeoff.png)
  - Using the currently measured exact full-state resume costs:
    - `m5-large` calibration (`70 ms/save`): optimal interval is about `1.83 min` at `1` resume/day and `0.92 min` at `4` resumes/day
    - `m5-xlarge` calibration (`110 ms/save`): optimal interval is about `2.30 min` at `1` resume/day and `1.15 min` at `4` resumes/day
    - under the generalized human-factors scan anchored at `0.1%` save-only overhead, both grounded profiles recommend a practical maximum checkpoint interval of `2 minutes`
  - The current `2s` benchmark interval is intentionally much more aggressive than the modeled optimum for realistic interruption rates; it remains useful for stress-testing checkpoint overhead, not as the recommended steady-state policy.
  - The tool is structured for multiple robustness profiles, but today only the exact step-boundary full-state resume profile is grounded well enough to include by default.

### March 9, 2026 — `a0d765d` — Add MLX checkpoints and benchmark grounding — score `11`

**Human-driven (5)**

- Requested that, given the constrained Apple Silicon hardware, the canonical matched benchmark run length for grounding optimization changes be `60s` instead of `30s`.

**Human-directed, AI-shaped (4)**

- Requested stronger grounding on the checkpoint/resume work and asked that changelog/program guidance become more explicit about preferring high-signal validation, while leaving the exact comparison design and write-up to the agent.

**AI-identified within brief, human-approved (2)**

- Requested that work continue to the next optimization item after landing the lazy-cache checkpoint.
  - Added resumable checkpoint save/load for the MLX trainer, including model weights, optimizer state, runtime counters, and train-loader cursor state.
  - Added periodic step-boundary checkpoint saves plus explicit `--resume-from`, `--checkpoint-path`, and `--checkpoint-interval` flags.
  - Added train-loader serialization for both prepacked and live-packed paths so resume continues from the saved training position instead of restarting the data stream.

**Grounding**

- Files:
  - `autoresearch_mlx/checkpoints.py`
  - `autoresearch_mlx/data.py`
  - `train_mlx.py`
  - `README.md`
  - `docs/mlx-port-architecture.md`
  - `program_mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py autoresearch_mlx/data.py autoresearch_mlx/checkpoints.py`
  - `./.venv/bin/python train_mlx.py --smoke --checkpoint-path /tmp/autoresearch_resume_smoke2 --checkpoint-interval 0.5`
  - `./.venv/bin/python train_mlx.py --resume-from /tmp/autoresearch_resume_smoke2 --time-budget 1.5`
  - `./.venv/bin/python train_mlx.py --smoke --checkpoint-path /tmp/autoresearch_resume_smoke3 --checkpoint-interval 0.5`
  - matched `60s` preset reruns with `--eval-tokens 512 --canonical-eval-tokens 512` for:
    - `m5-fast`
    - `m5-balanced`
    - `m5-large`
    - `m5-xlarge`
- Measurements:
  - matched `m5-large` run (`20s`, `512` eval tokens) with and without periodic checkpoint saves every `2s`:
    - wall-clock overhead outside tracked training time: `0.1s -> 0.8s` (`+0.7s`) across ten checkpoint writes, or about `0.07s/save`
    - peak memory: `1944.3 MB -> 1944.3 MB` (flat)
    - steady-state per-step throughput stayed in the same `~15k tok/s` band
    - `num_steps` stayed flat at `76 -> 76`, which is the cleaner fixed-budget result
  - matched `m5-xlarge` run (`60s`, `512` eval tokens) with and without periodic checkpoint saves every `2s`:
    - wall-clock overhead outside tracked training time: `0.2s -> 3.5s` (`+3.3s`) across twenty-nine checkpoint writes, or about `0.11s/save`
    - peak memory: `4294.2 MB -> 4294.2 MB` (flat)
    - `num_steps` stayed flat at `115 -> 115`
    - steady-state per-step throughput stayed in the same `~7.8k-8.0k tok/s` band
  - `m5-fast`: `val_bpb=1.964834`, `proxy_val_bpb=1.900012`, `~70.1k tok/s`, `174.5 MB`, `8213` steps
  - `m5-balanced`: `val_bpb=1.757194`, `proxy_val_bpb=1.741310`, `~34.5k tok/s`, `949.9 MB`, `1011` steps
  - `m5-large`: `val_bpb=1.901965`, `proxy_val_bpb=1.935591`, `~14.6k tok/s`, `1944.3 MB`, `215` steps
  - `m5-xlarge`: `val_bpb=2.150093`, `proxy_val_bpb=2.122628`, `~7.8k tok/s`, `4294.2 MB`, `114` steps
- Confirmed behavior:
  - checkpoint saves succeeded during smoke runs without breaking evaluation
  - resume restored the saved state at `step=133`, continued training to `step=186`, and preserved the same train-loader position rather than restarting from zero
  - `m5-fast` and `m5-balanced` used the train-side prepacked cache path during the 60-second reruns
  - `m5-large` and `m5-xlarge` used the token-cache plus live-packing path during those reruns, so their throughput figures are conservative relative to a fully prepacked `1024/2048` cache setup

### March 9, 2026 — `f1d14e8` — Lazy-grow model caches — score `2`

**AI-identified within brief, human-approved (2)**

- Requested the next optimization pass on lazy-growing the model caches after the current priority review.
  - Replaced the fixed `sequence_len * 10` RoPE cache with a smaller startup cache that grows up to the configured model sequence length.
  - Reworked local-attention mask caching to keep one growable mask per window size and slice it for shorter requests instead of caching separate masks per exact sequence length.
  - Prewarmed runtime caches in `train_mlx.py` before compiling the train step so cache growth stays out of the compiled hot path.

**Grounding**

- Files:
  - `autoresearch_mlx/model.py`
  - `train_mlx.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py autoresearch_mlx/model.py`
  - `./.venv/bin/python train_mlx.py --smoke`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 0.01 --eval-tokens 512 --canonical-eval-tokens 512`
  - matched A/B benchmark on `m5-xlarge` against the last committed cache behavior from `2be14fe`
- Measurements:
  - matched `m5-xlarge` run (`5s`, `512` eval tokens):
    - completed updates in budget: `10 -> 11`
    - fixed-budget throughput: `7.88k tok/s -> 8.34k tok/s` (`+5.9%`)
    - step-0 latency: `583 ms -> 582 ms` (flat)
    - peak memory: `4298.7 MB -> 4294.2 MB` (effectively flat)
  - ultra-short startup probe (`0.01s`) was noisy and favored the baseline in total wall-clock time, so the longer fixed-budget run is the more reliable comparison
- Confirmed behavior:
  - the smoke path still trains and evaluates end-to-end
  - the `m5-xlarge` path completed with the lazily prewarmed `2048`-token cache setup instead of relying on the old eager `10x` RoPE allocation strategy

### March 9, 2026 — `2be14fe` — Add changelog score parser — score `4`

**Human-directed, AI-shaped (4)**

- Requested a changelog parser for future autonomy plotting and later tightened the requirement to include explicit commit-count tracking for daily per-commit views.
  - Added `tools/changelog_scores.py`.
  - Added per-entry and per-day output.
  - Added CSV/JSON/TSV formats, score verification, and daily `commit_count` plus per-commit score fields.

**Grounding**

- Files:
  - `tools/changelog_scores.py`
  - `CHANGELOG.md`
  - `README.md`
  - `program_mlx.md`
- Validation:
  - `python3 -m py_compile tools/changelog_scores.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-unreleased --verify`
  - `python3 tools/changelog_scores.py --group-by day --format csv --include-unreleased`
- Historical score corrections surfaced by the parser:
  - `9a4ef79`: `39 -> 40`
  - `e06f85c`: `35 -> 36`

### March 9, 2026 — `2bcdc0c` — Add optional prepacked row caches — score `6`

**Human-directed, AI-shaped (4)**

- Implemented split-and-sequence-length keyed prepacked row caches, an opt-in `prepare_mlx.py` build path, runtime preference with fallback, and a trainer flag to disable the caches for ablations.

**AI-identified within brief, human-approved (2)**

- Requested work on optional prepacked caches after landing the optimizer checkpoint.

**Grounding**

- Files:
  - `autoresearch_mlx/constants.py`
  - `autoresearch_mlx/data.py`
  - `prepare_mlx.py`
  - `train_mlx.py`
  - `README.md`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile prepare_mlx.py train_mlx.py autoresearch_mlx/data.py autoresearch_mlx/constants.py`
  - `./.venv/bin/python prepare_mlx.py --num-shards 1 --build-prepacked-cache --prepacked-seq-lens 256,512`
  - `./.venv/bin/python train_mlx.py --smoke`
  - matched A/B benchmark on `m5-fast` against the live token-cache packing path using `--no-prepacked-cache`
- Measured effect on `m5-fast` (`2s`, matched settings):
  - step-0 latency: `48 ms -> 37 ms` (`-22.9%`)
  - completed updates in budget: `233 -> 255`
  - fixed-budget throughput: `59.65k tok/s -> 65.28k tok/s` (`+9.4%`)
  - peak memory: `147.1 MB -> 147.1 MB` (flat)
  - validation metrics: effectively unchanged within short-run noise
  - confirmed runtime behavior: train and val loaders both switched to `prepacked cache` when the matching cache existed

### March 9, 2026 — `137ba69` — Remove optimizer tree churn — score `3`

**AI-identified within brief, human-shaped (3)**

- Surfaced optimizer tree churn as the next high-value optimization after streamed gradient accumulation and, after human selection, removed it from the hot path.
  - Reworked `MuonAdamW` to cache stable parameter slots on the model.
  - Fetched gradients by cached path tokens and wrote updated arrays back directly instead of flattening and unflattening the full parameter tree on every step.
  - Cached per-group slot lists so the Muon path no longer rebuilds them inside the hot loop.

**Grounding**

- Files:
  - `autoresearch_mlx/optim.py`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/optim.py`
  - matched A/B benchmark on `m5-large` against the last committed optimizer from `9a4ef79`
- Measured effect on `m5-large` (`5s`, matched settings):
  - step-0 latency: `558 ms -> 365 ms` (`-34.6%`)
  - completed updates in budget: `17 -> 18`
  - fixed-budget throughput: `13.65k tok/s -> 14.18k tok/s` (`+3.8%`)
  - peak memory: `1946.6 MB -> 1946.6 MB` (flat)
  - steady-state per-step throughput: roughly flat within run-to-run noise

### March 9, 2026 — `9a4ef79` — Add changelog and provenance policy — score `17`

**Human-driven (5)**

- Requested a grounded `CHANGELOG.md` rather than a lightweight release log, with explicit autonomy distinctions and progressively finer provenance tiers.
- Corrected provenance overclaims with a deliberate under-claiming bias and required a branch policy that fully human-authored code changes happen in a fork.

**Human-directed, AI-shaped (4)**

- Added `CHANGELOG.md` and integrated it into the repo workflow.
  - Seeded it with grounded history for the MLX port, evaluation split, preset/token-cache work, and the streamed-accumulation change.
  - Linked the changelog from `README.md`.
  - Updated `program_mlx.md` so future experiment loops keep the changelog current.

**AI-identified within brief, human-shaped (3)**

- Proposed intermediate provenance ladders and wording variants; the human materially reshaped them into the current six-tier taxonomy and policy wording.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program_mlx.md`
- Validation:
  - docs/process only; no code-path tests were needed

### March 9, 2026 — `077a187` — Stream gradient accumulation in MLX trainer — score `3`

**AI-identified within brief, human-shaped (3)**

- Surfaced streamed gradient accumulation as a likely next optimization target and, after human selection, replaced stacked microbatch gathering with a streamed training loop.
  - Split the train step into a compiled per-microbatch gradient pass plus a compiled gradient-application pass.
  - Updated the architecture report to reflect the new training flow.

**Grounding**

- Files:
  - `train_mlx.py`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile train_mlx.py`
  - matched A/B benchmark on `m5-large` against the previous committed training loop from `e06f85c`
- Measured effect on `m5-large` (`5s`, matched settings):
  - steady-state throughput: `15,278.9 tok/s -> 15,279.2 tok/s` (`+0.00%`, effectively flat)
  - peak memory: `2453.8 MB -> 1946.6 MB` (`-507.2 MB`, `-20.7%`)
  - step-0 latency: `533 ms -> 354 ms` (`-33.6%`)
  - steps completed in budget: `18 -> 19`
  - canonical `val_bpb`: `2.373981 -> 2.368727`

### March 9, 2026 — `e06f85c` — Add token caching and calibrate M5 presets — score `19`

**Human-driven (5)**

- Requested M5-oriented practical defaults rather than retaining H100-shaped settings.
- Requested measured investigation and documentation of the reference-machine presets, including upstream-scale behavior and the README baseline figures.

**Human-directed, AI-shaped (4)**

- Requested an `m5-xlarge` or "upstream-ish" preset for the tested M5 while leaving the exact shape to the agent, then used fixed-budget measurements to document the concrete throughput, memory, and baseline figures.

**AI-identified within brief, human-shaped (3)**

- Surfaced the time-budget bug during benchmarking and, after the human requested that specific fix, implemented the budget-stop correction for slow presets.

**AI-identified within brief, human-approved (2)**

- Added prepare-time token caches and cache-aware loading so training no longer has to re-tokenize parquet text on every run.

**Grounding**

- Files:
  - `prepare_mlx.py`
  - `train_mlx.py`
  - `autoresearch_mlx/data.py`
  - `README.md`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile prepare_mlx.py train_mlx.py tools/overnight_mlx.py autoresearch_mlx/*.py`
  - `./.venv/bin/python prepare_mlx.py --num-shards 1`
  - `./.venv/bin/python train_mlx.py --smoke`
  - multiple fixed-budget runs on the reference M5 MacBook Pro (`32 GB`, `10 GPU cores`)
- Measured effect of token caching, isolated by swapping only the data path back to the pre-cache implementation:
  - `m5-fast`:
    - throughput: `61.6k tok/s -> 64.2k tok/s` (`+4.2%`)
    - step-0 latency: `130 ms -> 55 ms` (`-57.7%`)
    - peak memory: `179.6 MB -> 147.1 MB` (`-18.1%`)
  - `m5-balanced`:
    - throughput: `35.2k tok/s -> 36.0k tok/s` (`+2.3%`)
    - step-0 latency: `180 ms -> 132 ms` (`-26.7%`)
    - peak memory: `951.0 MB -> 951.0 MB` (flat in this probe)
- Baseline 5-minute canonical `val_bpb` figures on the reference machine:
  - `m5-fast`: `1.929023`
  - `m5-balanced`: `1.575952`
  - `m5-large`: `1.601354`
  - `m5-xlarge`: `1.765520`
- Upstream-scale ablation on the same machine:
  - literal `upstream` preset ran at roughly `1.1k-1.9k tok/s`
  - the same `50.3M` / `2048` architecture with M5-sized batch (`m5-xlarge`) ran at roughly `7.4k-8.0k tok/s`

### March 9, 2026 — `eee26f5` — Separate canonical eval and demote local tooling — score `9`

**Human-directed, AI-shaped (4)**

- Requested that the fork stay centered on the core MLX research loop and approved demoting optional workstation automation so it would not define the repo.

**AI-identified within brief, human-shaped (3)**

- Surfaced tooling demotion as part of keeping the core MLX path front and center, then moved the overnight automation under `tools/` and rewrote docs to describe it as optional local tooling rather than core architecture.

**AI-identified within brief, human-approved (2)**

- Surfaced canonical-vs-proxy evaluation separation as part of the MLX optimization plan, then split evaluation into canonical `val_bpb` and preset-shaped `proxy_val_bpb` and updated the sweep runner to keep or discard runs using canonical `val_bpb`.

**Grounding**

- Files:
  - `train_mlx.py`
  - `tools/overnight_mlx.py`
  - `tools/launch_overnight_mlx.sh`
  - `tools/detach_exec.py`
  - `README.md`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile train_mlx.py overnight_mlx.py autoresearch_mlx/*.py`
  - `python3 -m py_compile tools/overnight_mlx.py tools/detach_exec.py`
  - `./tools/launch_overnight_mlx.sh tooling-dry-run 0.01 --dry-run`
  - detached one-experiment sweep via the launcher path
- Measured effect:
  - detached sweep produced distinct metrics on the same run:
    - canonical `val_bpb`: `2.509401`
    - proxy `val_bpb`: `2.473907`
  - the sweep runner kept or discarded runs based on canonical `val_bpb`, not the proxy metric

### March 9, 2026 — `c3b3d8d` — Add initial MLX port for Apple Silicon — score `4`

**Human-directed, AI-shaped (4)**

- Requested a clean, feature-complete, idiomatic, maintainable MLX reimplementation of upstream `karpathy/autoresearch` for Apple Silicon rather than the PyTorch/MPS SDPA path, then implemented the core MLX path and its Apple-Silicon-first docs.
  - Implemented the MLX data path, model, optimizer, and evaluation path.
  - Kept the upstream CUDA path in-tree for reference while making the MLX path the primary workflow.
  - Added an MLX-specific program file and smoke-test path.

**Grounding**

- Files:
  - `prepare_mlx.py`
  - `train_mlx.py`
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/model.py`
  - `autoresearch_mlx/optim.py`
  - `program_mlx.md`
  - `README.md`
  - `pyproject.toml`
- Validation:
  - `python3 -m py_compile prepare_mlx.py train_mlx.py autoresearch_mlx/*.py`
  - `./.venv/bin/python prepare_mlx.py --num-shards 1`
  - `./.venv/bin/python train_mlx.py --smoke`
- Smoke-test result immediately before the baseline commit:
  - canonical `val_bpb`: `2.187049`
  - `training_seconds`: `1.0`
  - `peak_vram_mb`: `194.8`
