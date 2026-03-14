# DGX Spark Setup

This is the concrete bring-up tutorial for a DGX Spark / GB10 box.

If you follow it end to end, you should end up with:

- a working CUDA trainer path
- a reusable FA4-enabled runtime image
- a working `calibrate.py --engine cuda` flow
- a running agent research loop that uses the calibrated CUDA result

In this repo, DGX Spark is treated as the `blackwell-gb10` CUDA reference family. That matters because it wants the GB10 / SM120 FlashAttention 4 path, it should not be lumped into generic Blackwell or B200 calibration, and it is the local Blackwell workstation reference point for CUDA bring-up.

## 1. Prepare The Host

Make sure the DGX Spark host already has:

- a working NVIDIA driver stack
- Docker with `--gpus all`
- enough shared memory for this path; reserve at least `100g`

If you want full CUDA `kernel-lab` support, also enable GPU performance counters for Nsight Compute. Without this, `ncu` will fail with `ERR_NVGPUCTRPERM`.

On the host:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/99-nvidia-profiling.conf
sudo update-initramfs -u
sudo reboot
```

After reboot:

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

Expected:

```text
RmProfilingAdminOnly: 0
```

For profiling runs below, also pass:

```bash
--cap-add=SYS_ADMIN
```

## 2. Clone The Repo On The Spark

Clone the repo directly on the DGX Spark box:

```bash
ssh <spark-user>@<spark-host>
cd "$HOME"
git clone https://github.com/Entrpi/autoresearch-everywhere autoresearch-everywhere
cd autoresearch-everywhere
```

Before later runs, just update it:

```bash
ssh <spark-user>@<spark-host>
cd "$HOME/autoresearch-everywhere"
git pull --ff-only
```

The examples below assume the checkout lives at:

- `$HOME/autoresearch-everywhere`

## 3. Start From The Known Base Image

The validated runtime lineage here started from:

- source repo: [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
- local March 1, 2026 snapshot of `vllm-node-tf5:latest`
- image ID `sha256:c1ba011f841cacdfc234e5b754b1cb5e8120b8d4bd6b896c6703b28a44ba185a`

Runtime stack inside that base image:

- NVIDIA PyTorch image family: `26.01`
- NVIDIA build ID: `256811084`
- NVIDIA build ref: `9fa5c48351cf93ac6e6972ca113a7e3c54675a76`
- PyTorch: `2.10.0a0+a36e1d39eb.nv26.01.42222806`
- CUDA runtime: `13.1`

That base image is good enough for the raw CUDA trainer, but not for the desired GB10 path. For this repo, you also want:

- FlashAttention 4 for SM120
- `rustbpe`

## 4. Build The FA4 Runtime Image

The working FA4 path here was the SM120 / GB10 branch from [FA4 PR](https://github.com/Dao-AILab/flash-attention/pull/2268).

The important compile settings are:

```bash
FLASH_ATTN_CUDA_ARCHS=120
MAX_JOBS=10
CMAKE_BUILD_PARALLEL_LEVEL=10
NVCC_THREADS=1
```

That keeps the build focused on GB10 instead of compiling unnecessary architectures.

Start a mutable builder container from the base image:

```bash
docker run --gpus all \
  --shm-size=100g \
  -it \
  --name fa4-sm120-builder \
  vllm-node-tf5:latest \
  bash
```

Inside that container:

```bash
set -eux
python -m pip install --upgrade pip packaging ninja
cd /tmp
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git fetch origin pull/2268/head:fa4-pr-2268
git checkout fa4-pr-2268
git submodule update --init --recursive
export FLASH_ATTN_CUDA_ARCHS=120
export MAX_JOBS=10
export CMAKE_BUILD_PARALLEL_LEVEL=10
export NVCC_THREADS=1
python -m pip install -v --no-build-isolation .
python -m pip install --no-cache-dir rustbpe
python - <<'PY'
import flash_attn.flash_attn_interface as fai
import rustbpe
print("flash_attn_func", hasattr(fai, "flash_attn_func"))
print("rustbpe_ok", rustbpe is not None)
PY
```

Back on the host, turn that mutated container into the image you will actually use for later runs:

```bash
docker commit fa4-sm120-builder vllm-node-tf5-fa4:sm120
docker rm -f fa4-sm120-builder
```

This is the important persistence step. Installing FA4 in an interactive builder container does not affect later `docker run vllm-node-tf5:latest ...` calls unless you commit the mutated container to a new image and start using that new image name.

From here on, the runtime image should be:

- `vllm-node-tf5-fa4:sm120`

And it should already contain:

- `flash_attn.flash_attn_interface`
- `rustbpe`

## 5. Verify The Runtime Image

Before running the repo, do one direct image sanity check:

```bash
docker run --gpus all --rm vllm-node-tf5-fa4:sm120 \
  python - <<'PY'
import torch
import flash_attn.flash_attn_interface as fai
import rustbpe
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("flash_attn_func", hasattr(fai, "flash_attn_func"))
print("rustbpe_ok", rustbpe is not None)
PY
```

If that fails, stop there and fix the image before trying trainer or calibration runs.

## 6. Run A CUDA Trainer Smoke

Once the image is healthy, bind-mount the repo and run a smoke check:

```bash
docker run --gpus all \
  --shm-size=100g \
  --rm \
  -v "$HOME/autoresearch-everywhere:/workspace/autoresearch-everywhere" \
  -v "$HOME/.cache/autoresearch:/root/.cache/autoresearch" \
  -w /workspace/autoresearch-everywhere \
  vllm-node-tf5-fa4:sm120 \
  sh -lc '
    python prepare.py --engine cuda &&
    python train.py --engine cuda --smoke
  '
```

On a correctly provisioned GB10 / FA4 image, the trainer should report something like:

- `hardware_key: nvidia-blackwell-gb10-120gb`
- `accelerator_architecture: blackwell-gb10`
- `preferred_flash_attention_generation: 4`
- `resolved_attention_backend: installed:flash_attn.flash_attn_interface`

If it says `torch-sdpa`, you are either on the wrong image or the FA4 install did not persist.

## 7. Run Fast Calibration

This is the basic persistent-output calibration pattern:

```bash
docker run --gpus all \
  --cap-add=SYS_ADMIN \
  --shm-size=100g \
  --rm \
  -v "$HOME/autoresearch-everywhere:/workspace/autoresearch-everywhere" \
  -v "$HOME/.cache/autoresearch:/root/.cache/autoresearch" \
  -v "$HOME/cuda_fast_projection_fa4:/output" \
  -v "$HOME/curve_runs:/truth_curves:ro" \
  -w /workspace/autoresearch-everywhere \
  vllm-node-tf5-fa4:sm120 \
  sh -lc '
    python -u calibrate.py \
      --engine cuda \
      --mode fast \
      --output-dir /output \
      > /output/run.log 2>&1
  '
```

Important parts:

- bind-mount the repo into `/workspace/autoresearch-everywhere`
- bind-mount a persistent output directory on the host
- bind-mount `~/.cache/autoresearch` into `/root/.cache/autoresearch`
- reserve at least `100g` of SHM for this GB10 / FA4 path

The latest full FA4-backed fast calibration on GB10 currently emits:

- preset: `m5-balanced`
- `seq_len=1024`
- `window_pattern=SSSSL`
- `device_batch_size=8`
- `total_batch_size=49152`
- `grad_accum_steps=6`

That is the operating point the current calibration flow is trying to hand to the research loop.

Do not compare the short local-search probe throughput from that stage against the table below. The local-search pass is only a brief shape-selection check. The grounded throughput, token-count, and memory numbers below come from full `300s` truth runs and are the right reference for practical comparisons.

Separately, the grounded `300s` truth-curve reference presets on GB10 are:

| Preset          | Best first use on GB10                 |  Seq len | Depth / d_model / heads |    Params | Batch (device / total tokens) | Window    | Approx. tok/s | 300s steps | 300s tokens | Approx. peak memory | `300s val_bpb` |
| --------------- | -------------------------------------- | -------: | ----------------------- | --------: | ----------------------------- | --------- | ------------: | ---------: | ----------: | ------------------: | ---------------: |
| `m5-small`    | fastest throughput, capacity limited   |  `512` | `4 / 256 / 2`         | `11.5M` | `32 / 32768`                | `L`     |   `~523.9k` |   `4528` |  `148.4M` |        `~1.16 GB` |     `1.257603` |
| `m5-balanced` | current strict `300s` quality winner | `1024` | `6 / 384 / 3`         | `26.3M` | `32 / 32768`                | `SSSSL` |   `~267.2k` |   `2286` |   `74.9M` |        `~3.59 GB` |     `1.162382` |
| `m5-xlarge`   | near-frontier scaling candidate        | `2048` | `8 / 512 / 4`         | `50.3M` | `16 / 32768`                | `L`     |   `~146.1k` |   `1241` |   `40.7M` |        `~6.00 GB` |     `1.169337` |

So the practical read is:

- `calibrate.py --engine cuda` currently emits `m5-balanced` at `1024 / SSSSL / db8 / tb49152`
- `m5-balanced` is also the best current answer if the objective is strictly lowest `val_bpb` at `300s`
- `m5-xlarge` is close enough to justify longer-horizon experiments if you want to push beyond the strict `300s` winner

## 8. Start The Research Loop

Once fast calibration completes, the main bring-up goal is no longer more setup work. It is to start the actual training loop using the calibrated result.

The normal repo behavior is:

- `calibrate.py --engine cuda` writes the recommended result into the local CUDA platform-default cache
- later `train.py --engine cuda` runs can reuse that calibrated default instead of forcing you to name a preset manually

So the main next step after calibration is just:

```bash
docker run --gpus all \
  --shm-size=100g \
  --rm \
  -v "$HOME/autoresearch-everywhere:/workspace/autoresearch-everywhere" \
  -v "$HOME/.cache/autoresearch:/root/.cache/autoresearch" \
  -w /workspace/autoresearch-everywhere \
  vllm-node-tf5-fa4:sm120 \
  sh -lc '
    python -u train.py --engine cuda
  '
```

If you want to hand the loop to an agent, point it at `program.md` after calibration and tell it to continue from the calibrated CUDA setup rather than redoing bring-up.

The grounded GB10 result so far is still:

- `m5-balanced` as the current FA4-backed calibrated default
- `m5-balanced` as the strict `300s` quality winner
- `m5-xlarge` as the closest scaling candidate beyond it

That means the real bring-up story here is:

1. prepare the host
2. build the FA4 image
3. run calibration
4. start the research loop

Kernel-lab is valuable, but it is not the next thing you need in order to be productive on the box.

## Appendix: CUDA Kernel-Lab On DGX Spark

If you want backend profiling and kernel work after the trainer loop is already healthy, the first useful CUDA `kernel-lab` sequence is:

```bash
uv run kernel-lab.py --engine cuda capture --preset upstream --time-budget 20 --output /tmp/cuda-upstream-trace
uv run kernel-lab.py --engine cuda trace-profile --metadata /tmp/cuda-upstream-trace.metadata.json --output /tmp/cuda-upstream-trace.profile.json
uv run kernel-lab.py --engine cuda auto-review --trace-profile /tmp/cuda-upstream-trace.profile.json
uv run kernel-lab.py --engine cuda deep-profile --trace-profile /tmp/cuda-upstream-trace.profile.json --rank 1
```

That path is now validated on real GB10 hardware.

The first real run surfaced:

- dominant issue: `sync-bound`
- top families:
  - `launch_fusion`
  - `data_movement`

And after parser/matcher fixes, the CUDA trace-family mapping no longer misclassifies generic setup kernels as `matmul_epilogue`.

## Troubleshooting

### FlashAttention Still Is Not Used

If the trainer says:

```text
resolved_attention_backend: torch-sdpa
```

check the runtime image directly:

```bash
python - <<'PY'
import importlib
for name in [
    "flash_attn",
    "flash_attn.flash_attn_interface",
    "hopper.flash_attn_interface",
    "rustbpe",
]:
    try:
        importlib.import_module(name)
        print(name, "ok")
    except Exception as exc:
        print(name, "fail", exc)
PY
```

On GB10 you want `flash_attn.flash_attn_interface` and `rustbpe` to import successfully.

### Nsight Compute Fails With `ERR_NVGPUCTRPERM`

The host GPU counter setting was not applied, or the container was started without:

```bash
--cap-add=SYS_ADMIN
```

### The Container Exits But No Final Report Appears

First check:

- `run.log`
- `logs/*.stderr.log`

## Recommended Reading

- [docs/kernel-lab.md](./kernel-lab.md)
- [docs/cuda-core-loop-parity.md](./cuda-core-loop-parity.md)
- [docs/platform-calibration.md](./platform-calibration.md)
