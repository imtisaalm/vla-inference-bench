# VLA Inference Benchmark

A Python harness that benchmarks vision-language-action (VLA) robot policy inference the same way an LLM benchmark measures token generation: per-action-chunk latency (p50/p95), throughput, and peak memory, across CPU, Apple Silicon (MPS), and CUDA.

This is the companion to [TTFT-ITL-Benchmark-suite](https://github.com/imtisaalm/TTFT-ITL-Benchmark-suite). That project measures LLM inference under load. This one measures robot policy inference, one action chunk at a time. Same measurement discipline, different workload.

## Why Python

The TTFT suite is written in Go because the load generator only needs HTTP, concurrency, and timers. VLA policies do not work that way. The weights live on the HuggingFace Hub in PyTorch format and the only maintained loaders (`lerobot`, `transformers`) are Python. Reimplementing the policy stack in Go would mean rewriting the model itself, so this harness is Python by necessity. The benchmarking layer (timing, statistics, result files) stays dependency-light on top of it.

## What it measures

For each configured policy, on each detected device:

- **Per-chunk latency**: wall time to generate one action chunk (p50, p95, mean, min, max). A chunk is the policy's native output unit (50 action steps for SmolVLA).
- **Throughput**: action chunks per second, and actions per second (chunks/sec x chunk size).
- **Peak memory**: CUDA device allocation on NVIDIA GPUs, peak process RSS on CPU/MPS. The result file labels which one it is.

Warm-up runs execute before the measured sweep and are excluded from statistics.

## Policies

| Policy | Status | Checkpoint |
| --- | --- | --- |
| SmolVLA (450M) | Working | `lerobot/smolvla_base` |
| ACT | Planned | extension point in `bench.py` |
| Diffusion Policy | Planned | extension point in `bench.py` |

Nothing like an open VLA latency benchmark exists today. The goal is to make this the standard place people check before deploying a policy to a real robot.

## Install

Requires Python 3.10 or newer. Built and tested Mac-first.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda works too (`conda create -n vla-bench python=3.11`, then pip install the requirements inside it). Either way, use a virtual environment; the `lerobot` dependency tree is large and does not belong in your system Python.

Apple Silicon vs Intel: the harness runs on both. On Apple Silicon it uses PyTorch MPS (GPU acceleration). On Intel Macs it falls back to CPU. No NVIDIA GPU is needed or expected on a Mac; CUDA is supported for Linux/Windows machines with NVIDIA cards.

Linux note: on a Linux box without a GPU, install the CPU-only PyTorch wheels to avoid downloading gigabytes of CUDA libraries:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

Full benchmark (downloads SmolVLA weights, ~1-2 GB, on first run):

```bash
python bench.py --policy smolvla --runs 20 --warmup 3
```

Dry run (no download, synthetic stand-in model, exercises the timing pipeline):

```bash
python bench.py --dry-run
```

Options:

```bash
python bench.py --policy smolvla --device cpu --runs 10 --warmup 2 --output results/my-run.json
```

`--device` accepts `auto` (default), `mps`, `cuda`, or `cpu`. Auto-detection prefers MPS on Apple Silicon, then CUDA, then CPU. Everything runs on CPU at minimum, so an Intel Mac works; it is just slower.

What to expect: SmolVLA is a 450M-parameter model generating 50-step action chunks through 10 denoising steps. On Apple Silicon with MPS, expect seconds per chunk. On CPU (Intel Mac or otherwise), expect tens of seconds per chunk. Slow is fine here; this is a benchmark, not a demo. The numbers are the product.

Terminal output is compact:

```text
policy    device  chunk_p50  chunk_p95  chunks/s  actions/s  peak_mem
smolvla   cpu     812.4ms    901.2ms    1.21      60.5       2140 MiB (rss)
```

Each run also writes a JSON result file to `results/` with the full configuration, per-run latencies, and hardware metadata. See `results/results_template.md` for the table format used to publish numbers.

## Roadmap

- Wire ACT and Diffusion Policy checkpoints into the policy factory
- Multi-camera and state-dimension sweeps (input scaling behavior)
- Batch-size sweep for server-side deployment
- SO-101 arm deployment numbers: sim vs real hardware, same harness
