#!/usr/bin/env python3
"""VLA inference benchmark harness.

Times vision-language-action policy inference the way an LLM benchmark times
token generation: per-action-chunk latency (p50/p95), throughput, peak memory.

    python bench.py --policy smolvla --runs 20 --warmup 3
    python bench.py --dry-run          # no weight download, synthetic stand-in

The observation is synthetic (random pixels, random joint state). This measures
inference cost, not task success. That is the point: deployment decisions need
latency numbers, not demo videos.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

CHECKPOINTS = {
    "smolvla": "lerobot/smolvla_base",
    # Extension point: add "act" and "diffusion" checkpoints here once wired.
}

CHUNK_SIZE = {
    "smolvla": 50,  # n_action_steps in lerobot/smolvla_base config
}

# Observation spec for lerobot/smolvla_base (from its config.json).
OBS_SPEC = {
    "observation.state": (6,),
    "observation.images.camera1": (3, 256, 256),
    "observation.images.camera2": (3, 256, 256),
    "observation.images.camera3": (3, 256, 256),
}


def detect_device() -> str:
    """Device preference: MPS (Apple Silicon) first, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Fixed task string used for every benchmark run. Latency is what is measured,
# not task success, but the language input must be real token IDs.
TASK = "pick up the cube"


def build_observation(device: str, dry_run: bool = False) -> dict:
    """One synthetic observation batch, shaped exactly like the policy expects."""
    obs = {}
    for key, shape in OBS_SPEC.items():
        t = torch.rand(1, *shape, dtype=torch.float32)
        if key == "observation.state":
            t = t * 2.0 - 1.0  # joint state in [-1, 1]
        obs[key] = t.to(device)
    if not dry_run:
        # SmolVLA expects tokenized language input alongside the observation.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        enc = tok(
            TASK,
            padding="max_length",
            max_length=48,
            truncation=True,
            return_tensors="pt",
        )
        obs["observation.language.tokens"] = enc["input_ids"].to(device)
        obs["observation.language.attention_mask"] = enc["attention_mask"].to(device)
    return obs


class DryRunPolicy(torch.nn.Module):
    """Synthetic stand-in with the same interface as a real policy.

    Random weights, exercises the timing pipeline honestly. Results from this
    are labeled dry_run=true and mean nothing about any real policy.
    """

    def __init__(self, chunk_size: int = 50, action_dim: int = 6):
        super().__init__()
        flat = 6 + 3 * 3 * 256 * 256
        self.net = torch.nn.Sequential(
            torch.nn.Linear(flat, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, chunk_size * action_dim),
        )
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def select_action(self, obs: dict) -> torch.Tensor:
        parts = [obs[k].reshape(obs[k].shape[0], -1) for k in sorted(obs)]
        x = torch.cat(parts, dim=1)
        return self.net(x).reshape(-1, self.chunk_size, self.action_dim)

    def predict_action_chunk(self, obs: dict) -> torch.Tensor:
        return self.select_action(obs)

    def reset(self) -> None:
        pass


def load_policy(name: str, device: str):
    if name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(CHECKPOINTS["smolvla"])
        policy.to(device)
        policy.eval()
        return policy
    if name in ("act", "diffusion"):
        raise NotImplementedError(
            f"Policy '{name}' is not wired yet. Add its checkpoint to "
            f"CHECKPOINTS and a loader branch in load_policy()."
        )
    raise ValueError(f"Unknown policy: {name}")


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    # CPU is synchronous; nothing to do.


def peak_memory_mib(device: str) -> tuple[float, str]:
    """Returns (MiB, kind) where kind is 'cuda' or 'rss'."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**2), "cuda"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_mib = rss / (1024**2)  # bytes on macOS
    else:
        rss_mib = rss / 1024  # kilobytes on Linux
    return rss_mib, "rss"


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark VLA policy inference latency.")
    ap.add_argument("--policy", default="smolvla", choices=["smolvla", "act", "diffusion"])
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip weight download; time a synthetic stand-in model.")
    ap.add_argument("--output", default=None, help="Result JSON path.")
    args = ap.parse_args()

    device = detect_device() if args.device == "auto" else args.device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available on this machine.", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available on this machine.", file=sys.stderr)
        return 2

    dry_run = args.dry_run
    if dry_run:
        policy_name, checkpoint = "dry-run-standin", None
        policy = DryRunPolicy().to(device).eval()
        chunk_size = 50
    else:
        policy_name, checkpoint = args.policy, CHECKPOINTS[args.policy]
        print(f"Loading {checkpoint} ...", file=sys.stderr)
        policy = load_policy(args.policy, device)
        chunk_size = CHUNK_SIZE[args.policy]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    obs = build_observation(device, dry_run=dry_run)

    with torch.no_grad():
        for _ in range(args.warmup):
            policy.reset()
            policy.predict_action_chunk(obs)
            synchronize(device)

        latencies: list[float] = []
        for _ in range(args.runs):
            policy.reset()
            t0 = time.perf_counter()
            policy.predict_action_chunk(obs)
            synchronize(device)
            latencies.append(time.perf_counter() - t0)

    mem_mib, mem_kind = peak_memory_mib(device)
    total = sum(latencies)
    chunks_per_s = len(latencies) / total if total > 0 else 0.0

    stats = {
        "runs": len(latencies),
        "chunk_latency_s": {
            "mean": float(np.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        },
        "chunks_per_s": chunks_per_s,
        "actions_per_s": chunks_per_s * chunk_size,
        "chunk_size": chunk_size,
        "peak_memory_mib": round(mem_mib, 1),
        "peak_memory_kind": mem_kind,
    }

    result = {
        "policy": policy_name,
        "checkpoint": checkpoint,
        "dry_run": dry_run,
        "device_requested": args.device,
        "device_used": device,
        "warmup_runs": args.warmup,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "observation": "synthetic (random pixels, random joint state)",
        "task": None if dry_run else TASK,
        "stats": stats,
        "latencies_s": [round(x, 4) for x in latencies],
    }

    out = args.output or os.path.join(
        "results", f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    c = stats["chunk_latency_s"]
    mem_label = f"{stats['peak_memory_mib']:.0f} MiB ({mem_kind})"
    tag = "DRY-RUN " if dry_run else ""
    print(f"{tag}policy    device  chunk_p50  chunk_p95  chunks/s  actions/s  peak_mem")
    print(
        f"{tag}{policy_name:<9} {device:<7} "
        f"{c['p50']*1000:>7.1f}ms {c['p95']*1000:>7.1f}ms "
        f"{stats['chunks_per_s']:>8.2f} {stats['actions_per_s']:>9.1f}  {mem_label}"
    )
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
