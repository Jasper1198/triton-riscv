#!/usr/bin/env python3
"""Connect the existing EvoAutotuner to the SSH vector-add benchmark.

Only BLOCK_SIZE varies. Compatibility metadata in triton.Config stays fixed;
the ELF helper's compiler options remain unchanged. Returns a benchmark report,
not a torch.Tensor. Existing EvoTuner files are not modified by this adapter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid


class RemoteVecAddKernel:
    arg_names = ["N"]

    def __init__(self, *, root, target, identity, budget, output_dir,
                 warmup=20, repeats=1000, samples=11):
        self.root = Path(root)
        self.target = target
        self.identity = Path(identity)
        self.budget = budget
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.warmup, self.repeats, self.samples = warmup, repeats, samples
        self.attempts = 0
        self.measured = {}
        self.failed = {}

    def run(self, N, *, BLOCK_SIZE, **compatibility_metadata):
        # EvoAutotuner.run() calls this once again with the chosen config.
        # Reuse that config's measured result instead of doing another benchmark.
        N, block = int(N), int(BLOCK_SIZE)
        key = (N, block)
        if key in self.measured:
            return dict(self.measured[key])
        if key in self.failed:
            raise RuntimeError(self.failed[key])
        if self.attempts >= self.budget:
            raise RuntimeError("Remote evaluation budget exhausted")

        self.attempts += 1
        command = [
            sys.executable, str(self.root / "python/examples/evo_bench_vecadd.py"),
            "--root", str(self.root), "--mode", "ssh", "--target", self.target,
            "--identity", str(self.identity), "--block-size", str(block),
            "--size", str(N), "--warmup", str(self.warmup),
            "--repeats", str(self.repeats), "--samples", str(self.samples),
        ]
        print(f"\n[REMOTE {self.attempts}/{self.budget}] N={N}, BLOCK_SIZE={block}", flush=True)
        base = self.output_dir / f"attempt-{self.attempts:02d}-b{block}"
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=600)
            base.with_suffix(".stdout.log").write_text(process.stdout)
            base.with_suffix(".stderr.log").write_text(process.stderr)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Benchmark exited with {process.returncode}; see {base}.stderr.log"
                )
            records = [line.removeprefix("EVO_BENCH_JSON ")
                       for line in process.stdout.splitlines()
                       if line.startswith("EVO_BENCH_JSON ")]
            if len(records) != 1 or "PASS" not in process.stdout.splitlines():
                raise RuntimeError("Benchmark output is missing JSON or PASS")
            report = json.loads(records[0])
            latency = float(report["median_ms"])
            if (report.get("correct") is not True or report.get("size") != N
                    or report.get("block_size") != block
                    or not math.isfinite(latency) or latency <= 0):
                raise RuntimeError(f"Invalid benchmark result: {report}")
            report.update({"mode": "ssh", "target": self.target,
                           "warmup": self.warmup,
                           "measurement": "warm_cache_serial_full_grid_ms"})
            self.measured[key] = report
            self._record({"status": "ok", "attempt": self.attempts, **report})
            print(f"[MEASURED] BLOCK_SIZE={block}: {latency * 1000:.6f} us", flush=True)
            return dict(report)
        except (subprocess.TimeoutExpired, OSError, ValueError, KeyError, RuntimeError) as exc:
            self.failed[key] = str(exc)
            self._record({"status": "failed", "attempt": self.attempts,
                          "size": N, "block_size": block, "error": str(exc)})
            raise

    def _record(self, value):
        with (self.output_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")


def remote_tuner_class(base_class):
    class RemoteEvoAutotuner(base_class):
        def _bench(self, config, *args, **meta):
            if len(args) != 1 or set(config.kwargs) != {"BLOCK_SIZE"}:
                raise ValueError("This adapter supports vecadd(N), tuning BLOCK_SIZE only")
            if "BLOCK_SIZE" in meta:
                raise ValueError("Let EvoTuner provide BLOCK_SIZE")
            report = self.fn.run(args[0], BLOCK_SIZE=config.kwargs["BLOCK_SIZE"])
            # Original EvoTuner and its cost models use milliseconds.
            return float(report["median_ms"])

    return RemoteEvoAutotuner


def unused_local_bench(*args, **kwargs):
    raise RuntimeError("Local timing must not be used for a remote benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/triton-riscv"))
    parser.add_argument("--target", default=os.environ.get("EVO_TARGET"))
    parser.add_argument("--identity", type=Path, default=Path("/workspace/evo-ssh/id_ed25519"))
    parser.add_argument("--size", type=int, default=4099)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--cost-model", choices=["random", "xgb"], default="xgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()
    if not args.target or args.target.startswith("-") or any(c.isspace() for c in args.target):
        parser.error("provide --target user@host")
    if not args.identity.is_file():
        parser.error(f"SSH key not found: {args.identity}")
    if not (3 <= args.budget <= 6) or args.size <= 0:
        parser.error("this smoke search requires budget 3..6 and positive size")
    if args.warmup < 0 or args.repeats <= 0 or args.samples < 3 or args.samples % 2 == 0:
        parser.error("warmup >= 0, repeats > 0, and odd samples >= 3 are required")
    benchmark = args.root / "python/examples/evo_bench_vecadd.py"
    if not benchmark.is_file():
        parser.error(f"Benchmark script not found: {benchmark}")

    sys.path.insert(0, str(args.root / "python/evo_vendor"))
    import triton
    from dev.search_plan.evo_search import EvoAutotuner

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_dir = args.root / "artifacts/riscv/evo-search" / run_id
    kernel = RemoteVecAddKernel(
        root=args.root, target=args.target, identity=args.identity,
        budget=args.budget, output_dir=output_dir, warmup=args.warmup,
        repeats=args.repeats, samples=args.samples,
    )
    # Endpoints define six possible tile values: 8,16,32,64,128,256.
    # Metadata stays constant and is not passed to the standalone ELF compiler.
    configs = [triton.Config({"BLOCK_SIZE": block}, num_warps=1, num_stages=1, num_ctas=1)
               for block in (8, 256)]
    tuner = remote_tuner_class(EvoAutotuner)(
        fn=kernel, arg_names=kernel.arg_names, configs=configs, key=["N"],
        pop_size=6, max_generation=3, max_tuned_configs=args.budget, top_k=1,
        cost_model=args.cost_model, seed=args.seed, do_bench=unused_local_bench,
        warmup_configs=2 if args.cost_model == "xgb" else 0,
    )
    print(f"Search output: {output_dir}", flush=True)
    print("Search space: BLOCK_SIZE=8,16,32,64,128,256; fixed compiler options", flush=True)
    best_report = tuner.run(
        args.size,
        grid=lambda meta: (triton.cdiv(meta["N"], meta["BLOCK_SIZE"]),),
    )
    if not kernel.measured:
        raise RuntimeError(f"No valid remote measurements; inspect {output_dir}")
    measured_best = min(kernel.measured.values(), key=lambda report: report["median_ms"])
    if best_report["median_ms"] != measured_best["median_ms"]:
        raise RuntimeError("EvoTuner's returned config differs from the measured best")
    summary = {
        "kernel": "vector_add", "target": args.target, "size": args.size,
        "cost_model": args.cost_model, "seed": args.seed, "budget": args.budget,
        "attempts": kernel.attempts, "valid_measurements": len(kernel.measured),
        "search_space": [8, 16, 32, 64, 128, 256],
        "best_among_measured": best_report,
        "note": "Small integration search; not proof of the global optimum or XGBoost benefit.",
    }
    destination = output_dir / "best.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nBEST MEASURED BLOCK_SIZE={best_report['block_size']}")
    print(f"MEDIAN={best_report['median_us']:.6f} us")
    print(f"REMOTE ATTEMPTS={kernel.attempts}, VALID={len(kernel.measured)}")
    print(f"SUMMARY: {destination}")


if __name__ == "__main__":
    main()
