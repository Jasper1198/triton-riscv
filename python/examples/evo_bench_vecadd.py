#!/usr/bin/env python3
"""Compile a timed RVV vector-add ELF and run it via SSH or QEMU.

Run from the configured triton-riscv venv. Timing covers one full serial
launch grid, including its C dispatch loops. Compilation, transfer, SSH,
initialization, verification and output are outside the timing region.
This is a warm-cache, single-process smoke benchmark, not an EvoTuner search.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


@contextmanager
def cross_compile_environment(toolchain):
    updates = {
        "TRITON_RISCV_CROSS_COMPILE": "1",
        "TRITON_RISCV_TARGET_TRIPLE": toolchain.triple,
        "TRITON_RISCV_LLC_FEATURES": toolchain.llc_features,
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def timed_runner(runner, *, size, block_size, warmup, repeats, samples):
    """Adapt this backend's generated vecadd runner; fail on unknown layout."""
    pattern = (
        r"^  for \(int x = 0; x < \d+; \+\+x\)\n"
        r"    for \(int y = 0; y < \d+; \+\+y\)\n"
        r"      for \(int z = 0; z < \d+; \+\+z\)\n"
        r"        [^\n]+;"
    )
    matches = list(re.finditer(pattern, runner, flags=re.MULTILINE))
    if len(matches) != 1:
        raise RuntimeError("Unsupported generated runner grid layout")
    match = matches[0]
    grid = match.group(0)
    print_start = runner.find('  printf("x_ptr=[");', match.end())
    if print_start < 0:
        raise RuntimeError("Unsupported generated runner output layout")
    # Keep verification, remove verbose buffer printing. Reject NaN as well.
    prefix = runner[:print_start].replace("if (diff > 0)", "if (!(diff <= 0))")
    checks = runner[match.end():print_start].replace(
        "if (diff > 0)", "if (!(diff <= 0))"
    )
    if "verification failed:" not in checks:
        raise RuntimeError("Generated runner does not contain output verification")

    helpers = r'''
static uint64_t evo_now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    perror("clock_gettime");
    exit(2);
  }
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int evo_compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

'''
    prefix = (
        "#define _POSIX_C_SOURCE 200809L\n#include <time.h>\n#include <stdlib.h>\n"
        + prefix
    )
    prefix = prefix.replace("int main(void) {", helpers + "int main(void) {", 1)

    timing = r'''
  for (int evo_w = 0; evo_w < @WARMUP@; ++evo_w) {
@GRID@
  }
  double evo_ms[@SAMPLES@];
  for (int evo_s = 0; evo_s < @SAMPLES@; ++evo_s) {
    uint64_t evo_begin = evo_now_ns();
    for (int evo_r = 0; evo_r < @REPEATS@; ++evo_r) {
@GRID@
    }
    uint64_t evo_end = evo_now_ns();
    evo_ms[evo_s] = (double)(evo_end - evo_begin) / @REPEATS@ / 1000000.0;
  }
@CHECKS@
  qsort(evo_ms, @SAMPLES@, sizeof(double), evo_compare_double);
  printf("EVO_BENCH_JSON {\"correct\":true,\"size\":@SIZE@,"
         "\"block_size\":@BLOCK@,\"samples\":@SAMPLES@,"
         "\"repeats\":@REPEATS@,\"median_ms\":%.9f,"
         "\"median_us\":%.6f,\"sample_ms_sorted\":[",
         evo_ms[@MEDIAN@], evo_ms[@MEDIAN@] * 1000.0);
  for (int evo_s = 0; evo_s < @SAMPLES@; ++evo_s)
    printf(evo_s ? ",%.9f" : "%.9f", evo_ms[evo_s]);
  puts("]}");
  puts("PASS");
  return 0;
}
'''
    replacements = {
        "@WARMUP@": str(warmup), "@SAMPLES@": str(samples),
        "@REPEATS@": str(repeats), "@SIZE@": str(size),
        "@BLOCK@": str(block_size), "@MEDIAN@": str(samples // 2),
        "@GRID@": grid, "@CHECKS@": checks,
    }
    for token, value in replacements.items():
        timing = timing.replace(token, value)
    return prefix + timing


def checked_run(command, timeout=180):
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=timeout
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/triton-riscv"))
    parser.add_argument("--mode", choices=["compile-only", "qemu", "ssh"], default="compile-only")
    parser.add_argument("--target", default=os.environ.get("EVO_TARGET"))
    parser.add_argument("--identity", type=Path, default=Path("/workspace/evo-ssh/id_ed25519"))
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=4099)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        parser.error("--block-size must be a positive power of two")
    if args.size <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("size/repeats must be positive; warmup must be nonnegative")
    if args.samples < 3 or args.samples % 2 == 0:
        parser.error("--samples must be an odd integer >= 3")
    if args.mode == "ssh":
        if not args.target or args.target.startswith("-") or re.search(r"\s", args.target):
            parser.error("provide a valid --target user@host")
        if not args.identity.is_file():
            parser.error(f"SSH private key not found: {args.identity}")

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared import riscv

    sys.path.insert(0, str(args.root / "python/examples"))
    from rvv_vec_add_elf import vector_add_kernel

    toolchain = riscv.RiscvToolchain.from_env()
    size = args.size
    x = [float(i % 97) for i in range(size)]
    y = [float((i * 2) % 89) for i in range(size)]
    arguments = {
        "x_ptr": x, "y_ptr": y, "output_ptr": [0.0] * size,
        "n_elements": size,
    }
    source = riscv.make_standalone_source(
        vector_add_kernel, arguments, constexprs={"BLOCK_SIZE": args.block_size}
    )
    print(f"Compiling vecadd: size={size}, BLOCK_SIZE={args.block_size}", flush=True)
    with cross_compile_environment(toolchain):
        compiled = triton.compile(source, target=GPUTarget("cpu", 0, 0))
    generated = riscv._generate_runner(
        compiled.metadata.name, source.signature, arguments,
        (triton.cdiv(size, args.block_size),),
        expected={"output_ptr": [a + b for a, b in zip(x, y)]}, atol=0.0,
    )
    runner = timed_runner(
        generated, size=size, block_size=args.block_size,
        warmup=args.warmup, repeats=args.repeats, samples=args.samples,
    )
    directory = args.root / f"artifacts/riscv/evo-bench/vecadd-n{size}-b{args.block_size}"
    directory.mkdir(parents=True, exist_ok=True)
    obj, c_file, elf = directory / "kernel.o", directory / "runner.c", directory / "candidate.elf"
    obj.write_bytes(compiled.asm["obj"])
    c_file.write_text(runner)
    riscv.link_elf([obj], [c_file], elf, toolchain=toolchain)
    print(f"ELF: {elf}", flush=True)
    if args.mode == "compile-only":
        return

    if args.mode == "ssh":
        options = ["-i", str(args.identity), "-o", "IdentitiesOnly=yes",
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        target_arch = checked_run(["ssh", *options, args.target, "uname -m"]).stdout.strip()
        if target_arch != "riscv64":
            raise RuntimeError(f"Expected riscv64 target, got {target_arch!r}")
        checked_run(["ssh", *options, args.target, 'mkdir -p "$HOME/evo-bench"'])
        checked_run(["scp", *options, str(elf), f"{args.target}:evo-bench/candidate.elf"])
        result = checked_run([
            "ssh", *options, args.target,
            'chmod u+x "$HOME/evo-bench/candidate.elf" && exec "$HOME/evo-bench/candidate.elf"',
        ])
    else:
        command = [toolchain.qemu_binary()]
        if toolchain.qemu_cpu:
            command.extend(["-cpu", toolchain.qemu_cpu])
        if toolchain.sysroot:
            command.extend(["-L", toolchain.sysroot])
        result = checked_run([*command, str(elf)])
        print("QEMU timing is for pipeline testing, not Banana Pi performance.")

    print(result.stdout, end="", flush=True)
    records = [line.removeprefix("EVO_BENCH_JSON ") for line in result.stdout.splitlines()
               if line.startswith("EVO_BENCH_JSON ")]
    if len(records) != 1 or "PASS" not in result.stdout.splitlines():
        raise RuntimeError("Missing benchmark JSON or PASS marker")
    report = json.loads(records[0])
    if (report.get("correct") is not True or report.get("size") != size
            or report.get("block_size") != args.block_size
            or not math.isfinite(report["median_ms"]) or report["median_ms"] <= 0):
        raise RuntimeError(f"Invalid benchmark result: {report}")
    report.update({"mode": args.mode, "target": args.target if args.mode == "ssh" else "qemu",
                   "warmup": args.warmup, "measurement": "warm_cache_serial_full_grid_ms",
                   "elf": str(elf)})
    output = directory / f"result-{args.mode}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Result saved: {output}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {exc.cmd}", file=sys.stderr)
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise SystemExit(1)
    except subprocess.TimeoutExpired as exc:
        print(f"Command timed out: {exc.cmd}", file=sys.stderr)
        raise SystemExit(1)
