#!/usr/bin/env python3
"""Does host bandwidth add up when several runs overlap?

Port counters answer "which link carried traffic" only if the link is visible as
a device with counters. On a 950 whose DRAM is behind UB ports it often is not,
so this script answers the question a different way: measure one process, then
measure N processes started at the same time and compare the totals.

    python benchmarks/kernel/host_link_scale.py --procs 2 --rows 1024 --iters 2000

Read the verdict at the end:

    sum ~= N x single   the host path scales, so a single run cannot reach the
                        ceiling; spreading the work over several engines (or two
                        registered windows inside one kernel) buys bandwidth
    sum ~= single       the host path or the shared adapter is already saturated
                        at one instance, and no amount of parallelism helps

AIV cores are the other way to accidentally serialize the launches: a kernel that
asks for every core leaves none for the second process. Pass ``--cores`` to split
them (e.g. 20 each on a 40-core device) when you want real overlap.

``numactl`` is used to pin each process to a different NUMA node when available,
because one port per node is the usual layout; ``--bind none`` disables it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scatter_copy_ub_bench.py")


def numa_nodes() -> list[int]:
    try:
        with open("/sys/devices/system/node/online") as handle:
            raw = handle.read().strip()
    except OSError:
        return []
    nodes: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            nodes.extend(range(int(lo), int(hi) + 1))
        else:
            nodes.append(int(part))
    return nodes


def bench_cmd(args, rows: int) -> list[str]:
    cmd = [
        sys.executable,
        args.bench,
        "--src",
        "host",
        "--rows",
        str(rows),
        "--iters",
        str(args.iters),
        "--rounds",
        str(args.rounds),
        "--warmup",
        str(args.warmup),
        "--json",
    ]
    if args.cores:
        cmd += ["--cores", str(args.cores)]
    return cmd


def wrap(cmd: list[str], node: int | None) -> list[str]:
    if node is None:
        return cmd
    numactl = shutil.which("numactl")
    if numactl is None:
        return cmd
    return [numactl, f"--cpunodebind={node}", f"--membind={node}", *cmd]


def parse_gbps(stdout: str) -> float | None:
    value = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = float(record.get("gbps", 0.0))
    return value


def run_sequential(cmd: list[str], node: int | None) -> float | None:
    out = subprocess.run(wrap(cmd, node), capture_output=True, text=True, check=False)
    if out.returncode != 0:
        print(out.stderr.strip()[-2000:], file=sys.stderr)
        return None
    return parse_gbps(out.stdout)


def run_concurrent(cmds: list[list[str]], nodes: list[int | None]) -> list[float | None]:
    procs = [
        subprocess.Popen(
            wrap(cmd, node), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for cmd, node in zip(cmds, nodes)
    ]
    results: list[float | None] = []
    for proc in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(stderr.strip()[-2000:], file=sys.stderr)
            results.append(None)
            continue
        results.append(parse_gbps(stdout))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--procs", type=int, default=2, help="concurrent processes (default: 2)")
    parser.add_argument("--rows", type=int, default=1024, help="rows per process")
    parser.add_argument("--iters", type=int, default=2000, help="launches per measurement")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--cores",
        type=int,
        default=0,
        help="AIV blocks per process (0 = all). With several processes, 0 lets the "
        "first one take every core and the rest queue -- pass a fraction instead",
    )
    parser.add_argument(
        "--bind",
        choices=("auto", "none"),
        default="auto",
        help="auto = one NUMA node per process through numactl (default)",
    )
    parser.add_argument("--bench", default=BENCH, help="benchmark script to drive")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.bench):
        parser.error(f"benchmark not found: {args.bench}")
    if args.procs > 1 and args.cores == 0:
        print(
            "warning: --cores 0 means every process asks for all AIV cores, so the "
            "launches will serialize and the total can never exceed one run.\n"
            "  pass e.g. --cores 20 for two processes on a 40-AIV device.",
            file=sys.stderr,
        )

    nodes = numa_nodes() if args.bind == "auto" else []
    if args.bind == "auto" and len(nodes) < args.procs:
        print(
            f"note: {len(nodes)} NUMA node(s) visible for {args.procs} process(es); "
            "binding only where possible",
            file=sys.stderr,
        )

    total_rows = args.rows * args.procs
    print(f"single: 1 process x {args.rows} rows")
    single = run_sequential(bench_cmd(args, args.rows), nodes[0] if nodes else None)
    if single is None:
        print("baseline run failed; nothing to compare", file=sys.stderr)
        return 1
    print(f"  {single:.1f} GB/s")

    # Keep the total payload comparable to the baseline so the per-run fixed costs
    # do not flatter the concurrent case.
    per_proc_rows = total_rows // args.procs
    print(f"\nconcurrent: {args.procs} processes x {per_proc_rows} rows")
    cmds = [bench_cmd(args, per_proc_rows) for _ in range(args.procs)]
    bindings: list[int | None] = [nodes[i % len(nodes)] if nodes else None for i in range(args.procs)]
    results = run_concurrent(cmds, bindings)
    if any(r is None for r in results):
        print("at least one concurrent run failed", file=sys.stderr)
        return 1

    print(f"{'proc':>5} {'node':>5} {'GB/s':>8}")
    print("-" * 20)
    for index, (value, node) in enumerate(zip(results, bindings)):
        print(f"{index:>5} {('-' if node is None else node):>5} {value:>8.1f}")
    total = sum(r for r in results if r is not None)
    print("-" * 20)
    print(f"{'sum':>5} {'':>5} {total:>8.1f}   (single run: {single:.1f} GB/s, x{total / single:.2f})")
    print()
    if total >= single * 1.5:
        print(
            "verdict: the host path SCALES. A single run cannot saturate it, so more "
            "concurrent engines (or two registered source windows inside one kernel) "
            "can use the bandwidth that is idle today."
        )
    else:
        print(
            "verdict: the host path does NOT scale past one instance. Either it is "
            "already saturated (single link / shared host adapter / a PCIe slot at "
            "line rate) or the extra processes never overlapped (check --cores)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
