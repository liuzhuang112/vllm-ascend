#!/usr/bin/env python3
"""Does host bandwidth add up when several runs overlap?

Port counters answer "which link carried traffic" only if the link shows up as a
device with counters. On a 950 whose DRAM sits behind UB ports it usually does
not, so this script answers the question a different way: measure one process,
then measure N processes started at the same time and compare the totals.

    python benchmarks/kernel/host_link_scale.py --procs 2 --rows 1024 --iters 2000

On a two-socket node the eight dies are split into two affinity groups and each
socket owns one host port (see ``npu_topology.py``). One process on one die can
therefore only reach one port, so the useful test puts a process on a die from
*each* group, bound to the NUMA node of its affinity range:

    python benchmarks/kernel/host_link_scale.py --procs 2 --devices 0,4 --cores 20

Read the verdict at the end:

    sum ~= N x single   the host path scales, so a single run cannot reach the
                        ceiling; spreading the work over several dies (or two
                        registered windows inside one kernel) buys bandwidth
    sum ~= single       the host path or the shared adapter is already saturated
                        at one instance, and no amount of parallelism helps

AIV cores are the other way to accidentally serialize the launches: a kernel that
asks for every core leaves none for the second process. Pass ``--cores`` to split
them (e.g. 20 each on a 40-core device) when you want real overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import npu_topology

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scatter_copy_ub_bench.py")


def bench_cmd(args, rows: int, device: int) -> list[str]:
    cmd = [
        sys.executable,
        args.bench,
        "--src",
        "host",
        "--device",
        str(device),
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


def parse_devices(text: str) -> list[int]:
    devices = [int(part) for part in text.replace(" ", ",").split(",") if part.strip()]
    if not devices:
        raise ValueError("empty --devices")
    return devices


def choose_devices(args, topo: list[dict]) -> list[int]:
    """One die per affinity group, because one group == one host port."""
    if args.devices:
        return parse_devices(args.devices)
    clusters = npu_topology.groups(topo)
    if not clusters:
        return list(range(args.procs))
    # One representative die from each cluster, in cluster order, then cycle.
    representatives = [min(cluster["devices"]) for cluster in clusters]
    devices = [representatives[i % len(representatives)] for i in range(args.procs)]
    if args.procs > len(representatives):
        print(
            f"note: only {len(representatives)} affinity group(s) but {args.procs} "
            "process(es); some processes will share a host port",
            file=sys.stderr,
        )
    return devices


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
        "--devices",
        default="",
        help="comma-separated NPU indices, one per process (default: one die per affinity "
        "group, which is one die per host port)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=0,
        help="AIV blocks per process (0 = all). With several processes, 0 lets the "
        "first one take every core and the rest queue -- pass a fraction instead",
    )
    parser.add_argument(
        "--bind",
        choices=("auto", "topo", "none"),
        default="auto",
        help="auto/topo = pin each process to the NUMA node of its die's affinity range "
        "(default); none = no numactl",
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

    topo_text = npu_topology.read_topo() if args.bind != "none" else None
    topo = npu_topology.parse_topo(topo_text) if topo_text else []
    if topo:
        print(npu_topology.summary(topo))
        print()

    devices = choose_devices(args, topo)
    node_of_device: dict[int, int | None] = {}
    for row in topo:
        node_of_device[row["device"]] = npu_topology.node_of(row["cpus"])
    if args.bind == "none":
        nodes: list[int | None] = [None] * len(devices)
    else:
        nodes = [node_of_device.get(device) for device in devices]
        if topo and all(node is None for node in nodes):
            print(
                "note: the topo affinity ranges do not map onto a visible NUMA node "
                "(single node?) so binding cannot steer a die to a second host port",
                file=sys.stderr,
            )
        elif not topo:
            print(
                "note: npu-smi topo unavailable, so processes are not bound to the NUMA "
                "node of their die; results may all land on one host port",
                file=sys.stderr,
            )

    single_node = nodes[0] if nodes else None
    print(f"single: 1 process x {args.rows} rows on NPU{devices[0]}"
          f"{'' if single_node is None else f' (numa node {single_node})'}")
    single = run_sequential(bench_cmd(args, args.rows, devices[0]), single_node)
    if single is None:
        print("baseline run failed; nothing to compare", file=sys.stderr)
        return 1
    print(f"  {single:.1f} GB/s")

    # Keep the total payload comparable to the baseline so the per-run fixed costs
    # do not flatter the concurrent case.
    total_rows = args.rows * args.procs
    per_proc_rows = total_rows // args.procs
    print(f"\nconcurrent: {args.procs} processes x {per_proc_rows} rows")
    cmds = [bench_cmd(args, per_proc_rows, device) for device in devices]
    results = run_concurrent(cmds, nodes)
    if any(r is None for r in results):
        print("at least one concurrent run failed", file=sys.stderr)
        return 1

    print(f"{'proc':>5} {'npu':>4} {'node':>5} {'GB/s':>8}")
    print("-" * 26)
    for index, (value, device, node) in enumerate(zip(results, devices, nodes)):
        print(f"{index:>5} {device:>4} {('-' if node is None else node):>5} {value:>8.1f}")
    total = sum(r for r in results if r is not None)
    print("-" * 26)
    print(f"{'sum':>5} {'':>4} {'':>5} {total:>8.1f}   (single run: {single:.1f} GB/s, x{total / single:.2f})")
    print()
    if total >= single * 1.5:
        print(
            "verdict: the host path SCALES. A single run cannot saturate it, so more "
            "concurrent engines (or two registered source windows inside one kernel) "
            "can use the bandwidth that is idle today."
        )
        print(
            "  next: if the two processes sit on two different dies, a single-die kernel "
            "cannot follow -- the win has to come from running more dies, not from "
            "restructuring one kernel's copy."
        )
    else:
        print(
            "verdict: the host path does NOT scale past one instance. Either it is "
            "already saturated (single link / shared host adapter) or the extra "
            "processes never overlapped (check --cores and that the two processes "
            "landed on dies from different affinity groups)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
