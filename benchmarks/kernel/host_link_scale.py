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

    python benchmarks/kernel/host_link_scale.py --procs 2 --devices 0,4

Two runs are made and compared against the single-die baseline:

    across groups   dies from two sockets -> uses two host ports
    same group      two dies on one socket -> one port, two DMA engines

That second run is what separates the two possible ceilings:

    same group ~= 2 x single   the port can feed more than one die, so the
                               per-die DMA engine is the limit; a single die
                               cannot go faster, but more dies on that socket can
    same group ~= single       the port itself is capped; only a second socket
                               (or a second registered window onto its DRAM)
                               adds bandwidth
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


def measure(devices: list[int], nodes: list[int | None], args, rows: int) -> list[float] | None:
    cmds = [bench_cmd(args, rows, device) for device in devices]
    results = run_concurrent(cmds, nodes)
    if any(value is None for value in results):
        return None
    return [float(value) for value in results]  # type: ignore[arg-type]


def print_run(title: str, devices: list[int], nodes: list[int | None], values: list[float]) -> None:
    print(f"{title}")
    print(f"{'proc':>5} {'npu':>4} {'node':>5} {'GB/s':>8}")
    print("-" * 26)
    for index, (value, device, node) in enumerate(zip(values, devices, nodes)):
        print(f"{index:>5} {device:>4} {('-' if node is None else node):>5} {value:>8.1f}")
    print("-" * 26)
    print(f"{'sum':>5} {'':>4} {'':>5} {sum(values):>8.1f}")
    print()


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
        help="AIV blocks per process (0 = all). Only processes that share a die compete "
        "for cores, so 0 is fine when each process has its own die",
    )
    parser.add_argument(
        "--bind",
        choices=("auto", "topo", "none"),
        default="auto",
        help="auto/topo = pin each process to the NUMA node of its die's affinity range "
        "(default); none = no numactl",
    )
    parser.add_argument(
        "--same-group",
        dest="same_group",
        action="store_true",
        default=True,
        help="also measure two dies on ONE socket, which tells a per-port ceiling from a "
        "per-die one (default: on)",
    )
    parser.add_argument("--no-same-group", dest="same_group", action="store_false")
    parser.add_argument("--bench", default=BENCH, help="benchmark script to drive")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.bench):
        parser.error(f"benchmark not found: {args.bench}")

    topo_text = npu_topology.read_topo() if args.bind != "none" else None
    topo = npu_topology.parse_topo(topo_text) if topo_text else []
    clusters = npu_topology.groups(topo) if topo else []
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

    if len(set(devices)) < len(devices):
        print(
            "warning: the same die appears twice in --devices; two processes on one die "
            "contend for AIV cores and the second one mostly queues",
            file=sys.stderr,
        )

    single_node = nodes[0] if nodes else None
    where = "" if single_node is None else f" (numa node {single_node})"
    print(f"baseline: 1 process, NPU{devices[0]}, {args.rows} rows{where}")
    single = run_sequential(bench_cmd(args, args.rows, devices[0]), single_node)
    if single is None:
        print("baseline run failed; nothing to compare", file=sys.stderr)
        return 1
    print(f"  {single:.1f} GB/s\n")

    # Keep the total payload comparable to the baseline so the per-run fixed costs
    # do not flatter the concurrent case.
    total_rows = args.rows * args.procs
    per_proc_rows = total_rows // args.procs

    across = measure(devices, nodes, args, per_proc_rows)
    if across is None:
        print("a concurrent run failed (across affinity groups)", file=sys.stderr)
        return 1
    print_run(
        f"across groups: {args.procs} processes x {per_proc_rows} rows on dies {devices}",
        devices,
        nodes,
        across,
    )
    across_total = sum(across)

    same_total = None
    same_devices: list[int] = []
    if args.same_group and clusters and len(devices) >= 2:
        # Two dies from the same affinity group share one host port, so they answer
        # "can the port feed more than one die" independently of the socket layout.
        group = next((c for c in clusters if devices[0] in c["devices"]), None)
        if group is not None and len(group["devices"]) >= 2:
            same_devices = group["devices"][: len(devices)]
            same_nodes = [node_of_device.get(device) for device in same_devices]
            if same_devices != devices:
                same = measure(same_devices, same_nodes, args, per_proc_rows)
                if same is None:
                    print("a concurrent run failed (same affinity group)", file=sys.stderr)
                else:
                    print_run(
                        f"same group: {len(same_devices)} processes x {per_proc_rows} rows "
                        f"on dies {same_devices} (one socket)",
                        same_devices,
                        same_nodes,
                        same,
                    )
                    same_total = sum(same)

    print("summary")
    print(f"  single die                        {single:>7.1f} GB/s")
    print(
        f"  {len(devices)} dies across groups        "
        f"{across_total:>7.1f} GB/s  (x{across_total / single:.2f})"
    )
    if same_total is not None:
        print(
            f"  {len(same_devices)} dies in one group        "
            f"{same_total:>7.1f} GB/s  (x{same_total / single:.2f})"
        )
    print()

    if across_total >= single * 1.5:
        print(
            "verdict: the host path SCALES across sockets -- each socket's port is usable "
            "on its own, so a single die is limited to roughly one port's worth."
        )
    else:
        print(
            "verdict: two sockets do NOT add up; the two dies share one port or one "
            "adapter, so spreading work across dies buys nothing."
        )

    if same_total is not None:
        if same_total >= single * 1.5:
            print(
                "  one socket's port still feeds several dies at full rate: the ceiling "
                "you hit on a single die is the per-die DMA engine, not the port. More "
                "dies on that socket keep scaling; one die cannot go faster."
            )
        else:
            print(
                "  one socket's port saturates at a single die's rate: the port itself is "
                "the ceiling. Only a second socket (or a second registered window onto "
                "that socket's DRAM, if the die can address it) adds bandwidth."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
