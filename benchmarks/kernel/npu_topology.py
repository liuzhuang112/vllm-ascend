#!/usr/bin/env python3
"""Read the NPU interconnect topology and work out which host link each die has.

``npu-smi info -t topo`` prints one row per die::

            NPU0   NPU1  ...  NPU7    CPU    Affinity
    NPU0      X     UB   ...   UB     UB     288-383
    NPU1     UB      X   ...   UB     UB     288-383
    ...
    NPU4     UB     UB   ...    X     UB      96-191

Two columns matter for host bandwidth:

``CPU``
    how the die reaches host DRAM -- ``UB`` here, ``PCIe`` on an A2/A3 box.
``Affinity``
    the CPU cores the die is affine to. When the dies fall into two disjoint
    ranges, the node is two sockets, and each socket normally owns one host
    port. A process talking to a die in the first group only reaches that
    group's port.

The helper turns that into ``[{device, affinity, cpus, node, cpu_link}]``, where
``node`` is the NUMA node holding the affinity range (from
``/sys/devices/system/node/node*/cpulist``), which is what ``numactl`` needs.
"""

from __future__ import annotations

import re
import shutil
import subprocess

TOPO_RE = re.compile(r"^NPU(\d+)\b")
# The header line of the same table starts with 'NPU0' too, so a data row is only
# a row when its last column really is a CPU list ('288-383', '0-31,64-95').
AFFINITY_RE = re.compile(r"^[0-9][0-9,\-]*$")


def expand_cpu_list(text: str) -> set[int]:
    """'' -> {}; '0-3,8,10-11' -> {0,1,2,3,8,10,11}."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            if lo.strip().isdigit() and hi.strip().isdigit():
                cpus.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cpus.add(int(part))
    return cpus


def read_topo(timeout: int = 30) -> str | None:
    tool = shutil.which("npu-smi")
    if tool is None:
        return None
    try:
        out = subprocess.run(
            [tool, "info", "-t", "topo"], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout


def parse_topo(text: str) -> list[dict]:
    """Rows of the topo table -> [{device, affinity, cpus, cpu_link}]."""
    rows: list[dict] = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 3:
            continue
        match = TOPO_RE.match(tokens[0])
        if match is None:
            continue
        affinity = tokens[-1]
        if AFFINITY_RE.match(affinity) is None:
            # Header row ('... CPU Affinity'), or a line this parser does not
            # understand yet -- skip rather than invent a die with no cores.
            continue
        cpu_link = tokens[-2]
        rows.append(
            {
                "device": int(match.group(1)),
                "affinity": affinity,
                "cpus": expand_cpu_list(affinity),
                "cpu_link": cpu_link,
            }
        )
    return rows


def numa_nodes() -> dict[int, set[int]]:
    """{node index: cpus} from sysfs; empty when the kernel exposes none."""
    import glob
    import os

    nodes: dict[int, set[int]] = {}
    for path in glob.glob("/sys/devices/system/node/node*/cpulist"):
        try:
            index = int(os.path.basename(os.path.dirname(path))[4:])
        except ValueError:
            continue
        try:
            with open(path) as handle:
                nodes[index] = expand_cpu_list(handle.read())
        except OSError:
            continue
    return nodes


def node_of(cpus: set[int], nodes: dict[int, set[int]] | None = None) -> int | None:
    """The NUMA node that covers most of `cpus`, or None when unknown."""
    if not cpus:
        return None
    nodes = numa_nodes() if nodes is None else nodes
    best, best_hits = None, 0
    for index, node_cpus in nodes.items():
        hits = len(cpus & node_cpus)
        if hits > best_hits:
            best, best_hits = index, hits
    return best


def groups(rows: list[dict]) -> list[dict]:
    """Cluster dies by affinity range -- one cluster per socket / host port."""
    buckets: dict[str, list[int]] = {}
    for row in rows:
        buckets.setdefault(row["affinity"], []).append(row["device"])
    nodes = numa_nodes()
    out: list[dict] = []
    for affinity, devices in sorted(buckets.items(), key=lambda kv: min(kv[1])):
        cpus = expand_cpu_list(affinity)
        out.append(
            {
                "affinity": affinity,
                "cpus": cpus,
                "node": node_of(cpus, nodes),
                "devices": sorted(devices),
            }
        )
    return out


def summary(rows: list[dict]) -> str:
    """Human-readable table: which host port each die can use."""
    if not rows:
        return "npu-smi info -t topo is unavailable; cannot tell the dies apart"
    lines = [f"{'die':>4} {'cpu link':>9} {'affinity':>14} {'numa node':>10}  host port"]
    lines.append("-" * len(lines[0]))
    clustered = groups(rows)
    port_of_device: dict[int, int] = {}
    for port, group in enumerate(clustered):
        for device in group["devices"]:
            port_of_device[device] = port
    for row in rows:
        node = node_of(row["cpus"])
        lines.append(
            f"{row['device']:>4} {row['cpu_link']:>9} {row['affinity']:>14} "
            f"{('-' if node is None else node):>10}  #{port_of_device[row['device']]}"
        )
    lines.append("")
    if len(clustered) > 1:
        lines.append(
            f"{len(clustered)} affinity groups -> the node most likely has {len(clustered)} "
            "host ports, one per group:"
        )
        for port, group in enumerate(clustered):
            node = "-" if group["node"] is None else group["node"]
            lines.append(
                f"  port #{port}: dies {group['devices']} (affinity {group['affinity']}, "
                f"numa node {node})"
            )
        lines.append(
            "  a single process on one die can only reach its own group's port; use dies "
            "from two different groups to load two ports at once"
        )
    else:
        lines.append("one affinity group -> one host port for all dies; no second port to use")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dump", action="store_true", help="also print the raw npu-smi table")
    args = parser.parse_args(argv)
    text = read_topo()
    if text is None:
        print("npu-smi info -t topo is unavailable on this machine", flush=True)
        return 1
    if args.dump:
        print(text)
    print(summary(parse_topo(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
