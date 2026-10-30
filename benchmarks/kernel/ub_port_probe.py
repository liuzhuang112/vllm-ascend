#!/usr/bin/env python3
"""Report which host links a device-side run actually put traffic on.

A host-to-device ``DataCopyPad`` reads host DRAM through the host link (UB ports
on ascend950, PCIe elsewhere), so "how many ports did it use" is a question about
counters, not about the kernel: sample every port before and after the run and
look at the deltas.

Usage::

    # what the link can do at all (vendor tool, best reference point)
    ascend-dmi -bw -d 0 --type h2d -q

    # which ports the benchmark used
    python benchmarks/kernel/ub_port_probe.py -- \\
        python benchmarks/kernel/scatter_copy_ub_bench.py --src host --iters 2000 --rounds 1

    # a restricted port range, and without the topology dump
    python benchmarks/kernel/ub_port_probe.py --ports 0-3 --no-topo -- <command>

Counters come from two places, and either one may be the right one on a given
box: ``hccn_tool -i N -stat -g`` for Ascend's own port devices, and
``/sys/class/net/<if>/statistics`` for ordinary netdevs. Ports that do not move
are shown as 0 so the answer to "how many" is visible at a glance.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

SKIP_IFACE_PREFIXES = ("lo", "docker", "veth", "br-", "virbr", "tun", "tap")
COUNTER_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*[:=]\s*(\d+)")
BYTE_COUNTERS = ("rx_bytes", "tx_bytes", "rx_byte", "tx_byte", "rx_packets", "tx_packets")


def parse_ports(spec: str) -> list[int]:
    ports: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ports.extend(range(int(lo), int(hi) + 1))
        else:
            ports.append(int(part))
    return ports


def netdev_counters() -> dict[str, int]:
    counters: dict[str, int] = {}
    root = "/sys/class/net"
    if not os.path.isdir(root):
        return counters
    for iface in sorted(os.listdir(root)):
        if iface.startswith(SKIP_IFACE_PREFIXES):
            continue
        for name in ("rx_bytes", "tx_bytes"):
            path = os.path.join(root, iface, "statistics", name)
            try:
                with open(path) as handle:
                    counters[f"net:{iface}:{name}"] = int(handle.read().strip())
            except (OSError, ValueError):
                continue
    return counters


def hccn_counters(ports: list[int]) -> dict[str, int]:
    tool = shutil.which("hccn_tool")
    counters: dict[str, int] = {}
    if tool is None:
        return counters
    for port in ports:
        try:
            out = subprocess.run(
                [tool, "-i", str(port), "-stat", "-g"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode != 0:
            continue
        for line in out.stdout.splitlines():
            for name, value in COUNTER_RE.findall(line):
                if any(name.lower().startswith(k) for k in BYTE_COUNTERS):
                    counters[f"hccn{port}:{name}"] = int(value)
    return counters


def sample(ports: list[int]) -> dict[str, int]:
    counters = netdev_counters()
    counters.update(hccn_counters(ports))
    return counters


def device_of(key: str) -> str:
    # "net:eth0:rx_bytes" -> "net:eth0", "hccn0:rx_bytes" -> "hccn0": the port
    # identity is everything except the trailing counter name.
    return key.rsplit(":", 1)[0]


def direction_of(key: str) -> str:
    return "rx" if ":rx" in key else "tx"


def summarize(before: dict[str, int], after: dict[str, int], seconds: float) -> list[tuple[str, str, int, int, float]]:
    """Per device: (device, direction, delta bytes, delta packets, GB/s)."""
    rows: list[tuple[str, str, int, int, float]] = []
    devices = sorted({device_of(k) for k in after} | {device_of(k) for k in before})
    for device in devices:
        for direction in ("rx", "tx"):
            bytes_delta = 0
            packets_delta = 0
            for key, value in after.items():
                if device_of(key) != device or direction_of(key) != direction:
                    continue
                # Only the byte/packet counters that actually exist; pick the
                # most specific one so rx_bytes is not double counted with
                # rx_packets.
                if not key.endswith(("bytes", "byte", "packets")):
                    continue
                delta = value - before.get(key, value)
                if key.endswith(("bytes", "byte")):
                    bytes_delta = max(bytes_delta, delta)
                else:
                    packets_delta = max(packets_delta, delta)
            rate = bytes_delta / seconds / 1e9 if seconds > 0 else 0.0
            rows.append((device, direction, bytes_delta, packets_delta, rate))
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows


def topo_dump() -> None:
    tool = shutil.which("npu-smi")
    if tool is None:
        return
    try:
        out = subprocess.run(
            [tool, "info", "-t", "topo"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return
    if out.returncode != 0 or not out.stdout.strip():
        return
    print("topology (npu-smi info -t topo):")
    for line in out.stdout.strip().splitlines()[:40]:
        print(f"  {line}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ports",
        default="0-7",
        help="hccn_tool port ids to sample, e.g. 0-7 or 0,2,4 (default: 0-7)",
    )
    parser.add_argument(
        "--no-topo", action="store_true", help="skip the npu-smi topology dump"
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.2,
        help="seconds to wait after the command before the second sample",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given; pass it after --")

    ports = parse_ports(args.ports)
    if not args.no_topo:
        topo_dump()

    before = sample(ports)
    if not before:
        print(
            "no counters found: neither hccn_tool nor /sys/class/net is readable.\n"
            "  install/point at hccn_tool, or run as a user that can read "
            "/sys/class/net/*/statistics",
            file=sys.stderr,
        )
        return 2
    print(f"sampled {len(before)} counters over {len({device_of(k) for k in before})} devices")

    started = time.monotonic()
    code = subprocess.call(command)
    elapsed = time.monotonic() - started
    if args.settle > 0:
        time.sleep(args.settle)
    after = sample(ports)

    rows = summarize(before, after, elapsed)
    header = f"{'device':>12} {'dir':>4} {'GB':>10} {'MB/s':>9} {'packets':>12}"
    print()
    print(header)
    print("-" * len(header))
    used = 0
    for device, direction, delta, packets, rate in rows:
        if delta == 0:
            print(f"{device:>12} {direction:>4} {0.0:>10.3f} {0.0:>9.1f} {packets:>12}")
            continue
        if direction == "rx":
            used += 1
        print(
            f"{device:>12} {direction:>4} {delta / 1e9:>10.3f} "
            f"{delta / elapsed / 1e6:>9.1f} {packets:>12}"
        )

    rx_total = sum(delta for _d, direction, delta, _p, _r in rows if direction == "rx")
    print()
    print(
        f"{used} device(s) carried receive traffic, "
        f"{rx_total / elapsed / 1e9:.1f} GB/s in total over {elapsed:.2f}s"
    )
    print(
        "  one device carrying all of it -> the run is single-ported; "
        "two roughly equal -> both ports are in use"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
