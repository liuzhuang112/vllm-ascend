#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRAM -> HBM copy microbenchmark.

Measures only the copy stage of
``npu_fused_scatter_copy_sparse_flash_attention``:

    DRAM src --MTE2--> UB bank --MTE3--> HBM dst

One token row is `--ckv-dim` elements of the CKV plane plus `--kpe-dim`
elements of the KPE plane, split across two source/destination tensors exactly
like the paged K/V cache layout, so the per-token cost is two DataCopyPad calls
per direction.

Defaults reproduce the bf16 MLA cache: CKV 512 + KPE 64 = 1152 B per token.
Two other shapes matter in practice and are one flag away:

    # DeepSeek-style quantised MLA cache: one 656 B row, no separate rope plane.
    # The row is copied byte-wide, so uint8 is the dtype to use -- the op never
    # interprets the payload and this way it needs no fp8 support from the part.
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 \
        --ckv-dim 656 --kpe-dim 0 --dtype uint8

    # where does the per-row DMA slow down? sweep the row length instead of
    # guessing: 512 and 1024 are 128 B multiples, 656/672 are not (see below)
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 \
        --kpe-dim 0 --dtype uint8 --src host --sweep ckv-dim

Source placement decides what is being measured:

    --src npu    source lives in HBM       -> HBM to HBM ceiling
    --src host   source is host memory registered with aclrtHostRegisterV2
                 and published by aclrtHostGetDevicePointer
                                               -> real DRAM to HBM transfer

Examples:
    # production configuration: 2 banks, one token per DMA, table addressing
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 --bank 2 --group-rows 1

    # how much does coalescing rows into one DMA buy?
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 --sweep group-rows

    # how much do the scalar row-table reads cost?
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 --sweep table-mode

    # DRAM (host pinned) -> HBM
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 --src host

    # pick the card: which die you land on decides which host link is used
    python benchmarks/kernel/scatter_copy_ub_bench.py --list-devices
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 --src host --device 3

    # real cache shape: 131072 rows in DDR, only 4096 of them copied per launch
    # --rotate keeps the page tables cold, i.e. different rows every launch
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --src-rows 131072 \
        --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src host --rotate 16

    # is the host link the limiter, or the DMA engine? Same rows, both sources:
    # 656 B measures 33 GB/s over the host link and 91 GB/s from HBM, so the
    # 256 B billing is the link's, not the engine's
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --ckv-dim 656 \
        --kpe-dim 0 --dtype uint8 --sweep src

    # a launch costs 3.3 us before any row moves: this slope is the per-row cost
    # and the intercept is that constant, which is why small --rows reads low
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --ckv-dim 656 \
        --kpe-dim 0 --dtype uint8 --src host --sweep rows

    # the production shape: 78 DDR partitions of 131072 x 32 rows, 196608 rows
    # copied out of every one per pass (topk 2048 x mtp 3 x 32 requests), 78
    # launches per pass over 200 GiB of DDR. --dry-run first: it prints the
    # allocation plan and exits without taking the memory.
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --spaces 78 \
        --space-rows '131072*32' --ckv-dim 656 --kpe-dim 0 --dtype uint8 \
        --src host --dry-run

    # same shape at the four row lengths the cache uses (the pool is reallocated
    # per point, so this is four sequential 78-pass setups)
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --spaces 78 \
        --space-rows '131072*32' --kpe-dim 0 --dtype uint8 --src host \
        --sweep row-bytes

    # does polling more partitions cost anything? 1/2/4/8/16/32/64/78 spaces with
    # the same per-space pool: total DDR grows with the count, the payload per
    # launch does not. --sweep grid is the row-length x space-count matrix in one
    # run, and --xlsx puts it in a workbook (one sheet per dimension)
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --space-rows \
        '131072*32' --kpe-dim 0 --dtype uint8 --src host --sweep spaces
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --space-rows \
        '131072*32' --kpe-dim 0 --dtype uint8 --src host --sweep grid \
        --xlsx bench_spaces.xlsx

Two separate products must exist before the first run, from the repository
root::

    # 1. the aclnn kernel package (what the op actually executes)
    SOC_VERSION=ascend950 bash csrc/build_aclnn.sh
    # 2. the torch binding, i.e. torch.ops._C_ascend.scatter_copy_ub_bench
    SOC_VERSION=ascend950 pip install -e . --no-build-isolation

Both can be produced in one go; ``VLLM_ASCEND_CUSTOM_OPS`` keeps the kernel step
cheap::

    SOC_VERSION=ascend950 VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench \
        pip install -e . --no-build-isolation

Do NOT pass ``COMPILE_CUSTOM_KERNELS=0`` here: setup.py:422 empties
``ext_modules`` for it, so pip then compiles neither the kernel package nor
``vllm_ascend_C`` -- it is the "install the python part only" switch, not a
"skip the kernels" one.

On SoCs whose hardware profile does not declare ``RUNTIME_CUSTOM_OPS``
(ascend950 / A5 today) ``vllm_ascend.utils.enable_custom_op()`` deliberately
skips importing the extension, so ``torch.ops._C_ascend`` stays empty even
after a correct build. This script imports the extension by hand in that case;
a still-missing op then really does mean a stale ``vllm_ascend_C``.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from functools import cache

import torch

torch_npu = None


CKV_DIM = 512
KPE_DIM = 64
OP_NAME = "scatter_copy_ub_bench"
# Element size of each supported cache dtype, in bytes. A 1-byte cache row (a
# quantised MLA row, for instance) is measured through uint8: the op moves bytes
# and never interprets them, so no fp8 dtype has to exist on the part.
DTYPES = {"bf16": (torch.bfloat16, 2), "fp16": (torch.float16, 2), "uint8": (torch.uint8, 1)}
# Above this many elements a source pool is allocated but only the rows a table
# points at are given payload. The copy never reads the rest, and the verifier
# reads the same host bytes, so filling 205 GiB of DDR would cost minutes per
# space and buy nothing. Below it, filling everything keeps the mismatch report
# able to say "the destination holds source row N instead of row M".
FULL_FILL_ELEMS = 32 << 20
# ``torch.pin_memory()`` only locks pages; it does not publish the range to the
# device. ``aclrtHostRegisterV2`` does, and the payload is then aligned to this
# boundary -- engram/npu.py aligns to 64 KiB because unaligned device reads of a
# host range were observed to return wrong data on A3 / CANN 9.2.
HOST_ALIGN_BYTES = 64 * 1024
# One allocation per this many bytes while imitating a sweep's memory history
# (--churn-gib). 1 GiB keeps the number of live allocations small enough that the
# pinned allocator itself does not become the thing being measured.
CHURN_CHUNK_BYTES = 1 << 30
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000
# torch_npu's pinned allocator only maps its pages into the device address space
# when this option is on; without it aclrtHostGetDevicePointer fails and
# get_npu_view_from_cpu_tensor returns None. tests/.../test_uva_view.py gates
# itself on exactly this string. The option is parsed while torch_npu initialises,
# so it has to be set before the import.
ALLOC_CONF_ENV = "PYTORCH_NPU_ALLOC_CONF"
PINNED_REGISTER_OPT = "pinned_mem_register:True"
_alloc_conf_added = False
# Set by the re-exec below so the restarted process does not restart again.
INTERLEAVE_ENV = "SCATTER_COPY_UB_INTERLEAVED"
# CANN's own card filter. It is read before torch_npu initialises and renumbers
# the cards, which is why --device cannot validate against physical ids.
DEVICE_ENV = "ASCEND_RT_VISIBLE_DEVICES"
REBUILD_BINDING = (
    "  build the torch binding; the kernel package alone does not register it:\n"
    "    SOC_VERSION=<soc> VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench \\\n"
    "        pip install -e . --no-build-isolation\n"
    "  (COMPILE_CUSTOM_KERNELS=0 would build nothing: setup.py empties "
    "ext_modules for it.)\n"
    "  If vllm_ascend_C*.so already exists, it is stale -- remove it first:\n"
    "    find . -name 'vllm_ascend_C*.so' -delete"
)


def _op_registered() -> bool:
    return hasattr(torch.ops._C_ascend, OP_NAME)


# --- what one row's DMA costs -------------------------------------------------
# Empirical model of the host path (DDR -> UB -> HBM, one A5 die), fitted to three
# sweeps. Measured ns/row, and what the model says after the launch constant is
# taken out (the launch cost is 3.3 us per launch, i.e. 1.6 ns/row at 2048 rows):
#
#   len B   meas ns/row  U   tail   pure ns/row   ns per 256 B   (--sweep ckv-dim, 2048 rows)
#     384      12.4       2   128      10.7          5.35
#     512      12.4       2     0      10.7          5.35
#     576      17.7       3    64      16.0          5.34
#     640      17.7       3   128      16.0          5.34
#     656      20.5       3   144      18.9          5.34  + 2.9 tail
#     672      20.6       3   160      18.9          5.35  + 2.9 tail
#     704      20.6       3   192      18.9          5.35  + 2.9 tail
#     768      18.4       3     0      16.8          5.59
#     896      23.1       4   128      21.4          5.35
#    1024      23.1       4     0      21.4          5.34
#
#   the same 656 B rows merged k at a time (--table-mode 0 --sweep group-rows, 2048 rows):
#
#   group_rows=1  3 granules    20.1 ns/row    32.7 GB/s
#   group_rows=2  6 granules    34.7 ns/DMA    37.8 GB/s   (17.4 ns/row)
#   group_rows=4 11 granules    63.7 ns/DMA    41.2 GB/s   (15.9 ns/row)
#   group_rows=8 21 granules   122.3 ns/DMA    42.9 GB/s   (15.3 ns/row)
#
#   and rows per launch at 656 B (--sweep rows, same row, host):
#
#   1024 rows 29.8 | 2048 31.9 | 4096 33.2 | 8192 33.9 | 16384 34.3 GB/s
#   -> t = 3.33 us + 18.93 ns * rows, i.e. a fixed 3.3 us per launch and 18.9 ns/row
#
# Least squares over all eighteen points:
#
#   t = 3300 ns + rows * (5.35 ns * ceil(len/256) + 2.9 ns * (len % 256 > 128))
#
# max |error| 4% (the 768 B point), most points within 1%. Two things follow, and
# both were mistakes in an earlier reading of the same data:
#
#   * the ~0.4 ns/row "constant" of the first fit was the launch cost in disguise.
#     It belongs to the launch, so a small --rows always reads low: 656 B is 31.9
#     GB/s at 2048 rows and 34.3 at 16384 for exactly the same per-row work.
#   * the per-row asymptote for 656 B is 34.6 GB/s, not 32 -- 32 was 2048 rows of
#     a number that has a 3.3 us floor in it.
#
# Scope: every number above is --src host. With --src npu the same 656 B row runs
# at 90.9 GB/s (2.6x), so the 256 B billing below is a property of the host link,
# not of the DMA engine. That is why the column is called %host, not %model.
# It is an empirical model of one machine, not a documented rule -- which is why it
# is only ever printed as a hint.
DMA_GRANULE_BYTES = 256
DMA_NS_PER_GRANULE = 5.35
DMA_TAIL_SURCHARGE_NS = 2.9
LAUNCH_NS = 3300.0


def dma_ns(nbytes: int) -> tuple[float, int, int]:
    """(pure ns, charged bytes, tail bytes) for one DMA of `nbytes`, no launch cost."""
    units = -(-nbytes // DMA_GRANULE_BYTES) if nbytes > 0 else 0
    tail = nbytes % DMA_GRANULE_BYTES
    ns = units * DMA_NS_PER_GRANULE
    if tail > DMA_GRANULE_BYTES // 2:
        ns += DMA_TAIL_SURCHARGE_NS
    return ns, units * DMA_GRANULE_BYTES, tail


def predict_gbps(row_bytes: int, group_rows: int = 1, rows: int | None = None) -> float:
    """Model throughput of moving `group_rows` rows in one DMA.

    With `rows` the launch constant is amortised over that many tokens, which is
    the only way to compare against a measurement (a run never has infinite rows).
    Without it the answer is the asymptote (rows -> infinity).
    """
    ns, _, _ = dma_ns(row_bytes * group_rows)
    if rows:
        # One launch covers rows/group_rows DMAs, so this many ns land on each DMA
        # (per row it is LAUNCH_NS / rows either way).
        ns += LAUNCH_NS * group_rows / rows
    return float("inf") if ns <= 0 else row_bytes * group_rows / ns


def merge_curve(
    row_bytes: int, ks: tuple[int, ...] = (1, 2, 3, 4, 6, 8), rows: int | None = None
) -> str:
    """The predicted GB/s of merging k rows, for k in `ks`."""
    return " | ".join(f"k={k} {predict_gbps(row_bytes, k, rows):.0f}" for k in ks)


def dma_cost_hint(row_bytes: int, group_rows: int = 1, rows: int | None = None) -> str:
    """What one row's DMA is charged for, and what merging rows would buy."""
    ns, charged, tail = dma_ns(row_bytes)
    units = charged // DMA_GRANULE_BYTES
    text = f"DMA: {row_bytes} B/row is charged as {charged} B ({units} x 256 B"
    if tail > DMA_GRANULE_BYTES // 2:
        text += f", and the {tail} B tail (> 128 B) adds ~{DMA_TAIL_SURCHARGE_NS:.1f} ns"
    text += f") -> ~{ns:.1f} ns/row (~{row_bytes / ns:.0f} GB/s useful)"
    if charged > row_bytes:
        text += f"; the other {charged - row_bytes} B of the granule are free"
    curve = merge_curve(row_bytes, rows=rows)
    ceiling = DMA_GRANULE_BYTES / DMA_NS_PER_GRANULE  # B/ns == GB/s
    lines = [
        text,
        f"     model t = {LAUNCH_NS / 1000:.1f} us per launch + rows * ({DMA_NS_PER_GRANULE:.2f} ns "
        f"per 256 B + {DMA_TAIL_SURCHARGE_NS:.1f} ns if tail > 128 B); {ceiling:.1f} GB/s per granule",
        f"     merging rows (one DMA per group, at {rows or 'infinite'} rows): {curve} GB/s"
        + ("" if group_rows == 1 else f"  <- now running k={group_rows}, model says "
                                      f"{predict_gbps(row_bytes, group_rows, rows):.0f}"),
        f"     where this row can go: "
        f"{max(predict_gbps(row_bytes, k, rows) for k in range(1, 9)):.0f} GB/s "
        f"at k=8 (needs 8 adjacent rows on both sides)",
    ]
    return "\n".join(lines)


def _custom_op_gate() -> str | None:
    """Why ``enable_custom_op()`` refuses to load the extension, if it does.

    ``vllm_ascend.utils.enable_custom_op()`` returns early -- without importing
    ``vllm_ascend_C`` -- whenever ``VLLM_BATCH_INVARIANT`` is set or the current
    hardware profile does not declare ``RUNTIME_CUSTOM_OPS`` (ascend950 / A5
    today, where the production custom ops are only partially usable). In that
    case no rebuild can make ``torch.ops._C_ascend.<op>`` appear: the module is
    simply never imported. A standalone microbenchmark only needs the torch
    library entry points, so it imports the extension itself; this helper only
    explains why the normal path was skipped.
    """
    try:
        import vllm.envs as envs
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability,
            get_current_hardware_profile,
        )
    except Exception:  # optional diagnostics only
        return None
    if envs.VLLM_BATCH_INVARIANT:
        return "VLLM_BATCH_INVARIANT is set, so enable_custom_op() skips vllm_ascend_C"
    if get_current_hardware_profile().supports(HardwareCapability.RUNTIME_CUSTOM_OPS):
        return None
    return (
        "this SoC's hardware profile does not declare RUNTIME_CUSTOM_OPS "
        "(ascend950/A5), so enable_custom_op() skips vllm_ascend_C"
    )


def _import_extension() -> None:
    """Import ``vllm_ascend_C`` directly, bypassing the enable_custom_op() gate.

    Importing the module is what runs its ``TORCH_LIBRARY`` registrations, so
    this makes every ``torch.ops._C_ascend.*`` entry point available. The same
    manual import is used by ``vllm_ascend/platform.py`` and
    ``vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py``.
    """
    import importlib.util

    from vllm_ascend import utils as ascend_utils

    # enable_custom_op() normally publishes the op package path before the
    # import; it may have bailed out earlier, so do it here as well.
    try:
        ascend_utils.bootstrap_custom_op_env(include_vendor_lib=True)
    except Exception:  # noqa: BLE001 - env plumbing must not mask the import
        pass

    if importlib.util.find_spec("vllm_ascend.vllm_ascend_C") is None:
        raise SystemExit(
            "vllm_ascend.vllm_ascend_C is not built, so the torch binding cannot "
            "be registered at all.\n" + REBUILD_BINDING
        )
    try:
        import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"vllm_ascend.vllm_ascend_C exists but cannot be imported: {exc}\n"
            "  its shared-library dependencies do not resolve; check "
            "ASCEND_HOME_PATH and LD_LIBRARY_PATH, then\n" + REBUILD_BINDING
        ) from exc


def _warn_if_kernel_package_missing() -> None:
    """The binding is only half of it: the aclnn kernel must be loadable too."""
    import vllm_ascend

    vendor = os.path.join(
        os.path.dirname(os.path.abspath(vllm_ascend.__file__)),
        "_cann_ops_custom",
        "vendors",
        "custom_transformer",
    )
    if not os.path.isdir(vendor):
        print(
            f"warning: kernel package missing at {vendor}; the op will fail at "
            "launch time.\n"
            "  build it first: SOC_VERSION=<soc> bash csrc/build_aclnn.sh",
            file=sys.stderr,
        )


def _request_mapped_pinned_memory() -> None:
    """Turn on ``pinned_mem_register:True`` before torch_npu is imported.

    ``torch.pin_memory()`` only locks pages; ``get_npu_view_from_cpu_tensor``
    needs the range to be mapped for the device, which torch_npu does only when
    ``PYTORCH_NPU_ALLOC_CONF`` carries ``pinned_mem_register:True``. Without it
    the view is ``None`` no matter how the tensor was allocated.
    """
    global _alloc_conf_added

    conf = os.environ.get(ALLOC_CONF_ENV, "").strip()
    if PINNED_REGISTER_OPT in conf:
        return
    os.environ[ALLOC_CONF_ENV] = f"{conf},{PINNED_REGISTER_OPT}" if conf else PINNED_REGISTER_OPT
    _alloc_conf_added = True


def _reexec_interleaved(argv: list[str]) -> None:
    """Restart the script under ``numactl --interleave=all``.

    A host buffer whose pages all come from one NUMA node is reachable through
    one host link, which is what a 950 with two UB ports shows up as: the run
    saturates a single port no matter how many AIV cores issue the DMA. Spreading
    the pages over every node is the cheapest way to put half of the source
    behind each port, and it has to happen before the allocation, hence the
    re-exec rather than a call in ``Bench``.

    Verify the effect with ``benchmarks/kernel/ub_port_probe.py``; if the pages
    were already interleaved, or the allocator ignores the task mempolicy, the
    probe will still show one port and the split has to be done in the kernel
    instead (two registered windows, chosen per row).
    """
    if os.environ.get(INTERLEAVE_ENV) == "1":
        return
    numactl = shutil.which("numactl")
    if numactl is None:
        print(
            "warning: --host-interleave needs numactl, which is not installed; "
            "host pages stay on a single NUMA node and only one host link will "
            "be used",
            file=sys.stderr,
        )
        return
    os.environ[INTERLEAVE_ENV] = "1"
    print(f"note: restarting under {numactl} --interleave=all", file=sys.stderr)
    os.execv(
        numactl,
        [numactl, "--interleave=all", sys.executable, os.path.abspath(__file__), *argv],
    )


def _load_runtime():
    global torch_npu
    # Must run before the import below: torch_npu reads the allocator options
    # while it initialises.
    _request_mapped_pinned_memory()
    import torch_npu as _torch_npu

    torch_npu = _torch_npu
    from vllm_ascend.utils import enable_custom_op

    gate = _custom_op_gate()
    enable_custom_op()
    loaded_manually = False
    if not _op_registered():
        _import_extension()
        loaded_manually = True
    if not _op_registered():
        raise SystemExit(
            f"_C_ascend.{OP_NAME} is missing even after importing "
            "vllm_ascend_C: the built extension is older than the binding.\n"
            + REBUILD_BINDING
        )
    if loaded_manually:
        reason = f" ({gate})" if gate else ""
        print(f"note: imported vllm_ascend_C directly{reason}.", file=sys.stderr)
    _warn_if_kernel_package_missing()


def _count(text: str) -> int:
    """Row/token count, allowing the multiplicative form used in production.

    ``--space-rows '131072*32'`` is 4194304 rows and keeps the shape readable:
    32 requests x 128K tokens. Quote it, or the shell will expand the ``*``.
    Underscores are ignored, so ``1_048_576`` works too.
    """
    try:
        total = 1
        for part in text.replace("_", "").split("*"):
            total *= int(part)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a count: {text!r}") from None
    if total <= 0:
        raise argparse.ArgumentTypeError(f"count must be positive: {text!r}")
    return total


def _parse_int_list(text: str, flag: str) -> list[int]:
    """``"256,512,1024"`` -> ``[256, 512, 1024]``, for the grid row lengths."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(_count(part))
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(f"{flag}: {exc}") from None
    if not out:
        raise SystemExit(f"{flag} needs at least one value, e.g. 256,512,1024")
    return out


def _parse_device(text: str) -> int:
    """``3`` and ``npu:3`` both mean "the fourth visible NPU"."""
    value = text.strip().lower()
    if value.startswith("npu:"):
        value = value[len("npu:") :]
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an NPU index such as 0 or npu:0, got {text!r}") from exc
    if index < 0:
        raise argparse.ArgumentTypeError(f"NPU index must be >= 0, got {index}")
    return index


def _device_name(index: int) -> str:
    try:
        return torch.npu.get_device_name(index)
    except Exception:  # noqa: BLE001 - a diagnostic must not fail the run
        return "name unavailable"


def _physical_ids(count: int) -> list[str]:
    """The physical card ids behind ``--device``, when the filter tells us.

    ``ASCEND_RT_VISIBLE_DEVICES=2,5`` makes this process see two cards, numbered
    0 and 1. Reading the variable back is the only way to name the real ids: the
    runtime does not expose the mapping.
    """
    entries = [part.strip() for part in os.environ.get(DEVICE_ENV, "").split(",") if part.strip()]
    return entries if len(entries) == count else []


def _list_devices() -> int:
    """Print the cards this process can see, as ``--device`` numbers them."""
    count = torch.npu.device_count()
    mapped = os.environ.get(DEVICE_ENV, "").strip()
    if count <= 0:
        print("no NPU is visible to this process", file=sys.stderr)
        return 1
    print(f"{count} NPU(s) visible" + (f" via {DEVICE_ENV}={mapped}" if mapped else ""))
    physical = _physical_ids(count)
    print(f"{'--device':>8}  {'name':<28}" + ("  physical" if physical else ""))
    for index in range(count):
        suffix = f"  {physical[index]:>8}" if physical else ""
        print(f"{index:>8}  {_device_name(index):<28}{suffix}")
    if mapped:
        print(
            f"\n{DEVICE_ENV} renumbers the cards, so --device N is the N-th entry "
            f"of that list, not a physical card id.",
            file=sys.stderr,
        )
    return 0


def _select_device(index: int) -> int:
    """Make ``index`` the current card, and say which card that turned out to be.

    Every tensor in this benchmark is allocated on the current device and the
    op launches on it, so this single call is what decides the die -- and on a
    950 the die decides which UB port reaches host DRAM.
    """
    count = torch.npu.device_count()
    if count <= 0:
        raise SystemExit(
            "no NPU is visible to this process; check `npu-smi info` and "
            f"{DEVICE_ENV} (a bad filter hides every card)."
        )
    if index < 0:
        raise SystemExit(f"--device must be >= 0, got {index}")
    if index >= count:
        mapped = os.environ.get(DEVICE_ENV, "").strip()
        where = f"{DEVICE_ENV}={mapped} renumbers them. " if mapped else ""
        raise SystemExit(
            f"--device {index} is out of range: only {count} NPU(s) are visible "
            f"(valid: 0..{count - 1}). {where}Run --list-devices to see the mapping."
        )
    torch.npu.set_device(index)
    physical = _physical_ids(count)
    where = f", physical {physical[index]}" if physical else ""
    mapped = os.environ.get(DEVICE_ENV, "").strip()
    print(
        f"device: npu{index} ({_device_name(index)}){where}, {count} visible"
        + (f", {DEVICE_ENV}={mapped}" if mapped else ""),
        file=sys.stderr,
        flush=True,
    )
    return index


@cache
def _acl_runtime() -> ctypes.CDLL:
    """The CANN entry points that publish host memory to the device."""
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


def _host_register_flags() -> int:
    """Flags this SoC accepts in ``aclrtHostRegisterV2``.

    A5 rejects the PINNED hint with error 107000 (device_op.py says so
    explicitly), so ask the device adaptor rather than hard-coding:
    MAPPED on A5, MAPPED|PINNED elsewhere.
    """
    try:
        from vllm_ascend.device.device_op import DeviceOperator

        return DeviceOperator.host_register_flags()
    except Exception:  # noqa: BLE001 - diagnostics only
        return ACL_HOST_REG_MAPPED


def _elem_size(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def _random_values(rows: int, dim: int, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    """Random payload of the right dtype.

    Only the byte pattern matters to the copy, and it must be non-degenerate so
    the verifier can tell a torn row from a row that was never written. A 1-byte
    dtype is therefore filled with raw bytes rather than through float
    arithmetic, which keeps the payload free of NaN and of any dtype-specific
    random kernel.
    """
    if _elem_size(dtype) == 1:
        raw = torch.randint(0, 256, (rows, dim), generator=gen, dtype=torch.uint8)
        return raw.view(dtype) if dtype is not torch.uint8 else raw
    return torch.randn(rows, dim, generator=gen).to(dtype)


def _comparable(t: torch.Tensor) -> torch.Tensor:
    """View a 1-byte dtype as uint8 so the comparison is a plain byte compare."""
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _run_table(total: int, count: int, run_len: int, gen: torch.Generator) -> torch.Tensor:
    """Row indices for a ``count``-token gather/scatter over ``total`` rows.

    ``run_len == 1`` is the worst case for the copy and the default: a full
    permutation, so two consecutive tokens never share neighbouring rows and no
    group can ever be coalesced. ``run_len > 1`` emits random *runs* of
    consecutive rows, which is what a topk selection looks like when the picked
    blocks are adjacent. That makes the mergeable fraction a measured input
    instead of an assumption -- the whole point of ``--run-len``.
    """
    if run_len <= 1 or count < run_len:
        return torch.randperm(total, generator=gen)[:count]
    n_runs = count // run_len
    # Distinct run starts mean the table stays a set of unique row indices, so
    # two tokens can never write the same destination slot (which would make
    # --verify compare a row that a later token overwrote).
    blocks = total // run_len
    starts = torch.randperm(blocks, generator=gen)[:n_runs] * run_len
    idx = (starts[:, None] + torch.arange(run_len)[None, :]).reshape(-1)
    if idx.numel() < count:
        # Token count not divisible by run_len: the tail tokens are single rows.
        used = torch.zeros(total, dtype=torch.bool)
        used[idx] = True
        free = torch.nonzero(~used).flatten()
        idx = torch.cat([idx, free[: count - idx.numel()]])
    return idx[:count]


@dataclass
class Case:
    rows: int
    bank: int
    group_rows: int
    table_mode: int
    cores: int
    dtype: torch.dtype
    src: str
    device: int = 0
    ckv_dim: int = CKV_DIM
    kpe_dim: int = KPE_DIM
    run_len: int = 1
    # Rows allocated in the source (the gather pool). 0 means "same as rows",
    # i.e. every allocated row is copied once. A larger pool is the real shape
    # of a KV cache: 131072 rows in DRAM, 4096 of them picked by topk, so the
    # copy walks a working set 32x bigger than the payload it moves.
    pool_rows: int = 0
    # Distinct source tables to rotate through, one per launch (0/1 = reuse the
    # same 4096 rows every launch, which keeps the page tables warm).
    rotate: int = 0
    # Independent source regions polled in round-robin, one launch each. The
    # production shape is 78 of them (78 paged-cache partitions), each holding
    # 131072 x 32 rows, and every step copies 196608 rows out of every one, so a
    # full pass reads 78 x 123 MiB = 9.6 GiB spread over 205 GiB of DDR. A single
    # big pool cannot reproduce that: the address span between one space's rows
    # and the next one's is what the walk actually sees.
    spaces: int = 1
    # Rows per space. 0 -> pool_rows (and, if that is 0 too, rows).
    space_rows: int = 0
    # Write every space's result over the same destination window. Off by
    # default: production concatenates the 78 gathers into 9.6 GiB of HBM, and
    # overlapping writes would hide that traffic in the caches.
    dst_shared: bool = False
    # Destination rows allocated per copied row (the scatter is spread over this
    # many slots so it is not a disguised sequential write). 0 = auto: 4 for one
    # space, 1 for many, because 78 x 196608 rows x 4 x 656 B is 37 GiB of HBM
    # for a 9.4 GiB payload.
    dst_slack: int = 0
    # GiB of pinned host memory to allocate and release before this point is set
    # up. Not part of the copy at all: it reproduces the pinned-arena history of
    # a long sweep, which is the only thing that separates a 34.4 GB/s lone run
    # from the same flags reading 28 GB/s at rung 17. Only meaningful with
    # --isolate-points, where each point starts from a fresh process.
    churn_gib: int = 0

    def __post_init__(self) -> None:
        if self.dst_slack <= 0:
            self.dst_slack = 1 if self.spaces > 1 else 4
        if self.dst_slack < 1:
            raise ValueError("dst_slack must be >= 1")

    @property
    def src_total(self) -> int:
        """Rows in *one* source region (never smaller than the copy count)."""
        if self.space_rows > 0:
            return self.space_rows
        return self.pool_rows if self.pool_rows > 0 else self.rows

    @property
    def total_pool_bytes(self) -> int:
        """Bytes of source memory this case allocates, across all spaces."""
        return self.spaces * self.src_total * self.row_bytes

    @property
    def copy_rope(self) -> bool:
        return self.kpe_dim > 0

    @property
    def elem_size(self) -> int:
        return _elem_size(self.dtype)

    @property
    def row_bytes(self) -> int:
        """Bytes actually moved per token (the KPE plane may be off)."""
        return (self.ckv_dim + (self.kpe_dim if self.copy_rope else 0)) * self.elem_size


class Bench:
    """Allocates one problem size once and reuses it for every measurement."""

    def __init__(self, case: Case, seed: int = 0) -> None:
        self.case = case
        # Which die to run on matters for host bandwidth: on a 950 node the eight
        # dies are split across two sockets (see ``npu-smi info -t topo``), and
        # each socket attaches host DRAM through its own UB port. One process on
        # one die therefore touches one port at most. main() has already called
        # _select_device(), so this only matters when Bench is used directly.
        torch.npu.set_device(case.device)
        self.device = torch.device(f"npu:{case.device}")
        self._host_regions: list[tuple[int, int, torch.Tensor]] = []
        gen = torch.Generator().manual_seed(seed)
        rows = case.rows
        pool = case.src_total
        spaces = case.spaces

        # Sparse topk selection is a random gather from DRAM into a random slot,
        # so both tables are full permutations. Destinations are spread over 4x
        # the row count so the scatter is not a disguised sequential copy.
        # --run-len > 1 keeps that shape but makes runs of consecutive rows,
        # which is the only case a group of rows can be coalesced into one DMA.
        # The source table walks the whole pool: with --src-rows 131072 and 4096
        # tokens the 4096 rows are spread over 32x more pages than a 4096-row
        # buffer, which is the whole point of the bigger allocation.
        #
        # --rotate N pre-builds N source tables and hands the kernel a different
        # one for every launch, at zero cost (the table is already on the device,
        # so run() only picks a different slice). Without it every launch re-reads
        # the same rows, so the IOMMU/page tables stay warm and a large pool
        # looks cheaper than it is in production, where each step selects
        # different rows.
        #
        # --spaces N multiplies the source into N independent regions that are
        # polled one launch at a time, and gives every region its own table set.
        # The tables are all uploaded up front: 78 spaces x 196608 rows x 4 B is
        # 58 MiB of HBM, and slicing is free, so switching space inside the timed
        # loop costs nothing (spaces>1 only cycles the space index, not a copy).
        n_tables = max(1, case.rotate)
        tables = torch.cat(
            [
                torch.stack([_run_table(pool, rows, case.run_len, gen) for _ in range(n_tables)])
                for _ in range(spaces)
            ]
        ).to(torch.int32)
        self.dst_rows = _run_table(rows * case.dst_slack, rows, case.run_len, gen).to(torch.int32)

        # Only the rows a table points at are ever read, so a pool too large to
        # fill is left uninitialised elsewhere: allocating 205 GiB is affordable,
        # writing 205 GiB of random payload per space is not.
        self._src_kv: list[torch.Tensor] = []
        self._src_kv_cpu: list[torch.Tensor | None] = []
        for s in range(spaces):
            touched = torch.unique(tables[s * n_tables : (s + 1) * n_tables].reshape(-1).long())
            dev, cpu = self._make_source(pool, case.ckv_dim, gen, touched)
            self._src_kv.append(dev)
            self._src_kv_cpu.append(cpu)

        # Which nodes the pinned pages actually came from. A sweep cannot hold
        # this fixed -- each point takes its buffer out of whatever the previous
        # point released -- so it is recorded per point instead of assumed.
        self.numa_hint = ""
        if case.src == "host" and self._src_kv_cpu[0] is not None:
            first = self._src_kv_cpu[0]
            self.numa_hint = _numa_profile(first.data_ptr(), first.numel() * case.elem_size)

        dst_rows_per_space = rows * case.dst_slack
        n_dst = dst_rows_per_space if case.dst_shared else dst_rows_per_space * spaces
        self.dst_kv = torch.zeros(n_dst, case.ckv_dim, dtype=case.dtype, device=self.device)
        if case.copy_rope:
            self._src_rope: list[torch.Tensor] = []
            self._src_rope_cpu: list[torch.Tensor | None] = []
            for s in range(spaces):
                touched = torch.unique(
                    tables[s * n_tables : (s + 1) * n_tables].reshape(-1).long()
                )
                dev, cpu = self._make_source(pool, case.kpe_dim, gen, touched)
                self._src_rope.append(dev)
                self._src_rope_cpu.append(cpu)
            self.dst_rope = torch.zeros(n_dst, case.kpe_dim, dtype=case.dtype, device=self.device)
        else:
            # copy_rope=0: the rope tensors are still required by the op schema
            # but the kernel never touches them, so a one-element placeholder on
            # the right device is all it takes. A [rows, 0] tensor would be the
            # semantically cleaner argument, but zero-sized inputs are rejected
            # by some graph paths.
            self._src_rope = [torch.zeros(1, 1, dtype=case.dtype, device=self.device)] * spaces
            self._src_rope_cpu = [None] * spaces
            self.dst_rope = torch.zeros(1, 1, dtype=case.dtype, device=self.device)

        if case.table_mode == 0:
            # Identity addressing: the kernel never reads the row tables. The
            # scatter tables are then only meaningful for contiguity, so keep
            # them linear to make this a pure copy. This also means the copy
            # touches rows 0..rows-1 only: --src-rows changes the allocated pool
            # but not what this mode addresses, so it cannot show a working-set
            # effect. main() warns about that combination.
            tables = torch.arange(rows, dtype=torch.int32).reshape(1, rows).repeat(spaces, 1)
            n_tables = 1
            self.dst_rows = torch.arange(rows, dtype=torch.int32)

        # Per-space table count, and the flat index run() advances through.
        self.n_tables = n_tables
        self.src_tables_cpu = tables
        # The first table, for interactive use; run() is what advances the index
        # and verify() reads src_tables_cpu[self._last_table] instead.
        self.src_rows = tables[0]
        self.src_tables_dev = tables.to(self.device)
        # Pre-sliced views: run() must not create a tensor view per launch, that
        # alone would cost ~1 us inside a 80 us measurement.
        self._table_views = [self.src_tables_dev[i] for i in range(tables.shape[0])]
        # Same for the per-space source and destination slices.
        self._src_kv_views = list(self._src_kv)
        self._src_rope_views = list(self._src_rope)
        if case.dst_shared:
            self._dst_kv_views = [self.dst_kv] * spaces
            self._dst_rope_views = [self.dst_rope] * spaces
        else:
            self._dst_kv_views = [
                self.dst_kv[s * dst_rows_per_space : (s + 1) * dst_rows_per_space]
                for s in range(spaces)
            ]
            self._dst_rope_views = (
                [
                    self.dst_rope[s * dst_rows_per_space : (s + 1) * dst_rows_per_space]
                    for s in range(spaces)
                ]
                if case.copy_rope
                else [self.dst_rope] * spaces
            )
        self._space = 0
        self._table = 0
        self._last_space = 0
        self._last_table = 0
        self.dst_rows_dev = self.dst_rows.to(self.device)

    # The single-space names keep working for callers that only ever look at
    # space 0 (verify, interactive pokes); run() is the one that cycles.
    @property
    def src_kv_dev(self) -> torch.Tensor:
        return self._src_kv[0]

    @property
    def src_kv_cpu(self) -> torch.Tensor | None:
        return self._src_kv_cpu[0]

    @property
    def src_rope_dev(self) -> torch.Tensor:
        return self._src_rope[0]

    @property
    def src_rope_cpu(self) -> torch.Tensor | None:
        return self._src_rope_cpu[0]

    def _make_source(self, pool: int, dim: int, gen: torch.Generator, touched: torch.Tensor | None):
        """One source region of ``pool`` rows; only ``touched`` rows get payload.

        ``touched=None`` fills everything (small pools, best diagnostics). A
        large pool is allocated but left uninitialised outside ``touched``,
        because the copy never reads those rows and the reference check reads
        the very same host bytes -- so the comparison stays exact without
        writing tens of GiB of random data per space.
        """
        values = None
        if touched is None or pool * dim <= FULL_FILL_ELEMS:
            values = _random_values(pool, dim, self.case.dtype, gen)
            touched = None
        if self.case.src == "npu":
            if touched is None:
                return values.to(self.device), None
            big = torch.empty(pool, dim, dtype=self.case.dtype, device=self.device)
            big.index_copy_(
                0,
                touched.to(self.device),
                _random_values(int(touched.numel()), dim, self.case.dtype, gen),
            )
            return big, None
        if self.case.src != "host":
            raise ValueError(f"unknown source placement: {self.case.src}")
        return self._publish_host_source(values, pool, dim, gen, touched)

    def _publish_host_source(self, values, rows: int, dim: int, gen, touched):
        """Return (device view, host copy) of a host buffer the NPU can read.

        Two things are needed and only the second one is optional:

        1. the buffer must be pinned *and registered for device access* -- with
           ``pinned_mem_register:True`` in ``PYTORCH_NPU_ALLOC_CONF`` (set by
           ``_request_mapped_pinned_memory``) torch_npu's own allocator does the
           registration, which is the path ``tests/.../test_uva_view.py``
           exercises;
        2. a 64 KiB aligned payload, because unaligned device reads of host
           memory were observed to return wrong data on A3 / CANN 9.2 (see
           ``engram/npu.py::HostUvaBuffer``).

        If the allocator option did not take effect (torch_npu was already
        imported before this script ran) the range is registered here instead,
        mirroring HostUvaBuffer.
        """
        elt = _elem_size(self.case.dtype)
        # Over-allocate so the payload starts on a 64 KiB boundary. The whole
        # storage is registered, so the view's storage_offset stays reachable and
        # aclrtHostGetDevicePointer(storage base) remains valid.
        slack = HOST_ALIGN_BYTES // elt
        raw = torch.empty(rows * dim + slack, dtype=self.case.dtype, pin_memory=True)
        if not raw.is_pinned():
            raise SystemExit(
                "torch pin_memory() returned an ordinary CPU buffer, so it cannot "
                "be published to the device. Check the torch_npu pinned allocator "
                "on this machine, or run without --src host."
            )
        offset = ((-raw.data_ptr()) % HOST_ALIGN_BYTES) // elt
        payload = raw.narrow(0, offset, rows * dim).view(rows, dim)
        if values is not None:
            payload.copy_(values)
        else:
            # Large pool: only the rows a table points at carry payload. Writing
            # all 205 GiB per space would dominate the setup and tell the copy
            # nothing -- it reads nothing else.
            payload[touched] = _random_values(
                int(touched.numel()), dim, self.case.dtype, gen
            )

        view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(payload)
        if view is not None:
            return view, payload

        # Fallback: publish the pinned range ourselves.
        base = raw.data_ptr()
        size = raw.numel() * elt
        flags = _host_register_flags()
        rc = -1
        rc_dev = -1
        try:
            lib = _acl_runtime()
        except OSError as exc:
            lib = None
            print(f"warning: libascendcl.so not loadable ({exc})", file=sys.stderr)
        if lib is not None:
            rc = lib.aclrtHostRegisterV2(ctypes.c_void_p(base), size, flags)
            if rc == 0:
                self._host_regions.append((base, size, raw))
                view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(payload)
                if view is not None:
                    return view, payload
                dev_ptr = ctypes.c_void_p()
                rc_dev = lib.aclrtHostGetDevicePointer(
                    ctypes.c_void_p(payload.data_ptr()), ctypes.byref(dev_ptr), 0
                )
                lib.aclrtHostUnregister(ctypes.c_void_p(base))
                self._host_regions.clear()

        raise SystemExit(
            "the host source could not be published to the device:\n"
            f"  pinned={bool(raw.is_pinned())}, alloc_conf={os.environ.get(ALLOC_CONF_ENV, '')!r}"
            f", alloc_conf_added_by_us={_alloc_conf_added}\n"
            f"  aclrtHostRegisterV2 rc={rc} flags=0x{flags:x}\n"
            f"  aclrtHostGetDevicePointer rc={rc_dev}\n"
            "  if alloc_conf_added_by_us is False, this script could not add "
            "pinned_mem_register:True (torch_npu was already imported), so the "
            "pinned pages were never mapped. Export it and rerun:\n"
            f"    export {ALLOC_CONF_ENV}={PINNED_REGISTER_OPT}\n"
            "  a non-zero register rc means the runtime refused the range itself "
            "(A5 takes only ACL_HOST_REG_MAPPED; PINNED fails with 107000).\n"
            "  run without --src host to measure HBM->HBM instead."
        )

    def close(self) -> None:
        """Unregister the host sources, never while a launch may still read them."""
        if not self._host_regions:
            return
        torch.npu.synchronize()
        lib = _acl_runtime()
        for base, _size, _raw in self._host_regions:
            rc = lib.aclrtHostUnregister(ctypes.c_void_p(base))
            if rc:
                print(f"warning: aclrtHostUnregister rc={rc} for 0x{base:x}", flush=True)
        self._host_regions.clear()

    def launch(self, space: int, table: int | None = None) -> None:
        """One launch on one space, with an explicit table slot.

        Everything is pre-sliced, so this is: three list indexes, one op call.
        """
        t = self._table if table is None else table
        torch.ops._C_ascend.scatter_copy_ub_bench(
            self._src_kv_views[space],
            self._src_rope_views[space],
            self._table_views[space * self.n_tables + t],
            self.dst_rows_dev,
            self._dst_kv_views[space],
            self._dst_rope_views[space],
            self.case.bank,
            self.case.group_rows,
            self.case.table_mode,
            self.case.cores,
            int(self.case.copy_rope),
        )
        self._last_space = space
        self._last_table = space * self.n_tables + t

    def run(self) -> None:
        # Picking a different pre-staged table or a different pre-published
        # source costs nothing (no copy, no host round trip), so cycling both
        # inside the timed loop is free. With --spaces N every launch moves to
        # the next region and every full pass moves to the next table, which is
        # the production pattern: one gather per partition per step.
        self.launch(self._space)
        if self.case.spaces > 1:
            self._space = (self._space + 1) % self.case.spaces
            if self._space == 0 and self.n_tables > 1:
                self._table = (self._table + 1) % self.n_tables
        elif self.n_tables > 1:
            self._table = (self._table + 1) % self.n_tables

    def time_spaces(self) -> list[float]:
        """Per-space latency (us) of one pass, in space order.

        Worth one pass of extra work: a mean hides the case where one region is
        slow (NUMA placement, a different page-table walk depth), and that is
        exactly what --spaces is there to expose.
        """
        out: list[float] = []
        for s in range(self.case.spaces):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            self.launch(s)
            end.record()
            torch.npu.synchronize()
            out.append(start.elapsed_time(end) * 1000.0)
        return out

    def verify(self, sample: int = 64, repeats: int = 1) -> None:
        """Compare the copied rows against the source, and report *how* they differ.

        ``sample=0`` checks every row. ``repeats > 1`` re-runs the op and
        re-checks every time, which is the only way to catch a
        non-deterministic kernel: a launch that passes once and fails later
        points at a missing fence or uninitialised state, not at an addressing
        bug (that one fails every time, on the same rows).

        With ``kpe_dim == 0`` the rope plane is never written, so only the CKV
        plane is checked; the placeholder rope tensors hold no reference data.

        Every space is checked against its own source and its own table, because
        --spaces makes them independent: a wrong base pointer would otherwise
        pass as long as space 0 happens to be right.
        """
        for space in range(self.case.spaces):
            self._verify_space(space, sample, repeats)

    def _verify_space(self, space: int, sample: int, repeats: int) -> None:
        total = self.case.rows
        # Checking every row of 78 x 196608 rows means 9.6 GiB of host reads and
        # a 78-way tensor compare; keep the default sample, it is already more
        # than the 64-row default and covers each space's own address base.
        sample = total if sample <= 0 else min(sample, total)
        idx = torch.arange(sample, dtype=torch.int64)
        src_flat = _comparable(self._source_cpu(self._src_kv_cpu[space], self._src_kv[space]))
        dst_flat = self._dst_kv_views[space]
        dst_idx = self.dst_rows[idx].to(torch.int64)
        check_rope = self.case.copy_rope
        if check_rope:
            rope_flat = _comparable(
                self._source_cpu(self._src_rope_cpu[space], self._src_rope[space])
            )

        for attempt in range(repeats):
            self.launch(space)
            torch.npu.synchronize()
            # The table this launch used: with --rotate the rows differ per
            # launch, so the reference has to be rebuilt from the same table.
            used = self.src_tables_cpu[self._last_table]
            want_kv = src_flat[used[idx].to(torch.int64)]
            if check_rope:
                want_rope = rope_flat[used[idx].to(torch.int64)]
            got_kv = _comparable(dst_flat[dst_idx].cpu())
            bad_kv = (got_kv != want_kv).any(dim=1)
            if check_rope:
                got_rope = _comparable(self._dst_rope_views[space][dst_idx].cpu())
                bad_rope = (got_rope != want_rope).any(dim=1)
            else:
                got_rope = want_rope = None
                bad_rope = torch.zeros_like(bad_kv)
            if bool(bad_kv.any()) or bool(bad_rope.any()):
                summary = self._report_mismatch(
                    attempt, repeats, bad_kv, bad_rope, got_kv, want_kv, got_rope, want_rope,
                    src_flat, used, space
                )
                raise AssertionError(
                    f"scatter_copy_ub_bench verify failed on space {space} attempt "
                    f"{attempt + 1}/{repeats}: {summary}"
                )
            print(
                f"verify: attempt {attempt + 1}/{repeats} ok (space {space + 1}/{self.case.spaces}, "
                f"{sample}/{total} rows, "
                f"{self.case.ckv_dim}+{self.case.kpe_dim} elems x{self.case.elem_size}B, "
                f"{self.case.bank} banks, group {self.case.group_rows}, table_mode {self.case.table_mode}"
                + ("" if self.n_tables == 1 else f", table {self._table + 1}/{self.n_tables}")
                + ")",
                flush=True,
            )

    def _report_mismatch(self, attempt, repeats, bad_kv, bad_rope, got_kv, want_kv, got_rope, want_rope,
                         src_flat, used, space=0) -> str:
        """Print one block per bad row and return a one-line summary.

        The summary is also embedded in the raised AssertionError, so the
        diagnosis survives a log that keeps only the traceback.
        """
        rows = torch.nonzero(bad_kv | bad_rope).flatten().tolist()
        total_kv = int((got_kv != want_kv).sum())
        total_rope = int((got_rope != want_rope).sum()) if got_rope is not None else 0
        print(
            f"\nverify: space {space} attempt {attempt + 1}/{repeats} mismatched "
            f"{len(rows)}/{got_kv.shape[0]} sampled rows "
            f"(kv {total_kv} elems, rope {total_rope} elems; first rows {rows[:8]})",
            flush=True,
        )
        first = ""
        for r in rows[:4]:
            dst_tok = int(self.dst_rows[r])
            src_tok = int(used[r])
            got, want = got_kv[r], want_kv[r]
            cols = torch.nonzero(got != want).flatten()
            # Which source row (if any) did the destination actually receive?
            # A hit means broken address generation; a miss plus all-zero means
            # the row was never written; a miss plus non-zero means a torn DMA.
            hits = torch.nonzero((src_flat == got).all(dim=1)).flatten()
            hint = "no other source row matches -> torn or stale DMA"
            if hits.numel():
                hint = f"payload equals source row(s) {hits[:4].tolist()} (expected {src_tok})"
            elif bool((got == 0).all()):
                hint = "all zeros -> row never written"
            span = f"[{int(cols[0])}..{int(cols[-1])}]" if cols.numel() else "none"
            if got_rope is None:
                rope_state = "n/a (copy_rope=0)"
            else:
                rope_state = "bad" if bool((got_rope[r] != want_rope[r]).any()) else "ok"
            print(
                f"  sample row {r}: dst_tok={dst_tok} src_tok={src_tok}, "
                f"kv bad {cols.numel()}/{want.numel()} cols {span}, "
                f"rope {rope_state} | {hint}",
                flush=True,
            )
            if not first:
                first = (
                    f"sample row {r} dst_tok={dst_tok} src_tok={src_tok}, kv bad "
                    f"{cols.numel()}/{want.numel()} cols {span}, rope {rope_state}, {hint}"
                )
        return f"{len(rows)}/{got_kv.shape[0]} sampled rows bad; {first}"

    def _source_cpu(self, host, view):
        return host if host is not None else view.cpu()

    def bytes(self) -> int:
        return self.case.rows * self.case.row_bytes


def time_case(
    bench: Bench, warmup: int, iters: int, rounds: int
) -> tuple[float, float]:
    """Return (min, median) of per-launch latency in microseconds."""
    samples: list[float] = []
    bench.run()
    torch.npu.synchronize()
    for _ in range(rounds):
        for _ in range(warmup):
            bench.run()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            bench.run()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    return min(samples), statistics.median(samples)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rows",
        type=int,
        default=2048,
        help="token rows to copy per launch (the gather count); the source tensor may be "
        "larger, see --src-rows. 196608 is topk 2048 x mtp 3 x 32 requests",
    )
    parser.add_argument(
        "--src-rows",
        type=_count,
        default=0,
        help="rows allocated in the source (the gather pool), default = --rows. Use "
        "131072 with --rows 4096 to measure a real KV cache shape: the copy then "
        "walks a working set far bigger than the payload it moves. Accepts a product, "
        "e.g. '131072*32'",
    )
    parser.add_argument(
        "--spaces",
        type=_count,
        default=1,
        metavar="N",
        help="independent source regions, polled one launch at a time (1 = a single "
        "pool, the old behaviour). The production shape is 78 cache partitions, each "
        "holding --space-rows rows, with --rows copied out of every one per pass, so "
        "a pass reads N x rows across a span N times larger than one region",
    )
    parser.add_argument(
        "--space-rows",
        type=_count,
        default=0,
        metavar="R",
        help="rows per space (default: --src-rows, else --rows). The production space "
        "is '131072*32' = 4194304 rows, i.e. 32 requests x 128K tokens",
    )
    parser.add_argument(
        "--dst-shared",
        action="store_true",
        help="write every space's result into the same destination window. Off by "
        "default: production concatenates the per-space gathers, and overlapping "
        "writes into one 492 MiB window would be served by the caches",
    )
    parser.add_argument(
        "--dst-slack",
        type=int,
        default=0,
        metavar="N",
        help="destination rows per copied row (0 = auto: 4 with one space, 1 with "
        "--spaces > 1). The slack makes the scatter a scatter instead of a sequential "
        "write, but N x --spaces multiplies HBM: 78 x 196608 x 4 x 656 B is 37 GiB "
        "for a 9.4 GiB payload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the allocation plan (bytes per space, total DDR, table size) and "
        "exit without allocating -- check this before a 200 GiB run",
    )
    parser.add_argument(
        "--ckv-dim",
        type=int,
        default=CKV_DIM,
        help=f"elements per row in the CKV plane (default {CKV_DIM}); multiplied by the "
        "dtype size to get the row bytes, e.g. 656 B for the DeepSeek-style MLA "
        "row (512 nope + 16 scale + 128 rope) with --dtype uint8",
    )
    parser.add_argument(
        "--kpe-dim",
        type=int,
        default=KPE_DIM,
        help=f"elements per row in the KPE plane (default {KPE_DIM}); 0 disables the plane "
        "entirely (copy_rope=0), which is the layout where rope is part of the CKV row",
    )
    parser.add_argument("--bank", type=int, default=2, help="UB pipeline depth (1 disables overlap)")
    parser.add_argument("--group-rows", type=int, default=1, help="rows merged into one DataCopyPad")
    parser.add_argument("--table-mode", type=int, default=1, help="1 = gather/scatter via row tables, 0 = identity")
    parser.add_argument(
        "--run-len",
        type=int,
        default=1,
        help="rows per run of consecutive indices in the row tables (default 1 = a full "
        "permutation, i.e. nothing is coalescible). Use it to measure --group-rows at a "
        "realistic topk locality instead of the worst case",
    )
    parser.add_argument("--cores", type=int, default=0, help="AIV blocks to launch, 0 = all")
    parser.add_argument("--src", choices=("npu", "host"), default="npu", help="source placement")
    parser.add_argument(
        "--device",
        type=_parse_device,
        default=0,
        metavar="N",
        help="NPU index to run on, as `3` or `npu:3` (default: 0). Indices are into the "
        f"visible set, so {DEVICE_ENV}=5 makes --device 0 mean card 5. Use different "
        "indices per process when measuring how many host links a node has: topo "
        "groups them by socket. --list-devices prints the mapping",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print the visible NPUs with the index --device expects, then exit",
    )
    parser.add_argument(
        "--host-interleave",
        action="store_true",
        help="with --src host, restart under numactl --interleave=all so the source pages "
        "span every NUMA node (one host link per node: this is what uses both UB ports)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        metavar="N",
        help="pre-stage N source row tables and use a different one per launch (0/1 = "
        "reuse the same rows every launch). With a big --src-rows this keeps the "
        "page/IOMMU tables cold like production, where every step selects different "
        "rows; the rotation costs nothing because the tables are already on the device",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bf16",
        help="cache element type; a 1-byte cache row is measured through uint8, which is "
        "copied byte-wide like any other row",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print one JSON record per case on stdout (for host_link_scale.py); the "
        "human-readable table is suppressed",
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--verify", action="store_true", help="check copied rows before timing")
    parser.add_argument(
        "--verify-rows",
        type=int,
        default=64,
        help="rows to check with --verify (0 = every row; a full sweep is the only way to prove "
        "that every owning core wrote its share)",
    )
    parser.add_argument(
        "--repeat-verify",
        type=int,
        default=1,
        help="with --verify, re-run and re-check N times (N>1 exposes flaky kernels)",
    )
    parser.add_argument(
        "--sweep",
        choices=(
            "none", "group-rows", "bank", "table-mode", "cores", "src", "ckv-dim",
            "row-bytes", "src-rows", "rows", "spaces", "dst-slack", "churn", "grid",
        ),
        default="none",
        help="iterate one knob instead of measuring a single point; 'grid' is the "
        "row-length x space-count matrix",
    )
    parser.add_argument(
        "--grid-row-bytes",
        default=",".join(str(v) for v in GRID_ROW_BYTES),
        help="row lengths in bytes for --sweep grid (default %(default)s); add 656 to "
        "put the production row in the same matrix",
    )
    parser.add_argument(
        "--sweep-values",
        default="",
        help="replace this run's ladder, comma separated: '--sweep grid "
        "--sweep-values 1' measures the row-length ladder with no space ladder, "
        "'--sweep churn --sweep-values 0,48,96' is a three-point dose-response. "
        "Applies to --sweep grid (the values are space counts) or to the named sweep",
    )
    parser.add_argument(
        "--xlsx",
        default="",
        help="write every measured point to this .xlsx workbook: one row per point, "
        "plus a GB/s matrix and a pass-time matrix when the sweep covers two "
        "dimensions. Needs openpyxl; without it a .csv of the same rows is written",
    )
    parser.add_argument(
        "--churn-gib",
        type=int,
        default=0,
        metavar="GiB",
        help="allocate and release this much pinned host memory before setting the "
        "point up. The same 656 B copy reads 34.4 GB/s in a fresh process and "
        "27.9-28.5 GB/s as a rung of a long sweep with every flag identical; this "
        "reproduces that history on purpose instead of inheriting it. Use with "
        "--sweep churn --isolate-points for a dose-response",
    )
    parser.add_argument(
        "--isolate-points",
        action="store_true",
        help="measure every sweep point in its own child process. A sweep otherwise "
        "reuses one process, so rung k is measured on memory an earlier point "
        "already allocated and freed: 656 B rows read 34.2 GB/s as a lone run and "
        "25-28 GB/s as the 17th point of a grid. Isolating makes a rung mean what a "
        "standalone run of the same flags means, at one interpreter start per point",
    )
    parser.add_argument(
        "--isolate-timeout",
        type=int,
        default=60,
        metavar="MIN",
        help="per-point timeout for --isolate-points, in minutes (a 78-space point "
        "allocates 200 GiB of pinned memory, so this is not a tight bound)",
    )
    return parser.parse_args(argv)


# --sweep-values replaces one ladder for this run: "--sweep grid --sweep-values 1"
# is "the row-length ladder only, no space ladder" (7 points instead of 56), and
# "--sweep churn --sweep-values 0,48,96" is a three-point dose-response.
_SWEEP_OVERRIDES: dict[str, list] = {}


def sweep_values(name: str) -> list:
    if name in _SWEEP_OVERRIDES:
        return _SWEEP_OVERRIDES[name]
    return {
        "group-rows": [1, 2, 4, 8],
        "bank": [1, 2, 3, 4, 6, 8],
        "table-mode": [0, 1],
        "cores": [8, 16, 24, 32, 48],
        "src": ["npu", "host"],
        # 384/512/640/768/896/1024 are multiples of 128 B, 576/704 are only
        # multiples of 64 B, and 656/672 are the two real MLA rows (512 nope +
        # 16 scale + 128 rope, and the same row padded to 32 B). Comparing the
        # three groups is what tells a length penalty from an alignment one.
        "ckv-dim": [384, 512, 576, 640, 656, 672, 704, 768, 896, 1024],
        # The four row lengths the cache actually uses, expressed in bytes rather
        # than elements so the sweep reads as the layout discussion does. Every
        # one is a whole number of granules except 656, which is 768 + a 144 B
        # tail: 256 and 512 are the clean controls, 1024 the 4-granule one.
        "row-bytes": [256, 512, 656, 1024],
        # Rows per launch. A launch carries a fixed cost (dispatch + tiling lookup,
        # a few us), so a small --rows reports a low GB/s for the same row length:
        # 512 B rows measure 41.2 GB/s at 2048 rows and 44.4 at 4096. Fit
        # t = a * rows + b on this sweep to get that constant on the local card;
        # it is also why every absolute number must be compared at equal --rows.
        "rows": [1024, 2048, 4096, 8192, 16384],
        # Working-set sweep: same 4096-row payload, pool growing 32x. This is the
        # only way to tell "DDR got bigger and slower" from "the pool does not
        # matter at all", and it must be run with --table-mode 1.
        "src-rows": [4096, 16384, 32768, 65536, 131072],
        # Destination rows per copied row. 1 = every destination row is written
        # exactly once, which is what production does when it concatenates the
        # per-partition gathers; 4 = the writes are a random quarter of a 4x
        # larger window. It is the one knob a single-point run resolves
        # differently from a ladder (auto: 4 with one space, 1 with many), so a
        # standalone number and a matrix cell only compare when this is equal.
        "dst-slack": [1, 2, 4, 8],
        # How many independent DDR partitions take part in one pass. Production
        # polls 78 of them; 1 is the control (a single base pointer, one
        # contiguous region). --space-rows is held fixed, so total DDR grows with
        # the count while the payload per launch does not -- that is what makes
        # this the measurement of "does the address span cost anything".
        "spaces": [1, 2, 4, 8, 16, 32, 64, 78],
        # How much pinned host memory the process has been through before this
        # point is set up. A ladder rung inherits its predecessor's freed arena;
        # a lone run does not. Sweeping this in isolated processes separates
        # "the copy got slower" from "the process got older" -- the two readings
        # that have to be told apart before either number means anything.
        "churn": [0, 8, 32, 96, 192],
    }[name]


# --sweep grid walks these row lengths x every space count above. 256 and 1024
# are whole granules, 512 is two of them; add 656 with --grid-row-bytes to see
# the production row in the same matrix (it is the only one with a tail).
GRID_ROW_BYTES = (256, 512, 1024)


# A sweep name is normally the field name with dashes flattened; --src-rows is
# the source pool, which lives in Case.pool_rows.
SWEEP_FIELDS = {"src-rows": "pool_rows", "churn": "churn_gib"}


def _numa_profile(ptr: int, size: int) -> str:
    """Where the *resident* pages of [ptr, ptr+size) physically live, per NUMA node.

    This is the diagnostic for the one thing a host-sourced copy cannot control
    from inside the op: a 2.6 GiB pinned buffer is faulted in by whichever node
    the allocator happened to get the pages from, and the device reads it over
    the UB port of the socket that owns those pages. A sweep that keeps one
    process alive hands point k a buffer carved out of what point k-1 freed, so
    "the same copy" can run 20% slower for a reason that is not in the flags.

    ``""`` when /proc/self/numa_maps is missing (non-Linux), the range has no
    resident pages yet, or nothing overlaps it.
    """
    try:
        with open("/proc/self/numa_maps", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    lo, hi = ptr, ptr + size
    pages: dict[int, int] = {}
    page_kb = 4
    for line in lines:
        field = line.split()
        if len(field) < 3:
            continue
        try:
            # The first field is "start-end" in hex, not a bare address.
            start_hex, _, end_hex = field[0].partition("-")
            start, end = int(start_hex, 16), int(end_hex, 16)
        except ValueError:
            continue
        overlap = min(hi, end) - max(lo, start)
        if overlap <= 0:
            continue
        # A pinned buffer is normally one VMA per space, so this weight is 1;
        # it only matters if the allocator hands back a range inside a bigger
        # mapping, where counting the whole VMA would overstate the footprint.
        weight = overlap / max(1, end - start)
        for token in field[2:]:
            if token.startswith("kernelpagesize_kB="):
                page_kb = int(token.split("=", 1)[1] or page_kb)
            elif token.startswith("N") and "=" in token:
                node, _, count = token[1:].partition("=")
                if node.isdigit() and count.isdigit():
                    pages[int(node)] = pages.get(int(node), 0) + round(int(count) * weight)
    if not pages:
        return ""
    total = sum(pages.values())
    share = " ".join(
        f"N{node} {100.0 * count / total:.0f}%"
        for node, count in sorted(pages.items(), key=lambda kv: -kv[1])
    )
    return f"{share} of {total * page_kb / 2**10:.0f} MiB resident ({page_kb} KiB pages)"


def _guard_memory(case: Case) -> None:
    """Refuse a case that cannot fit, before any of it is allocated.

    Checked per point rather than once: --sweep spaces grows total DDR with the
    count, so the last point is the one that does not fit, and finding that out
    after 20 points of setup is a waste of an hour.
    """
    need = case.total_pool_bytes
    try:
        phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return
    if phys and need > 0.85 * phys:
        raise SystemExit(
            f"{case.spaces} space(s) x {case.src_total} rows x {case.row_bytes} B needs "
            f"{need / 2**30:.0f} GiB of {'pinned host memory' if case.src == 'host' else 'HBM'} "
            f"but the machine has {phys / 2**30:.0f} GiB; lower --spaces or --space-rows, "
            "or check --dry-run first"
        )


def _churn_bytes(total: int) -> int:
    """Allocate, fault in, and release ``total`` bytes of pinned host memory.

    Returns the bytes actually churned: 200 GiB is not always available, and the
    number is part of the reading ("churned 96 GiB, not 192").

    The page faulting is the point. An allocation nobody writes to is free pages,
    so it changes nothing; what the next allocation gets depends on how much
    *resident* memory was just handed back to the kernel.
    """
    if total <= 0:
        return 0
    page = 1 << 12
    held: list[torch.Tensor] = []
    done = 0
    try:
        while done < total:
            # The last chunk is trimmed, so a caller that asks for 3.25 GiB gets
            # 3.25 GiB and not a rounded-up 4.
            chunk = min(CHURN_CHUNK_BYTES, total - done)
            try:
                buf = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
            except RuntimeError as exc:
                print(
                    f"note: --churn-gib stopped at {done / 2**30:.0f} GiB ({exc})",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if not buf.is_pinned():
                raise SystemExit(
                    "--churn-gib needs the torch_npu pinned allocator; pin_memory() "
                    "returned an ordinary CPU buffer"
                )
            # One byte per 4 KiB makes every page distinct; the rest of the page
            # follows as the kernel copies the page in.
            buf.view(chunk // page, page)[:, 0] = 1
            held.append(buf)
            done += chunk
    finally:
        del held
        gc.collect()
    return done


def _churn_host(gib: int) -> None:
    """Reproduce the pinned-arena history of point N before measuring point N.

    This is the one difference between a lone run and the same flags measured as
    rung 17 of a matrix, made explicit. A ladder keeps one interpreter alive, so
    each point carves its pinned source out of what the previous point released;
    a lone run gets a virgin arena. 656 B rows read 34.4 GB/s in a fresh process
    and 27.9-28.5 GB/s as a rung, reproducibly, with every flag identical -- so
    "which number is real" is a question about process history, not about the
    copy, and it deserves a knob instead of a footnote.

    Churned before the case is set up, so the measurement that follows takes the
    same code path as always. Meaningful with --isolate-points: without it the
    churn of point k stays in the process for point k+1.
    """
    if gib <= 0:
        return
    try:
        phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        phys = 0
    if phys:
        budget = int(0.6 * phys / 2**30)
        if gib > budget:
            print(
                f"note: --churn-gib {gib} clipped to {budget} GiB (60% of "
                f"{phys / 2**30:.0f} GiB physical)",
                file=sys.stderr,
                flush=True,
            )
            gib = budget
    done = _churn_bytes(gib << 30)
    print(
        f"note: churned {done / 2**30:.0f} GiB of pinned host memory before setup",
        file=sys.stderr,
        flush=True,
    )


def _plan_line(case: Case) -> str:
    """One line per space count: what the run will allocate and move."""
    pass_rows = case.spaces * case.rows
    dst_rows = pass_rows * case.dst_slack
    churn = f", churn {case.churn_gib} GiB" if case.churn_gib else ""
    return (
        f"  {case.spaces:>3} space(s) x {case.src_total:>8} rows = "
        f"{case.total_pool_bytes / 2**30:>7.1f} GiB DDR, "
        f"pass {pass_rows * case.row_bytes / 2**30:>6.2f} GiB, "
        f"dst {dst_rows * case.row_bytes / 2**30:>6.2f} GiB "
        f"(x{case.dst_slack}), "
        f"tables {case.spaces * max(1, case.rotate) * case.rows * 4 / 2**20:>5.0f} MiB"
        f"{churn}"
    )


def _print_matrix(records: list[dict], key: str, title: str, fmt: str) -> None:
    """Rows = row length, columns = space count; blank where nothing was measured."""
    ok = [r for r in records if r.get("status") == "ok"]
    lens = sorted({r["row B"] for r in ok})
    spaces = sorted({r["spaces"] for r in ok})
    if len(lens) < 2 or len(spaces) < 2:
        return
    cell = {(r["row B"], r["spaces"]): r.get(key) for r in ok}
    width = max(len(title), 8) + 2
    print(f"\n{title}")
    print(f"{'B/row':>{width}}" + "".join(f"{s:>9}" for s in spaces))
    for rb in lens:
        line = f"{rb:>{width}}"
        for s in spaces:
            v = cell.get((rb, s))
            line += f"{'':>9}" if v is None else f"{format(v, fmt):>9}"
        print(line)


def _write_workbook(path: str, records: list[dict], meta: dict) -> None:
    """Dump the measured points: one sheet per dimension, plus the raw rows.

    openpyxl is optional -- the harness has to run on a node where it may not be
    installed -- so a missing wheel degrades to a .csv with exactly the same rows
    instead of losing the measurements.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        csv_path = os.path.splitext(path)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(
            f"xlsx: openpyxl is not installed, wrote {csv_path} instead "
            "(pip install openpyxl for the matrix workbook)",
            file=sys.stderr,
            flush=True,
        )
        return

    head_fill = PatternFill("solid", fgColor="1F3864")
    head_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style_header(ws, row: int, ncols: int) -> None:
        for c in range(1, ncols + 1):
            cell = ws.cell(row=row, column=c)
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border

    wb = Workbook()

    # --- sheet 1: one row per measured point -------------------------------
    ws = wb.active
    ws.title = "结果"
    ws.append(RESULT_COLUMNS)
    style_header(ws, 1, len(RESULT_COLUMNS))
    for rec in records:
        ws.append([rec.get(col) for col in RESULT_COLUMNS])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(RESULT_COLUMNS))}{ws.max_row}"
    width = {
        "case": 16, "row B": 8, "spaces": 8, "pool rows": 12, "launch MiB": 11,
        "src": 6, "banks": 7, "group": 7, "tbl": 6, "dst slack": 9, "cores": 7, "us": 10,
        "ns/row": 9, "GB/s": 9, "model GB/s": 12, "%host": 8, "pass ms": 11,
        "space min us": 13,
        "space max us": 13, "max/min": 9, "total DDR GiB": 14, "dst MiB": 10,
        "src numa": 34, "status": 10,
    }
    for i, col in enumerate(RESULT_COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width.get(col, 11)
    numfmt = {
        "GB/s": "0.0", "model GB/s": "0.0", "ns/row": "0.00", "%host": "0", "us": "0.0",
        "pass ms": "0.00", "space min us": "0.0", "space max us": "0.0",
        "max/min": "0.000", "launch MiB": "0.0", "dst MiB": "0",
        "total DDR GiB": "0.00",
    }
    for i, col in enumerate(RESULT_COLUMNS, start=1):
        if col in numfmt:
            for cell in ws[get_column_letter(i)][1:]:
                cell.number_format = numfmt[col]

    # --- sheets 2 and 3: the two dimensions as a matrix -------------------
    ok = [r for r in records if r.get("status") == "ok"]
    lens = sorted({r["row B"] for r in ok})
    spaces = sorted({r["spaces"] for r in ok})
    if len(lens) > 1 and len(spaces) > 1:
        # Sheet names may not contain / \ * ? : [ ], hence "GBps" and "pass_ms".
        # nsrow is the length-independent view: GB/s folds the row length into the
        # answer, so 256 B at 25 GB/s and 512 B at 46 GB/s look like "2x" when the
        # cost per row is actually 1.1x (10.1 vs 11.2 ns). Read that sheet before
        # concluding anything about a row length.
        for title, key, fmt in (
            ("GBps", "GB/s", "0.0"),
            ("nsrow", "ns/row", "0.00"),
            ("pass_ms", "pass ms", "0.00"),
            ("max_min", "max/min", "0.000"),
        ):
            ws = wb.create_sheet(title)
            with_model = key == "GB/s"
            ws.append(
                [f"B/row \\ spaces"] + spaces + (["model k=1 (host)"] if with_model else [])
            )
            style_header(ws, 1, len(spaces) + (2 if with_model else 1))
            cell = {(r["row B"], r["spaces"]): r[key] for r in ok}
            model = {r["row B"]: r["model GB/s"] for r in ok}
            for rb in lens:
                ws.append(
                    [rb]
                    + [cell.get((rb, s)) for s in spaces]
                    + ([model.get(rb)] if with_model else [])
                )
            for row in ws.iter_rows(min_row=2):
                for c in row[1:]:
                    c.number_format = fmt
            ws.column_dimensions["A"].width = 16
            for i in range(2, len(spaces) + (3 if with_model else 2)):
                ws.column_dimensions[get_column_letter(i)].width = 10
            ws.freeze_panes = "B2"

    # --- last sheet: what was run, and how to read it ---------------------
    ws = wb.create_sheet("说明")
    rows = [
        ("生成时间", meta.get("when", "")),
        ("命令", meta.get("cmd", "")),
        ("主机", meta.get("host", "")),
        ("device", meta.get("device", "")),
        ("源位置 src", meta.get("src", "")),
        ("每 launch 行数", meta.get("rows", "")),
        ("每 space 行数", meta.get("space_rows", "")),
        ("table_mode", meta.get("table_mode", "")),
        ("group_rows", meta.get("group_rows", "")),
        ("run_len", meta.get("run_len", "")),
        ("rotate", meta.get("rotate", "")),
        ("计时", meta.get("timing", "")),
        ("isolate", meta.get("isolate", "off")),
        ("", ""),
        ("GB/s", "rows x B/row / 时延，只算有效载荷；读一份 + 写一份的实际链路带宽约为 2 倍"),
        ("ns/row", "每行耗时，与行长无关的比较口径。GB/s 把行长折进了答案：256 B 的 25 GB/s "
                   "和 512 B 的 46 GB/s 看起来差一倍，实际每行成本只差 10%"),
        ("%host", "实测 / host 路径模型（predict_gbps）。100 = 正好在模型线上；"
                  "远低于 100 = 按合并的价付钱却没合并成；远高于 100 = 源不走 host 链路"),
        ("pass ms", "spaces x us，即一次轮询所有分区的时间"),
        ("max/min", "一个 pass 内各 space 时延的 max/min，接近 1 说明分区均匀"),
        ("model k=1", "单行单次 DMA 的 256 B 计费模型预测（host 路径）"),
        ("", ""),
        ("注意", "spaces 变化时每 space 的池子和每次 launch 的载荷都不变，"
                 "唯一变化是总 DDR 与 base 指针个数；所以这一列若有斜率，就是地址跨度在付钱"),
        ("dst_slack", "目的行数 / 拷贝行数。auto 是 1 个 space 用 4、多个 space 用 1；"
                      "spaces/grid 扫描时整条阶梯统一钉成 1，否则 1-space 那一点的目标区"
                      "比其余点大 4 倍，阶梯的第一级会同时变两个变量"),
        ("src numa", "源（锁页）内存的常驻页落在哪些 NUMA 节点，以及常驻了多少。"
                     "同一个 copy 换一个进程就能差 20%，差的就是这一列：扫一条 78 分区的阶梯"
                     "要在同一个进程里反复分配/释放上百 GiB 锁页内存，后面几个点的源往往已经"
                     "不是干净的一段本节点内存。想比较两条不同命令的结果，先看这一列是否一致"),
        ("isolate", "--isolate-points 时为 on：每个点单独起一个进程，点与点之间不共享"
                    "分配器状态，矩阵里的格子才等于单独跑同一条命令的结果"),
    ]
    for k, v in rows:
        ws.append([k, v])
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 100
    for row in ws.iter_rows(min_row=1):
        row[0].font = Font(bold=True)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(path)
    print(f"xlsx: wrote {path} ({len(records)} point(s))", file=sys.stderr, flush=True)


RESULT_COLUMNS = [
    "case", "row B", "spaces", "pool rows", "launch MiB", "src", "banks", "group",
    "tbl", "dst slack", "cores", "us", "ns/row", "GB/s", "model GB/s", "%host",
    "pass ms", "space min us", "space max us", "max/min", "total DDR GiB", "dst MiB",
    "src numa", "status",
]



def _measure_and_report(bench: Bench, case: Case, args, label: str) -> dict:
    """One point: verify if asked, time it, and print the row the table shows.

    Shared by the in-process loop and by --isolate-points, so a rung of a sweep
    is measured by exactly the code that measures a standalone run -- the only
    difference is what else the process has allocated.
    """
    if args.verify:
        bench.verify(sample=args.verify_rows, repeats=max(1, args.repeat_verify))
    us_min, us_med = time_case(bench, args.warmup, args.iters, args.rounds)
    gbps = bench.bytes() / (us_min * 1e-6) / 1e9 if us_min > 0 else 0.0
    # One pass = every space once. A single pass is also the only place
    # the spread between spaces shows up, and a 78-way mean would hide a
    # region that is slow for a reason of its own (NUMA placement, a
    # deeper page-table walk). Two passes: the first warms the walk.
    per_space: list[float] = []
    if case.spaces > 1:
        bench.time_spaces()
        per_space = bench.time_spaces()
    model = predict_gbps(case.row_bytes, case.group_rows, case.rows)
    lo = min(per_space) if per_space else None
    hi = max(per_space) if per_space else None
    dst_rows = case.rows * case.dst_slack * (1 if case.dst_shared else case.spaces)
    rec = {
        "case": label,
        "row B": case.row_bytes,
        "spaces": case.spaces,
        "pool rows": case.src_total,
        "launch MiB": case.rows * case.row_bytes / 2**20,
        "src": case.src,
        "banks": case.bank,
        "group": case.group_rows,
        "tbl": case.table_mode,
        "dst slack": case.dst_slack,
        "cores": case.cores,
        "us": us_min,
        "ns/row": us_min * 1000.0 / case.rows,
        "GB/s": gbps,
        "model GB/s": model,
        "%host": 100.0 * gbps / model,
        "pass ms": us_min * case.spaces / 1000.0,
        "space min us": lo,
        "space max us": hi,
        "max/min": (hi / lo) if (lo and hi) else None,
        "total DDR GiB": case.total_pool_bytes / 2**30,
        "dst MiB": dst_rows * case.row_bytes / 2**20,
        "src numa": bench.numa_hint,
        "status": "ok",
    }
    if args.json:
        # One record per point on stdout, nothing else, so a driver script
        # (host_link_scale.py, or this script's own --isolate-points) can
        # aggregate several processes. "rec" is the same row the xlsx holds.
        print(
            json.dumps(
                {
                    "label": label,
                    "src": case.src,
                    "device": case.device,
                    "rows": case.rows,
                    "src_rows": case.src_total,
                    "spaces": case.spaces,
                    "space_rows": case.src_total,
                    "total_pool_bytes": case.total_pool_bytes,
                    "dst_shared": case.dst_shared,
                    "dst_slack": case.dst_slack,
                    "pool_bytes": case.src_total * case.row_bytes,
                    "row_bytes": case.row_bytes,
                    "dtype": args.dtype,
                    "ckv_dim": case.ckv_dim,
                    "kpe_dim": case.kpe_dim,
                    "group_rows": case.group_rows,
                    "run_len": case.run_len,
                    "rotate": case.rotate,
                    "bank": case.bank,
                    "table_mode": case.table_mode,
                    "cores": case.cores,
                    "us": us_min,
                    "us_median": us_med,
                    "ns_per_row": us_min * 1000.0 / case.rows,
                    "gbps": gbps,
                    "pass_us": us_min * case.spaces,
                    "space_us_min": lo,
                    "space_us_max": hi,
                    "src_numa": bench.numa_hint,
                    "status": "ok",
                    "host_model_gbps": model,
                    "rec": rec,
                }
            ),
            flush=True,
        )
        return rec
    # ns/row is the number that matters when comparing row lengths:
    # GB/s hides the fact that the DMA is charged in 256 B granules.
    # %host is measured / host-path model at this row length, group
    # size and row count: ~100 means the run lands on the model, well
    # below 100 means it pays for a merge it never got (the whole group
    # is all-or-nothing, so run_len=1 never merges), and far above 100
    # means the source is not crossing the host link (--src npu).
    print(
        f"{label:>18} {case.rows:>7} {case.spaces:>3} {case.src_total:>7} "
        f"{case.row_bytes:>6} {case.group_rows:>5} "
        f"{case.bank:>4} {case.table_mode:>3} {case.cores:>5} "
        f"{us_min:>9.1f} {us_min * 1000.0 / case.rows:>7.1f} {gbps:>8.1f} "
        f"{100.0 * gbps / model:>5.0f}"
    )
    if us_med != us_min:
        print(f"{'':>18} median {us_med:.1f} us")
    if per_space:
        print(
            f"{'':>18} pass {case.spaces} x {us_min:.1f} us = "
            f"{us_min * case.spaces / 1000.0:.2f} ms; per-space "
            f"{lo:.1f}..{hi:.1f} us (max/min {hi / lo:.3f})"
        )
    if bench.numa_hint:
        print(f"{'':>18} src pages on {bench.numa_hint}")
    return rec


def _case_argv(case: Case, args) -> list[str]:
    """The flags that reproduce exactly one Case, for --isolate-points."""
    argv = [
        "--rows", str(case.rows),
        "--spaces", str(case.spaces),
        "--space-rows", str(case.src_total),
        "--dst-slack", str(case.dst_slack),
        "--churn-gib", str(case.churn_gib),
        "--ckv-dim", str(case.ckv_dim),
        "--kpe-dim", str(case.kpe_dim),
        "--dtype", args.dtype,
        "--src", case.src,
        "--device", str(case.device),
        "--bank", str(case.bank),
        "--group-rows", str(case.group_rows),
        "--table-mode", str(case.table_mode),
        "--cores", str(case.cores),
        "--run-len", str(case.run_len),
        "--rotate", str(case.rotate),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--rounds", str(args.rounds),
        "--verify-rows", str(args.verify_rows),
        "--repeat-verify", str(args.repeat_verify),
        "--sweep", "none",
        "--json",
    ]
    if case.dst_shared:
        argv.append("--dst-shared")
    if args.verify:
        argv.append("--verify")
    if args.host_interleave:
        argv.append("--host-interleave")
    return argv


def _run_isolated(points: list[tuple[str, Case]], args) -> tuple[list[dict], int]:
    """Measure every point in a fresh process, and keep its record.

    A sweep reuses one interpreter, so rung k inherits the pinned arena of every
    point before it. That is not a detail: 656 B rows read 25.4-28.5 GB/s as a
    rung of a 4 x 8 grid and 34.2 GB/s as a one-point run of the same flags,
    reproducibly, in both directions. Paying one interpreter start per point is
    what makes a matrix cell mean the same thing as the single command it will
    be held up against.
    """
    script = os.path.abspath(__file__)
    records: list[dict] = []
    failures = 0
    for index, (label, case) in enumerate(points, start=1):
        print(f"[{index}/{len(points)}] {label} (isolated)", file=sys.stderr, flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, script, *_case_argv(case, args)],
                capture_output=True,
                text=True,
                timeout=args.isolate_timeout * 60,
            )
        except subprocess.TimeoutExpired:
            failures += 1
            print(f"{label:>18} isolated run timed out", flush=True)
            records.append(
                {
                    "case": label,
                    "row B": case.row_bytes,
                    "spaces": case.spaces,
                    "status": "isolated run timed out",
                }
            )
            continue
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()
        rec = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("rec"), dict):
                rec = payload["rec"]
        if proc.returncode != 0 or rec is None:
            failures += 1
            tail = (proc.stdout.strip().splitlines() or ["no output"])[-1][:300]
            print(f"{label:>18} isolated run failed rc={proc.returncode}: {tail}", flush=True)
            records.append(
                {
                    "case": label,
                    "row B": case.row_bytes,
                    "spaces": case.spaces,
                    "status": f"isolated run failed rc={proc.returncode}",
                }
            )
            continue
        rec["case"] = label
        records.append(rec)
        if args.json:
            print(json.dumps(rec), flush=True)
        else:
            print(
                f"{label:>18} {case.rows:>7} {case.spaces:>3} {case.src_total:>7} "
                f"{case.row_bytes:>6} {case.group_rows:>5} "
                f"{case.bank:>4} {case.table_mode:>3} {case.cores:>5} "
                f"{rec['us']:>9.1f} {rec['ns/row']:>7.1f} {rec['GB/s']:>8.1f} "
                f"{rec['%host']:>5.0f}"
            )
    return records, failures


def main(argv=None) -> int:
    args = parse_args(argv)
    _SWEEP_OVERRIDES.clear()
    if args.sweep_values:
        if args.sweep == "none":
            raise SystemExit(
                "--sweep-values replaces a sweep's ladder, so it needs --sweep; "
                "without one there is no ladder to replace"
            )
        # --sweep grid drives the space ladder, so its override is the space list.
        name = "spaces" if args.sweep == "grid" else args.sweep
        values = (
            [t.strip() for t in args.sweep_values.split(",") if t.strip()]
            if name == "src"
            else _parse_int_list(args.sweep_values, "--sweep-values")
        )
        if not values:
            raise SystemExit("--sweep-values is empty")
        _SWEEP_OVERRIDES[name] = values
        print(
            f"note: the {name} ladder is overridden to {values}",
            file=sys.stderr,
            flush=True,
        )
    if args.host_interleave and argv is None:
        if args.src != "host":
            print("note: --host-interleave only affects --src host", file=sys.stderr)
        _reexec_interleaved(sys.argv[1:])
    _load_runtime()
    if args.list_devices:
        return _list_devices()
    # Bind the card before anything is allocated: every tensor below is created
    # on the current device, so this is the only place the die is chosen.
    _select_device(args.device)
    dtype = DTYPES[args.dtype][0]
    if args.kpe_dim < 0 or args.ckv_dim <= 0:
        raise SystemExit("--ckv-dim must be > 0 and --kpe-dim must be >= 0")
    if args.rotate < 0:
        raise SystemExit("--rotate must be >= 0")
    if args.spaces < 1:
        raise SystemExit("--spaces must be >= 1")
    # One region's size: --space-rows wins, then --src-rows, then --rows.
    space_rows = args.space_rows or args.src_rows or args.rows
    if space_rows < args.rows:
        raise SystemExit(
            f"the source region holds {space_rows} rows but --rows asks for "
            f"{args.rows}; one region must cover a single launch's copy count"
        )
    if args.src_rows and args.src_rows < args.rows:
        raise SystemExit(
            f"--src-rows ({args.src_rows}) is the size of the source pool and must be "
            f">= --rows ({args.rows}); lower --rows instead if you want a smaller copy"
        )
    if (args.src_rows or args.space_rows) and args.table_mode == 0:
        print(
            "note: --table-mode 0 addresses rows 0..rows-1 only, so neither --src-rows "
            "nor --space-rows can change the working set; use --table-mode 1 to measure "
            "the big pool",
            file=sys.stderr,
            flush=True,
        )

    base = Case(
        rows=args.rows,
        bank=args.bank,
        group_rows=args.group_rows,
        table_mode=args.table_mode,
        cores=args.cores,
        dtype=dtype,
        src=args.src,
        device=args.device,
        ckv_dim=args.ckv_dim,
        kpe_dim=args.kpe_dim,
        run_len=args.run_len,
        pool_rows=args.src_rows,
        rotate=args.rotate,
        spaces=args.spaces,
        space_rows=space_rows if (args.space_rows or args.spaces > 1) else 0,
        dst_shared=args.dst_shared,
        dst_slack=args.dst_slack,
        churn_gib=args.churn_gib,
    )
    row_bytes = base.row_bytes
    print(
        f"row: {base.ckv_dim} elems CKV"
        + (f" + {base.kpe_dim} elems KPE" if base.copy_rope else " (KPE plane off)")
        + f", {args.dtype}, {row_bytes} B/token"
        + ("" if base.run_len <= 1 else f", runs of {base.run_len} consecutive rows"),
        file=sys.stderr,
        flush=True,
    )
    # A spaces/grid sweep is only a measurement of the space count if everything
    # else is held. dst_slack is not: its auto value is 4 for one space and 1 for
    # many, so resolving it per point would give the spaces=1 rung -- and only
    # that rung -- a 4x, one-quarter-covered destination, and the ladder would
    # then move two variables between its first two points. Pin the whole ladder
    # to the many-space value, which is also the faithful one: production
    # concatenates the gathers, so every destination row is written exactly once.
    ladder_slack = args.dst_slack
    if args.sweep in ("spaces", "grid"):
        top = sweep_values("spaces")[-1]
        top_row_bytes = (
            max(_parse_int_list(args.grid_row_bytes, "--grid-row-bytes"))
            if args.sweep == "grid"
            else row_bytes
        )
        if ladder_slack <= 0:
            ladder_slack = 1
            print(
                "note: every point of a spaces/grid sweep uses --dst-slack 1; auto "
                "would give only the 1-space point a 4x destination, so its first rung "
                "would differ in two ways at once",
                file=sys.stderr,
                flush=True,
            )
        elif ladder_slack > 1:
            print(
                f"note: --dst-slack {ladder_slack} at {top} spaces needs "
                f"{top * base.rows * ladder_slack * top_row_bytes / 2**30:.0f} GiB of "
                "HBM for the destination alone",
                file=sys.stderr,
                flush=True,
            )
    if args.sweep == "src-rows":
        values = sweep_values("src-rows")
        print(
            f"pool: source rows swept {values[0]}..{values[-1]} "
            f"({values[0] * row_bytes / 2**20:.1f}..{values[-1] * row_bytes / 2**20:.1f} MiB "
            f"{'DDR' if args.src == 'host' else 'HBM'}), {base.rows} copied per launch",
            file=sys.stderr,
            flush=True,
        )
    elif base.spaces > 1:
        per_space_bytes = base.src_total * row_bytes
        dst_rows_total = base.rows * base.dst_slack * (1 if base.dst_shared else base.spaces)
        dst_bytes = dst_rows_total * row_bytes
        pass_rows = base.spaces * base.rows
        print(
            f"spaces: {base.spaces} x {base.src_total} rows, {per_space_bytes / 2**30:.2f} GiB "
            f"{'DDR' if base.src == 'host' else 'HBM'} per space, "
            f"{base.total_pool_bytes / 2**30:.1f} GiB total\n"
            f"        {base.rows} rows copied from every space "
            f"({100.0 * base.rows / base.src_total:.1f}% of a space, "
            f"{base.rows * row_bytes / 2**20:.0f} MiB per launch)\n"
            f"        one pass = {base.spaces} launches, {pass_rows} rows, "
            f"{pass_rows * row_bytes / 2**30:.2f} GiB moved, "
            f"dst {dst_bytes / 2**20:.0f} MiB"
            + (" (shared window)" if base.dst_shared else "")
            + f", tables {base.spaces * max(1, base.rotate) * base.rows * 4 / 2**20:.0f} MiB",
            file=sys.stderr,
            flush=True,
        )
        _guard_memory(base)
        print(
            "note: allocation is {:.0f} GiB, and only the rows each table points at are "
            "given payload, so setup is a malloc plus one fill per space, not a full "
            "rewrite".format(base.total_pool_bytes / 2**30),
            file=sys.stderr,
            flush=True,
        )
    elif base.src_total != base.rows:
        pool_bytes = base.src_total * row_bytes
        print(
            f"pool: {base.src_total} source rows ({pool_bytes / 2**20:.1f} MiB "
            f"{'DDR' if args.src == 'host' else 'HBM'}), {base.rows} copied per launch "
            f"({100.0 * base.rows / base.src_total:.1f}% of the pool)",
            file=sys.stderr,
            flush=True,
        )
    if args.dry_run:
        if args.sweep in ("grid", "spaces"):
            # The last space count allocates 78x what the first one does, so the
            # point of --dry-run here is to see the whole ladder before the
            # setup time is spent: one line per point, nothing allocated.
            lens = (
                _parse_int_list(args.grid_row_bytes, "--grid-row-bytes")
                if args.sweep == "grid"
                else [row_bytes]
            )
            for rb in lens:
                print(f"dry-run: {rb} B rows", file=sys.stderr, flush=True)
                for spaces in sweep_values("spaces"):
                    probe = Case(
                        **{
                            **base.__dict__,
                            "ckv_dim": rb // base.elem_size,
                            "kpe_dim": 0,
                            "spaces": spaces,
                            "space_rows": space_rows,
                            "dst_slack": ladder_slack,
                        }
                    )
                    print(_plan_line(probe), file=sys.stderr, flush=True)
                    _guard_memory(probe)
            print(
                f"dry-run: {len(lens)} row length(s) x {len(sweep_values('spaces'))} space "
                "counts, nothing was allocated",
                file=sys.stderr,
                flush=True,
            )
            return 0
        print(
            f"dry-run: {base.spaces} space(s) x {base.src_total} rows x {row_bytes} B = "
            f"{base.total_pool_bytes / 2**30:.2f} GiB source, "
            f"{base.rows} rows copied per launch; nothing was allocated",
            file=sys.stderr,
            flush=True,
        )
        return 0
    print(dma_cost_hint(row_bytes, base.group_rows, base.rows), file=sys.stderr, flush=True)
    if args.dst_slack <= 0 and args.sweep in ("none", "src", "rows", "ckv-dim", "row-bytes"):
        # The one knob whose default depends on the point: a lone run gets a 4x
        # destination, every rung of a spaces/grid ladder gets 1x. Two runs that
        # differ only in this are not the same measurement.
        print(
            f"note: --dst-slack is auto -> {base.dst_slack} destination rows per copied row "
            f"({base.rows * base.dst_slack * base.row_bytes / 2**20:.0f} MiB, only "
            f"1/{base.dst_slack} of it written). A --sweep spaces/grid point uses 1, so pass "
            "--dst-slack 1 (or 4) to both when comparing a single run with a matrix cell",
            file=sys.stderr,
            flush=True,
        )
    if base.src == "npu":
        print(
            "note: the model above is fitted on --src host. On --src npu the same row runs "
            "far faster (656 B: 90.9 GB/s, 2.6x), so %host >> 100 means 'the host link, not "
            "the DMA engine, was the limiter in the fit' -- not a broken measurement",
            file=sys.stderr,
            flush=True,
        )
    if base.rotate > 1:
        print(
            f"rows: rotating {base.rotate} pre-staged source tables, so consecutive "
            "launches read different subsets of the pool",
            file=sys.stderr,
            flush=True,
        )

    points: list[tuple[str, Case]] = []
    if args.sweep == "none":
        points.append(("baseline", base))
    elif args.sweep == "grid":
        # The row length x space count matrix. Every point keeps --space-rows, so
        # the per-space pool is identical across the grid and only the number of
        # base pointers (and the total DDR behind them) changes.
        grid_bytes = _parse_int_list(args.grid_row_bytes, "--grid-row-bytes")
        for rb in grid_bytes:
            if rb % base.elem_size:
                raise SystemExit(
                    f"--grid-row-bytes {rb} is not a whole number of {base.elem_size} B "
                    "elements; use --dtype uint8 for byte-exact row lengths"
                )
            for spaces in sweep_values("spaces"):
                case = Case(
                    **{
                        **base.__dict__,
                        "ckv_dim": rb // base.elem_size,
                        "kpe_dim": 0,
                        "spaces": spaces,
                        "space_rows": space_rows,
                        "dst_slack": ladder_slack,
                    }
                )
                points.append((f"{rb}B x{spaces}", case))
        print(
            f"grid: {len(grid_bytes)} row length(s) {grid_bytes} x "
            f"{len(sweep_values('spaces'))} space counts {sweep_values('spaces')} "
            f"= {len(points)} points, each point reallocates",
            file=sys.stderr,
            flush=True,
        )
        print(
            "      space count -> what one pass allocates and moves:",
            file=sys.stderr,
            flush=True,
        )
        for spaces in sweep_values("spaces"):
            sample = next(c for lb, c in points if c.spaces == spaces)
            print(_plan_line(sample), file=sys.stderr, flush=True)
    else:
        for value in sweep_values(args.sweep):
            if args.sweep == "row-bytes":
                # The sweep is in bytes so the numbers match the layout argument.
                # The op counts elements, and the KPE plane would add to the row,
                # so this one forces a single plane.
                if value % base.elem_size:
                    raise SystemExit(
                        f"row-bytes={value} is not a whole number of {base.elem_size} B "
                        "elements; use --dtype uint8 for byte-exact row lengths"
                    )
                case = Case(
                    **{**base.__dict__, "ckv_dim": value // base.elem_size, "kpe_dim": 0}
                )
                points.append((f"{value}B", case))
                continue
            field = SWEEP_FIELDS.get(args.sweep, args.sweep.replace("-", "_"))
            case = Case(**{**base.__dict__, field: value})
            label = f"x{value}" if args.sweep == "spaces" else f"{args.sweep}={value}"
            points.append((label, case))
    if args.sweep == "churn" and not args.isolate_points:
        raise SystemExit(
            "--sweep churn only means anything with --isolate-points: churn is not "
            "undone, so in one process rung k would start from the sum of all "
            "earlier rungs instead of from its own value"
        )
    if args.sweep in ("spaces", "grid") and ladder_slack != args.dst_slack:
        # --sweep spaces builds its points in the generic branch above, which
        # copies the base case's already-resolved dst_slack; rewrite it so the
        # whole ladder shares one destination layout.
        points = [
            (label, Case(**{**case.__dict__, "dst_slack": ladder_slack}))
            for label, case in points
        ]
    if args.sweep == "row-bytes" and base.kpe_dim > 0:
        print(
            "note: --sweep row-bytes measures the whole row in the CKV plane, so the "
            "KPE plane is off for every point",
            file=sys.stderr,
            flush=True,
        )
    if args.sweep == "dst-slack":
        biggest = max(sweep_values("dst-slack"))
        dst_bytes = (
            base.rows * biggest * base.row_bytes * (1 if base.dst_shared else base.spaces)
        )
        print(
            f"note: --sweep dst-slack spreads the writes over slack x --rows destination "
            f"rows, so at slack {biggest} this allocates {dst_bytes / 2**30:.1f} GiB of HBM "
            f"for {base.rows * base.row_bytes / 2**20:.0f} MiB of payload; slack 1 is the "
            "production shape (the per-partition gathers are concatenated, every row "
            "written exactly once)",
            file=sys.stderr,
            flush=True,
        )
    if args.sweep == "grid" and args.spaces != 1:
        print(
            f"note: --sweep grid drives its own space counts, --spaces {args.spaces} is ignored",
            file=sys.stderr,
            flush=True,
        )
    if args.sweep == "spaces" and not (args.space_rows or args.src_rows):
        print(
            "note: --space-rows was not given, so every space is only --rows big and the "
            "sweep changes the base-pointer count without growing DDR; pass --space-rows "
            "to reproduce the production pool",
            file=sys.stderr,
            flush=True,
        )

    # --sweep rows changes the copy count but the pool is fixed at --src-rows, so
    # a point can ask for more rows than the pool has; the table build would then
    # silently come up short.
    for label, case in points:
        if case.src_total < case.rows:
            raise SystemExit(
                f"{label}: --rows ({case.rows}) exceeds the source pool "
                f"({case.src_total}), the pool must cover the copy count"
            )

    # Under ~2 MiB per launch the fixed dispatch cost is a visible share of the
    # time, so GB/s reads low for reasons that have nothing to do with the copy.
    # Only for a single point: --sweep rows is the measurement of exactly this.
    if args.sweep == "none" and base.rows <= 2048:
        print(
            f"note: rows={base.rows} moves only {base.rows * row_bytes / 2**20:.1f} MiB per launch, "
            "so the fixed per-launch cost is a visible share of the latency and GB/s reads "
            "low; compare points at equal --rows (--sweep rows shows the floor)",
            file=sys.stderr,
            flush=True,
        )

    header = f"{'case':>18} {'rows':>7} {'spc':>3} {'pool':>7} {'B/row':>6} {'group':>5} {'bank':>4} {'tbl':>3} {'cores':>5} "
    header += f"{'us':>9} {'ns/row':>7} {'GB/s':>8} {'%host':>5}"
    print(header)
    print("-" * len(header))

    records: list[dict] = []
    failures = 0
    if args.isolate_points and len(points) > 1:
        # Every point is measured by _measure_and_report in a process of its own,
        # so `points` is consumed here instead of by the loop below. A single
        # point needs no isolation: it already *is* the standalone run.
        print(
            f"isolate: {len(points)} point(s), one process each "
            f"(timeout {args.isolate_timeout} min/point)",
            file=sys.stderr,
            flush=True,
        )
        records, failures = _run_isolated(points, args)
        points = []
    multi = len(points) > 1
    for index, (label, case) in enumerate(points, start=1):
        if multi:
            # A 24-point grid is mostly setup (200 GiB of pinned memory per
            # point), so the progress line says which point is paying for it.
            print(f"[{index}/{len(points)}] {label}", file=sys.stderr, flush=True)
        # Outside the try: a churn that cannot allocate is a broken setup, not a
        # measurement that failed, and it must stop the run rather than be
        # recorded as a blank cell.
        _churn_host(case.churn_gib)
        try:
            _guard_memory(case)
            bench = Bench(case)
        except (Exception, SystemExit) as exc:  # one point must not cost the sheet
            failures += 1
            print(f"{label:>18} setup failed: {exc}", flush=True)
            records.append(
                {
                    "case": label,
                    "row B": case.row_bytes,
                    "spaces": case.spaces,
                    "status": f"setup failed: {exc}",
                }
            )
            continue
        try:
            rec = _measure_and_report(bench, case, args, label)
            records.append(rec)
        except (Exception, SystemExit) as exc:  # keep the other points running
            failures += 1
            print(f"{label:>18} measurement failed: {exc}", flush=True)
            records.append(
                {
                    "case": label,
                    "row B": case.row_bytes,
                    "spaces": case.spaces,
                    "status": f"measurement failed: {exc}",
                }
            )
        finally:
            # Host sources stay registered (and the device may still be reading
            # them) until every launch has retired.
            bench.close()
            # 200 GiB per point: drop the tensors before the next point asks for
            # its own, or the second allocation fails on a machine that fit one.
            del bench
            gc.collect()

    _print_matrix(records, "GB/s", "GB/s by row length x space count", "0.1f")
    _print_matrix(records, "pass ms", "pass 时延 ms by row length x space count", "0.2f")
    if args.xlsx:
        _write_workbook(
            args.xlsx,
            records,
            {
                "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cmd": " ".join([os.path.basename(sys.argv[0]), *(list(argv) if argv else sys.argv[1:])]),
                "host": socket.gethostname(),
                "device": args.device,
                "src": args.src,
                "rows": args.rows,
                "space_rows": space_rows,
                "table_mode": args.table_mode,
                "group_rows": args.group_rows,
                "run_len": args.run_len,
                "rotate": args.rotate,
                "timing": f"warmup {args.warmup}, iters {args.iters}, rounds {args.rounds}",
                "isolate": "on" if args.isolate_points else "off",
            },
        )
    if failures:
        print(
            f"{failures}/{len(points)} point(s) failed; the rest are in the table"
            + (f" and {args.xlsx}" if args.xlsx else ""),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
