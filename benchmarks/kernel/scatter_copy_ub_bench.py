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

    # DeepSeek-style fp8 MLA cache: one 656 B row, no separate rope plane
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 \
        --ckv-dim 656 --kpe-dim 0 --dtype fp8

    # the padded variant of the same row, for comparison: 672 B is a whole
    # number of 32 B units, so it keeps the coalescing paths alive
    python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 \
        --ckv-dim 672 --kpe-dim 0 --dtype fp8 --sweep group-rows

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
import ctypes
import json
import os
import shutil
import statistics
import sys
from dataclasses import dataclass
from functools import cache

import torch

torch_npu = None


CKV_DIM = 512
KPE_DIM = 64
OP_NAME = "scatter_copy_ub_bench"
# Element size of each supported cache dtype, in bytes. fp8 has no arithmetic
# here -- the op moves bytes -- so uint8 is accepted as the same instantiation.
DTYPES = {"bf16": (torch.bfloat16, 2), "fp16": (torch.float16, 2), "uint8": (torch.uint8, 1)}
if hasattr(torch, "float8_e4m3fn"):
    DTYPES["fp8"] = (torch.float8_e4m3fn, 1)
# ``torch.pin_memory()`` only locks pages; it does not publish the range to the
# device. ``aclrtHostRegisterV2`` does, and the payload is then aligned to this
# boundary -- engram/npu.py aligns to 64 KiB because unaligned device reads of a
# host range were observed to return wrong data on A3 / CANN 9.2.
HOST_ALIGN_BYTES = 64 * 1024
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
    the verifier can tell a torn row from a row that was never written. Both
    1-byte dtypes are therefore filled with raw bytes and reinterpreted: fp8 has
    no randn/compare kernel on every torch build, and a NaN-heavy payload would
    still compare unequal if it went through float arithmetic.
    """
    if _elem_size(dtype) == 1:
        raw = torch.randint(0, 256, (rows, dim), generator=gen, dtype=torch.uint8)
        return raw.view(dtype) if dtype is not torch.uint8 else raw
    return torch.randn(rows, dim, generator=gen).to(dtype)


def _comparable(t: torch.Tensor) -> torch.Tensor:
    """View 1-byte dtypes as uint8 so comparisons never touch fp8 arithmetic."""
    return t.view(torch.uint8) if t.element_size() == 1 else t


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
        # one die therefore touches one port at most.
        torch.npu.set_device(case.device)
        self.device = torch.device(f"npu:{case.device}")
        self._host_regions: list[tuple[int, int, torch.Tensor]] = []
        gen = torch.Generator().manual_seed(seed)
        rows = case.rows

        # Sparse topk selection is a random gather from DRAM into a random slot,
        # so both tables are full permutations. Destinations are spread over 4x
        # the row count so the scatter is not a disguised sequential copy.
        self.src_rows = torch.randperm(rows, generator=gen).to(torch.int32)
        self.dst_rows = (torch.randperm(rows * 4, generator=gen)[:rows]).to(torch.int32)

        self.src_kv_dev, self.src_kv_cpu = self._make_source(rows, case.ckv_dim, gen)
        self.dst_kv = torch.zeros(rows * 4, case.ckv_dim, dtype=case.dtype, device=self.device)
        if case.copy_rope:
            self.src_rope_dev, self.src_rope_cpu = self._make_source(rows, case.kpe_dim, gen)
            self.dst_rope = torch.zeros(rows * 4, case.kpe_dim, dtype=case.dtype, device=self.device)
        else:
            # copy_rope=0: the rope tensors are still required by the op schema
            # but the kernel never touches them, so a one-element placeholder on
            # the right device is all it takes. A [rows, 0] tensor would be the
            # semantically cleaner argument, but zero-sized inputs are rejected
            # by some graph paths.
            self.src_rope_dev = torch.zeros(1, 1, dtype=case.dtype, device=self.device)
            self.src_rope_cpu = None
            self.dst_rope = torch.zeros(1, 1, dtype=case.dtype, device=self.device)

        if case.table_mode == 0:
            # Identity addressing: the kernel never reads the row tables. The
            # scatter tables are then only meaningful for contiguity, so keep
            # them linear to make this a pure copy.
            self.src_rows = torch.arange(rows, dtype=torch.int32)
            self.dst_rows = torch.arange(rows, dtype=torch.int32)

        self.src_rows_dev = self.src_rows.to(self.device)
        self.dst_rows_dev = self.dst_rows.to(self.device)

    def _make_source(self, rows: int, dim: int, gen: torch.Generator):
        values = _random_values(rows, dim, self.case.dtype, gen)
        if self.case.src == "npu":
            return values.to(self.device), None
        if self.case.src != "host":
            raise ValueError(f"unknown source placement: {self.case.src}")
        return self._publish_host_source(values, rows, dim)

    def _publish_host_source(self, values: torch.Tensor, rows: int, dim: int):
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
        elt = values.element_size()
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
        payload.copy_(values)

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

    def run(self) -> None:
        torch.ops._C_ascend.scatter_copy_ub_bench(
            self.src_kv_dev,
            self.src_rope_dev,
            self.src_rows_dev,
            self.dst_rows_dev,
            self.dst_kv,
            self.dst_rope,
            self.case.bank,
            self.case.group_rows,
            self.case.table_mode,
            self.case.cores,
            int(self.case.copy_rope),
        )

    def verify(self, sample: int = 64, repeats: int = 1) -> None:
        """Compare the copied rows against the source, and report *how* they differ.

        ``sample=0`` checks every row. ``repeats > 1`` re-runs the op and
        re-checks every time, which is the only way to catch a
        non-deterministic kernel: a launch that passes once and fails later
        points at a missing fence or uninitialised state, not at an addressing
        bug (that one fails every time, on the same rows).

        With ``kpe_dim == 0`` the rope plane is never written, so only the CKV
        plane is checked; the placeholder rope tensors hold no reference data.
        """
        total = self.case.rows
        sample = total if sample <= 0 else min(sample, total)
        idx = torch.arange(sample, dtype=torch.int64)
        src_flat = _comparable(self._source_cpu(self.src_kv_cpu, self.src_kv_dev))
        want_kv = src_flat[self.src_rows[idx].to(torch.int64)]
        dst_idx = self.dst_rows[idx].to(torch.int64)
        check_rope = self.case.copy_rope
        if check_rope:
            rope_flat = _comparable(self._source_cpu(self.src_rope_cpu, self.src_rope_dev))
            want_rope = rope_flat[self.src_rows[idx].to(torch.int64)]

        for attempt in range(repeats):
            self.run()
            torch.npu.synchronize()
            got_kv = _comparable(self.dst_kv[dst_idx].cpu())
            bad_kv = (got_kv != want_kv).any(dim=1)
            if check_rope:
                got_rope = _comparable(self.dst_rope[dst_idx].cpu())
                bad_rope = (got_rope != want_rope).any(dim=1)
            else:
                got_rope = want_rope = None
                bad_rope = torch.zeros_like(bad_kv)
            if bool(bad_kv.any()) or bool(bad_rope.any()):
                summary = self._report_mismatch(
                    attempt, repeats, bad_kv, bad_rope, got_kv, want_kv, got_rope, want_rope, src_flat
                )
                raise AssertionError(
                    f"scatter_copy_ub_bench verify failed on attempt {attempt + 1}/{repeats}: {summary}"
                )
            print(
                f"verify: attempt {attempt + 1}/{repeats} ok ({sample}/{total} rows, "
                f"{self.case.ckv_dim}+{self.case.kpe_dim} elems x{self.case.elem_size}B, "
                f"{self.case.bank} banks, group {self.case.group_rows}, table_mode {self.case.table_mode})",
                flush=True,
            )

    def _report_mismatch(self, attempt, repeats, bad_kv, bad_rope, got_kv, want_kv, got_rope, want_rope,
                         src_flat) -> str:
        """Print one block per bad row and return a one-line summary.

        The summary is also embedded in the raised AssertionError, so the
        diagnosis survives a log that keeps only the traceback.
        """
        rows = torch.nonzero(bad_kv | bad_rope).flatten().tolist()
        total_kv = int((got_kv != want_kv).sum())
        total_rope = int((got_rope != want_rope).sum()) if got_rope is not None else 0
        print(
            f"\nverify: attempt {attempt + 1}/{repeats} mismatched {len(rows)}/{got_kv.shape[0]} sampled rows "
            f"(kv {total_kv} elems, rope {total_rope} elems; first rows {rows[:8]})",
            flush=True,
        )
        first = ""
        for r in rows[:4]:
            dst_tok = int(self.dst_rows[r])
            src_tok = int(self.src_rows[r])
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
    parser.add_argument("--rows", type=int, default=2048, help="token rows to copy per launch")
    parser.add_argument(
        "--ckv-dim",
        type=int,
        default=CKV_DIM,
        help=f"elements per row in the CKV plane (default {CKV_DIM}); multiplied by the "
        "dtype size to get the row bytes, e.g. 656 fp8 bytes for the DeepSeek-style "
        "MLA cache",
    )
    parser.add_argument(
        "--kpe-dim",
        type=int,
        default=KPE_DIM,
        help=f"elements per row in the KPE plane (default {KPE_DIM}); 0 disables the plane "
        "entirely (copy_rope=0), which is the fp8 layout where the rope part is part of "
        "the 656 B row",
    )
    parser.add_argument("--bank", type=int, default=2, help="UB pipeline depth (1 disables overlap)")
    parser.add_argument("--group-rows", type=int, default=1, help="rows merged into one DataCopyPad")
    parser.add_argument("--table-mode", type=int, default=1, help="1 = gather/scatter via row tables, 0 = identity")
    parser.add_argument("--cores", type=int, default=0, help="AIV blocks to launch, 0 = all")
    parser.add_argument("--src", choices=("npu", "host"), default="npu", help="source placement")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU index to run on (default: 0). Use different indices per process when "
        "measuring how many host links a node has: topo groups them by socket",
    )
    parser.add_argument(
        "--host-interleave",
        action="store_true",
        help="with --src host, restart under numactl --interleave=all so the source pages "
        "span every NUMA node (one host link per node: this is what uses both UB ports)",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bf16",
        help="cache element type; fp8 and uint8 are copied byte-wide, so they measure the "
        "same DMA path (uint8 is the escape hatch when no fp8 tensor factory is available)",
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
        choices=("none", "group-rows", "bank", "table-mode", "cores", "src"),
        default="none",
        help="iterate one knob instead of measuring a single point",
    )
    return parser.parse_args(argv)


def sweep_values(name: str) -> list:
    return {
        "group-rows": [1, 2, 4, 8],
        "bank": [1, 2, 3, 4, 6, 8],
        "table-mode": [0, 1],
        "cores": [8, 16, 24, 32, 48],
        "src": ["npu", "host"],
    }[name]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.host_interleave and argv is None:
        if args.src != "host":
            print("note: --host-interleave only affects --src host", file=sys.stderr)
        _reexec_interleaved(sys.argv[1:])
    _load_runtime()
    dtype = DTYPES[args.dtype][0]
    if args.kpe_dim < 0 or args.ckv_dim <= 0:
        raise SystemExit("--ckv-dim must be > 0 and --kpe-dim must be >= 0")

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
    )
    row_bytes = base.row_bytes
    print(
        f"row: {base.ckv_dim} elems CKV"
        + (f" + {base.kpe_dim} elems KPE" if base.copy_rope else " (KPE plane off)")
        + f", {args.dtype}, {row_bytes} B/token"
        + ("" if row_bytes % 32 == 0 else ", not a multiple of 32 B -> per-row DMA only"),
        file=sys.stderr,
        flush=True,
    )

    points: list[tuple[str, Case]] = []
    if args.sweep == "none":
        points.append(("baseline", base))
    else:
        for value in sweep_values(args.sweep):
            case = Case(**{**base.__dict__, args.sweep.replace("-", "_"): value})
            points.append((f"{args.sweep}={value}", case))

    header = f"{'case':>18} {'rows':>7} {'B/row':>6} {'group':>5} {'bank':>4} {'tbl':>3} {'cores':>5} "
    header += f"{'us':>9} {'GB/s':>8}"
    print(header)
    print("-" * len(header))
    for label, case in points:
        bench = Bench(case)
        try:
            if args.verify:
                bench.verify(sample=args.verify_rows, repeats=max(1, args.repeat_verify))
            us_min, us_med = time_case(bench, args.warmup, args.iters, args.rounds)
            gbps = bench.bytes() / (us_min * 1e-6) / 1e9 if us_min > 0 else 0.0
            if args.json:
                # One record per point on stdout, nothing else, so a driver script
                # (host_link_scale.py) can aggregate several processes.
                print(
                    json.dumps(
                        {
                            "label": label,
                            "src": case.src,
                            "device": case.device,
                            "rows": case.rows,
                            "row_bytes": case.row_bytes,
                            "dtype": args.dtype,
                            "ckv_dim": case.ckv_dim,
                            "kpe_dim": case.kpe_dim,
                            "group_rows": case.group_rows,
                            "bank": case.bank,
                            "table_mode": case.table_mode,
                            "cores": case.cores,
                            "us": us_min,
                            "us_median": us_med,
                            "gbps": gbps,
                        }
                    ),
                    flush=True,
                )
            else:
                print(
                    f"{label:>18} {case.rows:>7} {case.row_bytes:>6} {case.group_rows:>5} "
                    f"{case.bank:>4} {case.table_mode:>3} {case.cores:>5} "
                    f"{us_min:>9.1f} {gbps:>8.1f}"
                )
                if us_med != us_min:
                    print(f"{'':>18} median {us_med:.1f} us")
        finally:
            # Host sources stay registered (and the device may still be reading
            # them) until every launch has retired.
            bench.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
