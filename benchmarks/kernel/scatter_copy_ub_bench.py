#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRAM -> HBM copy microbenchmark.

Measures only the copy stage of
``npu_fused_scatter_copy_sparse_flash_attention``:

    DRAM src --MTE2--> UB bank --MTE3--> HBM dst

One token row is 1152 B of payload at bf16/fp16: CKV (512 elems) + KPE (64
elems), split across two source/destination tensors exactly like the paged K/V
cache layout, so the per-token cost is two DataCopyPad calls per direction.

Source placement decides what is being measured:

    --src npu    source lives in HBM       -> HBM to HBM ceiling
    --src host   source is a pinned host view published to the device
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
import statistics
import sys
from dataclasses import dataclass

import torch

torch_npu = None


CKV_DIM = 512
KPE_DIM = 64
ROW_BYTES_PER_ELEM = CKV_DIM + KPE_DIM  # elements copied per token
OP_NAME = "scatter_copy_ub_bench"
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
    import os

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


def _load_runtime():
    global torch_npu
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


@dataclass
class Case:
    rows: int
    bank: int
    group_rows: int
    table_mode: int
    cores: int
    dtype: torch.dtype
    src: str


class Bench:
    """Allocates one problem size once and reuses it for every measurement."""

    def __init__(self, case: Case, seed: int = 0) -> None:
        self.case = case
        self.device = torch.device("npu:0")
        gen = torch.Generator().manual_seed(seed)
        rows = case.rows

        # Sparse topk selection is a random gather from DRAM into a random slot,
        # so both tables are full permutations. Destinations are spread over 4x
        # the row count so the scatter is not a disguised sequential copy.
        self.src_rows = torch.randperm(rows, generator=gen).to(torch.int32)
        self.dst_rows = (torch.randperm(rows * 4, generator=gen)[:rows]).to(torch.int32)

        self.src_kv_dev, self.src_kv_cpu = self._make_source(rows, CKV_DIM, gen)
        self.src_rope_dev, self.src_rope_cpu = self._make_source(rows, KPE_DIM, gen)
        self.dst_kv = torch.zeros(rows * 4, CKV_DIM, dtype=case.dtype, device=self.device)
        self.dst_rope = torch.zeros(rows * 4, KPE_DIM, dtype=case.dtype, device=self.device)

        if case.table_mode == 0:
            # Identity addressing: the kernel never reads the row tables. The
            # scatter tables are then only meaningful for contiguity, so keep
            # them linear to make this a pure copy.
            self.src_rows = torch.arange(rows, dtype=torch.int32)
            self.dst_rows = torch.arange(rows, dtype=torch.int32)

        self.src_rows_dev = self.src_rows.to(self.device)
        self.dst_rows_dev = self.dst_rows.to(self.device)

    def _make_source(self, rows: int, dim: int, gen: torch.Generator):
        values = torch.randn(rows, dim, generator=gen).to(self.case.dtype)
        if self.case.src == "npu":
            return values.to(self.device), None
        if self.case.src != "host":
            raise ValueError(f"unknown source placement: {self.case.src}")
        # Pinned host memory is registered by the torch_npu host allocator, so
        # the device address is the mapped view rather than the CPU pointer.
        host = values.pin_memory()
        view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(host)
        if view is None:
            raise SystemExit(
                "get_npu_view_from_cpu_tensor returned None: the host buffer is "
                "not registered for device access."
            )
        return view, host

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
        )

    def verify(self, sample: int = 64, repeats: int = 1) -> None:
        """Compare the copied rows against the source, and report *how* they differ.

        ``sample=0`` checks every row. ``repeats > 1`` re-runs the op and
        re-checks every time, which is the only way to catch a
        non-deterministic kernel: a launch that passes once and fails later
        points at a missing fence or uninitialised state, not at an addressing
        bug (that one fails every time, on the same rows).
        """
        total = self.case.rows
        sample = total if sample <= 0 else min(sample, total)
        idx = torch.arange(sample, dtype=torch.int64)
        src_flat = self._source_cpu(self.src_kv_cpu, self.src_kv_dev)
        rope_flat = self._source_cpu(self.src_rope_cpu, self.src_rope_dev)
        want_kv = src_flat[self.src_rows[idx].to(torch.int64)]
        want_rope = rope_flat[self.src_rows[idx].to(torch.int64)]
        dst_idx = self.dst_rows[idx].to(torch.int64)

        for attempt in range(repeats):
            self.run()
            torch.npu.synchronize()
            got_kv = self.dst_kv[dst_idx].cpu()
            got_rope = self.dst_rope[dst_idx].cpu()
            bad_kv = (got_kv != want_kv).any(dim=1)
            bad_rope = (got_rope != want_rope).any(dim=1)
            if bool(bad_kv.any()) or bool(bad_rope.any()):
                summary = self._report_mismatch(
                    attempt, repeats, bad_kv, bad_rope, got_kv, want_kv, got_rope, want_rope, src_flat
                )
                raise AssertionError(
                    f"scatter_copy_ub_bench verify failed on attempt {attempt + 1}/{repeats}: {summary}"
                )
            print(
                f"verify: attempt {attempt + 1}/{repeats} ok ({sample}/{total} rows, "
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
        total_rope = int((got_rope != want_rope).sum())
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
            rope_bad = bool((got_rope[r] != want_rope[r]).any())
            print(
                f"  sample row {r}: dst_tok={dst_tok} src_tok={src_tok}, "
                f"kv bad {cols.numel()}/{want.numel()} cols {span}, "
                f"rope {'bad' if rope_bad else 'ok'} | {hint}",
                flush=True,
            )
            if not first:
                first = (
                    f"sample row {r} dst_tok={dst_tok} src_tok={src_tok}, kv bad "
                    f"{cols.numel()}/{want.numel()} cols {span}, rope {'bad' if rope_bad else 'ok'}, {hint}"
                )
        return f"{len(rows)}/{got_kv.shape[0]} sampled rows bad; {first}"

    def _source_cpu(self, host, view):
        return host if host is not None else view.cpu()

    def bytes(self) -> int:
        return self.case.rows * ROW_BYTES_PER_ELEM * self.dst_kv.element_size()


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
    parser.add_argument("--bank", type=int, default=2, help="UB pipeline depth (1 disables overlap)")
    parser.add_argument("--group-rows", type=int, default=1, help="rows merged into one DataCopyPad")
    parser.add_argument("--table-mode", type=int, default=1, help="1 = gather/scatter via row tables, 0 = identity")
    parser.add_argument("--cores", type=int, default=0, help="AIV blocks to launch, 0 = all")
    parser.add_argument("--src", choices=("npu", "host"), default="npu", help="source placement")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
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
    _load_runtime()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    base = Case(
        rows=args.rows,
        bank=args.bank,
        group_rows=args.group_rows,
        table_mode=args.table_mode,
        cores=args.cores,
        dtype=dtype,
        src=args.src,
    )

    points: list[tuple[str, Case]] = []
    if args.sweep == "none":
        points.append(("baseline", base))
    else:
        for value in sweep_values(args.sweep):
            case = Case(**{**base.__dict__, args.sweep.replace("-", "_"): value})
            points.append((f"{args.sweep}={value}", case))

    header = f"{'case':>18} {'rows':>7} {'group':>5} {'bank':>4} {'tbl':>3} {'cores':>5} "
    header += f"{'us':>9} {'GB/s':>8}"
    print(header)
    print("-" * len(header))
    for label, case in points:
        bench = Bench(case)
        if args.verify:
            bench.verify(sample=args.verify_rows, repeats=max(1, args.repeat_verify))
        us_min, us_med = time_case(bench, args.warmup, args.iters, args.rounds)
        gbps = bench.bytes() / (us_min * 1e-6) / 1e9 if us_min > 0 else 0.0
        print(
            f"{label:>18} {case.rows:>7} {case.group_rows:>5} {case.bank:>4} "
            f"{case.table_mode:>3} {case.cores:>5} {us_min:>9.1f} {gbps:>8.1f}"
        )
        if us_med != us_min:
            print(f"{'':>18} median {us_med:.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
