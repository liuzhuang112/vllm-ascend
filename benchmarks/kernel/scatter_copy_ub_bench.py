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

The operator must be built first, from the repository root::

    SOC_VERSION=ascend910b1 bash csrc/build_aclnn.sh
    COMPILE_CUSTOM_KERNELS=0 pip install -e . --no-build-isolation

A subset build is enough while debugging::

    VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench bash csrc/build_aclnn.sh

The benchmark fails with a clear message when ``scatter_copy_ub_bench`` is not
registered.
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


def _load_runtime():
    global torch_npu
    import torch_npu as _torch_npu

    torch_npu = _torch_npu
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    if not hasattr(torch.ops._C_ascend, "scatter_copy_ub_bench"):
        raise SystemExit(
            "_C_ascend.scatter_copy_ub_bench is not registered: the torch "
            "binding has not been rebuilt.\n"
            "  1) build the op package (VLLM_ASCEND_CUSTOM_OPS limits it to "
            "this op):\n"
            "     SOC_VERSION=ascend910b1 bash csrc/build_aclnn.sh\n"
            "  2) rebuild the extension:\n"
            "     COMPILE_CUSTOM_KERNELS=0 pip install -e . --no-build-isolation"
        )


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

    def verify(self, sample: int = 64) -> None:
        """Compare the first `sample` destination rows against the source."""
        self.run()
        torch.npu.synchronize()
        sample = min(sample, self.case.rows)
        idx = torch.arange(sample, dtype=torch.int64)
        want_kv = self._source_cpu(self.src_kv_cpu, self.src_kv_dev)[self.src_rows[idx].to(torch.int64)]
        want_rope = self._source_cpu(self.src_rope_cpu, self.src_rope_dev)[self.src_rows[idx].to(torch.int64)]
        got_kv = self.dst_kv[self.dst_rows[idx].to(torch.int64)].cpu()
        got_rope = self.dst_rope[self.dst_rows[idx].to(torch.int64)].cpu()
        torch.testing.assert_close(got_kv, want_kv, rtol=0, atol=0)
        torch.testing.assert_close(got_rope, want_rope, rtol=0, atol=0)

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
            bench.verify()
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
