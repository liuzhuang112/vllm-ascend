/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
/**
 * \file scatter_copy_ub_bench_tiling_data.h
 * \brief Tiling data struct, shared verbatim by host and kernel.
 *
 * Deliberately a plain POD struct (same style as
 * csrc/scripts/opgen/template/add/op_kernel/add_example_tiling_data.h): the
 * host fills it through gert::TilingContext::GetTilingData<T>(), the kernel
 * reads it through GET_TILING_DATA_WITH_STRUCT. No BEGIN_TILING_DATA_DEF /
 * REGISTER_TILING_DATA_CLASS machinery is involved, so both sides see exactly
 * the same layout without any tiling-key registration.
 */

#ifndef __SCATTER_COPY_UB_BENCH_TILING_DATA_H__
#define __SCATTER_COPY_UB_BENCH_TILING_DATA_H__

#include <stdint.h>

struct ScatterCopyUbBenchTilingData {
    int64_t numTokens;   // rows to move (src_rows length)
    int64_t ckvDim;      // elements per row in src_kv / dst_kv
    int64_t kpeDim;      // elements per row in src_rope / dst_rope
    int64_t groupRows;   // rows merged into one DMA
    int64_t bankNum;     // UB slots of the TQueBind pipeline
    int64_t usedCoreNum; // AIV blocks to launch
    int64_t tableMode;   // 0 = identity rows, 1 = read the row tables
    int64_t dataType;    // 1 = bf16, 2 = fp16
};

#endif
