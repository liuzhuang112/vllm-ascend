/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef SCATTER_COPY_UB_BENCH_TILING_H
#define SCATTER_COPY_UB_BENCH_TILING_H

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ScatterCopyUbBenchTilingData)
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(int64_t, ckvDim);
TILING_DATA_FIELD_DEF(int64_t, kpeDim);
TILING_DATA_FIELD_DEF(int64_t, groupRows);
TILING_DATA_FIELD_DEF(int64_t, bankNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tableMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterCopyUbBench, ScatterCopyUbBenchTilingData)

struct ScatterCopyUbBenchCompileInfo {};
}  // namespace optiling

#endif  // SCATTER_COPY_UB_BENCH_TILING_H
