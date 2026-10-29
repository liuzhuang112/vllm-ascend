/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef SCATTER_COPY_UB_BENCH_TILING_H
#define SCATTER_COPY_UB_BENCH_TILING_H

#include <cstdint>

// The struct lives next to the kernel so that the device compiler can include
// the very same definition; the host includes it too and fills it in place.
#include "../op_kernel/scatter_copy_ub_bench_tiling_data.h"

namespace optiling {

struct ScatterCopyUbBenchCompileInfo {};

}  // namespace optiling

#endif  // SCATTER_COPY_UB_BENCH_TILING_H
