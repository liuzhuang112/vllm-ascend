/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
/**
 * \file scatter_copy_ub_bench_tiling.cpp
 * \brief Tiling of the DRAM -> HBM copy microbenchmark.
 *
 * All shape validation happens in the torch adapter so a misuse reports a
 * readable torch error instead of a bare GRAPH_FAILED here. Tiling only maps
 * shapes and attributes onto the kernel parameters, and clamps the knobs to the
 * compile-time bounds of the kernel.
 */

#include "scatter_copy_ub_bench_tiling.h"

#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_SRC_KV = 0;
constexpr size_t INPUT_SRC_ROPE = 1;
constexpr size_t INPUT_SRC_ROWS = 2;

constexpr int64_t MAX_BANKS = 8;
constexpr int64_t MAX_GROUP_ROWS = 8;

constexpr size_t ATTR_BANK_NUM = 0;
constexpr size_t ATTR_GROUP_ROWS = 1;
constexpr size_t ATTR_TABLE_MODE = 2;
constexpr size_t ATTR_CORE_NUM = 3;

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (divisor <= 0) ? 0 : ((value + divisor - 1) / divisor);
}

int64_t Clamp(int64_t value, int64_t lower, int64_t upper)
{
    return std::max(lower, std::min(value, upper));
}
}  // namespace

ge::graphStatus Tiling4ScatterCopyUbBench(gert::TilingContext *context)
{
    auto srcKvShapePtr = context->GetInputShape(INPUT_SRC_KV);
    auto srcRopeShapePtr = context->GetInputShape(INPUT_SRC_ROPE);
    auto srcRowsShapePtr = context->GetInputShape(INPUT_SRC_ROWS);
    auto srcKvDesc = context->GetInputDesc(INPUT_SRC_KV);
    auto attrs = context->GetAttrs();
    if (srcKvShapePtr == nullptr || srcRopeShapePtr == nullptr || srcRowsShapePtr == nullptr ||
        srcKvDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *bankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_BANK_NUM);
    const int64_t *groupRowsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_ROWS);
    const int64_t *tableModePtr = attrs->GetAttrPointer<int64_t>(ATTR_TABLE_MODE);
    const int64_t *coreNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_CORE_NUM);
    if (bankNumPtr == nullptr || groupRowsPtr == nullptr || tableModePtr == nullptr || coreNumPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto srcKvShape = srcKvShapePtr->GetStorageShape();
    const auto srcRopeShape = srcRopeShapePtr->GetStorageShape();
    const auto srcRowsShape = srcRowsShapePtr->GetStorageShape();
    if (srcKvShape.GetDimNum() < 2 || srcRopeShape.GetDimNum() < 2 || srcRowsShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens = srcRowsShape.GetDim(0);
    const int64_t ckvDim = srcKvShape.GetDim(1);
    const int64_t kpeDim = srcRopeShape.GetDim(1);
    int64_t groupRows = Clamp(*groupRowsPtr, 1, MAX_GROUP_ROWS);
    int64_t bankNum = Clamp(*bankNumPtr, 1, MAX_BANKS);
    const int64_t tableMode = (*tableModePtr != 0) ? 1 : 0;

    // Both supported element types are 16 bit. Keep the whole pipeline inside
    // the vector buffer; when the user asks for too many banks, drop banks
    // rather than fail, and report the reduction through the profile.
    constexpr int64_t ELEM_BYTES = 2;
    constexpr int64_t UB_BUDGET_BYTES = 160 * 1024;
    int64_t slotBytes = groupRows * (ckvDim + kpeDim) * ELEM_BYTES;
    slotBytes = (slotBytes + 31) / 32 * 32 + 32;  // plane alignment slack
    while (bankNum > 1 && bankNum * slotBytes > UB_BUDGET_BYTES) {
        --bankNum;
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const int64_t totalGroups = CeilDiv(numTokens, groupRows);
    int64_t blockDim = std::min<int64_t>(static_cast<int64_t>(aivNum), totalGroups);
    if (*coreNumPtr > 0) {
        blockDim = std::min<int64_t>(blockDim, *coreNumPtr);
    }
    if (blockDim <= 0) {
        blockDim = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    // The element type is a plain field instead of a tiling key: the kernel
    // then instantiates both branches unconditionally, which keeps the op free
    // of any tiling-key registration in the build.
    int64_t dataType = 0;
    if (srcKvDesc->GetDataType() == ge::DT_BF16) {
        dataType = 1;
    } else if (srcKvDesc->GetDataType() == ge::DT_FLOAT16) {
        dataType = 2;
    } else {
        return ge::GRAPH_FAILED;
    }

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < sizeof(ScatterCopyUbBenchTilingData)) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<ScatterCopyUbBenchTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->numTokens = numTokens;
    tiling->ckvDim = ckvDim;
    tiling->kpeDim = kpeDim;
    tiling->groupRows = groupRows;
    tiling->bankNum = bankNum;
    tiling->usedCoreNum = blockDim;
    tiling->tableMode = tableMode;
    tiling->dataType = dataType;
    rawTiling->SetDataSize(sizeof(ScatterCopyUbBenchTilingData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ScatterCopyUbBench(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterCopyUbBench)
    .Tiling(Tiling4ScatterCopyUbBench)
    .TilingParse<ScatterCopyUbBenchCompileInfo>(TilingPrepare4ScatterCopyUbBench);

}  // namespace optiling
