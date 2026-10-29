/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
/**
 * \file scatter_copy_ub_bench.cpp
 * \brief DRAM -> HBM copy microbenchmark, extracted from
 *        fused_scatter_copy_sparse_flash_attention/op_kernel/first_fill_scatter_copy_kernel.h
 *
 * Data path (identical to the production first-fill stage):
 *
 *     DRAM src  --MTE2 DataCopyPad-->  UB bank  --MTE3 DataCopyPad-->  HBM dst
 *
 * Only the copy is kept. There is no vector compute, so the two transfers are
 * the whole pipeline: a TQue<VECIN> queue of `bank_num` slots keeps
 * `bank_num - 1` reads in flight while the oldest bank drains to HBM, which is
 * exactly the software pipeline used by FirstFillScatterCopyKernel.
 *
 * The queue is a plain VECIN queue rather than a TQueBind: the A5 (arch35)
 * kernels never use TQueBind, they allocate a VECIN buffer, fill it with MTE2,
 * write it out with MTE3 and then fence MTE3_MTE2 before releasing it. Both the
 * standalone UnionCopy() scatter of the A5 fused op and its per-token
 * source-gather path do exactly that, so this file follows them.
 *
 * Differences from the production kernel (all of them are knobs here):
 *   - bank_num / group_rows / core_num are runtime parameters instead of
 *     compile-time constants, so one binary can sweep the whole space;
 *   - group_rows > 1 coalesces up to SCB_MAX_GROUP_ROWS rows whenever their
 *     source rows are contiguous and their destination rows advance with a
 *     constant stride, the same trick the production kernel uses for adjacent
 *     miss pairs;
 *   - table_mode == 0 skips the scalar GM reads of the row tables to expose the
 *     pure transfer ceiling, table_mode == 1 keeps the production-style
 *     `GlobalTensor::GetValue` address generation.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_copy_ub_bench_tiling_data.h"

using namespace AscendC;

namespace {
constexpr int64_t SCB_ALIGN_BYTES = 32;
// Upper bounds of the runtime knobs; the tiling layer clamps to these.
constexpr uint32_t SCB_MAX_BANKS = 8;
constexpr uint32_t SCB_MAX_GROUP_ROWS = 8;

template <typename T, uint32_t BANK_COUNT>
class ScatterCopyUbBenchKernel {
public:
    __aicore__ inline ScatterCopyUbBenchKernel() = default;

    __aicore__ inline void Init(GM_ADDR srcKv, GM_ADDR srcRope, GM_ADDR srcRows, GM_ADDR dstRows,
                                GM_ADDR dstKv, GM_ADDR dstRope,
                                const ScatterCopyUbBenchTilingData &tiling, TPipe *pipe)
    {
        srcKvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcKv));
        srcRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcRope));
        dstKvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstKv));
        dstRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstRope));
        srcRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(srcRows));
        dstRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dstRows));

        pipe_ = pipe;
        numTokens_ = tiling.numTokens;
        ckvDim_ = tiling.ckvDim;
        kpeDim_ = tiling.kpeDim;
        groupRows_ = tiling.groupRows;
        bankNum_ = static_cast<uint32_t>(tiling.bankNum);
        usedCoreNum_ = static_cast<uint32_t>(tiling.usedCoreNum);
        tableMode_ = tiling.tableMode;

        // One slot holds `group_rows` CKV rows followed by the same number of
        // KPE rows, mirroring the two separate source tensors. The KPE plane
        // starts 32B aligned so both planes stay DMA friendly.
        kpeOffsetElems_ = AlignBytesToElems(groupRows_ * ckvDim_ * static_cast<int64_t>(sizeof(T)));
        const int64_t slotElems = kpeOffsetElems_ + groupRows_ * kpeDim_;
        pipe_->InitBuffer(copyQueue_, static_cast<uint8_t>(bankNum_),
                          static_cast<uint32_t>(slotElems * sizeof(T)));
    }

    __aicore__ inline void Process()
    {
        coreIdx_ = static_cast<int64_t>(GetBlockIdx());
        if (coreIdx_ >= static_cast<int64_t>(usedCoreNum_)) {
            return;
        }
        totalGroups_ = CeilDiv(numTokens_, groupRows_);
        if (coreIdx_ >= totalGroups_) {
            return;
        }
        // Group-strided ownership: core c owns groups c, c + coreNum, ... so a
        // group's rows stay adjacent and remain coalescible.
        groups_ = CeilDiv(totalGroups_ - coreIdx_, static_cast<int64_t>(usedCoreNum_));

        const int64_t lookahead = static_cast<int64_t>(bankNum_) - 1;
        for (int64_t k = 0; k < lookahead && k < groups_; ++k) {
            CopyIn(static_cast<uint32_t>(k), GroupOf(k));
        }
        for (int64_t k = 0; k < groups_; ++k) {
            if (k + lookahead < groups_) {
                CopyIn(static_cast<uint32_t>((k + lookahead) % bankNum_), GroupOf(k + lookahead));
            }
            CopyOut(static_cast<uint32_t>(k % bankNum_));
        }
    }

private:
    __aicore__ static inline int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return (divisor <= 0) ? 0 : ((value + divisor - 1) / divisor);
    }

    __aicore__ static inline int64_t AlignBytesToElems(int64_t bytes)
    {
        return CeilDiv(bytes, SCB_ALIGN_BYTES) * SCB_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
    }

    __aicore__ inline int64_t GroupOf(int64_t k) const
    {
        return coreIdx_ + k * static_cast<int64_t>(usedCoreNum_);
    }

    __aicore__ inline void CopyIn(uint32_t slot, int64_t groupIdx)
    {
        const int64_t rowStart = groupIdx * groupRows_;
        int64_t count = numTokens_ - rowStart;
        if (count > groupRows_) {
            count = groupRows_;
        }

        // Resolve this group. The read side only needs to know whether the run
        // is contiguous; the write side keeps every destination row because a
        // random scatter cannot be described by one descriptor.
        int64_t srcTok[SCB_MAX_GROUP_ROWS];
        for (int64_t r = 0; r < count; ++r) {
            const int64_t tokenIdx = rowStart + r;
            if (tableMode_ != 0) {
                srcTok[r] = static_cast<int64_t>(srcRowsGm_.GetValue(tokenIdx));
                dstTok_[slot][r] = static_cast<int64_t>(dstRowsGm_.GetValue(tokenIdx));
            } else {
                srcTok[r] = tokenIdx;
                dstTok_[slot][r] = tokenIdx;
            }
        }
        count_[slot] = count;

        bool srcContiguous = count > 1;
        for (int64_t r = 1; r < count && srcContiguous; ++r) {
            if (srcTok[r] - srcTok[r - 1] != 1) {
                srcContiguous = false;
            }
        }
        const int64_t dstStep = (count > 1) ? dstTok_[slot][1] - dstTok_[slot][0] : 0;
        bool dstStrided = (count > 1) && (dstStep >= 1);
        for (int64_t r = 2; r < count && dstStrided; ++r) {
            if (dstTok_[slot][r] - dstTok_[slot][r - 1] != dstStep) {
                dstStrided = false;
            }
        }
        coalescedIn_[slot] = srcContiguous;
        coalescedOut_[slot] = dstStrided;

        // The queue type depends on the class template parameter BANK_COUNT, so
        // the member template has to be disambiguated with `template`.
        LocalTensor<T> local = copyQueue_.template AllocTensor<T>();
        DataCopyPadExtParams<T> padParams;  // no padding, same default as production
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(ckvDim_ * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;

        if (srcContiguous) {
            // The whole group is one contiguous source run, so MTE2 issues one
            // descriptor whose length spans every row.
            params.blockLen = static_cast<uint32_t>(count * ckvDim_ * sizeof(T));
            DataCopyPad(local, srcKvGm_[srcTok[0] * ckvDim_], params, padParams);

            params.blockLen = static_cast<uint32_t>(count * kpeDim_ * sizeof(T));
            DataCopyPad(local[kpeOffsetElems_], srcRopeGm_[srcTok[0] * kpeDim_], params, padParams);
        } else {
            for (int64_t r = 0; r < count; ++r) {
                params.blockCount = 1;
                params.blockLen = static_cast<uint32_t>(ckvDim_ * sizeof(T));
                params.srcStride = 0;
                DataCopyPad(local[r * ckvDim_], srcKvGm_[srcTok[r] * ckvDim_], params, padParams);

                params.blockLen = static_cast<uint32_t>(kpeDim_ * sizeof(T));
                DataCopyPad(local[kpeOffsetElems_ + r * kpeDim_], srcRopeGm_[srcTok[r] * kpeDim_], params,
                            padParams);
            }
        }
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint32_t slot)
    {
        LocalTensor<T> local = copyQueue_.template DeQue<T>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(ckvDim_ * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;

        if (coalescedOut_[slot]) {
            const int64_t dstStep = dstTok_[slot][1] - dstTok_[slot][0];
            params.blockCount = static_cast<uint16_t>(count_[slot]);
            params.dstStride = static_cast<uint32_t>((dstStep - 1) * ckvDim_ * sizeof(T));
            DataCopyPad(dstKvGm_[dstTok_[slot][0] * ckvDim_], local, params);

            params.blockLen = static_cast<uint32_t>(kpeDim_ * sizeof(T));
            params.dstStride = static_cast<uint32_t>((dstStep - 1) * kpeDim_ * sizeof(T));
            DataCopyPad(dstRopeGm_[dstTok_[slot][0] * kpeDim_], local[kpeOffsetElems_], params);
        } else {
            for (int64_t r = 0; r < count_[slot]; ++r) {
                params.blockCount = 1;
                params.blockLen = static_cast<uint32_t>(ckvDim_ * sizeof(T));
                params.dstStride = 0;
                DataCopyPad(dstKvGm_[dstTok_[slot][r] * ckvDim_], local[r * ckvDim_], params);

                params.blockLen = static_cast<uint32_t>(kpeDim_ * sizeof(T));
                DataCopyPad(dstRopeGm_[dstTok_[slot][r] * kpeDim_],
                            local[kpeOffsetElems_ + r * kpeDim_], params);
            }
        }
        // MTE3 drains the bank asynchronously, so the queue must not hand the
        // buffer back to MTE2 before the store retires. This is the fence the
        // production A5 copy paths use before they release a VECIN buffer.
        const auto writeDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(writeDone);
        WaitFlag<HardEvent::MTE3_MTE2>(writeDone);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe *pipe_ = nullptr;
    GlobalTensor<T> srcKvGm_;
    GlobalTensor<T> srcRopeGm_;
    GlobalTensor<T> dstKvGm_;
    GlobalTensor<T> dstRopeGm_;
    GlobalTensor<int32_t> srcRowsGm_;
    GlobalTensor<int32_t> dstRowsGm_;
    // VECIN rather than TQueBind: see the file header. The element type is a
    // tiling field, so both instantiations below exist in one binary.
    TQue<QuePosition::VECIN, BANK_COUNT> copyQueue_;

    int64_t coreIdx_ = 0;
    int64_t numTokens_ = 0;
    int64_t ckvDim_ = 0;
    int64_t kpeDim_ = 0;
    int64_t groupRows_ = 1;
    int64_t totalGroups_ = 0;
    int64_t groups_ = 0;
    int64_t kpeOffsetElems_ = 0;
    int64_t tableMode_ = 0;
    uint32_t bankNum_ = 2;
    uint32_t usedCoreNum_ = 1;

    // Destination rows of each in-flight group, plus the two coalescing
    // decisions. This is the only scalar state the pipeline carries.
    int64_t dstTok_[SCB_MAX_BANKS][SCB_MAX_GROUP_ROWS];
    int64_t count_[SCB_MAX_BANKS];
    bool coalescedIn_[SCB_MAX_BANKS];
    bool coalescedOut_[SCB_MAX_BANKS];
};
}  // namespace

extern "C" __global__ __aicore__ void scatter_copy_ub_bench(GM_ADDR srcKv, GM_ADDR srcRope,
                                                            GM_ADDR srcRows, GM_ADDR dstRows,
                                                            GM_ADDR dstKv, GM_ADDR dstRope,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    // Pure MTE2/MTE3 work: ask for AIV-only so the A5 scheduler does not pair
    // the block with an idle AIC. Same macro the dual-arch AIV ops use.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ScatterCopyUbBenchTilingData);
    GET_TILING_DATA_WITH_STRUCT(ScatterCopyUbBenchTilingData, tilingData, tiling);

    TPipe pipe;
    // Both instantiations are compiled here on purpose: the element type is a
    // runtime field, so the op needs no tiling-key registration.
    if (tilingData.dataType == 2) {
        ScatterCopyUbBenchKernel<half, SCB_MAX_BANKS> op;
        op.Init(srcKv, srcRope, srcRows, dstRows, dstKv, dstRope, tilingData, &pipe);
        op.Process();
    } else {
        ScatterCopyUbBenchKernel<bfloat16_t, SCB_MAX_BANKS> op;
        op.Init(srcKv, srcRope, srcRows, dstRows, dstKv, dstRope, tilingData, &pipe);
        op.Process();
    }
}
