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
 * the whole pipeline: a queue of `bank_num` slots keeps `bank_num - 1` reads in
 * flight while the oldest bank drains to HBM, which is exactly the software
 * pipeline used by FirstFillScatterCopyKernel.
 *
 * The queue is a TQueBind<VECIN, VECOUT>: the buffer is a VECIN input for MTE2
 * and a VECOUT output for MTE3, so releasing it inserts the MTE3_MTE2 edge.
 * That edge is the whole correctness requirement here -- with a bare
 * TQue<VECIN> the release would wait on V_MTE2 ("vector consumed it"), which
 * retires immediately in a kernel that never issues a vector op, so MTE2 could
 * refill a bank while MTE3 was still draining it. Both A5 (arch35) copy paths
 * in this repo use TQueBind for the same reason: moe/scatter_nd_update_sk and
 * the production first_fill_scatter_copy kernel.
 *
 * Row layout. Two planes are copied per token, exactly like the paged K/V cache
 * the production op walks: a CKV plane (ckv_dim elements) and a KPE plane
 * (kpe_dim elements). Either plane may be byte-sized (uint8, i.e. a 1-byte
 * cache such as a quantised MLA row), and the KPE plane may be switched off
 * entirely (copy_rope = 0), which is the DeepSeek-style layout where rope lives
 * inside the single 656 B row.
 *
 * Differences from the production kernel (all of them are knobs here):
 *   - bank_num / group_rows / core_num are runtime parameters instead of
 *     compile-time constants, so one binary can sweep the whole space;
 *   - group_rows > 1 coalesces up to SCB_MAX_GROUP_ROWS rows: a group is split
 *     into runs of rows that are consecutive on both sides, and every run is one
 *     descriptor per direction, so partial adjacency still pays off. When a whole
 *     group qualifies, the write side picks between one contiguous descriptor and
 *     one strided descriptor (see GroupMode);
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

/**
 * How one group's rows are laid out in UB and moved by DMA.
 *
 * The distinction only matters when a row is not a multiple of 32 B: a 656 B
 * row cannot be placed at `r * 656` in UB, because DataCopyPad requires a 32 B
 * aligned LocalTensor. GROUP_MODE_ROWS therefore pads every run to the next
 * 32 B (a single 656 B row -> 672) and moves one run per DMA, while the two
 * coalesced modes keep the rows tight and never address one row individually,
 * so they work for any row length -- but they can only be used when the rows
 * really are consecutive on both sides.
 *
 * GROUP_MODE_ROWS is the default and is *not* one DMA per row: a 656 B 8-row
 * group with rows 0..3 and 4..7 adjacent arrives as two runs of four, i.e. two
 * DMAs of 2624 B rather than eight of 656 B. That is where nearly all of the
 * 656 B row's headroom comes from -- the DMA is charged per 256 B granule, so
 * 2624 B costs 11 granules for 4 rows (16.0 ns/row) against 3 granules plus a
 * tail surcharge for one row (20.4 ns/row).
 */
enum GroupMode : int64_t {
    GROUP_MODE_ROWS = 0,        // one descriptor per run of adjacent rows
    GROUP_MODE_CONTIGUOUS = 1,  // whole group is one tight copy in and out
    GROUP_MODE_STRIDED = 2,     // tight read, strided write (dst step > 1)
};

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
        copyRope_ = (tiling.copyRope != 0) && (tiling.kpeDim > 0);
        kpeDim_ = copyRope_ ? tiling.kpeDim : 0;
        groupRows_ = tiling.groupRows;
        bankNum_ = static_cast<uint32_t>(tiling.bankNum);
        usedCoreNum_ = static_cast<uint32_t>(tiling.usedCoreNum);
        tableMode_ = tiling.tableMode;

        // UB row stride, not the row length: a 656 B row occupies 672 B so that
        // the next row still starts on a 32 B boundary. For every row that is
        // already a multiple of 32 B (bf16/fp16 caches) this is a no-op.
        ubRowElems_ = AlignBytesToElems(ckvDim_ * static_cast<int64_t>(sizeof(T)));
        kpeUbRowElems_ = copyRope_ ? AlignBytesToElems(kpeDim_ * static_cast<int64_t>(sizeof(T))) : 0;
        // The KPE plane starts on the 32 B boundary after `groupRows_` CKV slots,
        // so both planes stay DMA friendly in every mode.
        kpeOffsetElems_ = ubRowElems_ * groupRows_;
        const int64_t slotElems = kpeOffsetElems_ + kpeUbRowElems_ * groupRows_;
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
        // random scatter cannot be described by one strided descriptor.
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

        // A 32 B multiple per row is what lets the tight UB rows be described by
        // one strided descriptor; it is not needed by the fully contiguous mode,
        // which never addresses a single row.
        int64_t mode = GROUP_MODE_ROWS;
        if (srcContiguous && dstStrided) {
            if (dstStep == 1) {
                mode = GROUP_MODE_CONTIGUOUS;
            } else if (rowAlignable()) {
                mode = GROUP_MODE_STRIDED;
            }
        }
        mode_[slot] = mode;

        // Split the group into runs of rows that are consecutive on both sides,
        // so the ROWS mode still merges whatever adjacency this group happens to
        // have. The previous all-or-nothing test threw the whole group away as
        // soon as one pair broke the run: with 4-row topk runs and group_rows=8
        // it degraded to eight single-row DMAs and measured 31.7 GB/s against
        // 40.7 at group_rows=4.
        //
        // Offsets are accumulated in UB element units and every run starts on a
        // 32 B boundary, which is what DataCopyPad requires. Because
        // align32(len * rowBytes) <= len * align32(rowBytes), the packed runs
        // never exceed the per-row layout the tiling budgeted for; when every run
        // holds one row the offsets coincide with the old `r * ubRowElems_`.
        int64_t runCount = 0;
        if (mode == GROUP_MODE_ROWS) {
            int64_t runBegin = 0;
            int64_t ubOffset = 0;
            int64_t kpeUbOffset = 0;
            while (runBegin < count && runCount < static_cast<int64_t>(SCB_MAX_GROUP_ROWS)) {
                int64_t runEnd = runBegin + 1;
                while (runEnd < count && srcTok[runEnd] == srcTok[runEnd - 1] + 1 &&
                       dstTok_[slot][runEnd] == dstTok_[slot][runEnd - 1] + 1) {
                    ++runEnd;
                }
                const int64_t len = runEnd - runBegin;
                runStart_[slot][runCount] = runBegin;
                runLen_[slot][runCount] = len;
                runUb_[slot][runCount] = ubOffset;
                runKpeUb_[slot][runCount] = kpeUbOffset;
                ubOffset += AlignedSpanElems(len, ckvDim_);
                kpeUbOffset += AlignedSpanElems(len, kpeDim_);
                ++runCount;
                runBegin = runEnd;
            }
        }
        runCount_[slot] = runCount;

        // The queue type depends on the class template parameter BANK_COUNT, so
        // the member template has to be disambiguated with `template`.
        LocalTensor<T> local = copyQueue_.template AllocTensor<T>();
        // Spell every field out. DataCopyPadExtParams does have a default
        // constructor that zero-initialises, but relying on it means a future
        // CANN that drops the constructor silently leaves isPad at whatever the
        // UB stack held, and a stray isPad=true turns the transfer into padded
        // mode with random left/right padding. Every other kernel in this repo
        // passes the four fields explicitly; do the same.
        DataCopyPadExtParams<T> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                          static_cast<T>(0)};
        // DataCopyExtParams is not an aggregate in CANN 9.2.0: it declares
        //   DataCopyExtParams(uint16_t blockCount, uint32_t blockLen,
        //                     uint32_t srcStride, uint32_t dstStride, uint32_t rsv)
        // so all five arguments must be given (the fifth is the reserved field
        // that kernels in this repo also pass explicitly as 0).
        DataCopyExtParams params{1, 0, 0, 0, 0};
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(ckvDim_ * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;

        if (mode != GROUP_MODE_ROWS) {
            // The whole group is one contiguous source run, so MTE2 issues one
            // descriptor whose length spans every row -- and the rows land in
            // UB back to back, which is what the two coalesced copy-out modes
            // expect.
            params.blockLen = static_cast<uint32_t>(count * ckvDim_ * sizeof(T));
            DataCopyPad(local, srcKvGm_[srcTok[0] * ckvDim_], params, padParams);

            if (copyRope_) {
                params.blockLen = static_cast<uint32_t>(count * kpeDim_ * sizeof(T));
                DataCopyPad(local[kpeOffsetElems_], srcRopeGm_[srcTok[0] * kpeDim_], params, padParams);
            }
        } else {
            // One descriptor per run. len == 1 is the plain per-row case, so this
            // loop also covers the fully scattered group.
            for (int64_t j = 0; j < runCount; ++j) {
                const int64_t len = runLen_[slot][j];
                const int64_t runSrcTok = srcTok[runStart_[slot][j]];
                params.blockCount = 1;
                params.blockLen = static_cast<uint32_t>(len * ckvDim_ * sizeof(T));
                params.srcStride = 0;
                params.dstStride = 0;
                DataCopyPad(local[runUb_[slot][j]], srcKvGm_[runSrcTok * ckvDim_], params, padParams);

                if (copyRope_) {
                    params.blockLen = static_cast<uint32_t>(len * kpeDim_ * sizeof(T));
                    DataCopyPad(local[kpeOffsetElems_ + runKpeUb_[slot][j]], srcRopeGm_[runSrcTok * kpeDim_],
                                params, padParams);
                }
            }
        }
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint32_t slot)
    {
        LocalTensor<T> local = copyQueue_.template DeQue<T>();
        // five arguments: blockCount, blockLen, srcStride, dstStride, rsv
        DataCopyExtParams params{1, 0, 0, 0, 0};
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(ckvDim_ * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        const int64_t count = count_[slot];

        if (mode_[slot] == GROUP_MODE_CONTIGUOUS) {
            // src and dst rows advance by one on both sides: the whole group is
            // a single tight transfer in each direction.
            params.blockLen = static_cast<uint32_t>(count * ckvDim_ * sizeof(T));
            DataCopyPad(dstKvGm_[dstTok_[slot][0] * ckvDim_], local, params);

            if (copyRope_) {
                params.blockLen = static_cast<uint32_t>(count * kpeDim_ * sizeof(T));
                DataCopyPad(dstRopeGm_[dstTok_[slot][0] * kpeDim_], local[kpeOffsetElems_], params);
            }
        } else if (mode_[slot] == GROUP_MODE_STRIDED) {
            const int64_t dstStep = dstTok_[slot][1] - dstTok_[slot][0];
            params.blockCount = static_cast<uint16_t>(count);
            params.blockLen = static_cast<uint32_t>(ckvDim_ * sizeof(T));
            // Zero gap in UB (the rows were read tight), constant gap in GM.
            params.srcStride = 0;
            params.dstStride = static_cast<uint32_t>((dstStep - 1) * ckvDim_ * sizeof(T));
            DataCopyPad(dstKvGm_[dstTok_[slot][0] * ckvDim_], local, params);

            if (copyRope_) {
                params.blockLen = static_cast<uint32_t>(kpeDim_ * sizeof(T));
                params.dstStride = static_cast<uint32_t>((dstStep - 1) * kpeDim_ * sizeof(T));
                DataCopyPad(dstRopeGm_[dstTok_[slot][0] * kpeDim_], local[kpeOffsetElems_], params);
            }
        } else {
            for (int64_t j = 0; j < runCount_[slot]; ++j) {
                const int64_t len = runLen_[slot][j];
                const int64_t runDstTok = dstTok_[slot][runStart_[slot][j]];
                params.blockCount = 1;
                params.blockLen = static_cast<uint32_t>(len * ckvDim_ * sizeof(T));
                params.srcStride = 0;
                params.dstStride = 0;
                DataCopyPad(dstKvGm_[runDstTok * ckvDim_], local[runUb_[slot][j]], params);

                if (copyRope_) {
                    params.blockLen = static_cast<uint32_t>(len * kpeDim_ * sizeof(T));
                    DataCopyPad(dstRopeGm_[runDstTok * kpeDim_], local[kpeOffsetElems_ + runKpeUb_[slot][j]],
                                params);
                }
            }
        }
        // FreeTensor returns the bank to MTE2 only after the TQueBind has
        // fenced MTE3_MTE2 (VECOUT consumer -> VECIN producer), so no manual
        // SetFlag/WaitFlag is needed here -- and none is used on purpose: a
        // hand-fetched event id can collide with the one the queue reserves
        // for the same HardEvent.
        copyQueue_.FreeTensor(local);
    }

    // UB span of `rows` rows of `dim` elements, rounded up to the next 32 B unit:
    // the following run has to start on a 32 B boundary, and the tiling budget
    // was computed with exactly this rounding.
    __aicore__ inline int64_t AlignedSpanElems(int64_t rows, int64_t dim) const
    {
        return AlignBytesToElems(rows * dim * static_cast<int64_t>(sizeof(T)));
    }

    // Both planes must consist of whole 32 B units for the strided copy-out:
    // only then are the tight UB rows still 32 B apart in the descriptor sense.
    __aicore__ inline bool rowAlignable() const
    {
        const int64_t ckvBytes = ckvDim_ * static_cast<int64_t>(sizeof(T));
        if ((ckvBytes % SCB_ALIGN_BYTES) != 0) {
            return false;
        }
        if (copyRope_ && ((kpeDim_ * static_cast<int64_t>(sizeof(T))) % SCB_ALIGN_BYTES) != 0) {
            return false;
        }
        return true;
    }

private:
    TPipe *pipe_ = nullptr;
    GlobalTensor<T> srcKvGm_;
    GlobalTensor<T> srcRopeGm_;
    GlobalTensor<T> dstKvGm_;
    GlobalTensor<T> dstRopeGm_;
    GlobalTensor<int32_t> srcRowsGm_;
    GlobalTensor<int32_t> dstRowsGm_;
    // VECIN buffers are released by TQueBind on the MTE3_MTE2 edge, because the
    // dst position is VECOUT: that is the sync this pipeline needs. A bare
    // TQue<QuePosition::VECIN> would release on V_MTE2 instead ("vector has
    // consumed the buffer"), and with no vector op in this kernel that flag
    // retires immediately, letting MTE2 overwrite a bank MTE3 is still
    // draining. Both A5 (arch35) copy paths in this repo --
    // moe/scatter_nd_update_sk and the production first_fill_scatter_copy
    // kernel -- therefore use TQueBind<VECIN, VECOUT> here, and neither adds a
    // manual SetFlag/WaitFlag pair.
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BANK_COUNT> copyQueue_;

    int64_t coreIdx_ = 0;
    int64_t numTokens_ = 0;
    int64_t ckvDim_ = 0;
    int64_t kpeDim_ = 0;
    int64_t groupRows_ = 1;
    int64_t totalGroups_ = 0;
    int64_t groups_ = 0;
    int64_t ubRowElems_ = 0;
    int64_t kpeUbRowElems_ = 0;
    int64_t kpeOffsetElems_ = 0;
    int64_t tableMode_ = 0;
    bool copyRope_ = false;
    uint32_t bankNum_ = 2;
    uint32_t usedCoreNum_ = 1;

    // Destination rows of each in-flight group, the layout decision, and the
    // row count. This is the only scalar state the pipeline carries.
    int64_t dstTok_[SCB_MAX_BANKS][SCB_MAX_GROUP_ROWS];
    int64_t count_[SCB_MAX_BANKS];
    int64_t mode_[SCB_MAX_BANKS];
    // Run decomposition of each in-flight group (ROWS mode): run j starts at
    // dstTok_[slot][runStart_[slot][j]], is runLen_[slot][j] rows long, and lives
    // at runUb_ / runKpeUb_ (element offsets, 32 B aligned) inside the bank.
    int64_t runStart_[SCB_MAX_BANKS][SCB_MAX_GROUP_ROWS];
    int64_t runLen_[SCB_MAX_BANKS][SCB_MAX_GROUP_ROWS];
    int64_t runUb_[SCB_MAX_BANKS][SCB_MAX_GROUP_ROWS];
    int64_t runKpeUb_[SCB_MAX_BANKS][SCB_MAX_GROUP_ROWS];
    int64_t runCount_[SCB_MAX_BANKS];
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
    // Every instantiation is compiled here on purpose: the element type is a
    // runtime field, so the op needs no tiling-key registration. A 1-byte cache
    // row has no arithmetic here -- the copy is byte-wide, so it rides the
    // uint8_t instantiation.
    if (tilingData.dataType == 2) {
        ScatterCopyUbBenchKernel<half, SCB_MAX_BANKS> op;
        op.Init(srcKv, srcRope, srcRows, dstRows, dstKv, dstRope, tilingData, &pipe);
        op.Process();
    } else if (tilingData.dataType == 3) {
        ScatterCopyUbBenchKernel<uint8_t, SCB_MAX_BANKS> op;
        op.Init(srcKv, srcRope, srcRows, dstRows, dstKv, dstRope, tilingData, &pipe);
        op.Process();
    } else {
        ScatterCopyUbBenchKernel<bfloat16_t, SCB_MAX_BANKS> op;
        op.Init(srcKv, srcRope, srcRows, dstRows, dstKv, dstRope, tilingData, &pipe);
        op.Process();
    }
}
