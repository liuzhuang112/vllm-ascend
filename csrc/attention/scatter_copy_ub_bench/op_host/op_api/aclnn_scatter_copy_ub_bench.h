/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef ACLNN_SCATTER_COPY_UB_BENCH_H_
#define ACLNN_SCATTER_COPY_UB_BENCH_H_

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief DRAM -> HBM copy microbenchmark (UB staged, TQueBind pipeline).
 *
 * dst_kv / dst_rope are caller-owned buffers: the operator only writes them.
 * \return ACLNN_SUCCESS on success.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterCopyUbBenchGetWorkspaceSize(
    const aclTensor *srcKv, const aclTensor *srcRope,
    const aclTensor *srcRows, const aclTensor *dstRows,
    int64_t bankNum, int64_t groupRows, int64_t tableMode, int64_t coreNum,
    int64_t copyRope,
    const aclTensor *dstKv, const aclTensor *dstRope,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnScatterCopyUbBench(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_SCATTER_COPY_UB_BENCH_H_
