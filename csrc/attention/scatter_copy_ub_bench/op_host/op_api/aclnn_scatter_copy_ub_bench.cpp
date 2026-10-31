/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "aclnn_scatter_copy_ub_bench.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerScatterCopyUbBenchGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, int64_t, int64_t, int64_t,
    const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerScatterCopyUbBench(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

aclnnStatus aclnnScatterCopyUbBenchGetWorkspaceSize(
    const aclTensor *srcKv, const aclTensor *srcRope,
    const aclTensor *srcRows, const aclTensor *dstRows,
    int64_t bankNum, int64_t groupRows, int64_t tableMode, int64_t coreNum,
    int64_t copyRope,
    const aclTensor *dstKv, const aclTensor *dstRope,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerScatterCopyUbBenchGetWorkspaceSize(
        srcKv, srcRope, srcRows, dstRows,
        bankNum, groupRows, tableMode, coreNum, copyRope,
        dstKv, dstRope, workspaceSize, executor);
}

aclnnStatus aclnnScatterCopyUbBench(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerScatterCopyUbBench(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
