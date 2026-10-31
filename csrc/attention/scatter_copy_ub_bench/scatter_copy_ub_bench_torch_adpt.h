/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef SCATTER_COPY_UB_BENCH_TORCH_ADPT_H
#define SCATTER_COPY_UB_BENCH_TORCH_ADPT_H

namespace vllm_ascend {

// Single-launch DRAM -> HBM copy, the isolated copy stage of
// npu_fused_scatter_copy_sparse_flash_attention.
//
// src_rows / dst_rows are per-token row indices into src_* / dst_*. Pass
// identity rows and table_mode=0 for a pure sequential copy, or random rows and
// table_mode=1 to reproduce the production sparse gather/scatter pattern.
// A DRAM source is a plain CPU tensor that has been published on the device:
//   src_cpu = torch.empty(..., pin_memory=True)
//   src_dev = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(src_cpu)
// Ordinary pageable CPU memory is not device addressable and is rejected by the
// runtime, not by this adapter.
//
// copy_rope = 0 copies the CKV plane only. The rope tensors are still required
// arguments (the aclnn signature has no optional inputs) but any legal shape is
// accepted -- the kernel never touches them -- so a caller that has no KPE plane
// can pass a one-element placeholder instead of a [rows, 0] tensor, which some
// graph paths reject.
inline void scatter_copy_ub_bench(
    const at::Tensor& src_kv,
    const at::Tensor& src_rope,
    const at::Tensor& src_rows,
    const at::Tensor& dst_rows,
    at::Tensor dst_kv,
    at::Tensor dst_rope,
    int64_t bank_num,
    int64_t group_rows,
    int64_t table_mode,
    int64_t core_num,
    int64_t copy_rope) {
  TORCH_CHECK(src_kv.dim() == 2 && src_rope.dim() == 2,
              "scatter_copy_ub_bench: src_kv/src_rope must be 2-D [rows, dim].");
  TORCH_CHECK(dst_kv.dim() == 2 && dst_rope.dim() == 2,
              "scatter_copy_ub_bench: dst_kv/dst_rope must be 2-D [rows, dim].");
  TORCH_CHECK(src_rows.dim() == 1 && dst_rows.dim() == 1,
              "scatter_copy_ub_bench: src_rows/dst_rows must be 1-D [num_tokens].");
  TORCH_CHECK(src_rows.numel() == dst_rows.numel() && src_rows.numel() > 0,
              "scatter_copy_ub_bench: row tables must be non-empty and equal length.");

  const auto dtype = src_kv.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kByte,
              "scatter_copy_ub_bench: only fp16, bf16 and uint8 are supported "
              "(a 1-byte cache row is measured through uint8: the op copies "
              "bytes and never interprets them).");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 3>{&src_rope, &dst_kv, &dst_rope}) {
    TORCH_CHECK(tensor->scalar_type() == dtype,
                "scatter_copy_ub_bench: all cache tensors must share one dtype.");
  }
  TORCH_CHECK(src_kv.size(1) == dst_kv.size(1),
              "scatter_copy_ub_bench: src/dst dims must match per plane.");
  TORCH_CHECK(src_rows.scalar_type() == at::kInt && dst_rows.scalar_type() == at::kInt,
              "scatter_copy_ub_bench: row tables must be int32.");

  const int64_t elem_size = static_cast<int64_t>(src_kv.element_size());
  // Row length is not required to be a multiple of 32 B: DataCopyPad pads the UB
  // side itself, which is what makes a 656 B single-plane row (512 nope + 16
  // scale + 128 rope) copyable at all. Rows that are not a multiple of 32 B take
  // the per-row path in the kernel (each row is placed at a 32 B aligned UB
  // offset), so group_rows cannot be coalesced for them -- a throughput
  // consequence, not an error.
  const int64_t ckv_row_bytes = src_kv.size(1) * elem_size;
  TORCH_CHECK(ckv_row_bytes > 0,
              "scatter_copy_ub_bench: the CKV plane must have a non-empty row.");
  if (copy_rope != 0) {
    const int64_t kpe_row_bytes = src_rope.size(1) * elem_size;
    TORCH_CHECK(src_rope.size(1) == dst_rope.size(1),
                "scatter_copy_ub_bench: src/dst rope dims must match.");
    TORCH_CHECK(kpe_row_bytes > 0,
                "scatter_copy_ub_bench: copy_rope=1 needs a non-empty rope row; "
                "pass copy_rope=0 to copy the CKV plane only.");
  }
  // The DMA base of each plane must itself be 32 B aligned; a sliced view that
  // starts mid-row is the one case this op cannot express.
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 4>{&src_kv, &dst_kv, &src_rope, &dst_rope}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 32 == 0,
                "scatter_copy_ub_bench: every cache tensor must start on a 32B "
                "boundary (pass a freshly allocated tensor, not a slice).");
  }
  TORCH_CHECK(bank_num >= 1 && bank_num <= 8,
              "scatter_copy_ub_bench: bank_num must be in [1, 8].");
  TORCH_CHECK(group_rows >= 1 && group_rows <= 8,
              "scatter_copy_ub_bench: group_rows must be in [1, 8].");

  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 6>{
           &src_kv, &src_rope, &src_rows, &dst_rows, &dst_kv, &dst_rope}) {
    TORCH_CHECK(tensor->is_contiguous(),
                "scatter_copy_ub_bench: all tensors must be contiguous.");
  }

  const auto device = dst_kv.device();
  TORCH_CHECK(device.is_privateuseone(),
              "scatter_copy_ub_bench: destination caches must be on NPU.");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 4>{&src_rope, &dst_rope, &src_rows, &dst_rows}) {
    TORCH_CHECK(tensor->device() == device,
                "scatter_copy_ub_bench: all but src_kv must share one NPU device.");
  }
  // src_kv may be a device view of registered host memory, which torch already
  // reports as a PrivateUse1 tensor; anything else must live on the same NPU.
  TORCH_CHECK(src_kv.device() == device || src_kv.device().is_cpu(),
              "scatter_copy_ub_bench: src_kv must be on the destination NPU or a "
              "registered host view.");

  EXEC_NPU_CMD(
      aclnnScatterCopyUbBench,
      src_kv,
      src_rope,
      src_rows,
      dst_rows,
      bank_num,
      group_rows,
      table_mode,
      core_num,
      copy_rope,
      dst_kv,
      dst_rope);
}

}  // namespace vllm_ascend

#endif  // SCATTER_COPY_UB_BENCH_TORCH_ADPT_H
