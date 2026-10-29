#!/bin/bash
# Usage: bash build_aclnn.sh [ROOT_DIR] [SOC_VERSION]
#
#   ROOT_DIR     repository root. Defaults to the parent of this script, so a
#                bare `bash csrc/build_aclnn.sh` works from any directory.
#   SOC_VERSION  chip name used to pick the op list (ascend910b1,
#                ascend910_9391, ...). Falls back to the SOC_VERSION env var.
#
# This script uses bash-only syntax ([[ ]], arrays), so `sh build_aclnn.sh`
# would fail; re-exec ourselves under bash when that happens.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOC_VERSION="${2:-${SOC_VERSION:-}}"
: "${ROOT_DIR:?ROOT_DIR is not set}"
: "${SOC_VERSION:?SOC_VERSION is not set: pass it as the second argument or export SOC_VERSION, e.g. ascend910b1}"

log() {
    echo "[build_aclnn] $*"
}

setup_catlass_dependency() {
    local catlass_path="${ROOT_DIR}/csrc/third_party/catlass/include"
    local catlass_commit
    local absolute_catlass_path

    git config --global --add safe.directory "$ROOT_DIR"
    catlass_commit=$(git config -f "${ROOT_DIR}/.gitmodules" --get submodule.csrc/third_party/catlass.commit)
    if [[ ! -d "${catlass_path}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        git submodule sync
        if ! git submodule update --init --recursive; then
            log "fetch failed"
            exit 1
        fi
        cd "${ROOT_DIR}/csrc/third_party/catlass" || exit 1
        git fetch origin
        git checkout "${catlass_commit}" || exit 1
        cd - || exit 1
    fi
    absolute_catlass_path=$(cd "${catlass_path}" && pwd)
    export CPATH="${absolute_catlass_path}${CPATH:+:${CPATH}}"
    log "catlass include=${absolute_catlass_path}"
}

resolve_op_dir() {
    local op_name=$1
    local candidate_dir
    for candidate_dir in \
        "${ROOT_DIR}/csrc/moe/${op_name}" \
        "${ROOT_DIR}/csrc/gmm/${op_name}" \
        "${ROOT_DIR}/csrc/attention/${op_name}" \
        "${ROOT_DIR}/csrc/mc2/${op_name}" \
        "${ROOT_DIR}/csrc/ffn/${op_name}" \
        "${ROOT_DIR}/csrc/posembedding/${op_name}"; do
        if [[ -d "${candidate_dir}" ]]; then
            echo "${candidate_dir}"
            return 0
        fi
    done
    find "${ROOT_DIR}/csrc" -maxdepth 3 -type d -name "${op_name}" -print -quit 2>/dev/null
}

log_selected_ops() {
    local op_name
    local op_path
    local kernel_cpp_file_count

    log "resolved SOC_ARG=${SOC_ARG}"
    log "resolved CUSTOM_OPS=${CUSTOM_OPS}"
    log "custom op count=${#CUSTOM_OPS_ARRAY[@]}"
    for op_name in "${CUSTOM_OPS_ARRAY[@]}"; do
        op_path=$(resolve_op_dir "${op_name}")
        if [[ -z "${op_path}" ]]; then
            log "op ${op_name}: dir=<missing>"
            continue
        fi
        kernel_cpp_file_count=0
        if [[ -d "${op_path}/op_kernel" ]]; then
            kernel_cpp_file_count=$(find "${op_path}/op_kernel" -maxdepth 1 -name '*.cpp' | wc -l | tr -d ' ')
        fi
        log "op ${op_name}: dir=${op_path} cmake=$([[ -f "${op_path}/CMakeLists.txt" ]] && echo yes || echo no) op_host_cmake=$([[ -f "${op_path}/op_host/CMakeLists.txt" ]] && echo yes || echo no) op_kernel_cpp_count=${kernel_cpp_file_count}"
    done
}

# opbuild only writes ${ASCEND_AUTOGEN_PATH}/{aclnn,inner,exc}/aic-<soc>-ops-info.ini
# for the op defs it is handed, and a def only carries the SoCs listed in its
# AICore().AddConfig() calls. A subset build made only of 910B defs therefore
# produces no `inner/aic-ascend950-ops-info.ini`, and the packaging step that
# expects that file fails long after the real cause. Detect it up front.
ops_capable_of_inner_ini() {
    local soc=$1
    local op_name op_dir
    for op_name in "${CUSTOM_OPS_ARRAY[@]}"; do
        op_dir=$(resolve_op_dir "${op_name}")
        [[ -n "${op_dir}" && -f "${op_dir}/op_host/CMakeLists.txt" ]] || continue
        if grep -qsE "ACLNNTYPE[[:space:]]+aclnn_inner" "${op_dir}/op_host/CMakeLists.txt" &&
            grep -qsF "AddConfig(\"${soc}\"" "${op_dir}/op_host/"*def*.cpp 2>/dev/null; then
            echo "${op_name}"
        fi
    done
}

check_subset_can_emit_inner_ini() {
    local soc=$1
    local producers first_candidate

    producers=$(ops_capable_of_inner_ini "${soc}")
    if [[ -n "${producers}" ]]; then
        log "inner ops-info for ${soc}: produced by [$(echo "${producers}" | paste -sd' ' -)]"
        return 0
    fi
    if [[ "${VLLM_ASCEND_ALLOW_MISSING_OPS_INFO:-0}" == "1" ]]; then
        log "WARNING: no ${soc}-capable aclnn_inner op selected; continuing anyway"
        return 0
    fi

    log "ERROR: the selected subset cannot generate inner/aic-${soc}-ops-info.ini"
    log "  opbuild creates that file only from op defs that are BOTH"
    log "    ACLNNTYPE aclnn_inner (in op_host/CMakeLists.txt) AND"
    log "    AICore().AddConfig(\"${soc}\") (in op_host/*_def.cpp)."
    log "  Without it the later packaging step fails on the missing file."
    first_candidate=$(
        find "${ROOT_DIR}/csrc" -maxdepth 4 -path '*/op_host/CMakeLists.txt' -print 2>/dev/null |
            while read -r op_host_cmake; do
                op_dir=$(dirname "${op_host_cmake}")
                if grep -qsE "ACLNNTYPE[[:space:]]+aclnn_inner" "${op_host_cmake}" &&
                    grep -qsF "AddConfig(\"${soc}\"" "${op_dir}/"*def*.cpp 2>/dev/null; then
                    basename "$(dirname "${op_dir}")"
                fi
            done | sort -u | head -n 1
    )
    if [[ -n "${first_candidate}" ]]; then
        log "  fix: add ${first_candidate} to the subset, e.g."
        log "    VLLM_ASCEND_CUSTOM_OPS='${first_candidate};${CUSTOM_OPS_ARRAY[0]}' \\"
        log "        bash csrc/build_aclnn.sh \"\$PWD\" ${soc}"
    fi
    log "  or drop VLLM_ASCEND_CUSTOM_OPS and build the full ${soc} list."
    log "  bypass: VLLM_ASCEND_ALLOW_MISSING_OPS_INFO=1"
    return 1
}

log "start: ROOT_DIR=${ROOT_DIR:-<unset>} SOC_VERSION=${SOC_VERSION:-<unset>} cwd=$(pwd)"
log "env: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    log "matched SOC branch: ascend310"
    # ASCEND310P series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "causal_conv1d_v310"
        "recurrent_gated_delta_rule_v310"
        "chunk_fwd_o_vllm"
        "chunk_gated_delta_rule_fwd_h_vllm"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend310p"
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    log "matched SOC branch: ascend910b"
    # ASCEND910B (A2) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "chunk_gated_delta_rule_fwd_h_vllm"
        "scatter_nd_update_sk"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "sparse_flash_attention"
        "kv_quant_sparse_flash_attention_vllm"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "rms_norm_cast"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "compressor"
        "compressor_metadata"
        "quant_lightning_indexer_v2"
        "quant_lightning_indexer_v2_metadata"
        "sparse_flash_mla"
        "sparse_flash_mla_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_situ_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "chunk_fwd_o_vllm"
        "kda_gate_cumsum_vllm"
        "kda_layout_swap12"
        "sparse_attention_score"
        "k2q_csr"
        "mla_preprocess"
        "fused_sparse_attention_overlap"
        "fused_lightning_indexer_manage"
        "fused_quant_lightning_indexer_manage"
        "fused_scatter_copy_sparse_flash_attention"
        # DRAM -> HBM copy microbenchmark extracted from the fused op above.
        "scatter_copy_ub_bench"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    log "matched SOC branch: ascend910_93"
    # ASCEND910C (A3) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "chunk_gated_delta_rule_fwd_h_vllm"
        "scatter_nd_update_sk"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "sparse_flash_attention"
        "kv_quant_sparse_flash_attention_vllm"
        "dispatch_ffn_combine"
        "dispatch_ffn_combine_w4_a8"
        "dispatch_ffn_combine_bf16"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "rms_norm_cast"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "compressor"
        "compressor_metadata"
        "quant_lightning_indexer_v2"
        "quant_lightning_indexer_v2_metadata"
        "sparse_flash_mla"
        "sparse_flash_mla_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_situ_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "attn_res_fwd"
        "chunk_fwd_o_vllm"
        "kda_gate_cumsum_vllm"
        "kda_layout_swap12"
        "sparse_attention_score"
        "k2q_csr"
        "mla_preprocess"
        "fused_sparse_attention_overlap"
        "fused_lightning_indexer_manage"
        "fused_quant_lightning_indexer_manage"
        "fused_scatter_copy_sparse_flash_attention"
        # DRAM -> HBM copy microbenchmark extracted from the fused op above.
        "scatter_copy_ub_bench"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
elif [[ "$SOC_VERSION" =~ ^ascend950 ]]; then
    log "matched SOC branch: ascend950"
    # ASCEND950 (A5) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "sparse_flash_attention"
        "chunk_gated_delta_rule_fwd_h_vllm"
        "scatter_nd_update_sk"
        "add_rms_norm_bias"
        "moe_gating_top_k_hash"
        "inplace_partial_rotary_mul"
        "kv_compress_epilog_v2"
        "compressor"
        "compressor_metadata"
        "quant_lightning_indexer_v2"
        "quant_lightning_indexer_v2_metadata"
        "kv_quant_sparse_attn_sharedkv"
        "kv_quant_sparse_flash_attention_vllm"
        "kv_quant_sparse_attn_sharedkv_metadata"
        "hc_post"
        "hc_pre"
        "swiglu_group_quant"
        "situ_mx_quant"
        "grouped_matmul_situ_quant"
        "indexer_compress_epilog_v2"
        "attn_res_fwd"
        "chunk_fwd_o_vllm"
        "kda_gate_cumsum_vllm"
        "kda_layout_swap12"
        "k2q_csr"
        "sparse_attention_score"
        # DRAM -> HBM copy microbenchmark (A5 entry is the same kernel source).
        "scatter_copy_ub_bench"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend950"
else
    # others
    # currently, no custom aclnn ops for other series
    log "no custom ACLNN ops configured for SOC_VERSION=${SOC_VERSION}; skip build_aclnn"
    exit 0
fi

# Optional: build only a subset of the ops listed above. Handy while bringing
# up a single debug/benchmark op, e.g.
#   VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench \
#       bash csrc/build_aclnn.sh "$PWD" ascend910b1
# The run package install clears vllm_ascend/_cann_ops_custom first, so a
# subset build leaves only those ops installed; re-run without the variable to
# restore the full set.
if [[ -n "${VLLM_ASCEND_CUSTOM_OPS:-}" ]]; then
    IFS=';,' read -ra CUSTOM_OPS_ARRAY <<< "${VLLM_ASCEND_CUSTOM_OPS}"
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    log "VLLM_ASCEND_CUSTOM_OPS override -> ${CUSTOM_OPS}"
    # A subset that carries no SoC-matching aclnn_inner def would silently drop
    # the inner ops-info ini and break packaging much later; fail fast instead.
    if ! check_subset_can_emit_inner_ini "${SOC_ARG}"; then
        exit 1
    fi
fi

log_selected_ops


# # build custom ops
# cd csrc
# rm -rf build output build_out
# echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
# bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="$SOC_ARG"

# # install custom ops to vllm_ascend/_cann_ops_custom
# ./build/cann-ops-transformer*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom


(
  set -euo pipefail

  : "${ROOT_DIR:?ROOT_DIR is not set}"

  log "subshell cwd before cd=$(pwd)"
  cd "${ROOT_DIR}/csrc"
  log "subshell cwd after cd=$(pwd)"
  log "preserving csrc/build and cleaning output dirs"
  rm -rf -- output build_out

  : "${CUSTOM_OPS:?CUSTOM_OPS is not set}"
  : "${SOC_VERSION:?SOC_VERSION is not set}"
  : "${SOC_ARG:?SOC_ARG is not set}"

  log "build command: bash build.sh --pkg --ops=\"${CUSTOM_OPS}\" --soc=\"${SOC_ARG}\""
  log "building custom ops ${CUSTOM_OPS} for ${SOC_VERSION}"
  bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"
  log "build.sh finished"

  custom_ops_install_dir="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"
  log "custom_ops_install_dir=${custom_ops_install_dir}"

  mkdir -p -- "$custom_ops_install_dir"

  # Remove all top-level entries under custom_ops_install_dir except .gitkeep, including hidden files and directories.
  find "$custom_ops_install_dir" -mindepth 1 -maxdepth 1 \
    ! -name '.gitkeep' \
    -exec rm -rf -- {} +

  shopt -s nullglob
  installer_candidates=(./build/cann-ops-transformer*.run)
  shopt -u nullglob

  log "installer candidate count=${#installer_candidates[@]}"
  for installer_file in "${installer_candidates[@]}"; do
    log "installer candidate: $(ls -lh "${installer_file}")"
  done

  (( ${#installer_candidates[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#installer_candidates[@]}" >&2; exit 1; }

  chmod +x -- "${installer_candidates[0]}" || true
  log "running installer: ${installer_candidates[0]}"
  "${installer_candidates[0]}" --install-path="${custom_ops_install_dir}"
  # CANN leaves generated vendor script dirs owner-read-only; keep repo-local
  # editable-build artifacts removable by the non-root user who built them.
  if [[ -d "${custom_ops_install_dir}/vendors/custom_transformer/scripts" ]]; then
    chmod u+w "${custom_ops_install_dir}/vendors/custom_transformer/scripts"
  fi
  log "installer finished"
  log "installed files under ${custom_ops_install_dir} (maxdepth=4, first 120 entries):"
  { find "${custom_ops_install_dir}" -mindepth 1 -maxdepth 4 -print | sort | head -n 120 | sed 's#^#[build_aclnn] install: #'; } || true

  # install batch_invariant run package and whl package
  if [[ "${VLLM_BATCH_INVARIANT:-0}" == "1" ]]; then
    log "VLLM_BATCH_INVARIANT=1, installing batch_invariant run package and whl package..."

    # call separate installation script
    batch_invariant_script="${ROOT_DIR}/csrc/build_batch_invariant_ops.sh"
    if [[ -f "${batch_invariant_script}" ]]; then
      log "Calling batch_invariant_ops build script: ${batch_invariant_script}"
      bash "${batch_invariant_script}" "${SOC_ARG}"
    else
      log "Warning: batch_invariant_ops build script not found at ${batch_invariant_script}"
    fi
  else
    log "VLLM_BATCH_INVARIANT is not set to 1, skipping batch_invariant ops build"
  fi
)
