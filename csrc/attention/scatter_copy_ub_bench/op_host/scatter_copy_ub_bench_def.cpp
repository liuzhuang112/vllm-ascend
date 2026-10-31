/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
/**
 * \file scatter_copy_ub_bench_def.cpp
 * \brief Operator definition of the DRAM -> HBM copy microbenchmark.
 *
 * The five metadata/attribute knobs are deliberately explicit so a single
 * Python call can describe one measurement point:
 *
 *   bank_num    UB slots of the TQueBind software pipeline (1 => no overlap)
 *   group_rows  rows merged into one DataCopyPad (1 => one token per DMA)
 *   table_mode  0 = identity rows, 1 = read src/dst row tables like production
 *   core_num    AIV blocks to launch, 0 = all available
 *   copy_rope   1 = also move the KPE plane, 0 = CKV plane only
 */

#include "register/op_def_registry.h"

namespace ops {

class ScatterCopyUbBench : public OpDef {
public:
    explicit ScatterCopyUbBench(const char *name) : OpDef(name)
    {
        // Each DataType entry is one complete dtype combination. fp8 carries no
        // arithmetic here -- the copy is byte wide -- so uint8 is supported as
        // the same instantiation, which also lets a caller avoid the fp8 tensor
        // factory restrictions when only the DMA cost matters.
        this->Input("src_kv").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("src_rope").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("src_rows").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dst_rows").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();

        // Attributes are declared before the outputs so the generated
        // aclnnInner signature is inputs -> attributes -> outputs, matching the
        // public wrapper in op_api/aclnn_scatter_copy_ub_bench.cpp.
        this->Attr("bank_num").AttrType(REQUIRED).Int(2);
        this->Attr("group_rows").AttrType(REQUIRED).Int(1);
        this->Attr("table_mode").AttrType(REQUIRED).Int(1);
        this->Attr("core_num").AttrType(REQUIRED).Int(0);
        this->Attr("copy_rope").AttrType(REQUIRED).Int(1);

        this->Output("dst_kv").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Output("dst_rope").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);

        // A5. The kernel body only uses MTE2/MTE3 + TQue, so the same source
        // entry serves arch22 and arch35; no _apt.cpp / opFile override needed
        // (the 950-only fused scatter-copy op is a single .cpp as well).
        OpAICoreConfig a5Config;
        a5Config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend950", a5Config);
    }
};

OP_ADD(ScatterCopyUbBench);

}  // namespace ops
