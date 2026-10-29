# ScatterCopyUbBench

从 `fused_scatter_copy_sparse_flash_attention` 里抽出来的 **DRAM → HBM 搬运** 微基准算子。
它只保留「缺失 KV 从 DRAM 搬进 HBM 稀疏 Cache」这一段，去掉 Attention、去掉了首填/稳态的
分支判断，剩下的就是一个纯粹的、可单独计时的 UB 中转拷贝。

## 抽象来源

| 生产代码 | 本 demo |
|:---|:---|
| `op_kernel/first_fill_scatter_copy_kernel.h` 的 `CopyIn` / `CopyOut` | `op_kernel/scatter_copy_ub_bench.cpp` 的同名函数 |
| `TQueBind<VECIN, VECOUT, 2>` + `CopyIn(next) / CopyOut(cur)` 两槽软件流水 | `TQueBind` 深度由 `bank_num` 指定，`bank_num - 1` 级预取 |
| `srcTokenId → dram_block_table → (physicalBlock*128 + offset)` | `src_rows` / `dst_rows` 两张行表（`table_mode=1`） |
| 每个 token 两次 `DataCopyPad`：CKV 1024B + KPE 128B | 不变（`group_rows=1` 时完全一致） |
| `usedCoreNum * 2` 个 AIV 按 `blockIdx + k*usedCoreNum` 取任务 | `cores` 个 AIV 按 group 步进取任务 |

数据通路（与生产完全一致）：

```text
DRAM src ──MTE2 DataCopyPad──▶ UB bank[i] ──MTE3 DataCopyPad──▶ HBM dst
                ▲                      │
                └──── bank[(i+1)%N] ◀──┘   共 bank_num 个 bank，无 Vector 计算
```

关键点：这条路径上**没有任何 Vector 运算**，MTE2 与 MTE3 就是全部耗时，所以
`bank_num` 决定了能重叠多少读延迟。生产算子的 ScatterCopy 时延（README 里 41~98 μs）
基本就是这条流水线的极限。

## 编译

算子已加入 `csrc/build_aclnn.sh` 的 `ascend910b` / `ascend910_93` 算子列表。

`build_aclnn.sh` 的签名是 `bash build_aclnn.sh [ROOT_DIR] [SOC_VERSION]`，两个参数都可以省略：
`ROOT_DIR` 默认取脚本所在目录的上一级，`SOC_VERSION` 默认读同名环境变量。

```bash
# 0) 让 CANN 环境可用（已 source 过可跳过）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd /path/to/vllm-ascend

# 1) 最常见：显式传两个参数
bash csrc/build_aclnn.sh "$PWD" ascend910b1

# 2) 或用环境变量 + 省略参数
export SOC_VERSION=ascend910b1
bash csrc/build_aclnn.sh

# 3) 只编这一个算子（迭代快很多，但会覆盖已装的算子包，见下方说明）
VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench bash csrc/build_aclnn.sh
```

- 旧的报错 `ROOT_DIR is not set` 就是因为 `ROOT_DIR=$1` 而没传参；现在裸跑也能用。
- `SOC_VERSION` 用**芯片级**名字（`npu-smi info -t board -i 0 -c 0` 的 Chip Name），
  脚本按前缀选算子列表：`ascend910b*` → 910B，`ascend910_93*` → A3。
- 建议统一用 `bash`；用 `sh` 调用时脚本会自己 `exec bash` 重启一次，但仍以 `bash` 为准。
- `VLLM_ASCEND_CUSTOM_OPS` 是给调试用的覆盖开关，默认不设置时行为完全不变。
  安装 run 包前会清空 `vllm_ascend/_cann_ops_custom`，所以只编一个算子时其余算子会
  暂时缺失；测完请不带该变量重跑一次，恢复完整算子包。

编译完成后 `torch.ops._C_ascend.scatter_copy_ub_bench` 即可用（需要
`vllm_ascend.utils.enable_custom_op()`）。

注意这里有**两个独立的产物**，缺一个都跑不起来：

| 产物 | 由谁编译 | 不编会怎样 |
|:---|:---|:---|
| 算子本体 `libcust_opapi.so`（`_cann_ops_custom/` 下） | `build_aclnn.sh` | 报 `aclnnScatterCopyUbBench ... not found` |
| `torch.ops._C_ascend` 里的 Python 入口 | `pip install -e .` → 根 `CMakeLists.txt` | 报 `_C_ascend has no attribute scatter_copy_ub_bench` |

所以流程是：

```bash
SOC_VERSION=ascend910b1 bash csrc/build_aclnn.sh              # 1. 算子包
COMPILE_CUSTOM_KERNELS=0 pip install -e . --no-build-isolation # 2. 重编 vllm_ascend_C
```

`COMPILE_CUSTOM_KERNELS=0` 是为了让第 2 步只编 C++ 扩展、不再重跑一遍算子包
（默认它是 1，会先把全量算子再编一次）。第 2 步只改了 `torch_binding.cpp` /
新算子目录时是增量编译，通常一两分钟。

## 运行

```bash
# 生产配置：2 bank、每 token 一次 DMA、走行表寻址
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 --verify

# 扫描：把多少行合并成一次 DMA
python benchmarks/kernel/scatter_copy_ub_bench.py --sweep group-rows

# 扫描：流水深度
python benchmarks/kernel/scatter_copy_ub_bench.py --sweep bank

# 扫描：行表标量读的代价（table_mode=1 vs 0）
python benchmarks/kernel/scatter_copy_ub_bench.py --sweep table-mode

# 真实 DRAM → HBM（源在 host pinned 内存，经 UVA 映射给 device）
python benchmarks/kernel/scatter_copy_ub_bench.py --src host --verify
```

输出示例：

```text
              case    rows group bank tbl cores        us     GB/s
------------------------------------------------------------------
          baseline    2048     1    2   1     0      41.2     57.2
```

`GB/s = rows × 1152B / 时延`，即只统计有效载荷（读 1152B + 写 1152B 的实际带宽约为此值的 2 倍）。

## 参数含义

| 参数 | 作用 | 建议扫描范围 |
|:---|:---|:---|
| `--rows` | 单次 launch 搬运的 token 行数 | 512 / 2048 / 8192 / 32768 |
| `--bank` | UB 流水槽数，`1` = 完全不重叠 | 1 / 2 / 3 / 4 / 6 / 8 |
| `--group-rows` | 多少行合并成一次 `DataCopyPad`（源行连续时） | 1 / 2 / 4 / 8 |
| `--table-mode` | `1` = 走行表 `GetValue`（同生产）；`0` = 恒等寻址 | 0 / 1 |
| `--cores` | 启动的 AIV 数，`0` = 全部 | 8 / 16 / 24 / 48 |
| `--src` | `npu` = 源在 HBM；`host` = 源在 host pinned 内存 | npu / host |

`--table-mode 0` 与 `--table-mode 1` 的差值，就是**每行两次 `GlobalTensor::GetValue`
标量读**的代价。生产 kernel 里这部分开销与 DMA 是串行的，通常就是 ScatterCopy 时延里
最容易被忽略的一块。

## 已知限制

- 仅支持 FP16 / BF16；CKV、KPE 的行字节数必须是 32B 对齐（默认 512/64 维天然满足）。
- `group-rows > 1` 只有在**源行连续**（`src_rows` 步长为 1）时才会合并读侧 DMA；
  写侧允许任意正向步长，与生产里 miss pair 的合并写一致。随机行表下会退化成逐行 DMA，
  这正是我们想看到的「合并能不能用」的结论。
- `--cores` 超过实际 AIV 数时会在 tiling 里被截断，标签仍显示请求值。
- host 源依赖 `get_npu_view_from_cpu_tensor`（`aclrtHostGetDevicePointer` 的映射视图），
  普通可分页 CPU 内存不受支持。
- 计时为 host 侧 `torch.npu.Event` + N 次连续 launch，包含 launch 间隙；需要剔除
  调度开销时请配合 `msprof` 或 `torch_npu.profiler` 看单算子 task 时间。
