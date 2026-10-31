# ScatterCopyUbBench

从 `fused_scatter_copy_sparse_flash_attention` 里抽出来的 **DRAM → HBM 搬运** 微基准算子。
它只保留「缺失 KV 从 DRAM 搬进 HBM 稀疏 Cache」这一段，去掉 Attention、去掉了首填/稳态的
分支判断，剩下的就是一个纯粹的、可单独计时的 UB 中转拷贝。

## 抽象来源

| 生产代码 | 本 demo |
|:---|:---|
| `op_kernel/first_fill_scatter_copy_kernel.h` 的 `CopyIn` / `CopyOut` | `op_kernel/scatter_copy_ub_bench.cpp` 的同名函数 |
| `TQueBind<VECIN, VECOUT, 2>` + `CopyIn(next) / CopyOut(cur)` 两槽软件流水 | `TQueBind<VECIN, VECOUT, BANK_COUNT>`，深度由 `bank_num` 指定，`bank_num - 1` 级预取 |
| `srcTokenId → dram_block_table → (physicalBlock*128 + offset)` | `src_rows` / `dst_rows` 两张行表（`table_mode=1`） |
| 每个 token 两次 `DataCopyPad`：CKV 1024B + KPE 128B | 不变（`group_rows=1` 时完全一致）；可用 `--ckv-dim / --kpe-dim / --dtype` 改成 1 字节的 656B 单平面 |
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

算子已加入 `csrc/build_aclnn.sh` 的 `ascend910b` / `ascend910_93` / `ascend950` 算子列表。

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
  脚本按前缀选算子列表：`ascend910b*` → 910B，`ascend910_93*` → A3，`ascend950*` → A5。
- 建议统一用 `bash`；用 `sh` 调用时脚本会自己 `exec bash` 重启一次，但仍以 `bash` 为准。
- `VLLM_ASCEND_CUSTOM_OPS` 是给调试用的覆盖开关，默认不设置时行为完全不变。
  安装 run 包前会清空 `vllm_ascend/_cann_ops_custom`，所以只编一个算子时其余算子会
  暂时缺失；测完请不带该变量重跑一次，恢复完整算子包。

编译完成后 `torch.ops._C_ascend.scatter_copy_ub_bench` 即可用（需要
`vllm_ascend.utils.enable_custom_op()`）。

### 换 SoC 前必须先清 `csrc/build`（脚本已自动处理）

`csrc/cmake/opbuild.cmake` 里生成算子信息的自定义命令只把
`autogen/<组>/aclnn*.{h,cpp}` 声明为 `OUTPUT`，**没有**把同样由它写出的
`autogen/inner/aic-<soc>-ops-info.ini` 列为输出，也没有 `DEPENDS` 任何
`*_def.cpp`。于是只要那些 aclnn 源文件还在，opbuild 就被判定为最新、不再执行，
而下游 `add_ops_info_target()` 仍然要读 `inner/aic-<soc>-ops-info.ini` —— 这正是
「先编过 910B，再编 950 时报缺 `aic-ascend950-ops-info.ini`，且 `inner/` 里只有
910b/910_93」的成因，`/Users/zhuang/Desktop/vllm-ascend` 那份之所以正常，只是因为
它的 `build/` 是全新的。

`build_aclnn.sh` 现在会在编译前检查三种情况并自动清理（都会打 `WARNING` 日志）：

| 情况 | 处理 |
|:---|:---|
| `CMakeCache.txt` 里的 `ASCEND_COMPUTE_UNIT` 与本次 SoC 不一致 | `rm -rf csrc/build`（完整重编） |
| 目标 SoC 的 `autogen/inner/aic-<soc>-ops-info.ini` 缺失 | `rm -rf csrc/build/autogen csrc/build/tbe`（强制 opbuild 重跑） |
| 某个 `*_def.cpp` 比该 ini 更新（改过算子定义） | 同上 |

想无条件全量重编：`VLLM_ASCEND_CLEAN_BUILD=1 bash csrc/build_aclnn.sh`。

注意这里有**两个独立的产物**，缺一个都跑不起来：

| 产物 | 由谁编译 | 不编会怎样 |
|:---|:---|:---|
| 算子本体 `libcust_opapi.so`（`_cann_ops_custom/` 下） | `build_aclnn.sh` | 报 `aclnnScatterCopyUbBench ... not found` |
| `torch.ops._C_ascend` 里的 Python 入口 | `pip install -e .` → 根 `CMakeLists.txt` | 报 `_C_ascend has no attribute scatter_copy_ub_bench` |

所以流程是：

```bash
SOC_VERSION=ascend910b1 bash csrc/build_aclnn.sh               # 1. 算子包
SOC_VERSION=ascend910b1 pip install -e . --no-build-isolation  # 2. 编 vllm_ascend_C
```

第 2 步默认 `COMPILE_CUSTOM_KERNELS=1`，`cmake_build_ext.run()` 会**先再跑一遍
`build_aclnn.sh`**（全量，慢），然后 cmake 编扩展。想只编这一个算子就加
`VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench`，两步合一：

```bash
SOC_VERSION=ascend950 VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench \
    pip install -e . --no-build-isolation
```

> **不要**用 `COMPILE_CUSTOM_KERNELS=0` 来“只编扩展”。`setup.py:422` 在它为 0 时把
> `ext_modules` 设成空列表，`build_extensions()` 直接 `return`——算子包和
> `vllm_ascend_C.so` **都不会生成**，`import vllm_ascend.vllm_ascend_C` 报
> `ModuleNotFoundError`。它只是“只装 Python 部分”的开关（官方 installation.md 里也是
> 这个语义：*skips building the native extensions and custom operators*）。
>
> 只改了 `torch_binding.cpp` / 新算子目录时，cmake 是增量编译，通常一两分钟；但若
> `vllm_ascend_C*.so` 已存在且时间戳比源文件新，会跳过重链，此时先
> `find . -name 'vllm_ascend_C*.so' -delete` 再装。

### A5 上 `_C_ascend` 是空的：不是没编，是被 gate 掉了

`vllm_ascend.utils.enable_custom_op()` 开头有一句提前返回：

```python
if envs.VLLM_BATCH_INVARIANT or not get_current_hardware_profile().supports(
        HardwareCapability.RUNTIME_CUSTOM_OPS):
    _CUSTOM_OP_ENABLED = False
    return             # 从未 import vllm_ascend_C
```

而 `vllm_ascend/device/hardware_profile.py` 里 **A5（ascend950）的 capabilities 没有
`RUNTIME_CUSTOM_OPS`**（代码注释写明「ASCEND950 上自定义算子编译/执行只部分可用，
暂全部禁用」）。所以 950 机器上 `vllm_ascend_C` 根本没被加载，`torch.ops._C_ascend`
是空命名空间——**这时重编多少次都没用**，报错文案里的「binding has not been rebuilt」
是误导。

`TORCH_LIBRARY(_C_ascend)` 的注册发生在 `.so` 的模块初始化里，所以只要 `import
vllm_ascend.vllm_ascend_C` 就能拿到全部入口（`vllm_ascend/platform.py`、
`models/minimax_m3/ops/msa_m3_npu.py` 都是这么直接 import 的）。压测脚本
`benchmarks/kernel/scatter_copy_ub_bench.py` 已按这个逻辑处理：先走
`enable_custom_op()`，拿不到 op 就自己 import 扩展，只有 import 之后仍然没有才判定
是 `.so` 过期。

手工确认当前机器是哪种情况：

```bash
python - <<'PY'
import torch, torch_npu
from vllm_ascend.device.hardware_profile import (
    HardwareCapability, get_current_hardware_profile)
from vllm_ascend.utils import enable_custom_op
print("gate ok:", get_current_hardware_profile().supports(
    HardwareCapability.RUNTIME_CUSTOM_OPS))
print("enable_custom_op:", enable_custom_op())
print("op present:", hasattr(torch.ops._C_ascend, "scatter_copy_ub_bench"))
PY
```

`gate ok: False` 就是上面的原因；此时只要在脚本里手动 import 扩展即可，不要去改
`hardware_profile.py`（那是运行时全局开关，会影响生产的自定义算子派发）。

### ascend950 / A5

A5 上没有 Vector 参与，所以队列用 `TQueBind<QuePosition::VECIN, QuePosition::VECOUT, N>`：
`FreeTensor` 释放 buffer 时走的是 MTE3_MTE2 这条边，正好是这条流水线需要的同步。**不要**
再手写 `SetFlag/WaitFlag<MTE3_MTE2>`——手取的 event id 会和队列为同一个 HardEvent 预留的
那个撞车，表现为「第一次过、第二次错」的偶发失败（排查过程见「校验」一节）。910B/A3/A5
共用同一份 kernel 源码（不需要 `_apt.cpp` 和 `opFile.value`，`fused_quant_scatter_copy_sparse_flash_attention`
这个 950 专用算子也是单 `.cpp`）：

```bash
# 全量 950（会生成 csrc/build/autogen/inner/aic-ascend950-ops-info.ini）
SOC_VERSION=ascend950 bash csrc/build_aclnn.sh "$PWD" ascend950

# 只编这个算子：它自己就是 950 可用的 aclnn_inner 算子，所以子集也能生成上面那个 ini
SOC_VERSION=ascend950 VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench \
    bash csrc/build_aclnn.sh "$PWD" ascend950
```

两个 A5 上需要注意的语义差异：

- `bank_num` 同时决定 MTE2 预取深度和 MTE3 的排空并行度：`FreeTensor` 等到写回完成才把
  bank 还给 MTE2，所以 `--bank 1` 是完全串行的基线，`--bank 2` 起才有「边写边读」。
- tiling 用 `GetCoreNumAiv()` 定核，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`
  让 A5 不配 AIC，所以 `--cores` 就是真实 AIV 数，不像生产的 MIX kernel 还要乘 2。

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

# 真实 DRAM → HBM（源在 host 锁页内存，经 UVA 映射给 device）
# 脚本会自己把 pinned_mem_register:True 加进 PYTORCH_NPU_ALLOC_CONF；
# 若 torch_npu 已被提前 import，则先手动 export 该变量再运行。
python benchmarks/kernel/scatter_copy_ub_bench.py --src host --verify

# 单平面行：每 token 656 B，没有 KPE 平面（1 字节 cache，走 uint8 逐字节搬）
python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 --verify
```

输出示例：

```text
              case    rows  B/row group bank tbl cores        us  ns/row     GB/s
--------------------------------------------------------------------------------
          baseline    2048   1152     1    2   1     0      41.2    20.1     57.2
```

`GB/s = rows × B/row / 时延`，即只统计有效载荷（读一份 + 写一份的实际带宽约为此值的 2 倍）。

## 校验（`--verify`）

```bash
# 全量校验 + 重复 20 次：抓非确定性（fence 缺失这类 bug 只会偶发）
python benchmarks/kernel/scatter_copy_ub_bench.py --verify --verify-rows 0 --repeat-verify 20 \
    --iters 1 --rounds 1
```

`--verify-rows` 默认只查前 64 行，`0` 表示查全部。**非确定性 bug 用全量 + 重复才能定住**：
每核负责的组是 `c, c + coreNum, ...`，只查前 64 行会漏掉大部分核。

失败时每个坏行会打印一行诊断，并给出三种归因：

| 报告 | 含义 |
|:---|:---|
| `payload equals source row(s) X` | 写到了错误的 token → 行表/寻址错 |
| `all zeros -> row never written` | 该行完全没写 → 拥有它的核没跑到，或 DMA 没发 |
| `no other source row matches -> torn or stale DMA` | 半行对半行错 → 搬运被覆盖/撕裂，同步问题 |

最后一种基本只有一个来源：MTE2 提前复用了 MTE3 还在读的 UB bank。kernel 里的队列因此是
`TQueBind<QuePosition::VECIN, QuePosition::VECOUT, N>` —— 释放 buffer 时走 MTE3_MTE2 这条边；
裸 `TQue<VECIN>` 释放时等的是 `V_MTE2`（“vector 读完了”），而这个 kernel 一条 vector 指令都没有，
该 flag 立刻成立，MTE2 就会踩上去。

定位手段：`--bank 1` 关掉流水（同一 slot 读完即写、无重叠）。`--bank 1` 全过而 `--bank 2` 偶发失败，
就锁定在 bank 复用/同步上；两者都错则与重叠无关，往寻址方向查。

## 参数含义

| 参数 | 作用 | 建议扫描范围 |
|:---|:---|:---|
| `--rows` | 单次 launch 搬运的 token 行数 | 512 / 2048 / 8192 / 32768 |
| `--ckv-dim` | CKV 平面每行元素数（默认 512）；乘 dtype 字节数就是行字节数，DeepSeek 系 1 字节 MLA 用 656 | 512 / 656 / 672 / 1024 |
| `--kpe-dim` | KPE 平面每行元素数（默认 64）；`0` = 不搬 KPE（`copy_rope=0`） | 64 / 0 |
| `--dtype` | `bf16` / `fp16` / `uint8`。1 字节 cache 行一律用 `uint8`：算子只搬字节、不做算术，所以不需要该芯片支持 fp8 | bf16 / uint8 |
| `--run-len` | 行表里连续行游程的长度。默认 1 = 全随机置换（合并率 0）；`4` 表示 4 行连续，接近 topk 命中相邻块时的形状 | 1 / 2 / 4 / 8 |
| `--bank` | UB 流水槽数，`1` = 完全不重叠 | 1 / 2 / 3 / 4 / 6 / 8 |
| `--group-rows` | 多少行合并成一次 `DataCopyPad`（源行连续时）；实测 4 行是效率平台（`--src host` 下 42.0 → 45.6 GB/s），8 行无收益 | 1 / 2 / 4 |
| `--table-mode` | `1` = 走行表 `GetValue`（同生产）；`0` = 恒等寻址 | 0 / 1 |
| `--cores` | 启动的 AIV 数，`0` = 全部 | 8 / 16 / 24 / 48 |
| `--src` | `npu` = 源在 HBM；`host` = 源在 host pinned 内存 | npu / host |
| `--device` | 跑在哪张卡上（默认 0）。测 host 带宽时两张卡分属不同 socket 才有意义 | 0 / 4 |
| `--host-interleave` | 配合 `--src host`：以 `numactl --interleave=all` 重启，让源页跨 NUMA node（仅当同卡能看到两个 node 时有效） | 开关 |
| `--json` | 每个 case 输出一行 JSON（供 `host_link_scale.py` 之类脚本汇总），不再打印表格 | 开关 |
| `--verify-rows` | 校验多少行，`0` = 全部 | 64 / 0 |
| `--repeat-verify` | 重复跑并重复校验的次数（抓偶发） | 20 |

`--table-mode 0` 与 `--table-mode 1` 的差值，就是**每行两次 `GlobalTensor::GetValue`
标量读**的代价。生产 kernel 里这部分开销与 DMA 是串行的，通常就是 ScatterCopy 时延里
最容易被忽略的一块。

## 单平面 1 字节行：CKV_DIM=656、KPE_DIM=0（`--dtype uint8`）

DeepSeek 系的量化 MLA 缓存把 rope 塞进同一行：**656 B = 512 B nope + 16 B scale(fp32×4) +
128 B rope(bf16)**。这时没有独立的 KPE 平面，每 token 只有**一次** `DataCopyPad`。

1 字节行一律用 `--dtype uint8`：算子只搬字节、不解释内容，所以既不需要芯片支持 fp8，也不需要
torch 的 fp8 张量工厂。fp8 已经从这个算子里删干净了（def 的 dtype 组合、tiling、torch 适配层
的校验都只认 bf16 / fp16 / uint8）。

```bash
python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 --verify
python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src host --sweep bank
python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src host --sweep table-mode
```

### 和 bf16 1152 B 行的差别

| 项 | bf16 1152 B（512+64） | 656 B 单平面 |
|:---|:---|:---|
| 每 token 的 DMA 次数 | 2（CKV + KPE） | 1 |
| 行字节 / 32 B | 1024、128，都整除 | 656，**不整除**（656 = 20×32 + 16） |
| UB 内行间距 | 等于行字节数 | 向上取整到 **672 B**，否则第 2 行的局部地址只有 16 B 对齐，`DataCopyPad` 不接受 |
| 能否合并 | 三种模式都能 | 只能用 `GROUP_MODE_CONTIGUOUS`（整组在两侧都连续）：它从不单独寻址某一行，所以行字节不整除也无所谓；strided 模式不行（UB 里行距 672 B，无法用 32 B 单位表达） |

### 实测：DMA 按 256 B 粒度计费，尾巴超过半格再加一笔

`--src host`、`rows 2048`、`bank 2`、`group 1`、`table 1`、`--sweep ckv-dim` 的全部十点
（脚本现在直接打印 `ns/row` 列，就是这个原因）：

| 行字节 | GB/s | ns/row | 计费字节 | 计费单元 | ns/256B | `%256` |
|:---|:---|:---|:---|:---|:---|:---|
| 384 | 30.9 | 12.4 | 512 | 2 | 6.21 | 128 |
| 512 | 41.2 | 12.4 | 512 | 2 | 6.21 | 0 |
| 576 | 32.5 | 17.7 | 768 | 3 | 5.91 | 64 |
| 640 | 36.1 | 17.7 | 768 | 3 | 5.91 | 128 |
| 656 | 32.0 | **20.5** | 768 | 3 | **6.83** | **144** |
| 672 | 32.7 | **20.6** | 768 | 3 | **6.85** | **160** |
| 704 | 34.2 | **20.6** | 768 | 3 | **6.86** | **192** |
| 768 | 41.7 | 18.4 | 768 | 3 | 6.14 | 0 |
| 896 | 38.8 | 23.1 | 1024 | 4 | 5.77 | 128 |
| 1024 | 44.3 | 23.1 | 1024 | 4 | 5.78 | 0 |

规则非常干净，一句话：**计费字节 = 256 × ⌈len/256⌉，每 256 B 收 ~5.8 ns（≈44.2 GB/s），
若 `len % 256 > 128`（尾巴超过半格）再加 ~2.65 ns。**

拟合 `t = 0.4 + 5.79 × ⌈len/256⌉ + (2.65 if len%256 > 128)` 对上表十点加下一节合并四点的
平均误差 1.5%：

- **384 和 512 一样快（12.4 ns）**：384 也是按 512 计费 —— 粒度确实是 256 的倍数，不是 32/128。
- **576 / 640 / 768 都在 768 这一格**，17.7–18.4 ns；**656 / 672 / 704 也在 768 这一格，却要多花 2.7 ns**，
  差别只在它们的尾巴（144 / 160 / 192 B）超过了 128 B。128 B 整数倍那条假设因此被否掉
  （896 = 7×128 只有 38.8，576 = 4.5×128 反而和 640 一样）。
- 结论：**656 的 32.0 GB/s 是「768 B + 超半格尾巴」的必然结果**，跟数据量、对齐、行数都无关。

脚本启动时会把这条行本身的计费、以及「合并 k 行」的整条预测曲线一起打出来：

```text
row: 656 elems CKV (KPE plane off), uint8, 656 B/token
     DMA: 656 B/row is charged as 768 B (3 x 256 B, and the 144 B tail (> 128 B) adds ~2.6 ns)
     -> ~20.4 ns/row (~32 GB/s useful); the other 112 B of the granule are free
     model t = 0.4 + 5.79 ns/granule + 2.65 ns if tail > 128 B; per-granule ceiling 44.2 GB/s
     merging rows (one DMA per group): k=1 32 | k=2 37 | k=3 40 | k=4 41 | k=6 42 | k=8 43 GB/s
     where this row can go: 43 GB/s at k=8 (needs 8 adjacent rows on both sides)
```

### 合并实测：模型逐点命中，但撞上了「整组 all-or-nothing」

三条命令，`--src host --rows 2048 --ckv-dim 656 --kpe-dim 0 --dtype uint8`：

| group_rows | `--table-mode 0`（两侧恒等连续） | `--run-len 4`（4 行游程） | `--run-len 1`（全随机） |
|:---|:---|:---|:---|
| 1 | 32.7 | 32.0 | 32.1 |
| 2 | **37.8** | **37.3** | 32.1 |
| 4 | **41.2** | **40.7** | 32.5 |
| 8 | **42.9** | **31.7** ← 掉了 | 31.7 |

三件事：

1. **模型是对的**。用上面那组系数预测 k=2/4/8 是 35.1 / 64.1 / 122.0 ns（整组），
   实测 34.7 / 63.7 / 122.3 ns；换算 GB/s 是 37.3 / 40.9 / 43.0，实测 37.8 / 41.2 / 42.9。
   `k=8` 的 42.9 已经是「每格 44.2 GB/s」这个上限的 97%，**单行路径只剩 1–2% 的空间**。
2. **`--run-len 4` 在 k≤4 上几乎不掉**（37.3 vs 37.8、40.7 vs 41.2，差 1% 是每行多一次行表标量读）。
   也就是说：**4 行局部性就够吃满 4 行合并**，不需要更长的游程。
3. **`--run-len 4` 在 k=8 掉到 31.7**，比不合并还差。这不是带宽问题，是实现的取舍：那时 kernel
   还是「整组要么全连续、要么退回逐行」，8 行里只能凑出 4 行连续 → 整组作废。**group_rows 设成 8
   而合并是全 or 全不的，遇到小于 8 的游程就是负收益。**

这一条已经改掉了：kernel 现在把 group 拆成**行游程**（两侧同时相邻的极大段），每段一次 DMA。
所以同一个 `--run-len 4 --group-rows 8` 应该跑出 ≈40.7（两组 4 行），而不是 31.7。重编后可以
直接用上面那张表验收：k≤4 三列互相对上、k=8 的中间列从 31.7 抬到 40 附近，就说明拆游程生效了。

### 怎么把 656 提上去

这个模型只有两个变量，所以手段也只有两种：**让一次 DMA 覆盖更多行**（摊薄计费单元），
或者**把尾巴消掉**（补齐布局）。上限都是 **44.2 GB/s**（一个 256 B 单元的满速），
实测已经摸到 42.9（k=8，97% 上限）。

1. **合并行**（首选：不增加任何搬运量，已实测）：

   | k | 一次 DMA | 计费 | ns/行 | 实测 GB/s | 模型 |
   |:---|:---|:---|:---|:---|:---|
   | 1 | 656 | 768 + 尾巴 | 20.1 | 32.7 | 32.1 |
   | 2 | 1312 | 1536（尾 32 B，免加价） | 17.4 | 37.8 | 37.3 |
   | 4 | 2624 | 2816（尾 64 B） | 15.9 | 41.2 | 40.9 |
   | 8 | 5248 | 5376（尾 128 B） | 15.3 | **42.9** | 43.0 |

   k=2 起尾巴就都落在「不超半格」一侧，合并同时把那 2.65 ns 消掉了。收益分布很不均匀：
   k=2 已经 +16%，k=3 +24%，k=4 +27%，而 k=4 → k=8 只再 +5% —— 但 k 越大越吃局部性
   （`--run-len 4` 下 k=8 就掉回去了）。**生产上取 3–4 最划算**：拿到七成收益，只要求 3–4 行相邻。

   ```bash
   python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 \
       --table-mode 0 --src host --sweep group-rows
   ```

2. **生产上真正要改的是「怎么挑要合并的行」**。`--run-len k` 给出的答案是：游程 4 行时
   k=2/4 分别是 37.3 / 40.7（对上限列的 37.8 / 41.2，只差行表读的 1%）；游程 1 行时
   无论 group_rows 多大都是 32 —— **合并率 0 时合并一分不赚**。

   所以判断标准只有一条：**topk 选中的行里，相邻行占多大比例**。用 run-length 统计一下生产
   请求的 `srcTokenId` 序列（README 上文有片段），若「相邻对」占比 > 50%，就值得开 group_rows=4。

   ```bash
   # 4 行游程 + 合并 4 行：理想局部性的上限（实测 40.7）
   python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 \
       --run-len 4 --src host --sweep group-rows
   # 全随机置换：合并完全无效的底线（实测 32.1，与 k 无关）
   python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 \
       --run-len 1 --src host --sweep group-rows
   ```

3. **消除尾巴：把行补到 768 B**（`--ckv-dim 768` = 实测 **41.7 GB/s**，单行就能到）。
   代价是每行多搬 112 B：单行时间 20.4 → 17.8 ns（**-13%**），有效载荷带宽 656/17.8 ≈
   **36.9 GB/s（+13%）**，但 HBM 容量和链路流量都 +17%。
   注意只有补到 768 有用：704 / 672 仍在 768 格内且尾巴超半格，一分不省。
   和合并**叠加**时意义不大（k=4 已经是 41.2），所以这是「不做合并、只想单行变快」时的选项。

4. **加深流水**（零成本，值得一并扫）。每行固定开销占比高时，`bank` 从 2 加到 4 / 6 可能再吃回一块：
   ```bash
   python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 \
       --src host --sweep bank
   ```

5. **不要做的三件事**：用 strided 模式合并（UB 行距 672 B 表达不了）；把 656 拆成 512 + 144 两次
   DMA（512 → 2 格 12.0 ns，144 → 1 格 + 尾巴 8.9 ns，合计 20.9 ns，和一次 656 完全一样）；
   把行补到 704 / 672 这类「仍在 768 格内且尾巴超半格」的长度（一分不省）。

### 重编：kernel 变了就必须重打算子包

拆游程（run decomposition）改的是 `op_kernel/*.cpp`，**ABI 没变**，所以只重打算子包的就够；
但 `pip install -e .` 会一起重编两个产物，省事。判断自己为什么报错时按这个对照：

- `no matching function for call to 'aclnnScatterCopyUbBenchGetWorkspaceSize'` → 算子包是旧的
  （上一版加了第 11 个参数 `copy_rope` 时会出现）；
- `scatter_copy_ub_bench() expected 11 arguments` / `argument 'copy_rope' not found` → torch 扩展是旧的；
- **跑出来 k=8 的 `--run-len 4` 还是 31.7** → 算子包没重编（kernel 二进制没更新）。

```bash
SOC_VERSION=ascend950 VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench \
    pip install -e . --no-build-isolation
find . -name 'vllm_ascend_C*.so' -delete   # 若 .so 时间戳比源文件新，先删再装
```

## host 侧带宽只有 42 GB/s：先分清是哪条链路

典型读数：`--src npu` 130 GB/s、`--src host` 42 GB/s。42 GB/s 落在**单条**高速链路的
饱和区（PCIe Gen5 x16 有效 ~42–52、400G 类口 ~45–47），但"单条"不等于"只用了一个口" ——
如果每口是 200G 级（25 GB/s），42 也可能已经是两个口合计。所以顺序是先定性，再谈调优。

> **结论先行（已在本仓库的 950 双 socket 节点上实测）**：
> 单卡逐行 1152 B 走行表时 40.4 GB/s，把 4 行合成一次突发可到 **45.6 GB/s（≈400 Gbps 的 91%）**，
> 这才是这张卡的实际上限；**主机侧带宽随卡数严格线性到 8 卡 322.4 GB/s（99.7%）**——同 socket
> 两张卡 80.6、四张卡 161.2，说明口不是瓶颈，每张卡自己的 host 通路才是。
> 所以单卡用不上「第二个口」，要更多 host 带宽只能**多开卡**；单卡剩下的空间只有
> 「合并行（+8.6%）」和「少读行表（+4%）」这两项。推导见第 3 步的五行实测表。

**1) 先读拓扑：`CPU` 列告诉你 host 走什么，`Affinity` 列告诉你几个口**

```bash
python benchmarks/kernel/npu_topology.py            # 解析 + 汇总
python benchmarks/kernel/npu_topology.py --dump     # 连同 npu-smi 原始表
```

典型 8 die 双 socket 节点的输出：

```
 die  cpu link       affinity  numa node  host port
   0        UB        288-383          0  #0
   1        UB        288-383          0  #0
   2        UB        288-383          0  #0
   3        UB        288-383          0  #0
   4        UB         96-191          1  #1
   5        UB         96-191          1  #1
   6        UB         96-191          1  #1
   7        UB         96-191          1  #1

2 affinity groups -> the node most likely has 2 host ports, one per group:
  port #0: dies [0, 1, 2, 3] (affinity 288-383, numa node 0)
  port #1: dies [4, 5, 6, 7] (affinity 96-191, numa node 1)
```

判读规则：

| 拓扑现象 | 含义 |
|:---|:---|
| `CPU` 列为 `PCIe`，且 `lspci -vv` 显示 `Width x16, Speed 32GT/s` | 单条 PCIe Gen5 x16，42 GB/s 就是上限，没有第二个口可开 |
| `CPU` 列为 `UB`，`Affinity` 只有一组 | host 侧只有一条 UB 通路，die 再多也共用它 |
| `CPU` 列为 `UB`，`Affinity` 分成两组（如 288-383 / 96-191） | **两个 socket，各带一个 host 口**，每组 4 个 die 只能走自己那口 |

第三行就是"2 个 UB 口"的实际情况：口是**按 socket 划分**的，不是同一张卡上有两个口。
一个进程绑在一张卡上，最多只能用到这张卡所属 socket 的那一个口 —— 所以单卡 42 GB/s
不是 kernel 没写满，而是物理上只连了一条路。

**2) 端口计数器只在链路以"端口设备"形式暴露时才有意义**

```bash
python benchmarks/kernel/ub_port_probe.py -- \
    python benchmarks/kernel/scatter_copy_ub_bench.py --src host --iters 2000 --rounds 1
```

脚本会前后采样 `hccn_tool -i N -stat -g`（Ascend 端口设备，工具路径会自动在
`/usr/local/Ascend/driver/tools` 等位置查找）与 `/sys/class/net/*/statistics`，按设备打印
rx/tx 增量与 GB/s。

**关键判读**：如果输出里只有 `net:enpXX` 这类普通网卡、且速率只有几 MB/s，那是 SSH/管理
流量，**不代表 host→device 通路** —— 说明这条链路不以 netdev 形式暴露，计数器这条路走不
通，直接跳到第 3 步。只有出现 `hccn*` 行（或多块网卡有 GB 级增量）时，"几行为非零 = 几个口"
才成立。

**3) 不依赖计数器的决定性实验：多张卡同时跑，看带宽能不能叠加**

拓扑已经说明口是按 socket 分的，所以要两个口一起动，就必须让**两个 socket 上的卡同时跑**
（`--device` 选择卡号，默认 0）：

```bash
python benchmarks/kernel/host_link_scale.py --procs 2 --devices 0,4 --rows 1024
```

脚本会：打印拓扑摘要 → 按 affinity 把 NPU0 绑到它那组所在的 node、NPU4 绑到另一组（`numactl
--cpunodebind/--membind`）→ 测单卡基线 → 测**跨组**两张卡 → 再测**同组**两张卡 → 给判定。
不给 `--devices` 时按"每个 affinity 组取一个代表 die"自动选，正好是一组一个口。

同组对照是关键，它区分两种完全不同的天花板：

| 跨组（两个 socket） | 同组（一个 socket 两张卡） | 结论 |
|:---|:---|:---|
| ≈ 2× 单卡 | ≈ 2× 单卡 | 口不是瓶颈，**瓶颈是单卡的 DMA 引擎**；同 socket 上多开卡继续线性，单卡怎么改都上不去 |
| ≈ 2× 单卡 | ≈ 1× 单卡 | 一个 socket 的**口本身就封顶**在单卡速率；只有换到另一个 socket（或让这张卡寻址另一 socket 的 DRAM）才有增量 |
| ≈ 1× 单卡 | — | 两个 die 抢同一口/同一适配器，并行无收益 |

`--cores` 只在**两个进程共享同一张卡**时才需要给（那时它们抢 AIV）；各占一张卡时 `--cores 0`
即可，强行劈半反而略慢。

**实测（8 die / 4 NUMA node 的 950 节点，`node*/cpulist` = 0-95 / 96-191 / 192-287 / 288-383）**

| 配置 | 带宽 | 倍数 |
|:---|:---|:---|
| 单卡（NPU0，绑 node3） | 40.4 GB/s（`--cores 20` 时 39.2） | 1.00× |
| 同组两卡（NPU0 + NPU1，都在 node3） | 80.6 GB/s | 1.99× |
| 跨组两卡（NPU0 + NPU4，node3 + node1） | 80.7 GB/s（`--cores 20` 时 78.4） | 2.00× |
| 4 卡（单 socket，NPU0–3） | 161.2 GB/s | 3.99× |
| 8 卡（全部 die） | 322.4 GB/s | 7.98× |

五行连起来读，结论是唯一确定的：

1. 同组 2× → 一个 socket 的口本身远不止 40 GB/s（实测 ≥ 80.6），两张卡能从同一个口各取 40；
2. 4 卡 161.2（单 socket 一个口）→ 这个口至少能跑 161，远未封顶；
3. 8 卡 322.4 = 8 × 40.4 的 **99.7%** → 主机侧**没有任何共享瓶颈**，线性度一直保持到 8 卡；
4. 所以 40.4 GB/s 是**单张卡逐行 DMA 的速率**（合并到 45.6 见下一节），既不是口的上限，
   也不是整机上限。

推论：**单卡上换口、换地址区间、跨 socket 取页都拿不到额外带宽** —— 每张卡只有自己那一条
通路，同一个口还有余量也喂不进单卡。要更多主机带宽只有一条路：**多开卡**（40 GB/s × 卡数；
行合并后每卡可到 ~45.6，见下）。

**实测：短行合并能再拿回 ~9%，所以 40.4 不是硬件硬顶**

固定 `--src host --rows 2048 --table-mode 0`（源行连续，排除行表标量读），只扫 `group-rows`：

| `group-rows` | 每次突发长度 | `--rows 2048` | `--rows 1024` |
|:---:|---:|---:|---:|
| 1 | 1152 B | 42.0 GB/s | 40.5 GB/s |
| 2 | 2304 B | 44.7 GB/s | 42.9 GB/s |
| 4 | 4608 B | **45.6 GB/s** | **43.7 GB/s** |
| 8 | 9216 B | 45.5 GB/s | 43.5 GB/s |

三点结论：

1. **42.0 → 45.6（+8.6%）**：1152 B 的单行 DMA 确实有开销，合并 4 行就能拿回大部分；
2. **4 行之后不再涨**（45.6 → 45.5）：4608 B 已到效率平台，再长没有收益，还要冒「源行不连续
   就退化」的风险 —— 生产里 `group_rows` 取 4 就够；
3. **45.6 GB/s ≈ 400 Gbps 的 91%**，这才是这张卡的实际上限。而 `table-mode 1` 的 40.4 里，
   有 ~1.6 GB/s（约 4%）是每行两次 `GlobalTensor::GetValue` 标量读的开销（42.0 vs 40.4）。

生产里能不能吃到这 8.6%，取决于**行号里有多少连续段**：`group-rows > 1` 只在源行号连续时
才合并，随机 topk 下会退化成逐行。所以先量一下真实 miss pair 的 run-length 分布（平均 run ≥ 2
才值得写合并逻辑；只有零星相邻对，就按「逐行 + 减少表标量读」来做）：

```python
import torch

idx = ...  # 一次真实 decode 的 miss pair 源行号，按需排序后的 int64 张量
breaks = idx[1:] != idx[:-1] + 1
run_id = torch.cat([torch.zeros(1, dtype=torch.long), breaks.long().cumsum(0)])
lengths = torch.bincount(run_id)
mergeable = lengths[lengths >= 2].sum().item() / idx.numel()
print(f"rows={idx.numel()} runs={lengths.numel()} "
      f"mean_run={idx.numel() / lengths.numel():.2f} mergeable={mergeable:.1%}")
```

`mergeable` 就是「有多少行能跟相邻行合并」——它乘上 8.6%，才是这个优化在你们真实负载里的收益。

注意这台机器 4 个 NUMA node 只对应 2 个 affinity 组（NPU0-3 → node3、NPU4-7 → node1），
node0/node2 上没有挂 die。

**4) 页跨 NUMA node 只在"同一张卡能看见两个 node"时才有意义**

```bash
python benchmarks/kernel/scatter_copy_ub_bench.py --src host --host-interleave
```

`--host-interleave` 在分配源之前用 `numactl --interleave=all` 重启自己（mempolicy 必须早于
分配）。在 affinity 按 socket 分组的机器上实测**没有变化**，这是正常结果：单卡到另一个
socket 的 DRAM 要么路由不通、要么不按地址位分口，靠页放置改不了端口选择。

**5) 单卡双源窗口：实测已否掉，别做**

原本的设想是给算子加第二个源指针、两段锁页内存分别注册，让读请求分到两个口。
实测（见第 3 步）证明这条路没有收益：一个口能给两张卡各 40，而单卡无论读哪段地址都只有
40 —— 天花板在卡自己的 host 通路上，不在口上，也不在地址分布上。**结论：不要改 kernel。**

真要多拿 host 带宽，只有多开卡这一条路：

| 需求 | 做法 |
|:---|:---|
| 单卡下限 | 逐行 1152 B + 走行表：40.4 GB/s（≈400 Gbps 的 81%） |
| 单卡上限 | 合并 4 行 + 恒等寻址：**45.6 GB/s（≈400 Gbps 的 91%）**，`--rows` 越大越接近 |
| 656 B 单平面行（uint8） | 逐行 32.7 GB/s：DMA 按 256 B 计费（768 B）+ 超半格尾巴（+2.65 ns），与对齐无关。合并 4 行到 41.2、8 行到 42.9（上限 44.2）；或按 768 B 布局单行到 41.7（详见上一节） |
| 多卡 | 线性到 8 卡 322.4 GB/s，按 40 GB/s × 卡数 规划；无共享瓶颈，不需要特别绑定 |
| 生产 kernel 的清单 | ①相邻 miss pair 合并读（+8.6%）②减少每行表标量读（+4%）③其余都是噪声 |

## 已知限制

- 支持 BF16 / FP16 / UINT8（1 字节 cache 行走 uint8，算子只搬字节，不需要 fp8 支持）。
  行字节数**不要求** 32B 对齐：非 32B 整数倍的行（如 656B）在 UB 内按 32B 向上取整落位
  （656 → 672）；这类行仍可 `--group-rows > 1` 合并，但只能用「整组连续」模式
  （`GROUP_MODE_CONTIGUOUS`，两侧步长都为 1），strided 模式要求行距是 32B 整数倍。
- `group-rows > 1` 只有在**源行连续**（`src_rows` 步长为 1）时才会合并读侧 DMA；
  写侧允许任意正向步长，与生产里 miss pair 的合并写一致。随机行表下会退化成逐行 DMA，
  这正是我们想看到的「合并能不能用」的结论。
- `--cores` 超过实际 AIV 数时会在 tiling 里被截断，标签仍显示请求值。
- `--src host` 依赖 `get_npu_view_from_cpu_tensor`（`aclrtHostGetDevicePointer` 的映射视图）。
  **开关是 `PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True`**：torch_npu 的 pinned 分配器
  只有开了这个选项才会把锁页内存映射进 device 地址空间；没开时 `pin_memory()` 只锁页，
  `aclrtHostGetDevicePointer` 失败，接口返回 `None`（旧报错
  `the host buffer is not registered for device access`）。这也是
  `tests/e2e/nightly/single_node/ops/singlecard_ops/test_uva_view.py` 的 skip 条件
  （`"pinned_mem_register:True" not in os.getenv("PYTORCH_NPU_ALLOC_CONF")`）。
  脚本在 `import torch_npu` **之前**自动把它追加到该环境变量；如果 torch_npu 已经被别处
  import（例如用 `python -c` 包装运行），追加不会生效，脚本会给出提示，此时手动
  `export PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True` 再跑即可。
- 兜底路径：若映射仍未生效，脚本会自己对锁页区间调 `aclrtHostRegisterV2(ptr, size, flags)`
  （同 `models/deepseek_v41/engram/npu.py::HostUvaBuffer`）。flags 在 **A5 上只能给
  `ACL_HOST_REG_MAPPED`**（带 PINNED 会被拒，错误码 107000），A2/A3 是 `MAPPED|PINNED`，
  由 `DeviceOperator.host_register_flags()` 自动选。payload 会对齐到 64KiB（同
  `HostUvaBuffer`，A3/CANN 9.2 上未对齐的 device 读回数据不正确），跑完
  `torch.npu.synchronize()` 后再 `aclrtHostUnregister`。
- 计时为 host 侧 `torch.npu.Event` + N 次连续 launch，包含 launch 间隙；需要剔除
  调度开销时请配合 `msprof` 或 `torch_npu.profiler` 看单算子 task 时间。
