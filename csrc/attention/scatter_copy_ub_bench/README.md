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

# 扫描：每次 launch 搬多少行（斜率是每行成本，截距就是 launch 固定开销）
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --sweep rows

# 真实 DRAM → HBM（源在 host 锁页内存，经 UVA 映射给 device）
# 脚本会自己把 pinned_mem_register:True 加进 PYTORCH_NPU_ALLOC_CONF；
# 若 torch_npu 已被提前 import，则先手动 export 该变量再运行。
python benchmarks/kernel/scatter_copy_ub_bench.py --src host --verify

# 单平面行：每 token 656 B，没有 KPE 平面（1 字节 cache，走 uint8 逐字节搬）
python benchmarks/kernel/scatter_copy_ub_bench.py --ckv-dim 656 --kpe-dim 0 --dtype uint8 --verify
```

输出示例：

```text
              case    rows spc    pool  B/row group bank tbl cores        us  ns/row     GB/s %host
---------------------------------------------------------------------------------------------------
          baseline    2048   1    2048   1152     1    2   1     0      41.2    20.1     57.2   103
```

`GB/s = rows × B/row / 时延`，即只统计有效载荷（读一份 + 写一份的实际带宽约为此值的 2 倍）。
`pool` 是源张量分配的行数，默认等于 `rows`（见下文 `--src-rows`）。
`%host` 是**实测 / host 路径模型**（模型把每次 launch 的 3.3 µs 和 256 B 计费一起算进去）：
100 表示正好落在线上；明显低于 100 表示"按 k 行的价付了钱、却没真合并"（合并是整组
all-or-nothing，`--run-len 1` 时永远合不上）；**远高于 100 表示源没走 host 链路**（`--src npu`）。

## 选卡：`--device` 与 `ASCEND_RT_VISIBLE_DEVICES`

```bash
# 先看这个进程能摸到哪几张卡（顺序就是 --device 的取值）
python benchmarks/kernel/scatter_copy_ub_bench.py --list-devices
#   8 NPU(s) visible
#   --device  name                          physical
#          0  Ascend910_93xxx                       0
#          1  Ascend910_93xxx                       1
#   ...

# 跑第 4 张卡（0 基），`3` 和 `npu:3` 等价
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 2048 --src host --device 3
# stderr 第一行会回显实际选中的卡，日志里直接可查：
#   device: npu3 (Ascend910_93xxx), 8 visible
```

设了 `ASCEND_RT_VISIBLE_DEVICES` 时，`--list-devices` 会多一列 `physical`，把序号和物理卡号对上：

```text
2 NPU(s) visible via ASCEND_RT_VISIBLE_DEVICES=2,5
--device  name                          physical
       0  Ascend910_93xxx                       2
       1  Ascend910_93xxx                       5
```

两条规则：

1. **`--device N` 是"可见卡列表"里的序号，不是物理卡号。** 设了
   `ASCEND_RT_VISIBLE_DEVICES=2,5` 时，`--device 0` 指的是物理 2 号卡，`--device 1` 是物理 5 号卡；
   超出可见数量会直接报错并提示去看 `--list-devices`，不会静默跑错卡。
2. **`ASCEND_RT_VISIBLE_DEVICES` 必须在 `python` 启动前 export**（CANN 在初始化时装它的过滤器），
   而 `--device` 是进程内的 `torch.npu.set_device()`，在第一次分配之前调用一次。所以：
   - 只想换一张卡 → 用 `--device N`；
   - 想让进程**根本看不到**别的卡（避免别人占卡、或想固定编号）→
     `ASCEND_RT_VISIBLE_DEVICES=N python benchmarks/kernel/scatter_copy_ub_bench.py ...`。

为什么这件事在 host 带宽测试里不是小事：950 的 8 个 die 分属两个 socket，**每张卡只经由自己那个
UB 口访问 host DRAM**（见下文拓扑一节）。所以换卡就等于换一条链路，单卡读数会随卡号变（实测
40.4 vs 39.2 这类差别就来自 affinity 与 NUMA 绑定），多进程想叠出 2× 也必须挑**不同组**的卡：

```bash
# 两个进程各占一张卡（不同 socket），并发才能看到两条链路
python benchmarks/kernel/host_link_scale.py --procs 2 --devices 0,4 --rows 1024
```

`--device` 只影响本进程；`host_link_scale.py` 会把 `--devices` 里的每个号分别传给子进程的 `--device`。

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
| `--rows` | 单次 launch 搬运的 token 行数（gather 的数量，不是源的大小） | 512 / 2048 / 8192 / 32768 |
| `--src-rows` | 源张量分配多少行（gather 池），默认 = `--rows`。`131072` + `--rows 4096` 就是真实 KV cache 形状：工作集比搬的载荷大 32 倍。支持乘积写法，`'131072*32'` = 4194304 | 4096 / 131072 |
| `--spaces` | 独立源区域个数，**一次 launch 轮询一个**（默认 1 = 只有一个池，即旧行为）。生产形状是 78 个 cache 分区，每分区 `--space-rows` 行，每个分区搬 `--rows` 行，一个 pass 读完 78 个 | 1 / 2 / 4 / 8 / 16 / 32 / 64 / 78（`--sweep spaces`） |
| `--space-rows` | 每个分区的行数，默认取 `--src-rows`、再不行取 `--rows`。生产是 `'131072*32'` = 4194304 行（32 请求 × 128K token） | `'131072*32'` |
| `--dst-slack` | 目的端为"每行"分配几个槽位（散写程度）。`0` = 自动：单分区 4、多分区 1（78 × 196608 × 4 × 656 B 会是 37 GiB HBM，而载荷只有 9.4 GiB） | 0 / 1 / 4 |
| `--dst-shared` | 所有分区写同一段目的端（覆盖写）。默认关闭：生产把 78 次 gather 拼起来，覆盖写会被 cache 吸收掉一部分流量 | 开关 |
| `--dry-run` | 只打印分配规划（每分区字节数、总 DDR、表大小、pass 载荷）然后退出，不分配。配 `--sweep spaces` / `--sweep grid` 时逐点打印，跑 200~312 GiB 之前先看这个 | 开关 |
| `--sweep grid` | 行长 × 分区数 的矩阵（默认 256/512/1024 B × 1/2/4/8/16/32/64/78 = 24 点），每点重新分配池 | 开关 |
| `--grid-row-bytes` | 配合 `--sweep grid` 的行长列表，逗号分隔；加 656 就把生产行放进同一张矩阵 | `256,512,1024` / `256,512,656,1024` |
| `--xlsx` | 把每个测点写成 xlsx：`结果` + `GBps` / `pass_ms` / `max_min` 三张矩阵 + `说明`。没有 openpyxl 时退化成同名 `.csv` | 路径 |
| `--rotate` | 预生成 N 份行表、每次 launch 换一份（0/1 = 每次都用同一批行）。大池下用它把页表/IOMMU 缓存变冷，接近生产每步换行的情形；轮转本身零开销 | 0 / 8 / 16 |
| `--ckv-dim` | CKV 平面每行元素数（默认 512）；乘 dtype 字节数就是行字节数，DeepSeek 系 1 字节 MLA 用 656 | 512 / 656 / 672 / 1024 |
| `--kpe-dim` | KPE 平面每行元素数（默认 64）；`0` = 不搬 KPE（`copy_rope=0`） | 64 / 0 |
| `--dtype` | `bf16` / `fp16` / `uint8`。1 字节 cache 行一律用 `uint8`：算子只搬字节、不做算术，所以不需要该芯片支持 fp8 | bf16 / uint8 |
| `--run-len` | 行表里连续行游程的长度。默认 1 = 全随机置换（合并率 0）；`4` 表示 4 行连续，接近 topk 命中相邻块时的形状 | 1 / 2 / 4 / 8 |
| `--bank` | UB 流水槽数，`1` = 完全不重叠 | 1 / 2 / 3 / 4 / 6 / 8 |
| `--group-rows` | 多少行合并成一次 `DataCopyPad`（源行连续时）；实测 4 行是效率平台（`--src host` 下 42.0 → 45.6 GB/s），8 行无收益 | 1 / 2 / 4 |
| `--table-mode` | `1` = 走行表 `GetValue`（同生产）；`0` = 恒等寻址 | 0 / 1 |
| `--cores` | 启动的 AIV 数，`0` = 全部 | 8 / 16 / 24 / 48 |
| `--src` | `npu` = 源在 HBM；`host` = 源在 host pinned 内存 | npu / host |
| `--device` | 跑在哪张卡上（默认 0），写 `3` 或 `npu:3` 都行；**索引是"可见卡列表"里的序号**，不是物理卡号。测 host 带宽时两张卡分属不同 socket 才有意义 | 0 / 4 |
| `--list-devices` | 打印本进程能看到的卡、`--device` 序号与卡名的对应关系，然后退出 | 开关 |
| `--host-interleave` | 配合 `--src host`：以 `numactl --interleave=all` 重启，让源页跨 NUMA node（仅当同卡能看到两个 node 时有效） | 开关 |
| `--json` | 每个 case 输出一行 JSON（供 `host_link_scale.py` 之类脚本汇总），不再打印表格 | 开关 |
| `--verify-rows` | 校验多少行，`0` = 全部 | 64 / 0 |
| `--repeat-verify` | 重复跑并重复校验的次数（抓偶发） | 20 |

`--table-mode 0` 与 `--table-mode 1` 的差值，就是**每行两次 `GlobalTensor::GetValue`
标量读**的代价。生产 kernel 里这部分开销与 DMA 是串行的，通常就是 ScatterCopy 时延里
最容易被忽略的一块。

## 大 DDR 池：131072 行里挑 4096 行（`--src-rows` / `--rotate`）

真实 KV cache 不是"分配 4096 行、全搬一遍"，而是**DDR 里 131072 行，topk 挑 4096 行**。
默认的 `--rows` 同时决定了"搬多少"和"源有多大"，测不出池子变大的影响；`--src-rows` 把两者
拆开：

```bash
# 目标形状：DDR 82 MiB（131072 × 656 B），每次只搬 4096 行
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --src-rows 131072 \
    --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src host --rotate 16 --verify

# 池子本身做变量：同一份 4096 行载荷，工作集 32 倍增长
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --ckv-dim 656 --kpe-dim 0 \
    --dtype uint8 --src host --sweep src-rows
```

输出多一列 `pool`（源张量的行数），所以"搬的量没变、池子变了"在表里一眼可见（下面的数字
只是示意列的位置，等真机读数填进来）：

```text
              case    rows spc    pool  B/row group bank tbl cores        us  ns/row     GB/s %host
---------------------------------------------------------------------------------------------------
     src-rows=4096    4096   1    4096    656     1    2   1     0       ...     ...      ...   ...
   src-rows=131072    4096   1  131072    656     1    2   1     0       ...     ...      ...   ...
```

`spc` 是轮询的源分区数（默认 1），见下文 `--spaces`。

### 三个必须注意的点

1. **必须 `--table-mode 1`。** `--table-mode 0` 是恒等寻址（kernel 根本不读行表），永远只碰
   `0..rows-1` 这些行，`--src-rows` 对它没有任何影响 —— 这条组合脚本会直接打 note 提醒。
2. **不加 `--rotate` 时，每次 launch 都读同一批 4096 行。** 那 4096 行只占 2.6 MiB，页表/IOMMU
   表项在第一次 launch 之后就热了，测到的是"大池的**最好情况**"。`--rotate 16` 预生成 16 份行表
   放在 device 上，每次 launch 换一份（`run()` 只是换一个切片，零拷贝零同步），才接近生产里
   "每步 topk 都换行"的冷页表情形。两个数都测，差值就是**页表开销**，不是带宽。
3. **`GB/s` 仍只算载荷**（`rows × B/row`），池子大小不影响这个分母；要比较"同样载荷在不同池子里
   谁更快"，直接看 `GB/s` 或 `ns/row` 即可。内存代价是 `--src-rows × B/row` 的锁页内存：
   656 B 行 82 MiB，bf16 1152 B 行 144 MiB。

### 实测：512 B 行上池子完全不敏感（这就是结论）

`--rows 4096 --ckv-dim 512 --kpe-dim 0 --dtype uint8 --src host`（单卡，同一进程内三次运行）：

| 配置 | 池 | 每次 launch 读的行 | GB/s |
|:---|:---|:---|:---|
| `--src-rows 131072 --rotate 16` | 64 MiB DDR | 每 launch 换一份，16 份轮转 | **44.4** |
| `--src-rows 131072`（无 rotate） | 64 MiB DDR | 每次都同一批 4096 行 | **44.4** |
| 不给 `--src-rows` | 2 MiB DDR | 全部（`pool == rows`） | **44.5** |

三者差 0.2%，也就是**没有差别**，而且这是预期内的，原因不是"DDR 大池不慢"，而是**这个行长已经把
DMA 吃满了，没有局部性可赚**：

- 512 B = 正好 2 个 256 B 计费格、无尾巴、无浪费，模型 44.5 GB/s，实测 44.4（`%host` = 100）；
- 一行一次 DMA、每次 512 B，池子多大都是"一个 512 B 请求"；64 MiB 池子的页在 setup 阶段就被
  `_publish_host_source` 全量写过（页已实、映射已建），地址翻译和 DDR bank/row miss 都被多 outstanding
  请求的流水盖住了，而不是串行付钱；
- 报告的 `GB/s` 只除载荷（`rows × B/row`），池子不进分母，所以"池子变大"本身不会体现在这个数上。

**什么情况下才会看到池子的影响**（按可疑度排序）：

1. 换行字长。真正吃亏的是"计费格有浪费"的行长（656 付 768 B），但那是**长度**效应，不是池效应；
2. 池子跨越 MMU 覆盖范围：几百 MiB ~ GiB，或访问步长大于 DDR 页大小，才是地址翻译/bank 冲突的
   可测区；
3. 源不落在 host。`--src npu` 下同一行 656 B 是 90.9 GB/s（2.6 倍），说明 host 链路本身才是那条
   44 GB/s 的线；**池子效应和链路效应在这台机器上都轮不到排第一**。

**一个容易踩的坑**：不要拿不同 `--rows` 的读数比池子。上一节十点表里的 512 B = 41.2 是
`--rows 2048` 测的，本表 44.4 是 `--rows 4096`；这 7% 来自每次 launch 的固定开销（见下），与池子无关。

### 每次 launch 的固定开销：3.3 µs（五点拟合）

`--sweep rows` 在 656 B 行上跑出来的五点（`--src host --src-rows 131072 --rotate 16`）：

| rows | 1024 | 2048 | 4096 | 8192 | 16384 |
|:---|:---|:---|:---|:---|:---|
| GB/s | 29.8 | 31.9 | 33.2 | 33.9 | 34.3 |
| 旧模型 `%mdl` | 93 | 99 | 104 | 106 | 107 |
| 新模型 `%host` | 101 | 100 | 100 | 100 | 100 |

同一行长、同一份载荷，纯靠"多搬一点"就把 GB/s 从 29.8 抬到 34.3：**这不是带宽变了，是固定成本被摊薄**。
拟合 `t = a × rows + b`：

```text
t = 3.33 us + 18.93 ns × rows      (五点残差 ≤ 0.7%)
```

也就是说**每次 launch 有 3.3 µs 与行数无关的固定成本**（kernel 起块 / 首行延迟 / 收尾 drain），
在 1024 行时占 15%，正好是 29.8 / 34.6 = 86% 的来源。这也解释了旧模型里那个 0.41 ns/row 的
"常数项"：3.3 µs / 2048 行 = 1.6 ns/row，它在 2048 行的拟合里被摊进了斜率，于是"每 256 B 收 5.79 ns"
其实是"5.35 ns + 固定开销的毛刺"。现在模型把它拆成两段，18 个点全部落在 4% 内，上表 `%host`
也从 93→107 的漂移变成五点全 100。

实践含义：

- **绝对数只在相同 `--rows` 下可比**，跨表比行长时要么统一 `--rows`，要么只比 `ns/row`；
- 生产里每步搬的行越多（或把多步合一次 launch），这 3.3 µs 摊得越薄；
- 想在本卡上直接量它：`--sweep rows`（1024/2048/4096/8192/16384，对 `us` 列做线性拟合）；
- 656 B 行在 `--rows 4096` 上实测 **33.2 GB/s**（预测 33.2），渐近值 34.6 —— 仍然远低于合并 k 行的
  42 GB/s：**主因还是 768 B 计费，不是固定开销**。

### 预期与判读（通用）

- 池子变大 + `--rotate` 打开、行长达标（如 512 B）→ 数不动，属正常；
- 池子变大 + `--rotate` 打开、行长偏短（如 656 B）→ 每行可能踩一次地址翻译 miss，是**最坏情况**，
  但大概率仍被 DMA 计费盖住；
- `--src npu`（源在 HBM）做同样的对比，可以把"DDR 侧效应"和"NPU MMU 侧效应"分开：两者都有页表，
  但 HBM 没有 DDR 的 bank/row 局部性问题。

## 78 个 DDR 空间轮询：`--spaces` / `--space-rows`

生产形状不是"一个池、挑几行"，而是 **78 个独立 DDR 分区，每个分区 131072 × 32 行，
每次从每个分区搬 196608 行，78 个分区轮着来**：

```text
spaces                78            DDR 分区数
space_rows      131072×32 = 4194304  每分区行数（32 请求 × 128K token）
rows               196608            每次每分区搬的行数（topk 2048 × mtp 3 × 32 请求）
row_bytes             656            1 字节 MLA 行
一 个 pass          78 次 launch，15335424 行，9.37 GiB 载荷
池总量          78 × 2.56 GiB = 200 GiB DDR
```

单池测不出这件事：`--src-rows` 变大只是让**一个** base 指针覆盖更多页，而 78 个分区是
**78 个不同的 base**，相邻两次 launch 的地址跳 2.56 GiB。地址翻译层级、IOMMU 表项、
DDR bank/row 命中都可能因此变差，而 `GB/s` 的分母不变，所以差异会直接出现在数上。

```bash
# 0) 先看规划，不分配：200 GiB 之前务必跑这条
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --spaces 78 \
    --space-rows '131072*32' --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src host --dry-run

# 1) 单分区基线（2.56 GiB 池，一个 base 指针）
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --src-rows '131072*32' \
    --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src host --device 0

# 2) 78 分区（200 GiB DDR，78 个 base 指针）
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --spaces 78 \
    --space-rows '131072*32' --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src host --device 0

# 3) 四种行长的完整一轮（每个点都会重新分配池，串行释放）
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --spaces 78 \
    --space-rows '131072*32' --kpe-dim 0 --dtype uint8 --src host --sweep row-bytes
```

第 1 条和第 2 条的差别**只有地址跨度**（每分区搬的比例、行表形状、行长全都一样），
所以两者之差就是"78 个分区轮询"本身的代价。想再拆一层，可以把第 1 条换成
`--spaces 1 --src-rows '131072*32'`（同一条命令）与 `--spaces 78 --space-rows '131072*32'`
对比 —— 前者所有分区的数据挤在一个 2.56 GiB 区间里。

### 四种行长的资源预算

`--spaces 78 --space-rows '131072*32'`、每次每分区 196608 行：

| 行长 | 每分区池 | 总 DDR | 每 launch | 一个 pass | 预期 pass 时延 | 预期 GB/s |
|:---|:---|:---|:---|:---|:---|:---|
| 256 B | 1.00 GiB | 78 GiB | 48 MiB | 3.66 GiB | ~89 ms | ~44 |
| 512 B | 2.00 GiB | 156 GiB | 96 MiB | 7.31 GiB | ~178 ms | ~44 |
| 656 B | 2.56 GiB | **200 GiB** | 123 MiB | 9.37 GiB | ~292 ms | ~34.5 |
| 1024 B | 4.00 GiB | 312 GiB | 192 MiB | 14.62 GiB | ~355 ms | ~44.3 |

预期值来自 256 B 计费模型（`t = 3.3 µs + rows × (5.35 ns × ⌈len/256⌉ + 2.9 ns 若尾巴 > 128 B)`）：
256 / 512 / 1024 都是整格，落在 ~48 GB/s 的上限（实测 44 左右）；656 是 768 B + 超半格尾巴，
只有 ~34.6。**pass 时延里有 78 × 3.3 µs = 257 µs 的 launch 固定开销，占 0.07%~0.3%，可忽略** ——
"78 次轮询"本身不是成本，成本只可能来自地址跨度。

### 输出怎么读

表里多一列 `spc`（分区数），多分区时每个点还会打一行 pass 汇总：

```text
              case    rows spc    pool  B/row group bank tbl cores        us  ns/row     GB/s %host
---------------------------------------------------------------------------------------------------
          baseline  196608  78 4194304    656     1    2   1     0    3738.4    19.0     34.5   100
                       pass 78 x 3738.4 us = 291.60 ms; per-space 3712.5..3768.9 us (max/min 1.015)
```

- `us` 是**一次 launch** 的平均（200 次 iters 跨了 2~3 个 pass，均摊了分区差异）；
- `pass` 行是 `spc × us`，以及**一次 pass 里逐分区实测**的最小/最大值。`max/min` 是重点：
  接近 1 说明 78 个分区完全均匀；某个分区明显偏慢就是 NUMA 放置或页表深度的问题，
  用 `--host-interleave` 或绑核再看一次；
- `%host` 仍按单次载荷算，多分区不改变它的含义；`GB/s` 也是**单次 launch** 的，不是 pass 的
  （两者相等，因为 pass 就是 78 次同样的 launch）。

### 判读

| 现象 | 含义 | 下一步 |
|:---|:---|:---|
| 78 分区的 `GB/s` ≈ 单分区的 `GB/s` | 地址跨度无影响，计费模型照旧 | 直接按 256 B 模型规划容量 |
| 78 分区明显偏低（>5%） | 跨度/页表/IOMMU 真的在付钱 | 先确认整条阶梯的 `--dst-slack` 一致（扫描已自动钉成 1，单点仍是 4）；再看 `max/min`：均匀 → 系统性；不均 → 查 NUMA |
| `max/min` > 1.05 | 某一批分区偏慢 | `--host-interleave`、`numactl --cpunodebind`、或换 `--device` |
| pass 时间 **小于** 78 × `us` | 78 次 launch 在 device 侧重叠了 | 正常（run() 不同步），按 pass 时间算吞吐 |

### 分区数扫描：`--sweep spaces` / `--sweep grid`（结果导成 xlsx）

78 是生产的终值，但要判断"轮询分区数本身要不要钱"，得把它当变量：**1 / 2 / 4 / 8 / 16 / 32 / 64 / 78**，
每个 `--space-rows` 都不变，所以每分区的池子、每次 launch 的载荷、搬的比例全都固定，
**唯一变化的是 base 指针个数和它背后的总 DDR**。

```bash
# 一个维度：分区数，行长固定
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --space-rows '131072*32' \
    --ckv-dim 512 --kpe-dim 0 --dtype uint8 --src host --sweep spaces

# 两个维度：3 种行长 × 8 个分区数 = 24 点，直接写成 xlsx
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --space-rows '131072*32' \
    --kpe-dim 0 --dtype uint8 --src host --sweep grid --xlsx bench_spaces.xlsx

# 同样的矩阵，但先看每个点要多少内存（不分配）
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --space-rows '131072*32' \
    --kpe-dim 0 --dtype uint8 --src host --sweep grid --dry-run
```

`--sweep grid` 的行长默认 **256 / 512 / 1024 B**（都是 256 B 的整格，即最干净的对照组），
加 `--grid-row-bytes 256,512,656,1024` 就把生产的 656 B 一起放进同一张矩阵。

`--dry-run` 会逐点打印规划，因为**最后一个点是第一个点的 78 倍内存**：

```text
dry-run: 1024 B rows
    1 space(s) x  4194304 rows =     4.0 GiB DDR, pass   0.19 GiB, dst   0.19 GiB (x1), tables     1 MiB
    8 space(s) x  4194304 rows =    32.0 GiB DDR, pass   1.50 GiB, dst   1.50 GiB (x1), tables     6 MiB
   78 space(s) x  4194304 rows =   312.0 GiB DDR, pass  14.62 GiB, dst  14.62 GiB (x1), tables    58 MiB
```

`dst … (x1)` 是目的端预算和 `--dst-slack`：阶梯扫描时整条阶梯都是 x1（见下面"1 → 2 那一级"），
显式传 `--dst-slack 4` 时 78 分区那点会变成 58.5 GiB。

每个点都会完整地分配→测量→释放（`gc.collect()` + 逐段 unregister），所以 24 点就是 24 轮
setup；1024 B 那一行光分配/释放就可能占掉大半时间。中途某个点失败（内存不够、注册失败）
**不会终止整轮**：它记成一行 `status=setup failed: ...`，其余点照跑，最后一次写出表格，退出码 1。

#### 实测：分区数不是变量，行长度才是

`rows 196608`、每分区 `--space-rows 4194304`（21 倍 gather）、`--src host`、`--dtype uint8`，
四次 `--sweep grid` 的读数（GB/s）：

| 行 \ 分区数 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 78 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 256 B | 33.2 | 26.8 | 25.6 | 25.4 | 25.4 | 25.3 | 25.3 | 25.3 |
| 512 B | 46.2 | 45.4 | 45.2 | 45.6 | 45.6 | 45.6 | 45.8 | 45.9 |
| 656 B | 25.4 | 27.2 | 28.1 | 28.2 | 28.1 | 28.1 | 28.9 | 29.3 |
| 1024 B | 45.5 | 39.2 | 39.4 | 40.2 | 40.0 | 40.0 | 40.7 | 41.3 |

**只看 2 → 78 这一段**：256 −5.6%（16 以后就平了）、512 +1.1%、656 +7.7%、1024 +5.4%。
三个变快、一个变慢 5.6%，而 512 那一行的抖动只有 ±1%，那就是本卡的噪声底。结论：
**78 个分区的地址跨度不付费**，容量可以按行长直接算，不必为"分区多"留余量。

**1 → 2 那一级不能比**：`--dst-slack` 的 auto 在 1 个分区时取 4、多个分区时取 1，于是
ladder 的第一级是唯一一个目标区大 4 倍、只覆盖 1/4 的点，同时变了两个变量。现在
`--sweep spaces` / `--sweep grid` 会把**整条阶梯钉成 1**（也是和生产一致的口径：78 次
gather 是拼接，每个 dst 行写且只写一次），并打一行 note。所以这一列出现的
"33.2 → 26.8" 是目标区布局的差，不是分区数的差 —— 重跑一次 1 个分区就能确认。

**行长才是这里的变量**（同一批数据的每行耗时）：

| 行 B | ns/row @78 | 模型 | `%host` | 每 512 B 计费 | 判读 |
|:---|:---|:---|:---|:---|:---|
| 256 | **10.12** | 5.37 | 189 | 1 格，但花了 512 B 的 91% | 请求受限，25.3 GB/s 离 46 GB/s 的链路顶还有 45% |
| 512 | **11.15** | 10.72 | 104 | 1 格，45.9 GB/s | 甜点：正好顶到链路，矩阵平 |
| 656 | **22.39** | 18.97 | 118 | 2 格 | 花 1024 B 的 90% 时间只搬 64% 数据 |
| 1024 | **24.79** | 21.42 | 116 | 2 格 | 41.3 GB/s，距链路顶 10% |

三件事：

1. **256 B 只比 512 B 便宜 10%（10.12 vs 11.15 ns/行），却只装一半数据。** 这一规模下
   每行有一个 ~9 ns 的地板（≈ 一次请求的价，而不是一个 256 B 格的价），256 B 完全踩在
   地板上。要提升只能**合并行**（两行一次 DMA 就等于 512 B）或把布局做成 512 B。
2. **656 B 在这一规模是按 ~1024 B 计费的**（22.39 ≈ 4 格），不是 4096 行时那个 768 B 的
   3 格（18.97）。所以 29.3 GB/s 不是异常，是这行长在 196608 行上的真实价；比 512 B 差
   36%，而不是旧表里"只差 8.6%"。
3. **模型只在小批量上准**：`%host` 在 512 上是 104（命中），在 256 上是 189，在 656/1024
   上 116~118。模型是 2048~16384 行拟合的，外推到 196608 行 + 21 倍 gather 会偏乐观，
   只有已经顶到链路的 512 不受影响。

pass 时延是 `spaces × us`，随分区数线性增长，斜率就是"一次 launch 的价"（实测 78 分区）：
256 B 155 ms、512 B 171 ms、656 B 343 ms、1024 B 380 ms（含 78 × 3.3 µs 的 launch 固定
开销，占 0.1%）。

#### 下一步：先把 256 B 的地板找出来

上面第 1 条是假设（每行 ~9 ns 是"一次请求"的价），两条短测就能判定：

```bash
# a) 地板是不是行表的标量读？table_mode 0 完全不读表
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --spaces 8 --space-rows '131072*32' \
    --ckv-dim 256 --kpe-dim 0 --dtype uint8 --src host --sweep table-mode

# b) 地板是不是每行一次 DMA？4 行游程 + k=4 合并，两行就等于一个 512 B 请求
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 196608 --spaces 8 --space-rows '131072*32' \
    --ckv-dim 256 --kpe-dim 0 --dtype uint8 --src host --run-len 4 --sweep group-rows
```

(a) 把 25 抬到 ~40 → 地板是行表；仍是 25 → 地板是 DMA 请求数。(b) 的 k=4 若到 ~42，说明
256 B 只是"没合并"，布局上换成 512 B（或 topk 相邻行合并）就能拿回 1.8 倍。只跑 `--spaces 8`
是因为它已经和 78 差 0.4%，省 10 倍内存。

#### xlsx 里有什么

`--xlsx PATH` 一次写 6 个 sheet（需要 `openpyxl`，没有则退化成同名 `.csv` 并提示）：

| sheet | 内容 |
|:---|:---|
| `结果` | 每个点一行，22 列：配置（行 B / 分区数 / 池行数 / bank / group / table）、`us`、`ns/row`、`GB/s`、`model GB/s`、`%host`、`pass ms`、`space min/max us`、`max/min`、池总量、dst 大小、`status`；冻结首行 + 自动筛选 |
| `GBps` | **行长 × 分区数** 的吞吐矩阵，末列是单行模型预测 |
| `nsrow` | 同一个矩阵，值是 `ns/row`。**比行长时先看这张**：GB/s 把行长折进了答案（256 B 的 25 GB/s 和 512 B 的 46 GB/s 差一倍，每行其实只差 10%） |
| `pass_ms` | 同一个矩阵，值是 pass 时延（ms） |
| `max_min` | 同一个矩阵，值是一个 pass 内逐分区时延的 max/min（1.000 = 完全均匀） |
| `说明` | 生成时间、完整命令行、主机、device、src、rows/space-rows、bank/group/run-len/rotate、计时参数，以及每列（含 `dst_slack`）的含义 |

`结果` 里 `status != ok` 的行保留失败原因，不会静默丢掉；矩阵里缺的点是空格，不是 0。

### 三个实现细节（会影响读数，先知道再跑）

1. **池按需填充。** 每个分区的 2.56 GiB 只 malloc + 锁页，只有行表点到的那 196608 行
   （123 MiB）被写进随机数据；其余行是未初始化内存，而校验读的是同一块 host 内存，
   所以 `--verify` 依然逐字节严格。否则 78 × 2.56 GiB 的随机填充要几十分钟。
2. **行表是 78 张，共 58 MiB HBM**，全部预上传，`run()` 只换切片，切换分区零开销。
   `--rotate N` 会变成 78 × N 张，表内存和建表时间都乘 N（`randperm(4194304)` 每张约 30 ms）。
3. **目的端默认每分区一段**（`--dst-slack` 自动取 1，共 9.37 GiB HBM），因为生产把 78 次
   gather 拼起来。`--dst-shared` 改成覆盖写同一个 123 MiB 窗口，会少写 9.2 GiB，读数偏乐观。
   单分区时 `--dst-slack` 自动取 4，与旧行为一致；但**做阶梯扫描时 auto 会把第一级变成
   唯一一个 4 倍目标区的点**，所以 `--sweep spaces` / `--sweep grid` 会把整条阶梯钉成 1
   （并打 note）。想专门比较写侧布局，就显式传 `--dst-slack 4`，它会作用到每一个点。

### 200 GiB 锁页内存的前置条件

- `ulimit -l unlimited`（或至少大于池大小），否则 `pin_memory` 直接失败；
- 每分区一次独立的 `aclrtHostRegisterV2`，78 段；`close()` 会逐段 unregister；
- 分配和释放各要几十秒（78 × 2.56 GiB），`--sweep row-bytes` 跑四个点就要四轮；
- 物理内存不够时脚本会**在分配前**用 `SC_PHYS_PAGES` 拦下并报错，不会跑到一半 OOM。

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

### 实测（`--src host`）：DMA 按 256 B 粒度计费，尾巴超过半格再加一笔

`--src host`、`rows 2048`、`bank 2`、`group 1`、`table 1`、`--sweep ckv-dim` 的全部十点
（脚本现在直接打印 `ns/row` 列，就是这个原因）。"纯 ns/row" 是减去每次 launch 那 3.3 µs
摊到 2048 行上的 1.6 ns 之后的值（见下文「每次 launch 的固定开销」）：

| 行字节 | GB/s | ns/row | 纯 ns/row | 计费字节 | 单元 | ns/256B | `%256` |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 384 | 30.9 | 12.4 | 10.8 | 512 | 2 | 5.40 | 128 |
| 512 | 41.2 | 12.4 | 10.8 | 512 | 2 | 5.40 | 0 |
| 576 | 32.5 | 17.7 | 16.1 | 768 | 3 | 5.37 | 64 |
| 640 | 36.1 | 17.7 | 16.1 | 768 | 3 | 5.37 | 128 |
| 656 | 32.0 | **20.5** | 18.9 | 768 | 3 | 5.33 | **144** |
| 672 | 32.7 | **20.6** | 19.0 | 768 | 3 | 5.37 | **160** |
| 704 | 34.2 | **20.6** | 19.0 | 768 | 3 | 5.37 | **192** |
| 768 | 41.7 | 18.4 | 16.8 | 768 | 3 | 5.60 | 0 |
| 896 | 38.8 | 23.1 | 21.5 | 1024 | 4 | 5.38 | 128 |
| 1024 | 44.3 | 23.1 | 21.5 | 1024 | 4 | 5.38 | 0 |

规则非常干净，一句话：**计费字节 = 256 × ⌈len/256⌉，每 256 B 收 ~5.35 ns（≈47.9 GB/s），
若 `len % 256 > 128`（尾巴超过半格）再加 ~2.9 ns，另加每次 launch 3.3 µs。**

拟合 `t = 3300 ns + rows × (5.35 × ⌈len/256⌉ + 2.9 if len%256 > 128)` 对上表十点、下一节合并
四点、以及 `--sweep rows` 五点（共 18 点）的最大误差 4%（768 那点），多数 1% 以内：

- **384 和 512 一样快（12.4 ns）**：384 也是按 512 计费 —— 粒度确实是 256 的倍数，不是 32/128。
- **576 / 640 / 768 都在 768 这一格**，17.7–18.4 ns；**656 / 672 / 704 也在 768 这一格，却要多花 2.9 ns**，
  差别只在它们的尾巴（144 / 160 / 192 B）超过了 128 B。128 B 整数倍那条假设因此被否掉
  （896 = 7×128 只有 38.8，576 = 4.5×128 反而和 640 一样）。
- 结论：**656 单行的天花板是 34.6 GB/s**（= 656 / (3×5.35 + 2.9)，`rows → ∞`），跟数据量、对齐
  都无关；表里的 32.0 是 `rows 2048` 的读数，扣掉 launch 固定项后 2048 行是 31.9、4096 行是 33.2、
  16384 行是 34.3 —— **同一个行长，随 `rows` 变大而变快**，这正是「固定开销」的指纹。

脚本启动时会把这条行本身的计费、以及「合并 k 行」的整条预测曲线一起打出来（这里的 rows 是
基准点，曲线随 `--rows` 变化）：

```text
row: 656 elems CKV (KPE plane off), uint8, 656 B/token
     DMA: 656 B/row is charged as 768 B (3 x 256 B, and the 144 B tail (> 128 B) adds ~2.9 ns)
     -> ~18.9 ns/row (~35 GB/s useful); the other 112 B of the granule are free
     model t = 3.3 us per launch + rows * (5.35 ns per 256 B + 2.9 ns if tail > 128 B); 47.9 GB/s per granule
     merging rows (one DMA per group, at 4096 rows): k=1 33 | k=2 39 | k=3 41 | k=4 42 | k=6 44 | k=8 44 GB/s
     where this row can go: 44 GB/s at k=8 (needs 8 adjacent rows on both sides)
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

1. **模型是对的**。新系数预测 k=2/4/8 是 37.1 / 40.2 / 41.9 GB/s（`rows 2048`），实测
   37.8 / 41.2 / 42.9 —— 模型在这三点上低 2%，是全部 18 点里偏差最大的一档（大 DMA 的
   每格成本比小 DMA 略低）。`k=8` 的 42.9 已经到「每格 47.9 GB/s」这个单流上限的 90%，
   而 k=1 的 32.7 只有 68%：**单行路径没什么空间了，剩下的空间全在合并**。
2. **`--run-len 4` 在 k≤4 上几乎不掉**（37.3 vs 37.8、40.7 vs 41.2，差 1% 是每行多一次行表标量读）。
   也就是说：**4 行局部性就够吃满 4 行合并**，不需要更长的游程。
3. **`--run-len 4` 在 k=8 掉到 31.7**，比不合并还差。这不是带宽问题，是实现的取舍：那时 kernel
   还是「整组要么全连续、要么退回逐行」，8 行里只能凑出 4 行连续 → 整组作废。**group_rows 设成 8
   而合并是全 or 全不的，遇到小于 8 的游程就是负收益。**

这一条已经改掉了：kernel 现在把 group 拆成**行游程**（两侧同时相邻的极大段），每段一次 DMA。
所以同一个 `--run-len 4 --group-rows 8` 应该跑出 ≈40.7（两组 4 行），而不是 31.7。重编后可以
直接用上面那张表验收：k≤4 三列互相对上、k=8 的中间列从 31.7 抬到 40 附近，就说明拆游程生效了。

### HBM→HBM：同一行 90.9 GB/s —— 256 B 计费是 host 链路的账单

同样 656 B 行、`--rows 4096 --src-rows 131072 --rotate 16`，只换源的落点：

| 源 | GB/s | ns/row | `%host` |
|:---|:---|:---|:---|
| `--src host`（DDR → UB → HBM） | 33.2 | 19.8 | 103 |
| `--src npu`（HBM → UB → HBM） | **90.9** | 7.2 | **283** |

2.6 倍。这不是"NPU 侧少了一次搬运"那么简单，它说明：

1. **上面那套 256 B 计费模型是 host 路径的账单**，不是 DMA 引擎的物理上限。HBM→HBM 时每 256 B
   只等效 ~2.4 ns（不是 5.35），`%host = 283` 就是这个比值；
2. **「656 只能到 32~34 GB/s」只对 DDR→HBM 成立**。KV cache 本来就在 HBM、靠 device 侧
   gather/scatter 搬的场景没有这个惩罚，也就没必要为了 656 去合并行或补到 768 B；
3. **512 B 的 44.4 也不是 DMA 的上限，而是 host 链路的上限** —— 和下文「host 侧带宽只有 42 GB/s：
   先分清是哪条链路」一节对得上。之前把它当成"DMA 每 256 B 收 5.35 ns"的物理极限，是把链路
   计费当成了引擎规格。

要坐实 90.9 还需要排除两个解释：

- **片上缓存命中**：82 MiB 池 + 10 MiB dst 不算大，`--rotate 16` 轮转的 6.5 万行（41 MiB）有可能
  整个住在末级缓存里。验证：`--src npu --src-rows 1048576`（656 MiB，超出任何片上缓存），
  或 `--src npu --sweep src-rows` 看曲线是否在某处掉下来；
- **只有读侧变快**：dst 只有 10 MiB、本来就可缓存，所以"变快的那部分"可能全在源侧。

无论哪个解释成立，结论只会更强：DDR→HBM 的 34.6 GB/s 里套在头上的是 host 链路对小请求的
计费，不是 DMA 引擎的粒度。所以选型前先问一句——这一步是 DDR→HBM（走 host），还是
HBM→HBM（device 侧）？

```bash
# 排除片上缓存：同一行长，池子放到 656 MiB
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --src-rows 1048576 \
    --ckv-dim 656 --kpe-dim 0 --dtype uint8 --src npu --rotate 16
# 两条路径逐点对比
python benchmarks/kernel/scatter_copy_ub_bench.py --rows 4096 --ckv-dim 656 --kpe-dim 0 \
    --dtype uint8 --sweep src
```

### 怎么把 656 提上去

> 本节所有结论都以 **DDR → HBM（`--src host`）** 为前提；`--src npu` 下同一行已经是 90.9 GB/s，
> 下面的手段全部不需要（详见上一节）。

这个模型只有两个变量，所以手段也只有两种：**让一次 DMA 覆盖更多行**（摊薄计费单元），
或者**把尾巴消掉**（补齐布局）。上限是 **47.9 GB/s**（一个 256 B 单元的满速），
实测已经摸到 42.9（k=8，90% 上限）。

1. **合并行**（首选：不增加任何搬运量，已实测）：

   | k | 一次 DMA | 计费 | ns/行 | 实测 GB/s | 模型 |
   |:---|:---|:---|:---|:---|:---|
   | 1 | 656 | 768 + 尾巴 | 20.1 | 32.7 | 31.9 |
   | 2 | 1312 | 1536（尾 32 B，免加价） | 17.4 | 37.8 | 37.1 |
   | 4 | 2624 | 2816（尾 64 B） | 15.9 | 41.2 | 40.2 |
   | 8 | 5248 | 5376（尾 128 B） | 15.3 | **42.9** | 41.9 |

   k=2 起尾巴就都落在「不超半格」一侧，合并同时把那 2.9 ns 消掉了。收益分布很不均匀：
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
   DMA（512 → 2 格 10.7 ns，144 → 1 格 + 尾巴 8.3 ns，合计 19.0 ns，和一次 656 的 18.9 ns 完全一样）；
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

同样的对比放到 656 B 单平面行上是 **33.2（host） vs 90.9（npu）**，见上文「HBM→HBM」一节：
连"每 256 B 收 5.35 ns"这条看起来很像引擎规格的规则，本质也只是**这条链路**的账单。

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
| 656 B 单平面行（uint8） | 逐行 32.7 GB/s：host 链路按 256 B 计费（768 B）+ 超半格尾巴（+2.9 ns），与对齐无关。合并 4 行到 41.2、8 行到 42.9（单流上限 47.9）；或按 768 B 布局单行到 41.7。**`--src npu` 下同一行 90.9 GB/s，这些手段都不需要**（详见上一节） |
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
- `--spaces N` 会分配 N 段独立的源内存，不是一个张量的 N 个视图：`--spaces 78
  --space-rows '131072*32'` 在 656 B 行下是 200 GiB 锁页内存（256/512/1024 B 分别是
  78/156/312 GiB），需要 `ulimit -l` 放开，且分配/释放各几十秒。脚本在分配前用
  `SC_PHYS_PAGES` 拦截明显不够的情况；`--dry-run` 只打印规划不分配。
- 每个分区只给「行表点到的那几行」填随机数据（196608 行 = 123 MiB），其余是未初始化内存。
  校验读的是同一块 host 内存，所以 `--verify` 仍然逐字节严格；被填的行数决定了 setup 时间，
  `--rotate N` 会把被填的行集合并集放大到 N × 196608 行。
- `--sweep row-bytes`（或任何会改变行长/行数的 sweep）每个点都重新分配整个池，78 分区下
  四个点就是四轮 200 GiB 级别的 malloc + 锁页 + 释放。
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
