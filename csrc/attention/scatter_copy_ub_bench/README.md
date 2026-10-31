# ScatterCopyUbBench

从 `fused_scatter_copy_sparse_flash_attention` 里抽出来的 **DRAM → HBM 搬运** 微基准算子。
它只保留「缺失 KV 从 DRAM 搬进 HBM 稀疏 Cache」这一段，去掉 Attention、去掉了首填/稳态的
分支判断，剩下的就是一个纯粹的、可单独计时的 UB 中转拷贝。

## 抽象来源

| 生产代码 | 本 demo |
|:---|:---|
| `op_kernel/first_fill_scatter_copy_kernel.h` 的 `CopyIn` / `CopyOut` | `op_kernel/scatter_copy_ub_bench.cpp` 的同名函数 |
| `TQueBind<VECIN, VECOUT, 2>` + `CopyIn(next) / CopyOut(cur)` 两槽软件流水 | `TQue<VECIN>` 深度由 `bank_num` 指定，`bank_num - 1` 级预取；A5 无 `TQueBind`，写法对齐 `fused_quant_scatter_copy_sparse_flash_attention` 的 `UnionCopy()` |
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

A5 没有 Vector 参与的纯拷贝用的是 `TQue<QuePosition::VECIN>` + 显式 fence，而不是
`TQueBind`；本算子已按这个写法实现，所以 910B/A3/A5 共用同一份 kernel 源码（不需要
`_apt.cpp` 和 `opFile.value`，`fused_quant_scatter_copy_sparse_flash_attention` 这个
950 专用算子也是单 `.cpp`）：

```bash
# 全量 950（会生成 csrc/build/autogen/inner/aic-ascend950-ops-info.ini）
SOC_VERSION=ascend950 bash csrc/build_aclnn.sh "$PWD" ascend950

# 只编这个算子：它自己就是 950 可用的 aclnn_inner 算子，所以子集也能生成上面那个 ini
SOC_VERSION=ascend950 VLLM_ASCEND_CUSTOM_OPS=scatter_copy_ub_bench \
    bash csrc/build_aclnn.sh "$PWD" ascend950
```

两个 A5 上需要注意的语义差异：

- `CopyOut` 末尾有 `SetFlag/WaitFlag<MTE3_MTE2>` 再 `FreeTensor`，即每个 bank 的写回
  完成后才释放。生产代码的 source-gather 路径也是这么做的。因此 `bank_num` 在 A5 上
  主要影响 **MTE2 预取深度**，MTE3 是逐 bank 串行排空的；要测极致写侧流水请改用
  `UnionCopy()` 那种「每 slot 一对 `AllocEventID`」的写法。
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
```

输出示例：

```text
              case    rows group bank tbl cores        us     GB/s
------------------------------------------------------------------
          baseline    2048     1    2   1     0      41.2     57.2
```

`GB/s = rows × 1152B / 时延`，即只统计有效载荷（读 1152B + 写 1152B 的实际带宽约为此值的 2 倍）。

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
| `--bank` | UB 流水槽数，`1` = 完全不重叠 | 1 / 2 / 3 / 4 / 6 / 8 |
| `--group-rows` | 多少行合并成一次 `DataCopyPad`（源行连续时） | 1 / 2 / 4 / 8 |
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

## host 侧带宽只有 42 GB/s：先分清是哪条链路

典型读数：`--src npu` 130 GB/s、`--src host` 42 GB/s。42 GB/s 落在**单条**高速链路的
饱和区（PCIe Gen5 x16 有效 ~42–52、400G 类口 ~45–47），但"单条"不等于"只用了一个口" ——
如果每口是 200G 级（25 GB/s），42 也可能已经是两个口合计。所以顺序是先定性，再谈调优。

> **结论先行（已在本仓库的 950 双 socket 节点上实测）**：
> 单卡固定 ~40.4 GB/s，**主机侧随卡数严格线性到 8 卡 322.4 GB/s（99.7%）**——同一 socket
> 两张卡 80.6、四张卡 161.2，说明口不是瓶颈，每张卡自己的 host 通路才是。
> 所以单卡再怎么改都到不了 80，也用不上「第二个口」；要更多 host 带宽只能**多开卡**，
> 容量按 **40 GB/s × 卡数** 估算。推导见第 3 步的五行实测表。

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
4. 所以 40.4 GB/s 是**单张卡自己的 host 通路上限**，既不是口的上限，也不是整机上限。

推论：**单卡上再折腾（双源窗口、换地址区间、跨 socket 取页）都拿不到额外带宽** —— 每张卡
自己的通路固定 ~40，同一个口还有余量也喂不进单卡。要更多主机带宽只有一条路：**多开卡**，
容量规划就按 **40 GB/s × 卡数** 算。

数值上 40.4 GB/s 约等于 400 Gbps 链路的 81%，也就是「短行 DMA 效率」区间：如果想让单卡再
高一点，唯一的合法方向是把 1152 B 的短行合成更长的突发（`--table-mode 0 --sweep group-rows`），
而不是去找第二个口。这也正好是生产 kernel 里「能否合并相邻 miss pair 的读侧 DMA」那个问题。

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
| 单卡 | 最多 ~40.4 GB/s（≈400 Gbps 链路的 81%），`--bank`/`--group-rows` 只能优化「是否接近它」 |
| 多卡 | 线性到 8 卡 322.4 GB/s，按 40 GB/s × 卡数 规划；无共享瓶颈，不需要特别绑定 |
| 想再推高单卡 | 唯一方向是把 1152 B 短行合成更长突发：`--table-mode 0 --sweep group-rows`，看能否从 81% 往 90% 走 |

## 已知限制

- 仅支持 FP16 / BF16；CKV、KPE 的行字节数必须是 32B 对齐（默认 512/64 维天然满足）。
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
