# NInfer · Bonsai-2-27B 三元模型 · RTX 4070 Ti 12GB Linux 部署实录

> **一句话**：把 50+ GB 的 Qwen3.8-27B 三元模型（Bonsai-2-27B，权重只剩 -1/0/+1），
> 在一张 **12GB 显存的消费级显卡**上跑起来：**出字 80~100 tok/s、上下文 112K、
> 智力保留官方基准的 98.2%**。本仓库是全流程实录：依赖、编译、权重、启动、验证、
> 踩坑、性能数据——照着做即可复现；也提供 **Docker 容器化一键部署与云端自动打包**。

- **适用硬件**：RTX 40 系（sm_89）。本文在 **RTX 4070 Ti 12GB** 实测；
  官方在 **RTX 4080 SUPER（32GB 魔改）** 验证过（262K 上下文 / decode 226 tok/s）。
  RTX 50 系暂不支持：vendor 源码（Ada fork）的 device.h 仅认 SM86/89（见 Roadmap）。
- **适用系统**：**Linux x64**（Ubuntu 26.04 实测；Windows 请直接用官方交付包的 exe）。
- **最终形态**：OpenAI 兼容 HTTP 服务（`/v1/chat/completions`），支持思考模式。

---

## 目录

1. [项目说明（Description）](#项目说明description)
2. [安装（Installation）](#安装installation)
3. [使用（Usage）](#使用usage)
4. [支持（Support）](#支持support)
5. [路线图（Roadmap）](#路线图roadmap)
6. [贡献（Contributing）](#贡献contributing)
7. [作者与致谢（Authors and acknowledgment）](#作者与致谢authors-and-acknowledgment)
8. [许可证（License）](#许可证license)
9. [项目状态（Project status）](#项目状态project-status)

---

## 项目说明（Description）

[NInfer](https://github.com/Neroued/ninfer) 是一个单卡推理引擎（官方支持 Linux + RTX 5090）。
社区把它移植到了 Ada（sm_89）并接入了 **三元量化** 权重格式：

```
Qwen3.8-27B（全精度 FP16，~55 GB）
   │  PrismML Bonsai 2 训练管线（三元原生训练，非事后量化）
   ▼
Bonsai-2-27B 三元权重（GGUF，6.7 GB，值域 {-1,0,+1}，2.125 bpw）
   │  打包器（pack.py，字节级无损搬运 + Hadamard 旋转基 + MTP 头拼接）
   ▼
.ninfer 制品（7.74 GB，含视觉塔与草稿头）
   │  NInfer 引擎（本仓库教程：源码编译，sm_89）
   ▼
你的显卡 —— OpenAI 兼容 HTTP 服务
```

**三层精度账（每层都有实测）**：

| 层 | 损耗 | 证据 |
|---|---|---|
| 模型层：Bonsai 三元 vs FP16 全精度 | **1.8%** | PrismML 官方 20 基准：83.9 vs 85.4（保留 98.2%） |
| 模型层·社区独立复测 | ≈0% | 20 题基准 19/20 vs 19/20 打平；PPL 6.445 vs 参照 6.82（带内） |
| 打包层：GGUF → .ninfer | **0** | 字节级搬运，装配审计 15/15 |
| MTP 头借用 | 0.03% | 12/12 余弦 ≥ 0.99966 |
| 运行时 KV 量化（fp8） | 轻微 | 官方实测省 3.9 GB 显存、速度几乎不动 |

> 为什么三元损耗这么小？因为 Bonsai 不是"把全精度模型量化一下"，而是 PrismML
> 在 Qwen3.8-27B 架构上**原生以三元权重训练**的模型——训练时权重就一直活在
> {-1,0,+1} 的世界里。这是它和普通"量化版"的本质区别。

### 实测结果（RTX 4070 Ti 12GB）

**口径**：AD104，504 GB/s／显示器接核显／Ubuntu 26.04／单路

| 指标 | 数值 | 备注 |
|---|---|---|
| 持续解码 | **76~84 tok/s** | 512 tok 生成；MTP K=3 最优（接受率 36~52%，强依赖任务） |
| 短输出爆发 | 92~106 tok/s | 64~1024 tok 输出实测 |
| **prefill 吞吐** | **1700~1840 tok/s** | 1k→32k 提示几乎不衰减 |
| TTFT（短提示） | **81~190 ms** | 服务端口径 |
| TTFT（32k 文档） | ~12.7 s | 一次性读取 |
| 32k 上下文解码 | 83.9 tok/s | 对比空上下文几乎无衰减 |
| **容器化冒烟** | **85.2 tok/s / TTFT 66ms** | vendor 源码容器内编译，与宿主直跑持平 |
| 最大上下文 | **112K**（114,688） | fp8 KV + MTP K=3；131k 差 370MB，262k 需 9.2GB |
| 思考模式 | ✅ | 374 reasoning tok，推理题正确 |
| 贪心确定性 | ✅ | 同 prompt 两次输出逐字一致 |
| 并发聚合吞吐 | 169 / 336 tok/s @2/4 路 | 单路下降但聚合线性 |
| 稳定性 | 20/20 | 连发零错误 |

**官方对照（RTX 4080 SUPER，32GB）**：decode **226 tok/s**（贪心，16k ctx，
MTP 接受率 81.75%）、**262K 上下文**、prefill 2230 tok/s、运行时 8.54 GiB。
→ 显卡带宽决定上限，4080S（736 GB/s）是 4070 Ti（504 GB/s）的 1.46 倍，数字对得上。

---

## 安装（Installation）

### 依赖

**硬件**：

| 件 | 最低 | 本文实测 |
|---|---|---|
| 显卡 | RTX 40 系，**12 GB** 显存 | RTX 4070 Ti 12GB |
| 显示器 | **建议接核显**（独显跑推理实测损失 ~10% 带宽） | Intel UHD 770 |
| 内存 | 32 GB（路径 B 自打包需 64GB 或等量 swap） | 32 GB + 64GB swap |
| 磁盘 | 20 GB 空余（路径 A / 容器线）；80 GB（路径 B） | NVMe |

**软件**（Ubuntu 实测版本，其他发行版同理）：

| 件 | 版本 | 备注 |
|---|---|---|
| NVIDIA 驱动 | ≥ 580（本文 595.91） | 支持 CUDA 13.x 即可 |
| **CUDA Toolkit** | **13.1+**（本文 13.3） | 引擎 CMake 硬检查 `CUDA >= 13.1` |
| GCC / G++ | 15.2 | CUDA 13.3 支持的宿主编译器 |
| CMake | ≥ 3.28 | pip 装 `cmake` 即可，免 sudo |
| Ninja | 任意近期版 | 同上 |
| **pkg-config + ffmpeg 开发库** | libavformat≥60 / libavcodec≥60 / libavutil≥58 / libswscale≥7 | **Linux 宿主编译必装**（容器线不需要） |
| libcurl 开发库 | ≥ 7.85 | `libcurl4-openssl-dev` |
| Python | 3.10+ | 仅脚本用（urllib，无三方依赖） |
| Docker + nvidia-container-toolkit | 容器线需要 | 驱动 r580+ 即可，容器内自带 CUDA 13.3 |

```bash
# 宿主编译路径一次性装齐系统依赖（Ubuntu）；容器线跳过本步
sudo apt install -y build-essential libavformat-dev libavcodec-dev \
  libavutil-dev libswscale-dev libcurl4-openssl-dev pkg-config
```

### 快速开始

**三条路，按你的情况选**：

| | 容器化（本仓库自带源码） | 路径 A：官方交付包 | 路径 B：自打包 |
|---|---|---|---|
| 引擎源码 | **vendor/（随仓库分发）** | 交付包内 `src-tree/` | [CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)（公开） |
| 模型制品 | `models/`（fetch-model.sh 引导落位） | 包内现成 `.ninfer` | 自己从 GGUF 打包（pack.py） |
| 宿主机要装 | 驱动 + Docker | CUDA 13.1+ / ffmpeg 开发库 | 同左 |
| 适合 | **绝大多数人（推荐）** | 已有交付包者 | 想要全开源链路 / 极客 |
| 耗时 | 首次 ~40 分钟（含镜像下载） | ~30 分钟（含编译） | ~半天（含 50GB 下载） |

**容器化三步**（宿主机只需 NVIDIA 驱动 + Docker，无需装 CUDA/FFmpeg）：

```bash
./docker/build-image.sh      # vendor 源码容器内编译（apt 自动走清华源）
./docker/fetch-model.sh      # 模型制品校验/落位（夸克直链引导，SHA256 校验）
./docker/run-ninfer.sh       # 起服务（或用 docker compose -f docker/docker-compose.yml up -d）
./docker/test-serving.sh     # 冒烟：模型列表 + 一次真实生成
```

> 也可拉取 GitHub Actions 自动打包的现成镜像（见 `docker/README.md`《自动打包》），
> 跳过本地编译。容器参数逐项注释见 [docker/docker-compose.yml](docker/docker-compose.yml)；
> 制品的四种获取方式（含 12GB 卡与 ≥24GB 卡的差异）见
> [docker/README.md](docker/README.md)《模型制品》节。

**路径 A / B 说明**：两条宿主编译路径的编译与运行完全一致，只有"源码与制品从哪来"不同。
官方交付包（**Windows 版，含全部依赖与制品，开箱即跑**）作者网盘直链：

| 档位 | 内容 | 链接 |
|---|---|---|
| 极速档（PQ2 / ninfer 线） | 7.74 GB 制品 + Windows 引擎 + 源码树 | https://pan.quark.cn/s/f72b85b82626 |
| 均衡档（PTQ1 / ninfer 线） | 7.05 GB 制品（权重省 1.18 GiB，prefill −10%） | https://pan.quark.cn/s/0a799654ba7e |
| 超低显存档（llama.cpp + KVMem 线） | llama 线源码与制品，SM 75~120a | https://pan.quark.cn/s/fd20cf86d3ca |

指南与技术文档：[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)（魔搭）。

### 路径 A：官方交付包

**A-1 系统依赖**：按上面 apt 命令装齐（**Linux 编译必须 ffmpeg 开发库，没有关闭开关**）。

**A-2 编译引擎**：

```bash
# 1) 把交付包里的源码树拷到纯 ASCII 本地路径（如 ~/pyprojects/ninfer-4090w-ternary）
cp -a <交付包>/极速档（ninfer）/src-tree/ninfer-4090w-ternary ~/pyprojects/ninfer-4090w-ternary
cd ~/pyprojects/ninfer-4090w-ternary

# 2) 编译（TMPDIR 必须离开 tmpfs，否则 nvcc 并发编译段错误）
export TMPDIR=$PWD/tmp-nvcc && mkdir -p "$TMPDIR"
export CUDA_PATH=/usr/local/cuda-13.3
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
ninja -C build -j"$(nproc)"

# 3) 判据：build/apps/{ninfer,ninfer-serve,ninfer-perplexity} 三个产物齐全
#    （作者验证 549/549 步；偶发 cc1plus ICE 重试即可，或 -j6）
```

**A-3 制品**：交付包内 `model/bonsai2_27b_ternary_v2.ninfer`（PQ2 极速档，7.74 GB，
**含视觉塔**）开箱即用，**SHA256 校验后拷到纯 ASCII 路径**即可。均衡档 PTQ1（5.52 GiB 权重）
为可选项：本机实测 decode 反而慢 ~16%（trit 解包吃算力），仅省 0.76 GB 显存，不作推荐。

### 路径 B：从 GGUF 自打包（全开源工具链）

> 完整细节见 [shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)
> （工具链副本也在本仓库 `vendor/shensanshu-guide/`）。本节是 Linux 化的踩坑版流程。

**B-1 下载三个权重（全部走国内镜像）**：

```bash
pip install modelscope   # 或 uv venv 后安装
# 1) 三元 GGUF（6.7 GB，魔搭有 prism-ml 原生镜像）
modelscope download --model prism-ml/Ternary-Bonsai-2-27B-gguf \
  --include "Ternary-Bonsai-2-27B-PQ2_0.gguf" --local_dir <models>
# 2) 底座（~50 GB，供模板转换）
modelscope download --model Qwen/Qwen3.8-27B --local_dir <models>/Qwen3.8-27B
# 3) 草稿头（3.6 GB，魔搭镜像）
modelscope download --model incoai/Qwen3.8-27B-DFlash2 --local_dir <models>/Qwen3.8-27B-DFlash2
```

**B-2 编译引擎**：同路径 A，源码换 `git clone https://github.com/CraneBW/ninfer-ternary-bonsai-ada`。

**B-3 生成 groupwise-int 模板（吃内存的大头）**：

```bash
# 32GB 内存请先扩 swap（转换瞬时需求 >64GB）
sudo fallocate -l 64G /swapfile-ninfer && sudo chmod 600 /swapfile-ninfer \
  && sudo mkswap /swapfile-ninfer && sudo swapon /swapfile-ninfer

python -m tools.convert.qwen3_8_27b.convert \
  --model <models>/Qwen3.8-27B \
  --dflash2-model <models>/Qwen3.8-27B-DFlash2 \
  --out <models>/qwen3_8_27b_groupwise_int.ninfer \
  --device cpu          # ★ 12GB 显存的卡务必用 cpu（GPU 会 OOM）；实测 131 秒
```

判据：`identity.weights_id == "groupwise-int"`（用 `python -m tools.artifact.inspect <out> --objects` 查）。

**B-4 打包三元制品**：

```bash
# pack.py 里 NINFER_ROOT 常量改为你的引擎树路径（原值是 Windows 的 E:\...）
python3 tools/pack.py check  --gguf <models>/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                             --template <models>/qwen3_8_27b_groupwise_int.ninfer
python3 tools/pack.py build <models>/Ternary-Bonsai-2-27B.ninfer \
                            --gguf <models>/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                            --template <models>/qwen3_8_27b_groupwise_int.ninfer
```

判据：`check` 全绿（zero_share=0.3278 与理论精确一致）；制品文本部分 **6.696 GiB**（红线内）。

> **辅助工具**（本仓库 `extras/`，自包含零依赖）：
> `dump_gguf_meta.py` —— 校验 GGUF 元数据四项判据并导出 `prism.*` JSON
>（供 `check_signs.py` 使用）；`gguf_meta.py` + `_ternary_ref.py` ——
> 指南 verify 快照缺失的 `_ternary_ref` 模块的自包含实现
>（拷进 `tools/verify/` 与 check_*.py 同目录即可，无需 pack.py / 引擎源码树）。

**B-5 验证（四个脚本，缺一不可）**：

```bash
cd tools/verify
python check_payload_order.py <art> 'text/layers/0/gdn/output'   # row-split 三平面
python check_signs.py        <art> <hadamard-meta.json>          # 符号表逐值一致
python check_row_order.py    <art> <gguf>                        # 行序多重集
python check_assembly.py     <art> <gguf>                        # ★ 目标 15/15
```

判据：4 个全绿（尤其 `check_assembly` **15/15**）。

**B-6 数值金判据**：

```bash
./build/apps/ninfer-perplexity <art> --text <英文语料> --context 512 --stride 256
# PPL 落在个位数~低两位数 = 数值正确；几百 = 权重装配错了（"通顺的胡话"）
```

---

## 使用（Usage）

### 启动与验证

**容器线**（推荐，参数逐项注释见 [docker/docker-compose.yml](docker/docker-compose.yml)）：

```bash
./docker/run-ninfer.sh                        # work 档 64k，端口 8089
./docker/run-ninfer.sh -c 114688              # long 档 112k（12GB 卡上限）
docker compose -f docker/docker-compose.yml up -d    # 或 compose 管理
```

**宿主直跑线**（用本仓库脚本）：

```bash
# 先编辑 scripts/start-ninfer.sh 顶部的路径块：
#   SANYUAN   → 你的工作目录
#   ART_PQ2   → 你的 .ninfer 制品路径
#   ENG_NEW   → 引擎树路径
./scripts/start-ninfer.sh work      # 64k 上下文，思考开启（默认档）
./scripts/start-ninfer.sh long      # 112k 上下文（12GB 卡实测上限）
./scripts/start-ninfer.sh fast      # 8k 上下文，最省显存
./scripts/start-ninfer.sh status    # 查看状态
./scripts/start-ninfer.sh stop      # 停止
```

等价裸命令（work 档）：

```bash
./build/apps/ninfer-serve <你的制品>.ninfer \
  --host 127.0.0.1 --port 8088 --model-id qwen3.8-27b \
  --max-context 65536 --kv-capacity 65536 --kv-dtype fp8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 \
  --no-prefix-reuse        # ★ 必加！见下方踩坑表 #1
```

**验证**：

```bash
curl -s http://127.0.0.1:8088/v1/models          # 期望 200 + max_model_len
curl -s http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

- **请求体必须带 `model` 字段**（与 llama.cpp 不同，缺了会被拒）；
- 思考模式默认开启（模板决定），请求级开关：`"enable_thinking": true/false`、
  `"reasoning_effort": "low/medium/xhigh"`（OpenAI 系客户端默认发的 `high/max`
  会被 `patches/apply-compat.sh` 的别名补丁自动映射到 xhigh——源码树打一次即可）；
- 回归测试：`python3 scripts/bench_ninfer.py`（8 项，含生成的代码实际执行验证）。

### 性能测试

```bash
python3 scripts/perf_test.py            # 13 项全量（10-15 分钟），--quick 为 3 分钟快速档
python3 scripts/suite_ninfer.py         # 深度套件：K 矩阵/投机一致性/捞针/PPL（自管服务生命周期）
```

覆盖：TTFT、持续解码、输出长度影响、预填吞吐曲线、速度-上下文关系、
大海捞针（3 深度）、思考模式对比、采样影响、贪心确定性、effort 兼容回归、
并发吞吐、稳定性。结果存 `perf-results-<时间戳>.json`。典型输出见 [docs/性能实测.md](docs/性能实测.md)。

### 上下文与显存速查表

**公式**：`KV 池显存 ≈ 上下文 tokens × KV 密度`（fp8 ≈ 33-35 KiB/token、int8 ≈ 43、rk4v4 ≈ 17-24）
**硬限制**：`--kv-capacity ≥ --max-context`（本线引擎强制）；MTP 草稿窗口另需工作区（K 越大越多）。

| 显卡 | KV 档 | 实测最大上下文 | 备注 |
|---|---|---|---|
| 4070 Ti 12GB | fp8 + MTP K=3 | **112K** | 131k 差 370MB |
| 4070 Ti 12GB | fp8 无投机 | 131K（临界） | 日常不建议贴边 |
| 4070 Ti 12GB | int8 | 64K 稳 | KV 质量最好的档 |
| 4070 Ti 12GB | rk4v4（CraneBW 线） | 80~96K | 容量档，质量略降 |
| 4080 SUPER 16GB | fp8 | **262K** | 官方口径，运行时 8.54 GiB |
| 4080 SUPER 32GB 魔改 | fp8 | 262K + 更大并发余量 | 官方验证机 |

> 262K 需 ~9.2 GB 运行时预留（KV 池 + 草稿工作区），12GB 卡物理放不下——
> 这是显存上限，不是引擎上限。

---

## 支持（Support）

### 踩坑实录

全部是本文真实踩过的坑，按出现顺序。完整版（含 Windows 侧）见 [docs/踩坑实录.md](docs/踩坑实录.md)。

| # | 症状 | 真因 / 处方 |
|---|---|---|
| 1 | 第二个请求 HTTP 500，`candidate token ledger does not match prompt length` | **前缀复用与草稿账本冲突（上游 bug）** ⇒ 服务端必加 `--no-prefix-reuse` |
| 2 | 400 `reasoning effort 'high' is not supported` | 模板只认 low/medium/xhigh ⇒ `patches/apply-compat.sh`（P1 别名补丁：high/max→xhigh）；或客户端改发 medium |
| 3 | 转换时 CUDA OOM（Tried to allocate 4.74 GiB） | 12GB 卡跑不下 GPU 量化 ⇒ `--device cpu`（实测仅 131s） |
| 4 | 转换中途被 OOM-kill | 32GB 内存不够 ⇒ 64GB swap |
| 5 | nvcc 并发编译 GCC 段错误 | TMPDIR 在 tmpfs ⇒ `export TMPDIR=$PWD/tmp-nvcc` |
| 6 | 偶发 `cc1plus` ICE | 重试 / `-j6`（多路 nvcc 并发所致） |
| 7 | CMake 配置失败 `libswscale not found` | Linux 线强制 ffmpeg 开发库 ⇒ apt 装"依赖"节列出的包 |
| 8 | `--spec dflash2` 报 `artifact has no DFlash2 weight bundle` | 现有制品都不带 DFlash2 束 ⇒ 只能用 MTP（K≤5） |
| 9 | K=6/7 报 `T must be in [2,6]` | MTP 内核硬限：T=窗口+1≤6 ⇒ **K=5 是上限**（改三处校验也没用） |
| 10 | K=5 接受率远低于官方 226 t/s 的口径 | 交付源码树比官方二进制旧 3KB（草稿策略改进未开源）⇒ decode 有差距，prefill/TTFT/上下文收益照拿 |
| 11 | `pkill -f ninfer-serve` 把自己的脚本杀了 | 模式匹配到自身命令行 ⇒ 用 `pgrep -x ninfer-serve` |
| 12 | 重启服务 FATAL 显存不足 | 旧实例显存未释放 ⇒ 等 `nvidia-smi` 回落到桌面基线再启 |
| 13 | 中文路径 | Linux 编译/运行建议纯 ASCII 路径（Windows exe 则是硬性要求） |
| 14 | compose 起不来：`command.N must be a string` | YAML 把裸数字解析成 int ⇒ command 列表数值项加引号（本仓库 compose 已修） |

### FAQ

**Q: 和 llama.cpp 跑 GGUF 比有什么区别？**
A: 本线是 CUDA 原生内核（GEMV/mma 双路径 + MTP 投机解码 + CUDA Graph），官方口径
4080S 上 decode 226 t/s；三元 GGUF 需要 PrismML 特供的 llama.cpp fork，且没有官方
Linux 预编译。两条线制品不通用（.ninfer vs .gguf）。作者另发"超低显存档"
（llama.cpp + KVMem 三元线，显存下限更低、上下文更长）：
https://pan.quark.cn/s/fd20cf86d3ca ——那是另一条引擎线，与本文不通用。

**Q: 能开多路并发吗？**
A: 引擎支持，但官方口径单路优先（每加一路单路掉 ~28%，聚合吞吐仍涨：本机 4 路聚合 336 t/s）。
交互延迟优先就保持 1 路。

**Q: 视觉（多模态）能用吗？**
A: 制品含视觉塔（333 对象），启动加 `--vision` 即可（+0.27 GiB 权重 / +0.24 GiB 运行时）。
注意 `--vision` 与部分投机配置互斥，且一张图约吃 374 token 上下文。

**Q: 为什么不用官方生产引擎？**
A: 生产引擎二进制里没有三元量化支持（作者原话）。三元支持在 dev fork 线（v1.0.8 系），
这也是所有社区移植（CraneBW / zatfung / 本仓库）的共同基线。

**Q: 4090 / 5090 呢？**
A: 4090 同 sm_89，照抄本文参数即可。5090（sm_120a）官方基线约为 4090 的 1.4-1.5×，
但本仓库 vendor 源码（Ada fork）暂编不了 120a，需上游 Neroued 树（见 Roadmap）。

**Q: RTX 30 系 / 更老的卡能跑吗？**
A: NInfer 引擎只支持 sm_89/120a，容器化也不改变这一点。老卡走官方"超低显存档"的
llama.cpp 线（SM 75~120a）。

---

## 路线图（Roadmap）

- [x] 容器化部署（容器内编译 + 国内源 + GPU 架构参数化）
- [x] 云端自动打包（GitHub Actions → GHCR，arch89/120a 双变体）
- [ ] 官方 20 基准完整复测（对照 98.2% 保留口径，需从白皮书/eval 配置搭基准集）
- [ ] 大海捞针真实矩阵复测（根因已定位：测试脚本没关思考；关思考后 8k/32k 矩阵待跑）
- [ ] decode 追平官方 226 t/s 口径（差在官方二进制未开源的 3KB 草稿策略，等新源码）
- [ ] 262K 上下文（等上游 prefill 分页溢出实现；12GB 卡另有物理显存限制）
- [ ] CI 镜像层缓存加速
- [ ] RTX 50 系（sm_120a）镜像：vendor 的 device.h 当前仅认 SM86/89（CI 实测），
      需交付树同步上游 device.h 或改用上游树构建

## 贡献（Contributing）

Issue / PR 欢迎。改 `scripts/` 或 `docker/` 请先跑 `python3 scripts/bench_ninfer.py`
（8 项回归）确认无回归；改 `vendor/` 请保持与上游树的可追溯性（单独提交并注明来源 revision）。

## 作者与致谢（Authors and acknowledgment）

本仓库是**部署实录与教程**。`scripts/`、`docker/` 与 `extras/` 为本仓库原创；
引擎源码与指南工具链以 Apache-2.0 随仓库分发于 `vendor/`（原 LICENSE/NOTICE 保留在原位）。
不含任何模型权重。所有组件归属原作：

- **Bonsai-2-27B** © Prism ML, Inc.（Apache-2.0）—— *"Created using Bonsai by Prism ML."*
- **Qwen3.8-27B** © Alibaba Cloud（Apache-2.0）
- **NInfer 引擎** © [Neroued/ninfer](https://github.com/Neroued/ninfer)（Apache-2.0）
- 三元移植与工具链：[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)（魔搭）
- 交付包引擎线与制品：作者离线交付（路径 A）
- Linux 移植与性能调优：[CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)、
  [naamfung/zatfung](https://github.com/naamfung/zatfung)（KVMem 线）

## 许可证（License）

本仓库原创代码（`scripts/`、`docker/`、`extras/`、`docs/`）：Apache-2.0。
`vendor/` 内引擎源码遵循其原 Apache-2.0 许可（LICENSE/NOTICE 原样保留于目录内）。
权重与引擎许可不覆盖本仓库、反之亦然，详见 [NOTICE.md](NOTICE.md)。

## 项目状态（Project status）

- **宿主直跑线**：稳定——bench 8/8 回归、13 项性能全量通过（稳定 20/20）。
- **容器线**：已验证——vendor 源码容器内编译出镜像，双入口冒烟 83.8 / 85.2 tok/s，
  与宿主直跑持平（2026-09-27）。
- **已知限制**：MTP 接受率与官方口径有差距（未开源的 3KB 草稿策略）；
  大海捞针测试脚本需关思考运行；12GB 卡上下文上限 112K。
- 交付形态：OpenAI 兼容 HTTP 服务，compose 一键管理。
