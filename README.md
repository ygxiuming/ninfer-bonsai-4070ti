# NInfer · Bonsai-2-27B 三元模型 · RTX 4070 Ti 12GB Linux 部署套件

[![docker-build](https://github.com/ygxiuming/ninfer-bonsai-4070ti/actions/workflows/docker-build.yml/badge.svg)](https://github.com/ygxiuming/ninfer-bonsai-4070ti/actions/workflows/docker-build.yml)
[![release](https://github.com/ygxiuming/ninfer-bonsai-4070ti/actions/workflows/release.yml/badge.svg)](https://github.com/ygxiuming/ninfer-bonsai-4070ti/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![ModelScope 镜像](https://img.shields.io/badge/%E9%AD%94%E6%90%AD%E9%95%9C%E5%83%8F-xiuming%2Fninfer--bonsai--4070ti-8A2BE2)](https://modelscope.cn/models/xiuming/ninfer-bonsai-4070ti)

把 50+ GB 的 Qwen3.8-27B 三元模型（Bonsai-2-27B，权重只剩 -1/0/+1，2.125 bpw），在一张
**12GB 显存的消费级显卡**上跑成 OpenAI 兼容 HTTP 服务。本仓库是一套**照做即通的部署套件**：
引擎源码、启动脚本、Docker 容器、云端自动打包、全流程教程与实测数据都齐——依赖、编译、
权重、启动、验证、踩坑、性能，每一步都有判据。

**v3.0.0 起为双线形态**（兼容新旧模型与算法，同一张卡互斥切换）：

- **老线（A3 引擎）**：出字 75~145+ tok/s、上下文 112K、支持并发，生产可用；
- **fusion KVMem 环线（新，实验）**：逻辑上下文 **262K**（设备池可远小于它，超窗自动检索）、
  解码再快 **+45%~+98%**，代价是单并发 + 预填变慢（[双线对照](docs/性能实测.md#9-双线对照fusion-kvmem-环线-vs-老线2026-10-10同脚本同卡)）。

| | 说明 |
|---|---|
| 适用硬件 | RTX 40 系（sm_89），12GB 显存起步，本文在 **RTX 4070 Ti 12GB** 实测；RTX 50 系镜像由 CI 双架构构建（实卡未验） |
| 适用系统 | **Linux x64**（Ubuntu 26.04 实测；Windows 请直接用官方交付包的 exe） |
| 交付形态 | OpenAI 兼容 HTTP 服务（`/v1/chat/completions`），支持思考模式；两线模型 id 同为 `qwen3.8-27b` |

---

## 快速开始

**方式一：容器线（推荐——宿主机只需 NVIDIA 驱动 + Docker，无需装 CUDA/FFmpeg）**

```bash
git clone https://github.com/ygxiuming/ninfer-bonsai-4070ti.git
cd ninfer-bonsai-4070ti

# —— 老线（日常/生产）：出字 75~145+ tok/s、112K 上下文、可并发 ——
docker pull ghcr.io/ygxiuming/ninfer-bonsai:latest           # CI 容器内编译的成品镜像
./docker/fetch-model.sh --ms                                 # 下载 v2 制品 7.8GB（国内直连 + SHA256 校验）
./docker/run-ninfer.sh -i ghcr.io/ygxiuming/ninfer-bonsai:latest   # 起服务，宿主端口 8089
./docker/test-serving.sh                                     # 冒烟：模型列表 + 一次真实生成

# —— fusion KVMem 环线（实验：262K 逻辑上下文、解码 +45~98%、单并发）——
./docker/fetch-model.sh --ms-fusion                          # 下载 v3 三件约 22GB（可单件：--ms-fusion full）
docker pull ghcr.io/ygxiuming/ninfer-bonsai:latest-fusion
./docker/run-ninfer.sh -l fusion -i ghcr.io/ygxiuming/ninfer-bonsai:latest-fusion
```

**方式二：宿主直跑（已装 CUDA 13.1+/ffmpeg 开发库；或想连模型一起拿，走魔搭克隆）**

```bash
git clone https://www.modelscope.cn/xiuming/ninfer-bonsai-4070ti.git   # 国内直连，vendor+models 一次到位
cd ninfer-bonsai-4070ti
# 宿主直跑需先自编译引擎（vendor/ 只有源码），并把 scripts/start-ninfer.sh
# 顶部路径块改成你的实际路径（作者机 sanyuan 布局可跳过）——步骤见下文安装/使用
./scripts/start-ninfer.sh work       # 老线（端口 8088）；fusion 环线：./scripts/start-fusion.sh（引擎自编译同上）
```

> - 不想本地编译镜像：`docker pull` 拉的就是 CI 编译好的双线镜像（4 个 tag，见下表）；
>   本地自建等价物是 `./docker/build-image.sh [-l fusion]`。
> - GHCR 包首次发布默认私有，需在 GitHub 网页 → Packages → ninfer-bonsai → Package settings
>   改 **Public**（一次性，之后所有 tag 匿名可拉）。
> - 两条宿主线同端口 8088、同卡互斥，切换前先停对方（脚本有端口预检）。

### 双线镜像 tag 速查

| Tag（`ghcr.io/ygxiuming/ninfer-bonsai:` 前缀） | 线 | GPU |
|---|---|---|
| `latest` / `v3.0.0` / `cuda13.3-arch89` | 老线 | RTX 40 系 |
| `cuda13.3-arch120a` | 老线 | RTX 50 系 |
| `latest-fusion` / `v3.0.0-fusion` / `cuda13.3-arch89-fusion` | fusion 环线 | RTX 40 系 |
| `cuda13.3-arch120a-fusion` | fusion 环线 | RTX 50 系 |

### 文档地图

| 想做什么 | 看哪篇 |
|---|---|
| 从 v2.0.0 升级 / 新机完整搭建（含新旧兼容矩阵与回滚） | [docs/更新教程-v3.0.0.md](docs/更新教程-v3.0.0.md) |
| fusion 环线使用说明（三档/验收/调优/已知限制） | [docs/Fusion-KVMem-部署与运行.md](docs/Fusion-KVMem-部署与运行.md) |
| 性能实测与双线 13 项对照 | [docs/性能实测.md](docs/性能实测.md) |
| 踩坑与排障（双线 17+ 坑位） | [docs/踩坑实录.md](docs/踩坑实录.md) |
| 容器化细节（镜像/制品/参数逐项注释） | [docker/README.md](docker/README.md) |
| 更新记录 | [CHANGELOG.md](CHANGELOG.md) |

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

### 双线形态（v3.0.0）

同一个模型、两种引擎世代，按用途选线（同一张卡互斥切换）：

| | 老线（A3 引擎） | fusion KVMem 环线（新） |
|---|---|---|
| 制品 | `models/artifacts-pq2.ninfer`（v2 容器，7.74 GiB） | `models/fusion/` 三件（v3 容器，8.87/7.13/5.96 GiB） |
| KV 算法 | fp8，上下文/池 ≤112K | **rk4v4-e8 + 主机层**，逻辑上下文 **262,144**（设备池可远小于它） |
| 投机 | MTP K=3（高可预测任务 145-173 tok/s） | **DFlash2 + proposal 头**（解码 +45%~+98%） |
| 并发 | 支持（4 路聚合 336 t/s） | **单路**（环的硬约束） |
| 启动 | `./scripts/start-ninfer.sh work` | `./scripts/start-fusion.sh` |
| 适合 | 日常短问答/生产 | 超长上下文、长会话免冷预填 |
| 详细 | 下文 + [docs/性能实测.md](docs/性能实测.md) | [docs/Fusion-KVMem-部署与运行.md](docs/Fusion-KVMem-部署与运行.md) |

> 两代容器**硬校验互斥**（v2 魔数第 8 字节 02 / v3 为 03，引擎各认各代），无混搭路径 ——
> 升级是"并存"不是"替换"，兼容矩阵与升级步骤见 [docs/更新教程-v3.0.0.md](docs/更新教程-v3.0.0.md)。

### 三层精度账（每层都有实测）

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
| 持续解码 | **75~84 tok/s**（说明文/对话） | A3 引擎；接受率强依赖任务：数数/照抄类 88~99%，K=3 实测 **145 tok/s**、K=5 **173 tok/s** |
| 短输出爆发 | 92~106 tok/s | 64~1024 tok 输出实测 |
| **prefill 吞吐** | **2108~2209 tok/s** | A3 引擎（旧版 1694~1837，+24.6%，pkg5 口径 A/B + sha 逐位门） |
| TTFT（短提示） | **81~179 ms** | A3 引擎 179ms |
| TTFT（32k 文档） | ~12.7 s | 一次性读取 |
| 32k 上下文解码 | 83.9 tok/s | 对比空上下文几乎无衰减 |
| **容器化冒烟** | **85.2 tok/s / TTFT 66ms** | vendor 源码容器内编译，与宿主直跑持平 |
| 最大上下文 | **112K**（114,688） | fp8 KV + MTP K=3；131k 差 370MB，262k 需 9.2GB |
| 思考模式 | ✅ | 374 reasoning tok，推理题正确 |
| 贪心确定性 | ✅ | 同 prompt 两次输出逐字一致 |
| 并发聚合吞吐 | 169 / 336 tok/s @2/4 路 | 单路下降但聚合线性 |
| 稳定性 | 20/20 | 连发零错误 |

**A3 引擎更新（2026-09-27 深夜，pkg2 r12 树 + P1/P2/A3 补丁 + SM_COUNT 参数化）**：prefill +24.6%、
**前缀复用恢复**（r12 修复账本 bug，`--no-prefix-reuse` 不再需要）、A3 收益开关
`NINFER_TERNARY_TOKEN_GRID` 默认开（逐位无损已验证，须配 `NINFER_TERNARY_SMALL_T_ROWS=16`
钉行宽绕开上游 32 行档 PPL 缺陷）；decode 高可预测任务 145~173 tok/s。回滚：换用
`vendor/ninfer-4090w-ternary`（旧树，随仓库保留）并加回 `--no-prefix-reuse`。

**官方对照（RTX 4080 SUPER，32GB）**：decode **226 tok/s**（贪心，16k ctx，
MTP 接受率 81.75%）、**262K 上下文**、prefill 2230 tok/s、运行时 8.54 GiB。
→ 显卡带宽决定上限，4080S（736 GB/s）是 4070 Ti（504 GB/s）的 1.46 倍，数字对得上。

**fusion KVMem 环线实测（2026-10-10，同卡同脚本 13 项对照，完整表见 [docs/性能实测.md](docs/性能实测.md) §9）**：

| 指标 | 老线 | fusion 环线 | 变化 |
|---|---|---|---|
| 持续解码（512 tok） | 77.6 tok/s | **117.2 tok/s** | **+51%** |
| 输出 64 tok | 92.7 | **183.7** | **+98%** |
| 32k 上下文解码 | 69.7 | **134.3** | **+93%** |
| 思考轮墙钟 | 3486 ms | **1776 ms** | **−49%** |
| 预填 ~1k / ~32k | 1978 / 2050 | 1120 / 1789 | **−43% / −13%**（主机层搬运 21-24%） |
| 短提示 TTFT | 126 ms | 299 ms | +~170ms 固定开销 |
| 逻辑上下文 / 长会话复用 | 112K / 无主机层 | **262,144** / 追问轮 TTFT **19.5s→270ms**（44k）、**1.0s**（147k 超窗，cache 100%） | 本线新能力 |

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
| **CUDA Toolkit** | 老线宿主编译 **≥ 13.1**；fusion 环线宿主编译 **≥ 12.8**（本文 13.3） | 各引擎树 CMake 硬检查（老树 13.1 / fusion 树 12.8）；**容器线不需要装**（镜像自带 CUDA 13.3） |
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

### 获取本仓库

**① 推荐魔搭（国内直连且自带全部制品：v2 一件 + v3 三件，约 30GB）**：

```bash
git clone https://www.modelscope.cn/xiuming/ninfer-bonsai-4070ti.git
cd ninfer-bonsai-4070ti      # vendor/ 源码 + models/ 制品一次到位，全程无 GitHub
# 只下制品也行（按需单取）：
#   modelscope download xiuming/ninfer-bonsai-4070ti models/artifacts-pq2.ninfer --local_dir .          # 老线 v2
#   modelscope download xiuming/ninfer-bonsai-4070ti models/fusion/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local_dir .   # fusion 全量件
```

**② GitHub 仓库**（内容一致，不含模型）：https://github.com/ygxiuming/ninfer-bonsai-4070ti
（模型制品用 `./docker/fetch-model.sh [--ms|--ms-fusion]` 引导获取。）

### 三条部署路，按你的情况选

| | 容器化（本仓库自带源码） | 路径 A：官方交付包 | 路径 B：自打包 |
|---|---|---|---|
| 引擎源码 | **vendor/（随仓库分发）** | 交付包内 `src-tree/` | [CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)（公开） |
| 模型制品 | `models/`（fetch-model.sh 引导落位） | 包内现成 `.ninfer` | 自己从 GGUF 打包（pack.py） |
| 宿主机要装 | 驱动 + Docker | CUDA 13.1+ / ffmpeg 开发库 | 同左 |
| 适合 | **绝大多数人（推荐）** | 已有交付包者 | 想要全开源链路 / 极客 |
| 耗时 | 首次 ~40 分钟（含镜像下载） | ~30 分钟（含编译） | ~半天（含 50GB 下载） |

**容器化**（宿主机只需 NVIDIA 驱动 + Docker；双线由 `-l` 选）：

```bash
./docker/build-image.sh             # 老线镜像（vendor 源码容器内编译，apt 自动走清华源）
./docker/build-image.sh -l fusion   # fusion 环线镜像（→ tag …-arch89-fusion）
./docker/fetch-model.sh             # 老线 v2 制品校验/落位（SHA256）；fusion 用 --ms-fusion 取 v3 三件
./docker/run-ninfer.sh              # 起老线服务（端口 8089）；fusion：./docker/run-ninfer.sh -l fusion
./docker/test-serving.sh            # 冒烟：模型列表 + 一次真实生成
```

> **跳过本地编译**：CI 已自动构建并发布双线镜像，直接拉取（见文首[快速开始](#快速开始)与
> [tag 速查](#双线镜像-tag-速查)）。容器参数逐项注释见
> [docker/docker-compose.yml](docker/docker-compose.yml)（老线）与
> [docker/docker-compose.fusion.yml](docker/docker-compose.fusion.yml)（fusion）；
> 制品的获取方式见 [docker/README.md](docker/README.md)《模型制品》节。

**路径 A / B 说明**：两条宿主编译路径的编译与运行完全一致，只有"源码与制品从哪来"不同。
官方交付包（**Windows 版，含全部依赖与制品，开箱即跑**）作者网盘直链：

| 档位 | 内容 | 链接 |
|---|---|---|
| 极速档（PQ2 / ninfer 线）·压缩包 | 7.74 GB 制品 + Windows 引擎 + 源码树 | https://pan.quark.cn/s/ef30066ed62a |
| 极速档 · 文件夹 | 同上，文件夹视图 | https://pan.quark.cn/s/c57ab50cc616 |
| 省显存档 | llama.cpp + KVMem 线，SM 75~120a | https://pan.quark.cn/s/ad41b0e4dd52 |
| 均衡档（PTQ1 / ninfer 线） | 制品（权重省 1.18 GiB，decode −16%） | https://pan.quark.cn/s/0e62e85d30f7 |
| 补丁包 | P1/P2/A3 官方补丁（r12 树用） | https://pan.quark.cn/s/f5906c26283f |
| 文件哈希值 | 各档制品 SHA256 清单 | https://pan.quark.cn/s/d6789d82692e |

> 直链为 2026-09-27 深夜更新版（A3 引擎世代）；旧链接失效以本表为准。
> v3 三件（fusion 环线用）来自"模型包-20260930"：https://pan.quark.cn/s/ab6718583f2f

指南与技术文档：[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)（魔搭）。

**魔搭镜像仓库**：本仓库（含 `models/` 下全部制品：v2 一件 + v3 三件，约 30GB）可一键镜像到魔搭：
`./scripts/upload-modelscope.sh <魔搭用户名>` —— 国内克隆/下载免 GitHub 直连。

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

**容器线**（推荐，参数逐项注释见 [docker/docker-compose.yml](docker/docker-compose.yml) /
[docker/docker-compose.fusion.yml](docker/docker-compose.fusion.yml)）：

```bash
./docker/run-ninfer.sh                        # 老线 work 档 64k，端口 8089
./docker/run-ninfer.sh -c 114688              # 老线 long 档 112k（12GB 卡上限）
./docker/run-ninfer.sh -l fusion              # fusion 环线全量档（窗口 65536 + 逻辑上下文 262144）
./docker/run-ninfer.sh -l fusion-mtp          # fusion 环线瘦身档（窗口 131072）
docker compose -f docker/docker-compose.yml up -d            # 老线 compose
docker compose -f docker/docker-compose.fusion.yml up -d     # fusion compose
```

**宿主直跑线**（用本仓库脚本）：

```bash
# 老线：先编辑 scripts/start-ninfer.sh 顶部的路径块：
#   SANYUAN   → 你的工作目录
#   ART_PQ2   → 你的 .ninfer 制品路径
#   ENG_NEW   → 引擎树路径
./scripts/start-ninfer.sh work      # 64k 上下文，思考开启（默认档）
./scripts/start-ninfer.sh long      # 112k 上下文（12GB 卡实测上限）
./scripts/start-ninfer.sh fast      # 8k 上下文，最省显存
./scripts/start-ninfer.sh status    # 查看状态
./scripts/start-ninfer.sh stop      # 停止

# fusion 环线（引擎优先取 vendor 内自编译产物、模型优先取 models/fusion/，亦可自动回退）
./scripts/start-fusion.sh           # 默认档 fusion（全量件 + DFlash2 + 视觉，见 fusion 说明）
./scripts/start-fusion.sh fusion-mtp   # 瘦身档（窗口 131072）
./scripts/start-fusion.sh stop      # 停止（只杀本环线进程，不动老线/容器）
python3 scripts/fusion_accept_test.py  # 验收：短问答 + 40 万字符超窗针测两轮
```

> 两条宿主线同端口 8088、同卡互斥：切前先停对方（脚本会做端口预检并提示）。
> fusion 线首次启动有 10-50 秒自适应测卡并写存档（`~/ninfer-logs/ninfer-fusion-*.log`）。

等价裸命令（work 档）：

```bash
./build/apps/ninfer-serve <你的制品>.ninfer \
  --host 127.0.0.1 --port 8088 --model-id qwen3.8-27b \
  --max-context 65536 --kv-capacity 65536 --kv-dtype fp8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3
# ↑ A3/r12 树无需 prefix-reuse 开关（账本 bug 已修复，踩坑表 #1）；
#   若用 2026-09-26 旧树，须在参数里加回 --no-prefix-reuse
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
python3 scripts/fusion_accept_test.py   # fusion 环线验收：短问答 + 40 万字符超窗针测两轮（只读 HTTP）
```

覆盖：TTFT、持续解码、输出长度影响、预填吞吐曲线、速度-上下文关系、
大海捞针（3 深度）、思考模式对比、采样影响、贪心确定性、effort 兼容回归、
并发吞吐、稳定性。结果存 `perf-results-<时间戳>.json`。典型输出见 [docs/性能实测.md](docs/性能实测.md)
（含双线 13 项对照 §9；两代存档：`docker/perf-results-20260927-232502.json` 老线 /
`docker/perf-results-20261010-195050.json` fusion）。

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

> 262K 需 ~9.2 GB 运行时预留（KV 池 + 草稿工作区），**老线** 12GB 卡物理放不下——
> 这是显存上限，不是引擎上限。**fusion 环线则用"设备池 + 主机层 + 按需检索"绕开它**：
> 逻辑上下文 262,144 而设备池只花 ~2-3 GiB（12GB 实测池被显存封顶 ~5.5 万 token，属正常）。
> 环线速查：kv4v4-e8 ≈ 每页 1.10 MiB；`--kv-capacity auto` + `--host-kv-mib 8192`（内存侧预算）。

---

### 30 系（sm_86）：路线 A 省显存档（llama.cpp + KVMem）速查

30 系跑不了上面的 ninfer 线，走**官方另一条线**（llama.cpp + KVMem）。本仓库已在
**RTX 3060 12GB** 上完整落地、实测、并把日常运维都跑通了 —— 细节见
**[docs/省显存档-路线A-sm86实测.md](docs/省显存档-路线A-sm86实测.md)**，这里只放最常用的：

```bash
cd ninfer-bonsai-4070ti
./scripts/start-kvmem.sh long     # 起服务：256K 上下文档（含视觉 + MTP + 压缩代理）
./scripts/start-kvmem.sh status   # 看状态（引擎 + 代理分别报告）；logs 跟日志
./scripts/start-kvmem.sh stop     # 停掉（打游戏 / 跑视频生成前必做）
```

客户端接 **`http://127.0.0.1:18210/v1`**（自动压缩代理），不要直接接 18202 ——
代理会在历史超 4.5 万 token 时自动把早期对话压成滚动摘要，**长会话不再每轮等十几分钟**
（实测 1.4 万 token 会话 → 引擎只收 1534 tok，且关键记忆点保住）。只要引擎：`KVMEM_PROXY=0`。

| 默认开启 | 关闭方式 |
|---|---|
| 视觉（能贴图） | `KVMEM_VISION=0` |
| MTP1 投机解码（+15~22% 出字） | `KVMEM_MTP=0` |
| 上下文压缩代理（18210 端口） | `KVMEM_PROXY=0` |

| 档位 | 逻辑上下文 | GPU 窗口 | 单轮输出上限 | 提示上限 | 用途 |
|---|---|---|---|---|---|
| `work` | 131072 | 24K 检索 + 10K 生成 | 10240 | 120832 | 默认 |
| `long` | **262144** | 同上 | 10240 | **251904** | 长会话（已实测：12.1 万 token 请求通过） |
| `low` | 65536 | 12K + 6K | 6144 | 59392 | 边办公边跑 |
| `fast` | 8192 | 4K + 2K | 2048 | 6144 | 最省显存 |

**三条硬规则**（踩过才知道）：

1. **提示 token ≤ 上下文 − 单轮预留**（work 档 = 120832）。超了会被**秒拒**：
   `{"error":"prompt + max_tokens exceeds n_ctx"}` —— 这不是卡住、也不是模板问题，就是**会话太长了**。
   处置：新开会话 / `KVMEM_RESERVE=4096 ./scripts/start-kvmem.sh work` 让出空间 / 直接用 `long` 档。
2. **单轮输出 ≤ 预留**（10240，**思考 token 也算在内**），超了截断（`finish_reason: length`）。
3. **重启服务 = 清空 KVMem 缓存** → 长会话的下一个请求要全量重算（12 万 token ≈ 9.5 分钟）。别频繁重启。

**接客户端（ZCode / Chatbox 等）**：Base URL 填 `http://127.0.0.1:18210/v1`（压缩代理；
`KVMEM_PROXY=0` 只起引擎时改填 `http://127.0.0.1:18202/v1`），API Key 随便填，
模型名随便填。**第一次接之前先跑一次** `./patches/apply-kvmem-compat.sh models/Ternary-Bonsai-2-27B-PTQ1_0.gguf`
—— 它会给模型自带的聊天模板打两个兼容补丁，修掉两类"整轮 400"（都不需要重编引擎，
`start-kvmem.sh` 会自动挂上生成的模板）：

| 客户端行为 | 模板报错（服务端原文） | 补丁做了什么 |
|---|---|---|
| 发 `reasoning_effort:"high"`（ZCode 默认） | `Unexpected reasoning effort high. Supported types are xhigh…` | high/max→xhigh、minimal→low、未知→xhigh |
| 把技能清单/环境提醒当 **system 消息插在会话中间**（ZCode 实测在 messages[3]） | `System message must be at the beginning.`（**53 毫秒秒拒**） | 非首条 system 消息按 user 轮渲染，内容保留 |

ZCode 实测：首轮约 100 秒（2.2 万 token 系统提示 + 33 个工具），**第二轮起 1~2 秒**。

**判断它在干活还是卡住**：

```bash
tail -3 "$(ls -t ~/kvmem-logs/kvmem-*.log | head -1)"   # 自动挑最近在写的那个
# 日志文件名跟档位走：work→kvmem-work.log，long→kvmem-long.log，别翻错文件
```

一行日志怎么读（都是**毫秒**，`1384.52 ms` = 1.38 秒）：

| 字段 | 意思 |
|---|---|
| `prompt eval time = 1384 ms / 274 tokens (198 t/s)` | **读你的输入**用了 1.38 秒 —— 慢一般慢在这一项 |
| `cache = 0` / `cache = 12xxxx` | 没命中缓存 / 命中了（复用了几万 token）。**`cache = 0` 本身不可怕**：几百 token 照样秒回；可怕的是 `cache = 0` **配上几万 token 输入**，那要等几分钟 |
| `eval time = 236 ms / 9 tokens (38 t/s)` | **写回答**用了 0.24 秒 |
| `total time = 1620 ms` | 整个请求 1.62 秒 —— 几秒内都算正常 |
| `prompt processing ... progress = 0.64` | 正在预填充，`progress` 在涨就是正常（后面是当前速度） |
| `n_gen = ..., tg = xx t/s` | 正在出字 |
| 日志完全不动 | 这才叫卡住 |

**想用容器跑这条线**（换机器、隔离环境）—— **两种拿镜像的方式，任选**：

| 方式 | 怎么做 | 要编译吗 |
|---|---|---|
| ① **直接从 CI 拉编好的镜像**（推荐；别的机器零依赖） | `docker pull ghcr.io/<owner>/llama-kvmem-bonsai:cuda13.3-arch86`<br>（`<owner>` = 该镜像的构建仓库所有者，本仓库即 `ygxiuming`；包默认私有：先 `docker login ghcr.io`，或网页把它设为 Public） | ❌ **不用** |
| ② 本机自建 | `./docker/build-kvmem-image.sh`（约 15 分钟） | ✅ |

两种方式都还需要三样**镜像里没有**的东西：**模型**（放进 `models/`，镜像不含权重）、
**NVIDIA 驱动 + nvidia-container-toolkit**、**架构匹配**（`arch86` 只能 RTX 30 系）。
镜像内含引擎（sm_86 已编好）+ CUDA 用户态库 + 压缩代理 + 客户端兼容模板；
起服务用 `docker compose -f docker/kvmem/docker-compose.kvmem.yml up -d`
（拉来的镜像只需在同目录放一行 `.env` 指过去）。细节见 **[docker/kvmem/README.md](docker/kvmem/README.md)**。
（注意：`docker/` 下那套 ninfer 镜像只支持 sm_89/120a，与 30 系无关。）

**想做视频生成？** 12GB 卡能玩但只在"尝鲜档"（480p / 约 5 秒 / 每条十几分钟起步），
选型、约束与施工步骤见 **[docs/本地视频生成-方案与约束.md](docs/本地视频生成-方案与约束.md)**。
注意视频模型与文字模型**抢显存**，跑之前先 `./scripts/start-kvmem.sh stop`。

**看图（视觉）—— 默认开启**：视觉塔已装好（`models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`，0.63 GB，
sha256 已核对官方值），启动时**自动挂上**，贴图直接可用（显存 +约 0.6 GB）。

```bash
./scripts/start-kvmem.sh long                      # 默认就带视觉
KVMEM_VISION=0 ./scripts/start-kvmem.sh long       # 显存紧张时关掉视觉
KVMEM_VISION_CPU=1 ./scripts/start-kvmem.sh long   # 保留视觉但把编码器放 CPU
```

实测（RTX 3060 12GB）：首次带图请求 76 秒（含编码器加载），之后同一张图 **2.2 秒**；
中文表格截图能逐字读对表头（14.7 秒 / 2067 提示 token）。
不挂视觉时发图会直接失败（日志 `multimodal_error: multimodal prefill failed or cancelled`，
客户端表现为"一个 token 都不出"）。

**打游戏前先停服务**：游戏和模型都要显存，12GB 不够分（实测见 §8 症状表）。


## 支持（Support）

### 踩坑实录

全部是本文真实踩过的坑，按出现顺序。完整版（含 Windows 侧与双线坑位）见 [docs/踩坑实录.md](docs/踩坑实录.md)。

| # | 症状 | 真因 / 处方 |
|---|---|---|
| 1 | 第二个请求 HTTP 500，`candidate token ledger does not match prompt length` | **前缀复用与草稿账本冲突（上游 bug）** ⇒ 旧树加 `--no-prefix-reuse`；**A3/r12 树已修复**，前缀复用默认开 |
| 2 | 400 `reasoning effort 'high' is not supported` | 模板只认 low/medium/xhigh ⇒ `patches/apply-compat.sh`（P1 别名补丁：high/max→xhigh）；或客户端改发 medium |
| 3 | 转换时 CUDA OOM（Tried to allocate 4.74 GiB） | 12GB 卡跑不下 GPU 量化 ⇒ `--device cpu`（实测仅 131s） |
| 4 | 转换中途被 OOM-kill | 32GB 内存不够 ⇒ 64GB swap |
| 5 | nvcc 并发编译 GCC 段错误 | TMPDIR 在 tmpfs ⇒ `export TMPDIR=$PWD/tmp-nvcc` |
| 6 | 偶发 `cc1plus` ICE | 重试 / `-j6`（多路 nvcc 并发所致） |
| 7 | CMake 配置失败 `libswscale not found` | Linux 线强制 ffmpeg 开发库 ⇒ apt 装"依赖"节列出的包 |
| 8 | `--spec dflash2` 报 `artifact has no DFlash2 weight bundle` | 老线 v2 制品不带 DFlash2 束 ⇒ 老线只能用 MTP（K≤5）；要 DFlash2 走 fusion 环线 + v3 制品 |
| 9 | K=6/7 报 `T must be in [2,6]` | MTP 内核硬限：T=窗口+1≤6 ⇒ **K=5 是上限**（改三处校验也没用） |
| 10 | K=5 接受率远低于官方 226 t/s 的口径 | 交付源码树比官方二进制旧 3KB（草稿策略改进未开源）⇒ decode 有差距，prefill/TTFT/上下文收益照拿 |
| 11 | `pkill -f ninfer-serve` 把自己的脚本杀了 | 模式匹配到自身命令行 ⇒ 用 `pgrep -x ninfer-serve` |
| 12 | 重启服务 FATAL 显存不足 | 旧实例显存未释放 ⇒ 等 `nvidia-smi` 回落到桌面基线再启 |
| 13 | 中文路径 | Linux 编译/运行建议纯 ASCII 路径（Windows exe 则是硬性要求） |
| 14 | compose 起不来：`command.N must be a string` | YAML 把裸数字解析成 int ⇒ command 列表数值项加引号（本仓库 compose 已修） |
| 15 | fusion 引擎拒载老件 / 老引擎拒载 v3 件 | **容器代际硬校验**（魔数第 8 字节 02/03）⇒ 引擎与制品成对使用，无混搭路径（踩坑 5.1） |
| 16 | 环线报 `requires --max-concurrency 1` / 池小于窗口 | 单并发是设计约束；池被显存封顶属正常（看日志 runtime 行）；host 层 FATAL 按报文抬 `--host-kv-mib`（踩坑 5.2/5.3） |
| 17 | 超窗题面"答得通顺但是错的" | 检索被关会**静默答错**（HTTP 200）⇒ 别设 `NINFER_KV_RETRIEVE=0`，超窗针测必须做（踩坑 5.4） |

### FAQ

**Q: 和 llama.cpp 跑 GGUF 比有什么区别？**
A: 本线是 CUDA 原生内核（GEMV/mma 双路径 + MTP 投机解码 + CUDA Graph），官方口径
4080S 上 decode 226 t/s；三元 GGUF 需要 PrismML 特供的 llama.cpp fork，且没有官方
Linux 预编译。两条线制品不通用（.ninfer vs .gguf）。作者另发"超低显存档"
（llama.cpp + KVMem 三元线，显存下限更低、上下文更长）：
https://pan.quark.cn/s/fd20cf86d3ca ——那是另一条引擎线，与本文不通用。

**Q: 能开多路并发吗？**
A: 老线引擎支持，但官方口径单路优先（每加一路单路掉 ~28%，聚合吞吐仍涨：本机 4 路聚合 336 t/s）。
交互延迟优先就保持 1 路。fusion 环线是单并发硬约束（环的设计）。

**Q: 视觉（多模态）能用吗？**
A: 老线制品含视觉塔（333 对象），启动加 `--vision` 即可（+0.27 GiB 权重 / +0.24 GiB 运行时）。
注意 `--vision` 与部分投机配置互斥，且一张图约吃 374 token 上下文。
fusion 环线的全量件**默认开视觉**（瘦身件/PTQ1 件没有视觉塔；视觉路径本线作者未验，见已知限制）；
显存紧时三个旋钮：`NINFER_VISION=off` / `NINFER_VISION_RESIDENCY=overlay`（塔放内存按图借显存）/ `=cpu`。

**Q: 为什么不用官方生产引擎？**
A: 生产引擎二进制里没有三元量化支持（作者原话）。三元支持在 dev fork 线（v1.0.8 系），
这也是所有社区移植（CraneBW / zatfung / 本仓库）的共同基线。

**Q: 4090 / 5090 呢？**
A: 4090 同 sm_89，照抄本文参数即可。5090（sm_120a）官方基线约为 4090 的 1.4-1.5×；
老线 A3 树 device.h 已含 SM120 分支，fusion 树 CMake 亦支持 120a（双线 120a 镜像均由 CI 构建，实卡未验）。

**Q: RTX 30 系 / 更老的卡能跑吗？**
A: NInfer 引擎只支持 sm_89/120a（fusion 环线 CMake 另支持 80/86），容器化也不改变这一点。
老卡走官方"超低显存档"的 llama.cpp 线（SM 75~120a）。

**Q: fusion 环线（新线）和老线怎么选？**
A: 按任务选，同一张卡互斥切换：短问答/日常生产/要并发的场景用**老线**（快、稳、支持并发）；
超长上下文（>112K）、长会话免冷预填的场景用**fusion 环线**（逻辑上下文 262K、解码快 45-98%，
代价是单并发、预填慢 13-43%、短提示 TTFT +170ms）。两线对照与口径见
[docs/性能实测.md](docs/性能实测.md) §9。

**Q: 两条线能同时跑吗？/ 升级要重下模型吗？**
A: 不能同时跑——同卡约 10 GiB/线，12GB 装不下，脚本会做端口预检。升级**不需要重下老模型**：
v3 三件（约 22GB）只在启用 fusion 环线时需要；老线制品与镜像向后兼容，v3.0.0 对老线零改动
（见 [docs/更新教程-v3.0.0.md](docs/更新教程-v3.0.0.md)）。

---

## 路线图（Roadmap）

- [x] 容器化部署（容器内编译 + 国内源 + GPU 架构参数化）
- [x] 云端自动打包（GitHub Actions → GHCR）与 Release 自动发布（v* 标签→谷歌风格模板）
- [x] A3 引擎升级（r12 树：前缀复用修复 + prefill +24.6% + token grid 收益开关）
- [x] RTX 50 系（120a）恢复：A3 树 device.h 已含 SM120 分支（CI 双架构构建）
- [x] **262K 逻辑上下文**（v3.0.0 fusion 环线：设备池 + 主机层 + 按需检索达成，
      12GB 卡实测首跑；见 [docs/Fusion-KVMem-部署与运行.md](docs/Fusion-KVMem-部署与运行.md)）
- [x] **双线容器化**（v3.0.0：`-l old|fusion` 双线镜像 + CI 四腿矩阵 + release 双线发布）
- [x] **超窗检索首测**（147k token 题面 = 设备池 2.7 倍，针测两轮 PASS，追问轮 cache 100% / TTFT 1.0s）
- [ ] fusion 环线视觉路径实测（引擎与全量件都带视觉载荷，但环线作者未验；待本机红图对照）
- [ ] fusion 环线约 95k 题面"零命中"反例定位（上游账本未解，等新源码批或复现归档）
- [ ] 官方 20 基准完整复测（对照 98.2% 保留口径，需从白皮书/eval 配置搭基准集）
- [ ] 大海捞针真实矩阵复测（根因已定位：测试脚本没关思考；关思考后 8k/32k 矩阵待跑）
- [ ] decode 追平官方 226 t/s 口径（差在官方二进制未开源的 3KB 草稿策略，等新源码）
- [ ] CI 镜像层缓存加速（当前每腿全量构建 ~25 分钟+，fusion 腿更久）

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
- **fusion KVMem 环线引擎** © [1314521gjy/ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem)（Apache-2.0，
  @ `acba3d4` 随仓库分发于 `vendor/ninfer-fusion-kvmem/`）
- 三元移植与工具链：[shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)（魔搭）
- 交付包引擎线与制品：作者离线交付（路径 A）
- Linux 移植与性能调优：[CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)、
  [naamfung/zatfung](https://github.com/naamfung/zatfung)（KVMem 线）

## 许可证（License）

本仓库原创代码（`scripts/`、`docker/`、`extras/`、`docs/`）：Apache-2.0。
`vendor/` 内引擎源码遵循其原 Apache-2.0 许可（LICENSE/NOTICE 原样保留于目录内）。
权重与引擎许可不覆盖本仓库、反之亦然，详见 [NOTICE.md](NOTICE.md)。

## 项目状态（Project status）

- **老线（宿主直跑）**：稳定——bench 8/8 回归、13 项性能全量通过（稳定 20/20）；v3.0.0 对老线零改动。
- **fusion KVMem 环线（v3.0.0 新增）**：实验可用——首跑成功（全量件权重 8.43 GiB、
  KV 池 54,784 / rk4v4-e8）、超窗针测两轮 PASS、与老线 13 项同脚本对照完成；
  已知限制见 [Fusion 说明](docs/Fusion-KVMem-部署与运行.md) §7（单并发/视觉未验/95k 反例）。
- **容器线（双线）**：老线镜像沿用（v2.0.0 已验证，双入口冒烟 83.8 / 85.2 tok/s）；
  fusion 镜像由 CI 构建发布（`-arch89-fusion` / `-arch120a-fusion`），本地 `build-image.sh -l fusion` 等价。
- **发布**：`v3.0.0`（GitHub Release 谷歌风格模板 + GHCR 双线镜像
  `v3.0.0`/`latest`（老线）与 `v3.0.0-fusion`/`latest-fusion`；CI 四腿矩阵 89/120a × 老线/fusion）。
- **魔搭镜像**：https://modelscope.cn/models/xiuming/ninfer-bonsai-4070ti
  （含 v2 制品 7.8GB + v3 三件约 22GB，国内直连）。
- **更新记录**：[CHANGELOG.md](CHANGELOG.md)（v3.0.0 / v2.0.0 / v1.0.0）；
  升级步骤：[docs/更新教程-v3.0.0.md](docs/更新教程-v3.0.0.md)。
- **已知限制**：老线 MTP 接受率与官方口径有差距（未开源的 3KB 草稿策略）、
  大海捞针测试脚本需关思考运行、12GB 卡 fp8 上下文上限 112K；
  fusion 环线为单并发实验版（细节见其说明 §7）。
- 交付形态：OpenAI 兼容 HTTP 服务，compose 一键管理（双线各一份 compose）。
