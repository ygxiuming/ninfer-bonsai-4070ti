# Docker 容器化部署（容器内编译 + 国内源）

宿主机只要求: **x86_64 Linux + NVIDIA 驱动(r580+，支持 CUDA 13.x) + Docker + nvidia-container-toolkit**。
无需在宿主机安装 CUDA Toolkit、GCC、FFmpeg 开发库 —— 编译与依赖全部在容器内完成，
CUDA 用户态库随镜像走，显卡驱动用宿主的（NVIDIA 容器运行时透传，性能与宿主直跑无差别）。

## 三个脚本

| 脚本 | 作用 |
|---|---|
| `build-image.sh` | 收集本地引擎源码 → `docker build`（容器内编译 `ninfer` / `ninfer-serve`） |
| `run-ninfer.sh` | 启动服务容器，参数等价宿主 `start-ninfer.sh`（默认 work 档 64k，端口 8089 避让宿主 8088） |
| `test-serving.sh` | 冒烟测试: `/v1/models` + 一次真实生成（关思考、temp=0） |

```bash
./docker/build-image.sh            # 首次含基础镜像下载，见下表
./docker/fetch-model.sh            # 校验/落位模型制品（models/，不随 git 分发）
./docker/run-ninfer.sh             # 或 -c 114688（long 档）/ -p 8088（自定端口）
./docker/test-serving.sh
./docker/run-ninfer.sh stop
```

不在 docker 组时脚本自动改用 sudo；50 系卡构建: `./docker/build-image.sh -a 120a`（tag 后缀 arch120a）。

## 方式 B：docker compose（参数全部可见可改）

与 `run-ninfer.sh` 二选一（容器同名 `ninfer-serve`，不要同时用）：

```bash
docker compose -f docker/docker-compose.yml up -d      # 启动
docker compose -f docker/docker-compose.yml logs -f    # 跟随日志
docker compose -f docker/docker-compose.yml ps         # 状态
docker compose -f docker/docker-compose.yml down       # 停止并移除
```

`docker-compose.yml` 里**每个启动参数都带一行中文注释**，手改即可：
制品路径映射（volumes 左侧）、宿主端口（ports 左侧）、上下文档位（work 65536 /
long 114688 / fast 8192）、MTP K（draft-tokens，3 最优上限 5）、并发路数、
思考开关（删 `--no-thinking` 行首注释）、GPU 卡号（count）。

## 需要下载的镜像（构建前拉好）

| 镜像 tag | 用途 | 压缩体积(amd64) |
|---|---|---|
| `nvidia/cuda:13.3.0-devel-ubuntu24.04` | 构建层（含 nvcc） | ~3.5GB（cudnn-devel 实测 3.87GB 作参照） |
| `nvidia/cuda:13.3.0-runtime-ubuntu24.04` | 运行层（最终镜像底座） | ~0.3GB（cudnn-runtime 实测 1.85GB） |

- **不要选** `cudnn-` 变体（多带 cuDNN，引擎用不上）；`base-` 变体没有编译工具
- 国内直连慢: 用镜像站检索确认后拉取，或 daocloud 前缀
  `docker pull docker.m.daocloud.io/nvidia/cuda:13.3.0-devel-ubuntu24.04`
  然后 `docker tag` 回标准名；构建完 `sudo docker builder prune` 可清编译缓存
- 磁盘预留 ~15GB（镜像 + 构建缓存）

## CUDA 版本: 13.1 还是 13.3？

- 引擎 `CMakeLists.txt` 硬校验 **CUDA >= 13.1**（低于才 FATAL_ERROR），13.1.2 与 13.3.0 都能编
- 本镜像默认 **13.3.0** —— 对齐官方《兜底方案-从源码自己编译》依赖表与本机 549/549 实测编译链
- 引擎树内官方 Dockerfile 原配 13.1.2，需要时切回: `./docker/build-image.sh -b nvidia/cuda:13.1.2`
- KNOWN_ISSUES 里"CUDA 13.3 崩溃"是 **llama.cpp 线**（PrismML-Eng/llama.cpp #222）的问题，与 NInfer 引擎无关

## 模型制品：.ninfer 的来路与获取（能不能自己打包？）

`.ninfer` 是引擎容器格式（三元权重 + MTP 草稿头 + 元数据），由**转换器**从模型 checkpoint 生成
——它不是编译产物：编译只产出引擎二进制（已烤进镜像），制品按需获取、只读挂载进容器。

| 路线 | 下载量 | 谁来做 | 12GB 卡可用？ |
|---|---|---|---|
| ① PQ2 极速档交付 | 7.8GB（夸克手动） | 沈三殊字节级重打包 | ✅ 本机在用 |
| ② HF 现成制品 `neroued/Qwen3.8-27B-NInfer` | 20.4GB（`fetch-model.sh --hf`，走 hf-mirror） | 原引擎作者 Neroued 直转发布 | ❌ 权重 20.4GB > 12GB 显存（≥24GB 卡用） |
| ③ 自转·直转 | 底座 ~52GB（魔搭 Qwen/Qwen3.8-27B）+ DFlash2 3.6GB（z-lab） | 你自己：`vendor/ninfer-4090w-ternary/tools/convert/qwen3_8_27b`（依赖仅 safetensors） | ❌ 产出同为 20.4GB |
| ④ 自转·GGUF | 三元 GGUF 6.7GB（魔搭 prism-ml）+ 底座 + 模板 | 你自己：`vendor/shensanshu-guide/tools/pack.py`（魔搭指南，已适配 Linux） | ✅ 产出 ~7-9GB |

**CraneBW 仓库是怎么做的**：它就是 Neroued 引擎线的公开 git——`tools/convert/qwen3_8_27b` 吃
`Qwen/Qwen3.8-27B` + `z-lab/Qwen3.8-27B-DFlash2` 两个 checkpoint，一个命令转出完整 .ninfer
（配方与源 revision 记录在其 docs/maintainer/），成品发 HF 供"下载即用"；同一套转换器就在我们
vendor/ 里。**12GB 卡要小体积制品，走 ① 或 ④**（PQ2 的 7.8GB 是沈三殊的字节级重打包优化，
公开转换器做不出来）。

## 适配显卡范围（能否"适配任何显卡"？）

容器化解决的是**软件栈**可移植，不改变引擎代码的显卡支持范围 —— `CMakeLists.txt` 硬校验
只允许两种架构，且一次编译只能选一个（40 系与 50 系需各构建一个镜像）:

| 显卡 | 支持 | 构建参数 |
|---|---|---|
| RTX 40 系（Ada: 4090/4080/4070/4060 及 L40/L4） | ✅ | `-a 89`（默认） |
| RTX 50 系（Blackwell: 5090/5080/5070） | ✅ | `-a 120a` |
| RTX 30 系及更早 / AMD / Intel | ❌ | 引擎限制，与容器无关 |

- 老卡（SM 75~86）走官方交付包**超低显存档**的 llama.cpp 线（SM 75~120a），不是本容器
- 同代不同显存只影响档位: 12GB → work 64k / long 112k；更大显存 `run-ninfer.sh -c` 放大上下文

## 与宿主机直跑的关系

- 性能无差别（宿主驱动透传 + 容器内 CUDA 13.3 用户态库）
- 显存预算相同: 启动前 `nvidia-smi` 确认空闲（64k 档需 ~10GB），旧实例必须先停
- 制品 `.ninfer` 只读挂载不进镜像，镜像只含两个二进制与运行库
- 服务状态: **2026-09-27 脚本就绪，容器构建尚未实测**（等基础镜像拉取后一起测试）
