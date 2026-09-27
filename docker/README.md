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
./docker/run-ninfer.sh             # 或 -c 114688（long 档）/ -p 8088（自定端口）
./docker/test-serving.sh
./docker/run-ninfer.sh stop
```

不在 docker 组时脚本自动改用 sudo；50 系卡构建: `./docker/build-image.sh -a 120a`（tag 后缀 arch120a）。

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
