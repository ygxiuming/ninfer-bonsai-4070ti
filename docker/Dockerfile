# =============================================================================
# NInfer · Bonsai-2-27B 容器镜像 —— 引擎官方 Dockerfile 的国内源 + GPU 架构参数化版
#
# CUDA 版本说明（回应"13.1 还是 13.3"）:
#   · 引擎 CMakeLists.txt 硬校验: CUDA >= 13.1（低于则 FATAL_ERROR），两者都能编
#   · 本镜像默认 13.3.0 —— 对齐官方《兜底方案-从源码自己编译》依赖表与本机实测编译链
#   · 引擎树内官方 Dockerfile 钉的是 13.1.2（作者容器配方），需要时
#     --build-arg CUDA_BASE=nvidia/cuda:13.1.2 切回
#   · KNOWN_ISSUES 里"CUDA 13.3 崩溃"是 llama.cpp 线（PrismML-Eng/llama.cpp #222）
#     的问题，与 NInfer 引擎无关
#
# 与引擎树内官方 Dockerfile 的差异:
#   1. apt 换清华 TUNA 源（USE_MIRROR=1 默认；海外网络构建可传 0）
#   2. 禁用镜像内 NVIDIA CUDA apt 仓库: 容器编译所需的 nvcc 已随 devel 镜像内置，
#      本次依赖没有任何 cuda-* 包，该仓库保留只会拖慢/卡住国内 apt update
#   3. CUDA 架构改为构建参数 CUDA_ARCH: 89=RTX 40 系(Ada) / 120a=RTX 50 系(Blackwell)
#      （引擎 CMakeLists 硬校验只允许这两个值，一次编译只能选一个）
# =============================================================================
ARG CUDA_BASE=nvidia/cuda:13.3.0
FROM ${CUDA_BASE}-devel-ubuntu24.04 AS build

ARG CUDA_ARCH=89
ARG USE_MIRROR=1
# 编译并行度：nvcc 单作业峰值 ~2GB 内存，20 核/30GB 机器建议 6（全并行会 OOM/ICE）
ARG NINFER_JOBS=6

RUN set -eux; \
    if [ "$USE_MIRROR" = "1" ]; then \
        rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources; \
        for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/*.list; do \
            [ -f "$f" ] || continue; \
            sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://cn.archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "$f"; \
        done; \
    fi; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        cmake libavcodec-dev libavformat-dev libavutil-dev libcurl4-openssl-dev \
        libswresample-dev libswscale-dev ninja-build pkg-config; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY src/ /src/

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel "${NINFER_JOBS}" --target ninfer ninfer-serve

FROM ${CUDA_BASE}-runtime-ubuntu24.04

ARG USE_MIRROR=1

RUN set -eux; \
    if [ "$USE_MIRROR" = "1" ]; then \
        rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources; \
        for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/*.list; do \
            [ -f "$f" ] || continue; \
            sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://cn.archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "$f"; \
        done; \
    fi; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ca-certificates libavcodec60 libavformat60 libavutil58 libcurl4t64 libswscale7; \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8088
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "--help"]
