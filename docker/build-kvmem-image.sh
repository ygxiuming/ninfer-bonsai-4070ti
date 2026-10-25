#!/usr/bin/env bash
# =============================================================================
# 构建 3060（sm_86）容器镜像 —— llama.cpp + KVMem 线（路线 A「省显存档」）
#
# 用法:
#   ./docker/build-kvmem-image.sh                       # 默认 arch 86 + CUDA 13.3.0
#   ./docker/build-kvmem-image.sh -a 89                 # 89/120a 同分支官方编过，本机未验证，属尝鲜
#   ./docker/build-kvmem-image.sh -b nvidia/cuda:<13.2.x>-devel-ubuntu24.04 # 贴合作者工具链基线（tag 以 Docker Hub 为准）
#   ./docker/build-kvmem-image.sh -m 1                  # 国内构建（apt 走清华源）
#   ./docker/build-kvmem-image.sh -j 3                  # nvcc 并行度（默认 6；runner 4 核建议 3）
#   ./docker/build-kvmem-image.sh -n host               # 构建容器与宿主同网（国内走宿主代理时必加）
#   ./docker/build-kvmem-image.sh -n host -p http://127.0.0.1:10808   # 显式指定构建容器代理
#   KVMEM_DRY_RUN=1 ./docker/build-kvmem-image.sh       # 干跑：只准备上下文并自检，不调 docker
#
# 前置:
#   · docker 可用（不在 docker 组时脚本自动改用 sudo）
#   · 首次构建会拉取 CUDA devel 基础镜像（约 6~7 GB）
#   · 构建期需要访问 GitHub（克隆 kvmem-llama.cpp 与 llama.cpp 子模块，revision 已 pin）
# 产物: llama-kvmem-bonsai:cuda<版本>-arch<架构>（默认 cuda13.3-arch86）
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH=86
BASE=nvidia/cuda:13.3.0-devel-ubuntu24.04
JOBS="${KVMEM_JOBS:-6}"
MIRROR=0
TAG=""
NETWORK=""
PROXYARG=""
while getopts "a:b:j:m:t:n:p:" o; do case "$o" in
    a) ARCH="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    m) MIRROR="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    n) NETWORK="$OPTARG" ;;   # 构建网络：host = 构建容器与宿主同网（国内走宿主代理时用它）
    p) PROXYARG="$OPTARG" ;;  # 显式指定构建容器用的 http(s) 代理
    *) grep '^#' "$0" | sed -n '2,16p'; exit 1 ;;
esac; done

[[ "$ARCH" =~ ^(86|89|120a)$ ]] || { echo "错误: -a 只能是 86（RTX 30 系）/ 89（40 系）/ 120a（50 系）" >&2; exit 1; }
command -v docker >/dev/null || { echo "错误: 需要 docker" >&2; exit 1; }
# 从 nvidia/cuda:<版本>-<变体> 里取 <版本> 的前两段（如 13.3.0-devel-ubuntu24.04 → 13.3）
CUDASHORT="$(echo "${BASE#*:}" | cut -d- -f1 | grep -oE '^[0-9]+\.[0-9]+' || true)"
[[ -n "$CUDASHORT" ]] || { echo "错误: -b 需为 nvidia/cuda:<版本> 形式（如 nvidia/cuda:13.3.0-devel-ubuntu24.04）" >&2; exit 1; }
TAG="${TAG:-llama-kvmem-bonsai:cuda${CUDASHORT}-arch${ARCH}}"

DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# 只把需要的文件放进构建上下文（避免把 8 GB 的 models/ 整个塞进去）：
#   上下文根/ Dockerfile                          ← docker build 默认只认根目录的 Dockerfile
#   上下文根/docker/kvmem/docker-entrypoint.sh    ← 与 Dockerfile 里的 COPY 路径一一对应
#   上下文根/scripts/kvmem-proxy.py
#   上下文根/patches/bonsai-chat-template-alias.jinja
CTX="$(mktemp -d /tmp/kvmem-docker-ctx.XXXXXX)"
trap 'rm -rf "$CTX"' EXIT

# 仓库路径:上下文内路径 —— 与 Dockerfile 的每条 COPY 一一对应
CTX_FILES=(
    "docker/kvmem/Dockerfile:Dockerfile"
    "docker/kvmem/docker-entrypoint.sh:docker/kvmem/docker-entrypoint.sh"
    "scripts/kvmem-proxy.py:scripts/kvmem-proxy.py"
    "patches/bonsai-chat-template-alias.jinja:patches/bonsai-chat-template-alias.jinja"
)

# 先自检**再**复制：缺文件时给出可读的报错，而不是让 cp 在 `set -e` 下直接秒退。
# （踩过：PR #1 漏了 patches/bonsai-chat-template-alias.jinja，CI 日志只有一行
#   `cp: cannot stat ...`，下面那段"构建上下文不完整"的友好提示根本没机会执行）
missing=0
for pair in "${CTX_FILES[@]}"; do
    [[ -f "$ROOT/${pair%%:*}" ]] || { echo "   [缺] 仓库里没有 ${pair%%:*}" >&2; missing=1; }
done
[[ "$missing" == "0" ]] || { echo "错误: 构建所需的文件在仓库里缺失，已中止（见上面 [缺] 行）" >&2; exit 1; }

mkdir -p "$CTX/docker/kvmem" "$CTX/scripts" "$CTX/patches"
for pair in "${CTX_FILES[@]}"; do
    cp "$ROOT/${pair%%:*}" "$CTX/${pair##*:}"
done

# 复制后再确认一遍（防止 cp 静默失败）：Dockerfile 必须在上下文根，且它 COPY 的每个文件都要在
missing=0
for f in "Dockerfile" "docker/kvmem/docker-entrypoint.sh" "scripts/kvmem-proxy.py" "patches/bonsai-chat-template-alias.jinja"; do
    if [[ -f "$CTX/$f" ]]; then
        echo "   [ctx] $f  ($(wc -c < "$CTX/$f") 字节)"
    else
        echo "   [ctx] 缺 $f" >&2; missing=1
    fi
done
[[ "$missing" == "0" ]] || { echo "错误: 构建上下文不完整，已中止（先修脚本再跑）" >&2; exit 1; }

# 干跑模式：只准备上下文并打印清单，不调用 docker（CI/无 docker 环境可用来自检）
if [[ "${KVMEM_DRY_RUN:-0}" == "1" ]]; then
    echo ">> KVMEM_DRY_RUN=1：上下文已就绪，跳过 docker build。清理后退出。"
    exit 0
fi

echo ">> docker build（${BASE} , arch=${ARCH} , 并行=${JOBS} , 国内源=${MIRROR}）→ $TAG"

# 构建容器的网络与代理：
#   构建期需要访问 GitHub（克隆源码）与 apt 源。国内环境通常需要：
#     -n host            让构建容器与宿主同网络（这样 127.0.0.1:10808 的宿主代理才可达）
#     -p http://...      显式指定代理；不传则继承环境变量 HTTPS_PROXY/https_proxy
DOCKER_ARGS=()
[[ -n "$NETWORK" ]] && DOCKER_ARGS+=(--network "$NETWORK")
PROXY="${PROXYARG:-${HTTPS_PROXY:-${https_proxy:-}}}"
if [[ -n "$PROXY" ]]; then
    DOCKER_ARGS+=(--build-arg "http_proxy=$PROXY" --build-arg "https_proxy=$PROXY"
                  --build-arg "no_proxy=localhost,127.0.0.1")
    echo "   构建容器代理: $PROXY $([[ "$NETWORK" == "host" ]] && echo '(网络=host，127.0.0.1 可达 ✓)' || echo '(注意：非 host 网络时 127.0.0.1 指向容器自身，代理不可达 ✗)')"
else
    echo "   构建容器代理: 未设置（直连）"
fi

$DOCKER build \
    "${DOCKER_ARGS[@]}" \
    --build-arg CUDA_BASE="$BASE" \
    --build-arg CUDA_ARCH="$ARCH" \
    --build-arg KVMEM_JOBS="$JOBS" \
    --build-arg USE_MIRROR="$MIRROR" \
    -t "$TAG" \
    "$CTX"

echo ">> 构建完成: $TAG"
echo "   起服务:   docker compose -f docker/kvmem/docker-compose.kvmem.yml up -d"
echo "   或裸跑:   $DOCKER run --rm -it --gpus all -p 127.0.0.1:18210:18210 -p 127.0.0.1:18202:18202 \\"
echo "               -v $ROOT/models:/models:ro $TAG"
