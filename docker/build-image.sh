#!/usr/bin/env bash
# =============================================================================
# NInfer 容器镜像构建 —— 容器内完整编译（apt 走清华源，无需宿主机装 CUDA/编译器）
#
# 用法:
#   ./docker/build-image.sh                    # 默认: CUDA 13.3.0 + RTX 40 系(arch 89)
#   ./docker/build-image.sh -a 120a            # RTX 50 系(Blackwell) → tag 后缀 arch120a
#   ./docker/build-image.sh -b nvidia/cuda:13.1.2   # 切回引擎官方 Dockerfile 的 CUDA 版本
#   ./docker/build-image.sh -e <引擎树> -t <镜像tag> -m 0   # 自定路径/tag、关闭国内源
#
# 前置:
#   · docker 可用（不在 docker 组时脚本自动改用 sudo）
#   · 首次构建自动拉取基础镜像 ~7GB（精确 tag 与加速方式见 docker/README.md）
# =============================================================================
set -euo pipefail

# 默认路径相对本仓库推导（../sanyuan/），可用环境变量 NINFER_ENGINE 或 -e 覆盖
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENG="${NINFER_ENGINE:-$REPO_ROOT/vendor/ninfer-4090w-ternary}"
ARCH="89"
BASE="nvidia/cuda:13.3.0"
TAG=""
MIRROR="1"
while getopts "e:t:a:b:m:" o; do case "$o" in
    e) ENG="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    a) ARCH="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    m) MIRROR="$OPTARG" ;;
    *) grep '^#' "$0" | sed -n '2,14p'; exit 1 ;;
esac; done

[[ "$ARCH" == "89" || "$ARCH" == "120a" ]] || {
    echo "错误: CUDA_ARCH 只能是 89(RTX 40 系) 或 120a(RTX 50 系) —— 引擎 CMakeLists 硬限制" >&2; exit 1; }
[[ -f "$ENG/CMakeLists.txt" ]] || { echo "错误: 引擎树不存在: $ENG" >&2; exit 1; }
command -v rsync >/dev/null || { echo "错误: 需要 rsync（sudo apt install rsync）" >&2; exit 1; }
CUDASHORT="$(echo "$BASE" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?$' | cut -d. -f1,2 || true)"
[[ -n "$CUDASHORT" ]] || { echo "错误: -b 需为 nvidia/cuda:<版本> 形式" >&2; exit 1; }
TAG="${TAG:-ninfer-bonsai:cuda${CUDASHORT}-arch${ARCH}}"

DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

CTX="$(mktemp -d /tmp/ninfer-docker-ctx.XXXXXX)"
trap 'rm -rf "$CTX"' EXIT
echo ">> 收集引擎源码到构建上下文: $ENG"
mkdir "$CTX/src"
rsync -a --exclude 'build/' --exclude 'tmp-nvcc/' --exclude '*.o' --exclude '*.a' \
      --exclude '.git/' "$ENG/" "$CTX/src/"
cp "$(cd "$(dirname "$0")" && pwd)/Dockerfile" "$CTX/"

echo ">> docker build（基础镜像 ${BASE}.x, CUDA_ARCH=$ARCH, 国内源=$MIRROR）→ $TAG"
$DOCKER build \
    --build-arg CUDA_BASE="$BASE" \
    --build-arg CUDA_ARCH="$ARCH" \
    --build-arg NINFER_JOBS="${NINFER_JOBS:-6}" \
    --build-arg USE_MIRROR="$MIRROR" \
    -t "$TAG" \
    "$CTX"

echo ">> 构建完成: $TAG"
echo "   启动服务: ./docker/run-ninfer.sh -i $TAG"
