#!/usr/bin/env bash
# =============================================================================
# Route A「省显存档」引擎构建脚本 —— RTX 3060 12GB（sm_86）/ Ubuntu / 免 root
#
# 做什么:
#   1) 拉取 kvmem-llama.cpp 的 Bonsai 分支（PrismML 三元支持 + KVMem 分层 KV）
#   2) 用免 root 方式准备 CUDA 13.2 工具链（NVIDIA apt 仓库的 .deb 直接 dpkg -x 解包）
#   3) 以 CMAKE_CUDA_ARCHITECTURES=86-real 编译 llama-kvmem-server / llama-kvmem-cli
#
# 为什么不用官方 Linux 预编译包:
#   官方 rc3 Linux 预编译包编了 sm_86，但那是通用 Qwen3.8 线（IQ3/IQ4），
#   **不含** Bonsai 三元格式（ggml type 142/143 = PQ2_0/PTQ1_0）与 Hadamard 激活变换；
#   Bonsai 版（prism.3）官方只发了 Windows 预编译 —— Linux 必须自编。
#
# 前置: 磁盘 ~15 GB、NVIDIA 驱动（本机 595.91.07）、CPU 需 AVX2/FMA/F16C/BMI2
# 用法: ./build-kvmem-sm86.sh [KVROOT]      （默认 /mnt/D/code/kvmem-bonsai）
# =============================================================================
set -euo pipefail

KVROOT="${1:-/mnt/D/code/kvmem-bonsai}"
CUDA_VER="13.2.86-1"          # CUDA 13.2 Update 2（= prism.3 / rc3 的构建基线）
CUBLAS_VER="13.4.1.3-1"
REPO_URL="https://github.com/kvmem/kvmem-llama.cpp.git"
REPO_BRANCH="kvmem-bonsai-llama.cpp"
# 国内直连（不走系统代理）更快；GitHub 用 ghfast 镜像
NVIDIA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
CMAKE_URL="https://cmake.org/files/v3.31/cmake-3.31.6-linux-x86_64.tar.gz"
NINJA_URL="https://ghfast.top/https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip"

mkdir -p "$KVROOT"/{debs,downloads,tmp,tools}
cd "$KVROOT"

# ---------------------------------------------------------------- 1) 工具链
CUDA_ROOT="$KVROOT/cuda-stage/usr/local/cuda-13.2"
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "== 1/4 解包 CUDA $CUDA_VER 到 $CUDA_ROOT（免 root）"
  DEBS=(
    cuda-nvcc-13-2_${CUDA_VER}_amd64.deb cuda-crt-13-2_${CUDA_VER}_amd64.deb
    cuda-cudart-13-2_${CUDA_VER}_amd64.deb cuda-cudart-dev-13-2_${CUDA_VER}_amd64.deb
    cuda-cccl-13-2_${CUDA_VER}_amd64.deb cuda-driver-dev-13-2_${CUDA_VER}_amd64.deb
    cuda-culibos-13-2_${CUDA_VER}_amd64.deb cuda-culibos-dev-13-2_${CUDA_VER}_amd64.deb
    cuda-nvrtc-13-2_${CUDA_VER}_amd64.deb cuda-nvrtc-dev-13-2_${CUDA_VER}_amd64.deb
    libcublas-13-2_${CUBLAS_VER}_amd64.deb libcublas-dev-13-2_${CUBLAS_VER}_amd64.deb
    libnvvm-13-2_${CUDA_VER}_amd64.deb          # ← 漏了它 nvcc 找不到 nvvm/bin/cicc
    cuda-tileiras-13-2_${CUDA_VER}_amd64.deb
    libnvjitlink-13-2_${CUDA_VER}_amd64.deb libnvjitlink-dev-13-2_${CUDA_VER}_amd64.deb
    libnvfatbin-13-2_${CUDA_VER}_amd64.deb
  )
  for f in "${DEBS[@]}"; do
    [[ -s "debs/$f" ]] || curl -fsSL --noproxy '*' --retry 3 -o "debs/$f" "$NVIDIA_REPO/$f"
    dpkg -x "debs/$f" cuda-stage/
  done
fi

if [[ ! -x "$KVROOT/tools/ninja" ]]; then
  echo "== 2/4 取 cmake / ninja（用户级）"
  curl -fsSL --noproxy '*' -o tools/cmake.tar.gz "$CMAKE_URL" && tar -xzf tools/cmake.tar.gz -C tools/
  curl -fsSL --noproxy '*' -o tools/ninja.zip "$NINJA_URL" && (cd tools && unzip -oq ninja.zip && chmod +x ninja)
fi

# 写入环境文件（start-kvmem.sh 与交互 shell 共用）
cat > env.sh <<EOF
export KVROOT=$KVROOT
export CUDA_HOME=\$KVROOT/cuda-stage/usr/local/cuda-13.2
export PATH=\$CUDA_HOME/bin:\$KVROOT/tools/cmake-3.31.6-linux-x86_64/bin:\$KVROOT/tools:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$CUDA_HOME/nvvm/lib64:\${LD_LIBRARY_PATH:-}
export CUDACXX=\$CUDA_HOME/bin/nvcc
export CMAKE=\$KVROOT/tools/cmake-3.31.6-linux-x86_64/bin/cmake
export CMAKE_GENERATOR=Ninja
export CMAKE_CUDA_ARCHITECTURES=86-real
export TMPDIR=\$KVROOT/tmp
export NPROC=6
EOF
# shellcheck disable=SC1091
source "$KVROOT/env.sh"

"$CUDA_HOME/bin/nvcc" --version | tail -2

# ---------------------------------------------------------------- 2) 源码
if [[ ! -d "$KVROOT/kvmem-src" ]]; then
  echo "== 3/4 克隆 $REPO_BRANCH（含 llama.cpp 子模块，走代理）"
  git clone --branch "$REPO_BRANCH" --recurse-submodules --shallow-submodules "$REPO_URL" "$KVROOT/kvmem-src"
fi
cd "$KVROOT/kvmem-src"
./scripts/apply-patches.sh

# ---------------------------------------------------------------- 3) 编译
echo "== 4/4 配置 + 编译（sm_86）"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=86-real \
  -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="$KVROOT/kvmem-src"
ninja -C build -j"${NPROC:-6}" llama-kvmem-server llama-kvmem-cli

echo
echo "完成。产物: $KVROOT/kvmem-src/build/bin/llama-kvmem-server"
echo "启动:      KVROOT=$KVROOT ./scripts/start-kvmem.sh work"
echo "           （在本仓库根目录执行；KVROOT 与上面一致，start-kvmem.sh 才找得到引擎）"
