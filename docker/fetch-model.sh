#!/usr/bin/env bash
# =============================================================================
# 模型制品获取/校验（.ninfer 不随 git 分发：7.8GB 单文件远超 GitHub 100MB 硬限制）
#
# 来源（三选一）:
#   ① 官方交付包·夸克网盘直链（README §4）—— 极速档 https://pan.quark.cn/s/f72b85b82626
#      取包内 artifacts-pq2.ninfer（7.74GB）放到 models/ 目录后重跑本脚本校验
#   ② 本机已有制品: ./docker/fetch-model.sh <路径> —— 自动复制到 models/ 并校验
#   ③ 全量自打包（约 62GB，见 docker/README.md"自打包"节）: 底座+三元 GGUF+草稿头 → pack.py
#
# 用法:
#   ./docker/fetch-model.sh               # 校验 models/artifacts-pq2.ninfer 是否就绪
#   ./docker/fetch-model.sh <制品路径>    # 复制到 models/ 并校验
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/models/artifacts-pq2.ninfer"
SHA256="05bbbf01090c6f61113b54556daa22ad0b45036a078af6cd6f02a3fde47ad76c"

if [[ $# -ge 1 ]]; then
    [[ -f "$1" ]] || { echo "错误: 文件不存在: $1" >&2; exit 1; }
    mkdir -p "$(dirname "$DEST")"
    echo ">> 复制 $(du -h "$1" | cut -f1) → $DEST"
    cp "$1" "$DEST"
fi

if [[ ! -f "$DEST" ]]; then
    echo "未找到模型制品: $DEST"
    echo "获取方式:"
    echo "  ① 夸克网盘极速档: https://pan.quark.cn/s/f72b85b82626 （下载包内 artifacts-pq2.ninfer 放入 models/）"
    echo "  ② 本机已有制品:   $0 <制品路径>"
    echo "  ③ 全量自打包:     见 docker/README.md《自打包》节（约 62GB，全魔搭源可脚本化）"
    exit 1
fi

echo ">> 校验 SHA256（7.8GB，约 20-40s）..."
ACTUAL="$(sha256sum "$DEST" | cut -d' ' -f1)"
if [[ "$ACTUAL" == "$SHA256" ]]; then
    echo "✅ 制品就绪: $DEST（SHA256 匹配）"
else
    echo "❌ SHA256 不匹配" >&2
    echo "   实际: $ACTUAL" >&2
    echo "   期望: $SHA256" >&2
    exit 1
fi
