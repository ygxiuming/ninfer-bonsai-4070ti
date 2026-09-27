#!/usr/bin/env bash
# =============================================================================
# 一键镜像本仓库（含 models/ 下 .ninfer 制品）到魔搭 ModelScope
#
# 前置（一次性）: 魔搭 CLI 走 modelscope-hub 包（modelscope 本体已不自带入口）：
#   uvx --from modelscope-hub modelscope login --token <你的token>
#
# 用法:
#   ./scripts/upload-modelscope.sh <魔搭用户名> [仓库名，默认 ninfer-bonsai-4070ti]
#
# 行为: folder 模式上传整个仓库目录（排除 .git；models/*.ninfer 随之入库），
#       首次上传自动建仓；重复执行=增量同步。
# 页面: https://modelscope.cn/models/<用户名>/<仓库名>/files
# =============================================================================
set -euo pipefail
MS_USER="${1:?用法: $0 <魔搭用户名> [仓库名]}"
REPO="${2:-ninfer-bonsai-4070ti}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
MS="uvx --from modelscope-hub modelscope"

echo ">> 上传 ${SRC} → modelscope.cn/${MS_USER}/${REPO}（首次自动建仓，7.8GB 制品耐心等）"
$MS upload "${MS_USER}/${REPO}" "$SRC" . \
    --exclude ".git/*" "models/*.part" \
    --commit-message "mirror: sync from GitHub $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo local)"

echo ">> 完成: https://modelscope.cn/models/${MS_USER}/${REPO}/files"
