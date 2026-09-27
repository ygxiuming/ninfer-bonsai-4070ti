#!/usr/bin/env bash
# =============================================================================
# 一键镜像本仓库（含 models/ 下 .ninfer 制品，走 git LFS）到魔搭 ModelScope
#
# 前置（一次性）:
#   pip install modelscope && modelscope login   # 浏览器/Token 登录
#   git lfs version                              # 无则 sudo apt install git-lfs
#
# 用法:
#   ./scripts/upload-modelscope.sh <魔搭用户名> [仓库名，默认 ninfer-bonsai-4070ti]
#
# 行为:
#   1. 调魔搭 API 创建模型仓库（已存在则跳过）
#   2. git clone 到 /tmp、rsync 仓库内容（.git 与 models/.part 除外）、LFS 跟踪 *.ninfer
#   3. commit + push（模型 7.8GB，国内带宽几分钟~十几分钟）
#
# 完成后魔搭仓库页: https://modelscope.cn/models/<用户名>/<仓库名>/files
# =============================================================================
set -euo pipefail
MS_USER="${1:?用法: $0 <魔搭用户名> [仓库名]}"
REPO="${2:-ninfer-bonsai-4070ti}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
command -v git-lfs >/dev/null || { echo "错误: 需要 git-lfs（sudo apt install git-lfs）" >&2; exit 1; }
command -v modelscope >/dev/null || { echo "错误: 需要 modelscope CLI（pip install modelscope && modelscope login）" >&2; exit 1; }

echo ">> 创建/确认魔搭仓库 ${MS_USER}/${REPO}"
TOKEN="$(cat ~/.cache/modelscope/token 2>/dev/null || echo "")"
[[ -n "$TOKEN" ]] || { echo "错误: 未找到魔搭 token（先 modelscope login）" >&2; exit 1; }
curl -s -X PUT "https://www.modelscope.cn/api/v1/models/${MS_USER}/${REPO}" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"Visibility":2,"License":"Apache-2.0","Name":"'"$REPO"'"}' >/dev/null || true

WORK="$(mktemp -d /tmp/ms-mirror.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
echo ">> clone 魔搭仓库"
git clone "https://${MS_USER}:${TOKEN}@www.modelscope.cn/${MS_USER}/${REPO}.git" "$WORK/repo" 2>/dev/null \
  || git clone "https://www.modelscope.cn/${MS_USER}/${REPO}.git" "$WORK/repo"

echo ">> 同步仓库内容（git 历史与 .part 残片除外）"
rsync -a --exclude '.git/' --exclude '*.part' "$SRC/" "$WORK/repo/"
cd "$WORK/repo"
git lfs install >/dev/null
git lfs track '*.ninfer' >/dev/null
git add -A >/dev/null
git -c user.name="${MS_USER}" -c user.email="${MS_USER}@modelscope.cn" \
  commit -q -m "mirror: sync from GitHub $(cd "$SRC" && git rev-parse --short HEAD)（含 .ninfer 制品 LFS）" || echo "（无变更）"
echo ">> push（模型 7.8GB 走 LFS）"
git push -u origin master 2>/dev/null || git push -u origin main
echo ">> 完成: https://modelscope.cn/models/${MS_USER}/${REPO}/files"
