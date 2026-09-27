#!/usr/bin/env bash
# =============================================================================
# 模型制品获取/校验（.ninfer 不随 git 分发：远超 GitHub 单文件 100MB 硬限制）
#
# 用法:
#   ./docker/fetch-model.sh                # 校验 models/ 下的制品是否就绪
#   ./docker/fetch-model.sh --ms           # ★推荐：从魔搭镜像仓库下载（7.8GB，国内直连）
#   ./docker/fetch-model.sh <制品路径>     # 复制到 models/ 并校验
#   ./docker/fetch-model.sh --hf           # 下载 HF 现成制品（20.4GB，≥24GB 卡用，见下）
#
# 三种来源（详见 docker/README.md《模型制品》节）:
#   ① PQ2 极速档交付（7.8GB，本机 12GB 卡唯一现成可用）—— 夸克网盘手动下载:
#      https://pan.quark.cn/s/f72b85b82626 （包内 artifacts-pq2.ninfer 放入 models/）
#   ② HF 现成制品 neroued/Qwen3.8-27B-NInfer（qwen3_8_27b.ninfer，20.4GB）—— 可脚本
#      下载（走 hf-mirror），但权重 20.4GB > 12GB 显存，仅适合 ≥24GB 显存的卡
#   ③ 自打包（全开源，工具已在 vendor/，见 docker/README.md《模型制品》节）
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/models/artifacts-pq2.ninfer"
SHA_PQ2="05bbbf01090c6f61113b54556daa22ad0b45036a078af6cd6f02a3fde47ad76c"
SHA_HF="0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3"
HF_FILE="qwen3_8_27b.ninfer"
HF_REPO="neroued/Qwen3.8-27B-NInfer"

verify() {  # $1=文件 $2=期望sha256 $3=标签
    echo ">> 校验 SHA256..."
    local actual; actual="$(sha256sum "$1" | cut -d' ' -f1)"
    if [[ "$actual" == "$2" ]]; then
        echo "✅ 制品就绪: $1（$3, SHA256 匹配）"
    else
        echo "❌ SHA256 不匹配（$3）" >&2
        echo "   实际: $actual" >&2
        echo "   期望: $2" >&2
        exit 1
    fi
}

case "${1:-verify}" in
    --ms)
        mkdir -p "$REPO_ROOT/models"
        MS="uvx --from modelscope-hub modelscope"
        echo ">> 从魔搭下载 xiuming/ninfer-bonsai-4070ti（7.8GB，国内直连）"
        $MS download xiuming/ninfer-bonsai-4070ti models/artifacts-pq2.ninfer --local_dir "$REPO_ROOT"
        verify "$DEST" "$SHA_PQ2" "PQ2 极速档（魔搭镜像）"
        ;;
    --hf)
        mkdir -p "$REPO_ROOT/models"
        OUT="$REPO_ROOT/models/$HF_FILE"
        echo ">> 从 HF 镜像下载 $HF_REPO/$HF_FILE（20.4GB，hf-mirror 国内直连）"
        if command -v hf >/dev/null 2>&1; then
            HF_ENDPOINT=https://hf-mirror.com hf download "$HF_REPO" "$HF_FILE" --local-dir "$REPO_ROOT/models"
        else
            curl -L --fail -o "$OUT.part" \
              "https://hf-mirror.com/$HF_REPO/resolve/main/$HF_FILE"
            mv "$OUT.part" "$OUT"
        fi
        verify "$OUT" "$SHA_HF" "neroued groupwise-int（≥24GB 显存卡用）"
        echo "   ⚠ 12GB 卡放不下该制品（20.4GB 权重）；12GB 卡请用来源①的 PQ2。"
        ;;
    verify|"")
        if [[ ! -f "$DEST" ]]; then
            echo "未找到模型制品: $DEST"
            echo "获取方式:"
            echo "  ① 魔搭镜像仓库（推荐）:       $0 --ms（或 modelscope download xiuming/ninfer-bonsai-4070ti models/artifacts-pq2.ninfer --local_dir .）"
            echo "  ② 夸克网盘 PQ2:               https://pan.quark.cn/s/ef30066ed62a → 放入 models/"
            echo "  ③ 本机已有制品:               $0 <制品路径>"
            echo "  ④ HF 现成制品（≥24GB 卡）:     $0 --hf"
            echo "  ⑤ 自打包:                     见 docker/README.md《模型制品》节"
            exit 1
        fi
        ACTUAL="$(sha256sum "$DEST" | cut -d' ' -f1)"
        if [[ "$ACTUAL" == "$SHA_PQ2" ]]; then
            echo "✅ 制品就绪: $DEST（PQ2 极速档, SHA256 匹配）"
        elif [[ "$ACTUAL" == "$SHA_HF" ]]; then
            echo "✅ 制品就绪: $DEST（neroued groupwise-int, SHA256 匹配; ⚠ 12GB 卡放不下）"
        else
            echo "❌ SHA256 不匹配，非已知制品" >&2
            echo "   实际:     $ACTUAL" >&2
            echo "   PQ2 期望: $SHA_PQ2" >&2
            echo "   HF  期望: $SHA_HF" >&2
            exit 1
        fi
        ;;
    *)
        [[ -f "$1" ]] || { echo "错误: 文件不存在: $1" >&2; exit 1; }
        mkdir -p "$(dirname "$DEST")"
        echo ">> 复制 $(du -h "$1" | cut -f1) → $DEST"
        cp "$1" "$DEST"
        exec "$0" verify
        ;;
esac
