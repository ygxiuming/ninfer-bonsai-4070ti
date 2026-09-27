#!/usr/bin/env bash
# =============================================================================
# 引擎兼容性补丁（对官方交付树 src-tree/ninfer-4090w-ternary 施加，幂等可重跑）
#
# 用法:
#   ./apply-compat.sh <引擎树根路径>              # 应用 P1（推荐必打）
#   ./apply-compat.sh --with-k7 <引擎树根路径>    # 额外应用 P2（实验性）
#   ./apply-compat.sh --check  <引擎树根路径>     # 只检查不改文件
#
# P1 reasoning_effort 命名兼容（★ 推荐必打）
#    OpenAI 系客户端默认发 reasoning_effort="high"，而聊天模板只认
#    low/medium/xhigh ⇒ 400。映射: minimal→Low, high/max→XHigh。
#    文件: src/serve/translate.cpp
#
# P2 MTP 窗口上限 [1,5]→[1,7]（可选，实验性，对 K≤5 行为零影响）
#    放开三层限制中的前两层。第三层是内核硬限
#    （mtp_prepare_next_round: T∈[2,6]），K=6/7 启动仍会被拦（属预期）。
#    文件: src/product/speculative_options.h
#          src/targets/qwen3_6/impl/runtime/layouts_impl.h
#          src/targets/qwen3_6_27b/impl/config.h
#          src/targets/qwen3_6_35b_a3b/impl/config.h
#          src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h
# =============================================================================
set -uo pipefail

WITH_K7=0
MODE="apply"
ARGS=()
for a in "$@"; do
    case "$a" in
        --with-k7) WITH_K7=1 ;;
        --check)   MODE="check" ;;
        *)         ARGS+=("$a") ;;
    esac
done
TREE="${ARGS[0]:?用法: $0 [--with-k7] [--check] <引擎树根路径>}"
FAIL=0

# ---------------------------------------------------------------- P1 (python)
run_p1() {
    python3 - "$TREE" "$MODE" <<'PY'
import sys
from pathlib import Path

tree, mode = sys.argv[1], sys.argv[2]
p = Path(tree) / "src" / "serve" / "translate.cpp"
if not p.exists():
    print(f"  [失败] 找不到 {p}")
    sys.exit(1)
t = p.read_text(encoding="utf-8")
orig = t

LOW = "    case RequestedReasoningEffort::Low:\n"
MIN = "    case RequestedReasoningEffort::Minimal:\n"
XH = "    case RequestedReasoningEffort::XHigh:\n"
HIGH = "    case RequestedReasoningEffort::High:\n"
MAX = "    case RequestedReasoningEffort::Max:\n"

if MIN not in t.split(LOW, 1)[-1].split("break;", 1)[0]:
    t = t.replace(LOW, LOW + MIN, 1)
    print("  [P1a] minimal → Low 组: 已插入")
else:
    print("  [P1a] minimal → Low 组: 已应用，跳过")

if HIGH not in t.split(XH, 1)[-1].split("break;", 1)[0]:
    t = t.replace(XH, XH + HIGH + MAX, 1)
    print("  [P1b] high/max → XHigh 组: 已插入")
else:
    print("  [P1b] high/max → XHigh 组: 已应用，跳过")

# 旧拒绝分支（Minimal/High/Max → 400）若仍在则移除
reject = (
    MIN + HIGH + MAX
    + "        invalid_prompt_option(\"reasoning effort '\" +\n"
    + "                                  std::string(requested_reasoning_effort_name(requested)) +\n"
    + "                                  \"' is not supported by the loaded chat template\",\n"
    + "                              \"reasoning_effort\", \"reasoning_effort_not_supported\");\n"
)
if reject in t:
    t = t.replace(reject, "", 1)
    print("  [P1c] 旧拒绝分支: 已移除")
else:
    print("  [P1c] 旧拒绝分支: 不存在，跳过")

if t != orig:
    if mode == "apply":
        p.write_text(t, encoding="utf-8")
        print("  [P1] translate.cpp 已写回")
    else:
        print("  [P1] --check 模式: 有待应用改动（未写回）")
else:
    print("  [P1] 无需改动")
PY
    return $?
}

# ---------------------------------------------------------------- P2 (sed)
p2_sed() {  # $1=描述 $2=相对路径 $3=sed表达式 $4=已应用判据(grep)
    local desc="$1" file="$TREE/$2" expr="$3" guard="$4"
    if [[ ! -f "$file" ]]; then
        echo "  [失败] $desc —— 文件不存在: $2"; FAIL=1; return
    fi
    if grep -q "$guard" "$file"; then
        echo "  [$desc] 已应用，跳过"; return
    fi
    if [[ "$MODE" == "check" ]]; then
        echo "  [$desc] 待应用"; return
    fi
    if sed -i "$expr" "$file"; then
        echo "  [$desc] OK"
    else
        echo "  [$desc] 失败"; FAIL=1
    fi
}

run_p2() {
    p2_sed "前端校验 >5→>7" "src/product/speculative_options.h" \
        's/options.draft_tokens > 5) {/options.draft_tokens > 7) {/' \
        "draft_tokens > 7"
    p2_sed "前端校验文案" "src/product/speculative_options.h" \
        's/requires --draft-tokens in \[1,5\]/requires --draft-tokens in [1,7]/' \
        "requires --draft-tokens in \[1,7\]"
    p2_sed "27B 变体常量 5→7" "src/targets/qwen3_6_27b/impl/config.h" \
        's/inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;/inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 7;/' \
        "kMaximumMtpDraftTokens    = 7"
    p2_sed "35B 变体常量 5→7" "src/targets/qwen3_6_35b_a3b/impl/config.h" \
        's/inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;/inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 7;/' \
        "kMaximumMtpDraftTokens    = 7"
    p2_sed "解码帧域 5→7" "src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h" \
        's/inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 5;/inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 7;/' \
        "kMtpDecodeMaximumDrafts    = 7"
    p2_sed "引擎侧校验文案" "src/targets/qwen3_6/impl/runtime/layouts_impl.h" \
        's/MTP draft window must be in \[1,5\]/MTP draft window must be in [1,7]/' \
        "MTP draft window must be in \[1,7\]"
}

echo "== P1: reasoning_effort 命名兼容 =="
run_p1 || FAIL=1

if [[ "$WITH_K7" == 1 ]]; then
    echo "== P2: MTP 窗口上限 [1,5]→[1,7]（实验性） =="
    run_p2
else
    echo "== P2: 跳过（需要 --with-k7） =="
fi

echo
if [[ "$MODE" == "apply" ]]; then
    echo "完成。重新编译: ninja -C build && 重启服务。"
fi
exit $FAIL
