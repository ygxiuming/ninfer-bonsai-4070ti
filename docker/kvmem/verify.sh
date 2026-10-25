#!/usr/bin/env bash
# =============================================================================
# 3060 容器镜像自检 —— **不占显存**（容器内用 KVMEM_NGL=0 在 CPU 上跑整条链路）
# 前提：本机有 NVIDIA 驱动（容器要 --gpus all 注入驱动；只是不用留出模型要的显存）
#
# 检查项：
#   1) 镜像 ENV 里没有 KVMEM_MODEL（否则入口脚本会跳过「自动切 MTP 模型」——实测踩过）
#   2) 引擎二进制能加载（CUDA 运行库/驱动桩是否配齐）
#   3) 容器起来后：引擎端口与压缩代理端口 /v1/models 均 200
#   4) 启动横幅正确（模型 / 视觉 / 投机）
#   5) 经压缩代理真实生成一句话，回答非空
#
# 用法: ./docker/kvmem/verify.sh [镜像名]      # 默认 llama-kvmem-bonsai:cuda13.3-arch86
# 说明: 需要 models/ 下有 .gguf；CPU 生成一句约 1~2 分钟（GPU 上只需几秒）。
# =============================================================================
set -euo pipefail

IMG="${1:-llama-kvmem-bonsai:cuda13.3-arch86}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS="$ROOT/models"
EP=18399   # 宿主侧映射：引擎
PP=18398   # 宿主侧映射：压缩代理
NAME="kvmem-verify-$$"
fail=0
# 对齐按「显示列宽」算，不能按字节：中文字符是 2 列、UTF-8 下占 3 字节，
# `%-52s` 会把它当 3 列宽来补空格，整列越往右越歪。wc -L 是唯一按 wcwidth 算宽度的。
say() {
    local w pad
    w=$(printf '%s' "$1" | wc -L 2>/dev/null || echo 52)
    pad=$(( 52 > w ? 52 - w : 1 ))
    printf "  %s%*s %s\n" "$1" "$pad" "" "$2"
}
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== 3060 容器镜像自检 ($IMG) =="

# 1) ENV 里不能有 KVMEM_MODEL
if docker inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -q '^KVMEM_MODEL='; then
    say "① 镜像 ENV 不应含 KVMEM_MODEL" "✗（会让 MTP 自动切换失效）"; fail=1
else
    say "① 镜像 ENV 无 KVMEM_MODEL" "✓"
fi

# 2) 二进制能加载（--gpus all 让驱动注入；--help 不占显存）
if docker run --rm --gpus all --entrypoint /opt/kvmem/bin/llama-kvmem-server "$IMG" --help 2>&1 | grep -q "usage:"; then
    say "② 引擎二进制可加载（含 CUDA/驱动库）" "✓"
else
    say "② 引擎二进制可加载" "✗"; fail=1
fi

# 3) 缺模型时报错清晰
out="$(docker run --rm --gpus all -e KVMEM_MODEL=/nope.gguf "$IMG" 2>&1 || true)"
grep -q "缺模型" <<<"$out" && say "③ 缺模型时给出明确提示" "✓" || { say "③ 缺模型时给出明确提示" "✗"; fail=1; }

if [[ ! -r "$MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf" ]]; then
    echo "  （models/ 下没有模型，跳过 ④⑤⑥；把 .gguf 放进去可跑完整自检）"
    [[ "$fail" == "0" ]] && echo "== 结论：基础检查通过（未做推理链路）==" || echo "== 结论：有失败项 =="
    exit "$fail"
fi

# 4) 起容器（CPU 模式，不占显存）
docker run -d --name "$NAME" --gpus all \
    -p 127.0.0.1:${EP}:18202 -p 127.0.0.1:${PP}:18210 \
    -v "$MODELS:/models:ro" \
    -e KVMEM_NGL=0 -e KVMEM_CTX=4096 -e KVMEM_BUDGET=1024 -e KVMEM_RESERVE=512 \
    "$IMG" >/dev/null
for _ in $(seq 1 40); do
    curl -s --noproxy '*' -m 2 "http://127.0.0.1:${EP}/v1/models" >/dev/null 2>&1 && break; sleep 5
done
say "④ 引擎端口 /v1/models" "$(curl -s --noproxy '*' -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:${EP}/v1/models)"
say "⑤ 代理端口 /v1/models" "$(curl -s --noproxy '*' -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:${PP}/v1/models)"

banner="$(docker logs "$NAME" 2>&1 | grep -m1 'llama-kvmem-server 启动' || true)"
echo "     横幅: ${banner:-（没抓到）}"
grep -q "视觉=开" <<<"$banner" && say "⑥ 视觉塔已挂载" "✓" || say "⑥ 视觉塔已挂载" "（未挂，可忽略）"
grep -q "投机=开" <<<"$banner" && say "⑦ MTP 投机已开且换用合并模型" "✓" || say "⑦ MTP 投机" "（未开：可能只有无 MTP 的模型）"

ans="$(curl -s --noproxy '*' -m 900 "http://127.0.0.1:${PP}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"用一句话中文自我介绍。"}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:60])' 2>/dev/null || true)"
if [[ -n "$ans" ]]; then say "⑧ 经代理真实生成（CPU）" "✓ $ans"; else say "⑧ 经代理真实生成" "✗ 无输出"; fail=1; fi

echo "== 结论：$([[ "$fail" == 0 ]] && echo '全部通过 ✓（CUDA 内核路径需 GPU 模式另测）' || echo '有失败项 ✗') =="
exit "$fail"
