#!/usr/bin/env bash
# =============================================================================
# 容器服务冒烟测试（run-ninfer.sh 就绪后执行）
# 用法: ./docker/test-serving.sh [端口，默认 8089]
# 检查: ① /v1/models ② 一次真实生成（关思考、temp=0，验证内容与解码速度）
# =============================================================================
set -euo pipefail
PORT="${1:-8089}"
BASE="http://127.0.0.1:$PORT"

echo "① 模型列表:"
curl -sf -m 5 "$BASE/v1/models" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  ", d["data"][0]["id"], "| max_model_len", d["data"][0]["max_model_len"])'

echo "② 生成测试（enable_thinking=false, temperature=0）:"
resp=$(curl -sf -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
  "max_tokens": 96, "temperature": 0, "enable_thinking": false
}')
RESP="$resp" python3 - <<'PY'
import json, os
r = json.loads(os.environ["RESP"])
t, c = r["timings"], r["choices"][0]["message"]["content"]
print("  ", c.strip()[:80])
print(f'   decode {t["predicted_per_second"]:.1f} tok/s | TTFT {t["prompt_ms"]:.0f} ms')
PY
echo "冒烟通过 ✅"
