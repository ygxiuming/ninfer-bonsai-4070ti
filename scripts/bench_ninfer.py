#!/usr/bin/env python3
"""NInfer Ternary-Bonsai-2-27B 全参数基准测试（针对 OpenAI 兼容接口）。

用法:
    python3 bench_ninfer.py [base_url]          # 默认 http://127.0.0.1:8084
    （先用 start-ninfer.sh 启动服务）

测试项:
  1. 服务信息           /v1/models, max_model_len
  2. 正确性哨兵         英文常识 / 中文身份 / 素数函数（生成代码并实际执行验证）
  3. TTFT               短提示 ×5 的首字延迟
  4. 持续解码速度       512 token 生成 ×2（decode t/s + MTP 接受率/轮）
  5. 预填吞吐           ~6k token 提示（prefill t/s）
  6. 长上下文           自适应（≤24k 或 max_model_len-4k）大海捞针 + 预填速度
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8084").rstrip("/")
MODEL = "qwen3.8-27b"
TIMEOUT = 1200

WORDS = ("history empire river stone bridge forest signal window garden language "
         "memory pattern surface shadow flight anchor harvest lattice cipher beacon "
         "meadow current fabric harbor lantern marble orbit pillar quartz ridge "
         "summit timber umbrella velvet whisper xenon yarn zephyr anvil breeze "
         "candle dagger ember falcon granite hollow ivory jungle kettle lagoon").split()


def synthetic_text(n_chars: int) -> str:
    import random
    rng = random.Random(42)
    words = []
    total = 0
    while total < n_chars:
        w = rng.choice(WORDS)
        words.append(w)
        total += len(w) + 1
    text = " ".join(words)
    return " ".join(text[i:i + 80] for i in range(0, len(text), 80)).replace("  ", " ")


def chat(messages, max_tokens=256, timeout=TIMEOUT, **overrides):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    body.update(overrides)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fmt(x, nd=1, suffix=""):
    return f"{x:.{nd}f}{suffix}" if isinstance(x, (int, float)) else str(x)


results = []


def record(name, value, note=""):
    results.append((name, value, note))
    print(f"  [{name}] {value}  {note}")


def main() -> int:
    print(f"== NInfer 基准测试 @ {BASE} ==")

    # 1. 服务信息
    with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
        info = json.loads(r.read())["data"][0]
    max_len = info.get("max_model_len", 0)
    print(f"  模型: {info['id']} | max_model_len: {max_len}")

    # 2a. 英文常识
    r = chat([{"role": "user", "content": "The capital of France is what city? One word answer."}],
             max_tokens=16, temperature=0.0)
    ok1 = "paris" in r["choices"][0]["message"]["content"].lower()
    record("正确性·英文常识", "PASS" if ok1 else "FAIL", r["choices"][0]["message"]["content"][:40])

    # 2b. 中文身份
    r = chat([{"role": "user", "content": "用一句话介绍你是什么模型。"}], max_tokens=64)
    record("正确性·中文输出", "PASS" if len(r["choices"][0]["message"]["content"]) > 4 else "FAIL",
           r["choices"][0]["message"]["content"][:40])

    # 2c. 代码生成 + 实际执行
    r = chat([{"role": "user", "content":
               "写一个 Python 函数 is_prime(n) 判断素数，只输出代码，不要解释。"}],
             max_tokens=200, temperature=0.0)
    code = r["choices"][0]["message"]["content"]
    if "```" in code:
        code = code.split("```")[1].removeprefix("python").strip()
    probe = (code + "\nprint(is_prime(7), is_prime(10), is_prime(97), is_prime(1))")
    try:
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, timeout=15).stdout.strip()
        code_ok = out == "True False True False"
    except Exception:
        code_ok = False
    record("正确性·代码执行", "PASS" if code_ok else "FAIL",
           f"is_prime → {out!r}" if code_ok else "输出/执行不符")

    # 3. TTFT（短提示 ×5）
    ttfts = []
    for i in range(5):
        t0 = time.time()
        chat([{"role": "user", "content": f"用一句话说明数字 {i+2} 是不是质数。"}],
             max_tokens=48, temperature=0.0)
        ttfts.append((time.time() - t0) * 1000)
    record("TTFT(短提示×5)", fmt(sum(ttfts) / len(ttfts), 0, " ms"),
           f"min {min(ttfts):.0f} / max {max(ttfts):.0f}")

    # 4. 持续解码速度 ×2
    essay = [{"role": "user", "content":
              "Explain how a modern computer boots from power-on to a login prompt. "
              "Write a detailed step-by-step explanation in about 400 words."}]
    decs, accs, lens = [], [], []
    for _ in range(2):
        r = chat(essay, max_tokens=512)
        t = r["timings"]
        decs.append(t["predicted_per_second"])
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        accs.append(100.0 * da / dn if dn else 0.0)
        lens.append(t["predicted_n"] / max(1, dn - da + t["predicted_n"] - (dn - da)) if dn else 0)
    record("解码速度(512tok×2)", fmt(sum(decs) / len(decs), 1, " tok/s"),
           f"两轮 {fmt(decs[0])}/{fmt(decs[1])}")
    record("MTP 接受率", fmt(sum(accs) / len(accs), 1, "%"), f"draft 采纳统计")

    # 5. 预填吞吐（~6k token）
    r = chat([{"role": "user", "content":
               synthetic_text(24000) + "\n\nAnswer in one word: what color is grass?"}],
             max_tokens=8)
    t = r["timings"]
    record("预填吞吐(~6k tok)", fmt(t["prompt_per_second"], 0, " tok/s"),
           f"prompt_n={t['prompt_n']} TTFT={t['prompt_ms']:.0f}ms")

    # 6. 长上下文（大海捞针，自适应上限）
    if max_len > 10000:
        target_tokens = min(24000, max_len - 4000)
        chars = target_tokens * 4
        needle_prompt = (synthetic_text(chars) +
                         "\n\nIgnore the document above. In one sentence, "
                         "what city is the capital of France?")
        try:
            r = chat([{"role": "user", "content": needle_prompt}], max_tokens=64)
            t = r["timings"]
            ok = "paris" in r["choices"][0]["message"]["content"].lower()
            record(f"长上下文(~{target_tokens//1000}k tok)", "PASS" if ok else "FAIL",
                   f"prefill {t['prompt_per_second']:.0f} tok/s")
        except Exception as e:
            record(f"长上下文(~{target_tokens//1000}k tok)", "SKIP", str(e)[:60])
    else:
        record("长上下文", "SKIP", f"max_model_len={max_len} 太小")

    # 汇总
    print("\n== 汇总 ==")
    fails = sum(1 for _, v, _ in results if v == "FAIL")
    for name, value, note in results:
        print(f"  {name:<22} {value:<16} {note}")
    print(f"\n结果: {len(results) - fails}/{len(results)} 项通过"
          + ("" if fails == 0 else f"（{fails} 项失败）"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
