#!/usr/bin/env python3
"""NInfer Bonsai-2-27B 服务性能全面测试（针对运行中的 OpenAI 兼容服务）。

用法:
    python3 perf_test.py                       # 测 http://127.0.0.1:8088（全量，约 10-15 分钟）
    python3 perf_test.py --base-url http://127.0.0.1:8090
    python3 perf_test.py --quick               # 快速档（跳过长上下文与并发，约 3 分钟）

测试项:
  1  服务信息          max_model_len / 引擎在线
  2  TTFT 延迟         短提示 ×8（首字延迟 min/avg/p95）
  3  持续解码速度      512 tok ×3（t/s、MTP 接受率、tok/轮、波动）
  4  输出长度影响      64/256/512/1024 tok 的解码速度
  5  预填吞吐曲线      ~1k/4k/8k/16k/32k token 提示的 prefill t/s 与 TTFT
  6  速度-上下文关系   空上下文 vs 半满上下文的解码速度（KV 带宽代价）
  7  大海捞针          3 深度 × {8k, 32k, 最大-4k} 长度（精度）
  8  思考模式          开/关对比 + 推理题正确性（9.11 vs 9.9）
  9  采样参数影响      greedy vs temp0.7 的速度与接受率
 10  贪心确定性        同 prompt 两次输出逐字一致
 11  reasoning_effort 兼容回归   high/medium/low/none 全部 200
 12  并发吞吐          2/4 路并发（服务单路排队，看聚合吞吐与排队延迟）
 13  稳定性            20 连发错误率

结果: 控制台表格 + perf-results-<时间戳>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

AP = argparse.ArgumentParser()
AP.add_argument("--base-url", default="http://127.0.0.1:8088")
AP.add_argument("--quick", action="store_true", help="跳过长上下文/并发/稳定性，约 3 分钟")
AP.add_argument("--needle-max", type=int, default=32000, help="大海捞针最大长度（token）")
args = AP.parse_args()

BASE = args.base_url.rstrip("/")
MODEL = "qwen3.8-27b"
R = {"meta": {}, "tests": {}}
W = WORDS = ("history empire river stone bridge forest signal window garden language memory "
             "pattern surface shadow flight anchor harvest lattice cipher beacon meadow "
             "current fabric harbor lantern marble orbit pillar quartz ridge summit timber "
             "urn vessel willow xenon yarn zephyr anvil breeze candle dagger ember falcon").split()


def synthetic(n_chars, seed=42):
    import random
    rng = random.Random(seed)
    words, t = [], 0
    while t < n_chars:
        w = rng.choice(W)
        words.append(w)
        t += len(w) + 1
    return " ".join(words)


def chat(prompt, mx=256, timeout=1200, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": mx}
    body.update(kw)
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    resp["_wall"] = (time.time() - t0) * 1000
    return resp


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def show(name, **kv):
    R["tests"][name] = kv
    line = "  ".join(f"{k}={v}" for k, v in kv.items())
    print(f"  [{name}] {line}")


def main() -> int:
    # 1. 服务信息
    with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
        info = json.loads(r.read())["data"][0]
    max_len = info.get("max_model_len", 0)
    R["meta"] = {"base": BASE, "model": info["id"], "max_model_len": max_len}
    print(f"== 性能测试 @ {BASE} | 模型 {info['id']} | max_model_len {max_len} ==\n")

    essay = ("Explain how a modern computer boots from power-on to a login prompt. "
             "Write a detailed step-by-step explanation in about 400 words.")
    try:
        chat("hi", 8)
    except Exception as e:
        print(f"服务不可用: {e}")
        return 1

    # 2. TTFT
    ttfts, t0set = [], []
    for i in range(8):
        r = chat(f"用一句话说明数字 {i+2} 是不是质数。", 32, temperature=0)
        t = r["timings"]
        ttfts.append(t["prompt_ms"])
        t0set.append(r["_wall"])
    show("2_TTFT", avg_ms=round(sum(ttfts) / len(ttfts)), min_ms=round(min(ttfts)),
         p95_ms=round(pct(ttfts, 0.95)), wall_avg_ms=round(sum(t0set) / len(t0set)))

    # 3. 持续解码
    decs, accs, rounds = [], [], []
    for i in range(3):
        r = chat(essay, 512, temperature=0)
        t = r["timings"]
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        decs.append(t["predicted_per_second"])
        accs.append(100 * da / dn if dn else 0)
        rounds.append(t["predicted_n"] / max(1, dn / 3) if dn else 1)
    show("3_持续解码", avg=round(sum(decs) / len(decs), 1), min=round(min(decs), 1),
         max=round(max(decs), 1), unit="tok/s",
         mtp_acc=round(sum(accs) / len(accs), 1), tok_per_round=round(sum(rounds) / len(rounds), 2))

    # 4. 输出长度影响
    for mx in (64, 256, 512, 1024):
        r = chat(essay, mx, temperature=0)
        t = r["timings"]
        show(f"4_输出{mx}tok", decode=round(t["predicted_per_second"], 1), unit="tok/s",
             out_n=t["predicted_n"])

    # 5. 预填吞吐曲线
    print("  -- 预填曲线（每点一次请求）--")
    curve = {}
    for toks in (1000, 4000, 8000, 16000, 32000):
        if toks * 4 > max_len * 4 - 8000:
            break
        r = chat(synthetic(toks * 4, seed=toks) + " Answer in one word: what color is grass?", 8)
        t = r["timings"]
        curve[toks] = {"prefill": round(t["prompt_per_second"]), "ttft_ms": round(t["prompt_ms"]),
                       "prompt_n": t["prompt_n"]}
        print(f"    ~{toks//1000}k tok: prefill {t['prompt_per_second']:.0f} t/s | TTFT {t['prompt_ms']:.0f} ms")
    show("5_预填曲线", **{f"{k//1000}k": v["prefill"] for k, v in curve.items()}, unit="tok/s")

    # 6. 速度-上下文关系
    if 32000 <= max_len - 4000:
        fill = synthetic(30000 * 4, seed=99) + "\n\nContinue writing a 300-word technical story."
        r = chat(fill, 384, temperature=0)
        t = r["timings"]
        show("6_32k上下文解码", decode=round(t["predicted_per_second"], 1), unit="tok/s",
             prefill=round(t["prompt_per_second"]), note="对比第3项看KV带宽代价")
    else:
        print("  [6_速度-上下文] 跳过（上下文不足 32k）")

    # 7. 大海捞针
    if not args.quick:
        print("  -- 大海捞针（3 深度）--")
        needle_all = {}
        for length in sorted({8192, 32768, min(args.needle_max, max_len - 4000)}):
            if length < 6000 or length > max_len - 2000:
                continue
            oks = []
            for frac in (0.25, 0.5, 0.75):
                body = (synthetic(int(length * 4 * frac), seed=int(frac * 100))
                        + " The access code for the vault is BAMBOO-42."
                        + synthetic(int(length * 4 * (1 - frac)), seed=77)
                        + "\n\nWhat is the access code for the vault? Answer with the code only.")
                r = chat(body, 24, temperature=0)
                oks.append("BAMBOO-42" in r["choices"][0]["message"]["content"])
            needle_all[f"{length//1024}k"] = oks
            print(f"    {length//1024}k tok: {sum(oks)}/3 (深/中/浅)")
        show("7_大海捞针", **{k: f"{sum(v)}/3" for k, v in needle_all.items()})
    else:
        print("  [7_大海捞针] quick 模式跳过")

    # 8. 思考模式
    try:
        r_on = chat("9.11 和 9.9 哪个更大？请仔细思考后回答。", 2048)
        t_on = r_on["timings"]
        think_on = r_on.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        ans = r_on["choices"][0]["message"]["content"]
        correct = "9.9" in ans
        r_off = chat("9.11 和 9.9 哪个更大？", 256, enable_thinking=False)
        think_off = r_off.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        show("8_思考模式", reasoning_tok_on=think_on, reasoning_tok_off=think_off,
             答案正确=correct, wall_on_ms=round(r_on["_wall"]), wall_off_ms=round(r_off["_wall"]))
    except Exception as e:
        show("8_思考模式", error=str(e)[:60])

    # 9. 采样参数影响
    for tag, kw in (("greedy", dict(temperature=0)), ("temp0.7", dict(temperature=0.7))):
        r = chat(essay, 512, **kw)
        t = r["timings"]
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        show(f"9_采样_{tag}", decode=round(t["predicted_per_second"], 1),
             acc=round(100 * da / dn, 1) if dn else 0)

    # 10. 贪心确定性
    a = chat(essay, 256, temperature=0)["choices"][0]["message"]["content"]
    b = chat(essay, 256, temperature=0)["choices"][0]["message"]["content"]
    show("10_贪心确定性", identical=(a == b))

    # 11. reasoning_effort 兼容回归（关思考，只测 effort 路由）
    re_results = {}
    for effort in ("high", "medium", "low", "none"):
        try:
            r = chat("1+1=?", 8, reasoning_effort=effort, enable_thinking=False)
            re_results[effort] = r["choices"][0]["message"]["content"][:10]
        except urllib.error.HTTPError as e:
            re_results[effort] = f"HTTP{e.code}"
    show("11_effort回归", **re_results)

    # 12. 并发（服务端单路，会排队）
    if not args.quick:
        for n in (2, 4):
            lat, tps = [], []
            lock = threading.Lock()
            t0 = time.time()

            def one(i):
                r = chat(f"用一句话解释线程 {i} 的概念。", 128, temperature=0)
                with lock:
                    lat.append(r["_wall"])
                    tps.append(r["timings"]["predicted_per_second"])

            with ThreadPoolExecutor(max_workers=n) as ex:
                list(ex.map(one, range(n)))
            wall = time.time() - t0
            show(f"12_并发{n}路", wall_s=round(wall, 1), avg_lat_ms=round(sum(lat) / len(lat)),
                 聚合_decode=round(sum(tps), 1))
    else:
        print("  [12_并发] quick 模式跳过")

    # 13. 稳定性
    if not args.quick:
        errs, ok = 0, 0
        for i in range(20):
            try:
                chat(f"2 的 {i+1} 次方等于多少？只回答数字。", 24, temperature=0)
                ok += 1
            except Exception:
                errs += 1
        show("13_稳定性20连发", ok=ok, err=errs)

    # 汇总
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = f"perf-results-{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print(f"\n== 完成，结果已存 {out} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
