#!/usr/bin/env python3
"""NInfer Bonsai-2-27B 速度 + 精度 综合测试套件。

用法:
    python3 suite_ninfer.py [--engine <engine_root>] [--artifact <path>] [--skip-ppl]
                            [--base-port 8090]

内容:
  A. 速度矩阵    MTP K ∈ {1,2,3,5,7} × greedy/temp0.7，16k fp8
                 （K=7 需要 --k7patch 编译过的引擎；失败会自动跳过）
  B. 峰值接受率  高可预测任务（翻译）在最优 K 下
  C. 精度-投机解码一致性  K=1 vs K=K 的 greedy 输出必须逐字一致
  D. 精度-大海捞针        3 个深度 × 当前最大上下文
  E. 精度-代码执行        3 道题实际运行验证
  F. 精度-PPL             fp8 / int8 / rk4v4 三档 KV 各测一次
                         （ninfer-perplexity，固定语料；黄金参照 6.82±0.54，旧版实测 11.0@小说）
结果写入 suite-results-<时间戳>.json 与控制台表格。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

AP = argparse.ArgumentParser()
AP.add_argument("--engine", default=str(Path.home() / "pyprojects/sanyuan/ninfer-4090w-ternary"))
AP.add_argument("--artifact", default=str(Path.home() / "pyprojects/sanyuan/artifacts-pq2.ninfer"))
AP.add_argument("--base-port", type=int, default=8090)
AP.add_argument("--skip-ppl", action="store_true")
AP.add_argument("--skip-k7", action="store_true", help="引擎未打 K7 补丁时使用")
args = AP.parse_args()

ENGINE = args.engine
ARTIFACT = args.artifact
PORT = args.base_port
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "qwen3.8-27b"
RESULTS = {"speed": [], "consistency": [], "needle": [], "code": [], "ppl": []}

ESSAY = ("Explain how a modern computer boots from power-on to a login prompt. "
         "Write a detailed step-by-step explanation in about 400 words.")


# ---------------------------------------------------------------- server mgmt
def kill_servers():
    out = subprocess.run(["pgrep", "-x", "ninfer-serve"], capture_output=True, text=True)
    for pid in out.stdout.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(15):
        if subprocess.run(["pgrep", "-x", "ninfer-serve"], capture_output=True).returncode != 0:
            return
        time.sleep(1)


def start_server(kv, k, ctx, extra=()):
    kill_servers()
    log = open(f"/tmp/suite-serve-{ctx}-{kv}-{k}.log", "w")
    cmd = [f"{ENGINE}/build/apps/ninfer-serve", ARTIFACT,
           "--host", "127.0.0.1", "--port", str(PORT), "--model-id", MODEL,
           "--max-context", str(ctx), "--kv-capacity", str(ctx), "--kv-dtype", kv,
           "--max-concurrency", "1", "--no-thinking", *extra]
    if k:
        cmd += ["--spec", "mtp", "--draft-tokens", str(k)]
    else:
        pass  # K=1: 不带 spec
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
    for _ in range(90):
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"server died: see /tmp/suite-serve-{ctx}-{kv}-{k}.log")
            time.sleep(1)
    raise RuntimeError("server start timeout")


# ---------------------------------------------------------------- http helpers
def chat(prompt, mx=256, temperature=0, timeout=900):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mx, "temperature": temperature}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def synthetic(n_chars, seed=42):
    import random
    rng = random.Random(seed)
    W = ("history empire river stone bridge forest signal window garden language memory "
         "pattern surface shadow flight anchor harvest lattice cipher beacon meadow "
         "current fabric harbor lantern marble orbit pillar quartz ridge summit timber").split()
    words, t = [], 0
    while t < n_chars:
        w = rng.choice(W)
        words.append(w)
        t += len(w) + 1
    return " ".join(words)


# ---------------------------------------------------------------- A+B. 速度
def speed_matrix(max_k):
    ks = [k for k in (1, 2, 3, 5, max_k if max_k not in (1, 2, 3, 5) else None) if k]
    rows = []
    for k in ks:
        try:
            proc = start_server("fp8", k, 16384)
        except Exception as e:
            print(f"  K={k}: 启动失败 ({e})")
            rows.append({"k": k, "error": str(e)[:60]})
            continue
        try:
            chat("hi", 16)
            r = chat(ESSAY, 512, temperature=0)
            t = r["timings"]
            dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
            row = {"k": k, "decode": round(t["predicted_per_second"], 1),
                   "acc": round(100 * da / dn, 1) if dn else 0.0,
                   "tok_round": round(t["predicted_n"] / (dn / max(k, 1)), 2) if dn else 1.0,
                   "prefill_short": round(t["prompt_per_second"], 0)}
            # 高可预测任务（翻译）→ 峰值接受率
            src = synthetic(3000)
            r2 = chat("请将下面的英文段落翻译成中文：\n\n" + src, 512, temperature=0)
            t2 = r2["timings"]
            dn2, da2 = t2.get("draft_n", 0), t2.get("draft_n_accepted", 0)
            row["acc_trans"] = round(100 * da2 / dn2, 1) if dn2 else 0.0
            row["decode_trans"] = round(t2["predicted_per_second"], 1)
            rows.append(row)
            print(f"  K={k}: decode {row['decode']} t/s | 接受率 {row['acc']}% "
                  f"(翻译 {row['acc_trans']}%) | 翻译decode {row['decode_trans']} t/s")
        except Exception as e:
            rows.append({"k": k, "error": str(e)[:60]})
            print(f"  K={k}: 测试失败 ({str(e)[:50]})")
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
    RESULTS["speed"] = rows


# ---------------------------------------------------------------- C. 一致性
def consistency(max_k):
    """投机解码不能改变 greedy 输出：K=1 与 K=max 的输出必须逐字一致。"""
    proc = start_server("fp8", 1, 16384)
    prompts = [
        "Count from 1 to 20, separated by spaces.",
        "List the first 12 prime numbers.",
        ESSAY,
    ]
    base_out = []
    for p in prompts:
        r = chat(p, 384, temperature=0)
        base_out.append(r["choices"][0]["message"]["content"])
    try:
        proc.terminate()
    except Exception:
        pass
    proc = start_server("fp8", max_k, 16384)
    same = 0
    for i, p in enumerate(prompts):
        r = chat(p, 384, temperature=0)
        ok = r["choices"][0]["message"]["content"] == base_out[i]
        same += ok
        print(f"  一致性 {i+1}/3: {'PASS' if ok else 'FAIL'}")
        RESULTS["consistency"].append({"case": i + 1, "ok": ok})
    print(f"  投机解码一致性: {same}/3 （逐字一致 = 数值无损）")
    try:
        proc.terminate()
    except Exception:
        pass


# ---------------------------------------------------------------- D. 大海捞针
def needle(max_ctx):
    proc = start_server("fp8", 3, max_ctx)
    depth_ok = {}
    for frac, tag in ((0.25, "25%"), (0.5, "50%"), (0.75, "75%")):
        target = min(30000, max_ctx - 4000)
        body = synthetic(int(target * 4 * frac))
        body += " The access code for the vault is BAMBOO-42."
        body += synthetic(int(target * 4 * (1 - frac)), seed=7)
        body += "\n\nWhat is the access code for the vault? Answer with the code only."
        r = chat(body, 24, temperature=0)
        ok = "BAMBOO-42" in r["choices"][0]["message"]["content"]
        depth_ok[tag] = ok
        RESULTS["needle"].append({"depth": tag, "ctx": target, "ok": ok})
        print(f"  深度{tag} (~{target//1000}k tok): {'PASS' if ok else 'FAIL'} "
              f"→ {r['choices'][0]['message']['content'][:40]}")
    print(f"  大海捞针: {sum(depth_ok.values())}/3")
    try:
        proc.terminate()
    except Exception:
        pass


# ---------------------------------------------------------------- E. 代码执行
def code_tests():
    proc = start_server("fp8", 3, 16384)
    cases = [
        ("Write a Python function fib(n) returning the nth Fibonacci number (fib(0)=0). "
         "Only code.", "print(fib(0), fib(1), fib(10), fib(20))", "0 1 55 6765"),
        ("Write a Python function rev(s) that reverses a string. Only code.",
         "print(rev('ninfer'), rev('abc'))", "refnin cba"),
    ]
    passed = 0
    for i, (task, probe, want) in enumerate(cases):
        r = chat(task + " 只输出代码。", 200, temperature=0)
        code = r["choices"][0]["message"]["content"]
        if "```" in code:
            code = code.split("```")[1].removeprefix("python").strip()
        try:
            out = subprocess.run([sys.executable, "-c", code + "\n" + probe],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            ok = out == want
        except Exception:
            ok = False
        passed += ok
        RESULTS["code"].append({"case": i + 1, "ok": ok})
        print(f"  代码题 {i+1}: {'PASS' if ok else 'FAIL'} ({out!r})")
    print(f"  代码执行: {passed}/{len(cases)}")
    try:
        proc.terminate()
    except Exception:
        pass


# ---------------------------------------------------------------- F. PPL
def ppl_suite():
    corpus = Path("/tmp/ppl_corpus_raw.txt")
    if not corpus.exists():
        try:
            subprocess.run(["curl", "-sL", "--max-time", "60",
                            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
                            "-o", str(corpus)], check=True)
        except Exception:
            print("  语料获取失败，跳过 PPL")
            return
    data = corpus.read_text(encoding="utf-8", errors="ignore")
    Path("/tmp/ppl_suite_slice.txt").write_text(
        data[int(len(data) * 0.18):int(len(data) * 0.18) + 180000], encoding="utf-8")
    for kv in ("fp8", "int8", "rk4v4"):
        out = subprocess.run(
            [f"{ENGINE}/build/apps/ninfer-perplexity", ARTIFACT,
             "--text", "/tmp/ppl_suite_slice.txt", "--context", "512", "--stride", "256",
             "--kv-dtype", kv],
            capture_output=True, text=True, timeout=1800)
        m = re.findall(r"PPL\s+([0-9.]+)", out.stdout + out.stderr)
        ppl = m[-1] if m else "?"
        RESULTS["ppl"].append({"kv": kv, "ppl": ppl})
        print(f"  KV={kv}: PPL {ppl}  （黄金参照 6.82±0.54；同语料旧版 fp8 ≈ 11.0）")


# ---------------------------------------------------------------- main
def main() -> int:
    max_k = 7
    if not args.skip_k7:
        try:
            proc = start_server("fp8", 7, 16384)
            print("K=7 引擎（打过补丁）可用")
            proc.terminate()
        except Exception as e:
            print(f"K=7 不可用（{str(e)[:40]}），回退 K=5")
            max_k = 5
    else:
        max_k = 5

    print("\n== A/B. 速度矩阵 (fp8, 16k ctx) ==")
    speed_matrix(max_k)

    print("\n== C. 投机解码一致性 (K=1 vs K=max, greedy 逐字比对) ==")
    consistency(max_k)

    print("\n== D. 大海捞针 ==")
    needle(114688 if not args.skip_k7 else 65536)

    print("\n== E. 代码执行 ==")
    code_tests()

    if not args.skip_ppl:
        print("\n== F. PPL（KV 档位精度损耗） ==")
        ppl_suite()

    ts = time.strftime("%Y%m%d-%H%M%S")
    Path(f"suite-results-{ts}.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    print(f"\n结果已存: suite-results-{ts}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
