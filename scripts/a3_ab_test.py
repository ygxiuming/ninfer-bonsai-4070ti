#!/usr/bin/env python3
"""A3（token → grid.y）两臂对照验证 —— pkg5《验证-A3开关-A-B.ps1》的 Linux 移植版。

口径与官方 ps1 完全一致：
  * 两臂【逐轮交替】跑，丢首轮（热身），其余轮取【中位】；
  * 每个臂每轮【单独起一次引擎】（env 每进程只读一次，不能在同一进程里切）；
  * prefill 取【引擎返回的 timings.prompt_per_second】（不是客户端 TTFT 折算）；
  * 每轮输出【贪心 sha256 前 16 位】（覆盖 content + "\\0" + reasoning_content），
    两臂逐轮相同才算逐位相同（硬门槛 G1）；
  * 用 nonce 前缀强制【冷缓存】；
  * base 臂 = SMALL_T_ROWS=16 + TOKEN_GRID=0，sched3 臂 = 默认；两臂共同 NINFER_TERNARY_MMA=1。

门：G1 两臂逐轮 sha 相同；G3 |prefill 增幅| >= 1%；G2（PPL 逐位）本夹具不覆盖。

用法:
    python3 a3_ab_test.py [--exe PATH] [--artifact PATH] [--port 8137]
                          [--runs 3] [--warmup 1] [--tokens 400]
                          [--prompt-words 900] [--max-ctx 16384] [--draft 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HOME = Path.home()
SANYUAN = HOME / "pyprojects" / "sanyuan"
DEFAULT_EXE = SANYUAN / "ninfer-4090w-ternary-a3" / "build" / "apps" / "ninfer-serve"
DEFAULT_ART = SANYUAN / "artifacts-pq2.ninfer"
OUTDIR = SANYUAN / "verify-out"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def vram_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def start_engine(exe: Path, artifact: Path, port: int, args, envv: dict, tag: str):
    env = os.environ.copy()
    for k in ("NINFER_TERNARY_SMALL_T_ROWS", "NINFER_TERNARY_TOKEN_GRID",
              "NINFER_TERNARY_KSPLIT"):
        env.pop(k, None)
    env["NINFER_TERNARY_MMA"] = "1"
    env.update(envv)
    cmd = [str(exe), str(artifact),
           "--host", "127.0.0.1", "--port", str(port),
           "--model-id", "qwen3.8-27b",
           "--max-context", str(args.max_ctx),
           "--max-concurrency", "1",
           "--kv-dtype", "fp8", "--spec", "mtp", "--draft-tokens", str(args.draft),
           "--pending-timeout-ms", "3600000", "--max-pending-requests", "64"]
    out_log = open(OUTDIR / f"engine-{tag}.out.log", "wb")
    err_log = open(OUTDIR / f"engine-{tag}.err.log", "wb")
    proc = subprocess.Popen(cmd, stdout=out_log, stderr=err_log,
                            start_new_session=True, env=env)
    for _ in range(90):
        time.sleep(2)
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                        timeout=5) as r:
                if json.loads(r.read()).get("data"):
                    return proc, out_log, err_log
        except Exception:
            pass
    log(f"  引擎没起来（{tag}）—— err 尾部：")
    err_log.close()
    tail = (OUTDIR / f"engine-{tag}.err.log").read_text(errors="replace")[-800:]
    for line in tail.splitlines():
        log(f"    {line}")
    return None, out_log, err_log


def stop_engine(proc, out_log, err_log) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=10)
    out_log.close()
    err_log.close()
    time.sleep(4)


def measure(port: int, nonce: str, args) -> dict:
    words = " ".join(f"topic{i}" for i in range(1, args.prompt_words + 1))
    prompt = ("Summarise the following numbered list of identifiers in continuous prose, then "
              "write a long factual article about ternary weight quantisation. Do not stop early.\n"
              + words)
    if nonce:
        prompt = f"Run {nonce}.\n{prompt}"
    body = {"model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.tokens, "temperature": 0}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read())
    except Exception as e:
        return {"ok": False, "err": str(e)[:200]}
    tim = resp["timings"]
    msg = resp["choices"][0]["message"]
    text = f"{msg.get('content', '')}\0{msg.get('reasoning_content') or ''}"
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    dn = int(tim.get("draft_n") or 0)
    da = int(tim.get("draft_n_accepted") or 0)
    return {"ok": True,
            "decode": float(tim["predicted_per_second"]),
            "prefill": float(tim["prompt_per_second"]),
            "prompt_tokens": int(resp["usage"]["prompt_tokens"]),
            "gen": int(resp["usage"]["completion_tokens"]),
            "dn": dn, "da": da,
            "acc": round(100.0 * da / dn, 2) if dn > 0 else 0.0,
            "sha": sha}


def median(xs):
    return statistics.median(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ART)
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--prompt-words", type=int, default=900)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--draft", type=int, default=2)
    ap.add_argument("--base-env", type=str, default="",
                    help='JSON dict，覆盖 base 臂环境变量（默认 ps1 口径）')
    ap.add_argument("--sched-env", type=str, default="",
                    help='JSON dict，覆盖 sched3 臂环境变量（默认全缺省）')
    ap.add_argument("--same-nonce", action="store_true",
                    help="两臂用同一 nonce（同 prompt）；默认 nonce 含臂名")
    args = ap.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    assert args.exe.is_file(), f"exe 不存在: {args.exe}"
    assert args.artifact.is_file(), f"制品不存在: {args.artifact}"

    base_env = json.loads(args.base_env) if args.base_env else {
        "NINFER_TERNARY_SMALL_T_ROWS": "16", "NINFER_TERNARY_TOKEN_GRID": "0"}
    sched_env = json.loads(args.sched_env) if args.sched_env else {}
    arms = {
        "base": (base_env, f"A3 OFF ({base_env})"),
        "sched3": (sched_env, f"A3 ON ({sched_env or 'defaults'})"),
    }

    log("=" * 60)
    log("A3 (token -> grid.y) A/B -- interleaved rounds, median, sha gate")
    log("=" * 60)
    log(f"exe={args.exe}")
    log(f"artifact={args.artifact}")
    log(f"port={args.port} runs={args.runs} warmup={args.warmup} "
        f"tokens={args.tokens} promptWords={args.prompt_words} "
        f"maxCtx={args.max_ctx} draft={args.draft}")
    vram = vram_used_mib()
    log(f"preflight: vram used = {vram} MiB")
    if vram > 2500:
        log("[WARN] 显存已被占用：测量必须独占 GPU，否则读数作废。")

    all_results = {a: [] for a in arms}
    aborted = False
    for r in range(1, args.warmup + args.runs + 1):
        for a, (envv, what) in arms.items():
            tag = f"{a}-r{r}"
            log(f"---- arm={a} round={r}  [{what}]")
            proc, out_log, err_log = start_engine(args.exe, args.artifact,
                                                  args.port, args, envv, tag)
            if proc is None:
                aborted = True
                break
            nonce = f"r{r}-{stamp}" if args.same_nonce else f"r{r}-{a}-{stamp}"
            m = measure(args.port, nonce, args)
            stop_engine(proc, out_log, err_log)
            if not m["ok"]:
                log(f"     ERROR: {m['err']}")
                aborted = True
                break
            all_results[a].append(m)
            log(f"     decode={m['decode']:.1f} t/s  prefill={m['prefill']:.1f} t/s  "
                f"accept={m['acc']:.1f}% ({m['da']}/{m['dn']})  gen={m['gen']}  sha={m['sha']}")
        if aborted:
            break

    if aborted:
        log("\n[ABORT] 有臂没起来或测量失败 —— 数据不完整，不作为结论。")
        return 5

    log("\n=== per-arm medians (warmup rounds dropped) ===")
    summary = {}
    for a in arms:
        meas = all_results[a][args.warmup:]  # 轮号 > warmup 的才是测量轮（与 ps1 一致）
        summary[a] = {
            "decode_median": round(median([m["decode"] for m in meas]), 2),
            "prefill_median": round(median([m["prefill"] for m in meas]), 2),
            "acc_median": round(median([m["acc"] for m in meas]), 2),
            "prefill_all": [round(m["prefill"], 1) for m in meas],
            "decode_all": [round(m["decode"], 1) for m in meas],
            "sha_all": [m["sha"] for m in meas],
        }
        s = summary[a]
        log(f"  {a:<8} decode={s['decode_median']:7.1f} | "
            f"prefill={s['prefill_median']:9.1f} | accept={s['acc_median']:5.1f}%")

    b, s = summary["base"], summary["sched3"]
    pct = 100.0 * (s["prefill_median"] - b["prefill_median"]) / b["prefill_median"]
    pct_d = 100.0 * (s["decode_median"] - b["decode_median"]) / b["decode_median"]

    sha_ok = True
    for i in range(len(all_results["base"])):
        sb = all_results["base"][i]["sha"]
        ss = all_results["sched3"][i]["sha"]
        if sb != ss:
            sha_ok = False
            log(f"  [GATE-FAIL] round {i+1} sha differs: base={sb} sched3={ss}")

    log("")
    log(f"prefill: base {b['prefill_median']:.1f} -> A3 {s['prefill_median']:.1f} t/s  =  {pct:+.1f}%")
    log(f"decode : base {b['decode_median']:.1f} -> A3 {s['decode_median']:.1f} t/s  =  {pct_d:+.1f}%  (expect within +-2% noise)")
    log(f"gate G1 (greedy sha identical every round) : {'PASS' if sha_ok else 'FAIL -- 改错了地方，回退'}")
    log(f"gate G3 (gain >= 1%)                       : {'PASS' if abs(pct) >= 1 else 'FAIL -- 视为无收益，保持出厂引擎'}")
    log("gate G2 (PPL bit-identical, two windows)  : 本夹具不覆盖 —— 按主包 F0 的数值门单独做")

    payload = {"stamp": stamp, "exe": str(args.exe), "artifact": str(args.artifact),
               "port": args.port, "runs": args.runs, "warmup_dropped": args.warmup,
               "tokens": args.tokens, "prompt_words": args.prompt_words,
               "max_ctx": args.max_ctx, "draft": args.draft,
               "arms": summary, "delta_prefill_pct": round(pct, 2),
               "delta_decode_pct": round(pct_d, 2),
               "gate_g1_sha_identical": sha_ok,
               "gate_g3_gain_ge_1pct": abs(pct) >= 1,
               "note": "prefill from engine timings.prompt_per_second; arms interleaved; "
                       "first round discarded; median of the rest",
               "raw": all_results}
    json_path = OUTDIR / f"verify-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(f"\nJSON: {json_path}")
    return 0 if sha_ok and abs(pct) >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
