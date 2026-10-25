#!/usr/bin/env python3
"""kvmem-proxy —— 本地模型「上下文自动压缩」中间层。

链路：ZCode/客户端 → 本代理(18210) → llama-kvmem-server(18202)

做什么：
  透传所有请求（JSON 与 SSE 流式都支持）；当 /v1/chat/completions 的提示
  估算 token 超过阈值时，把「较早的消息」压缩成一段滚动摘要（摘要由上游
  模型自己生成，增量更新：每次只摘要新增部分），只保留 system 消息和最近
  N 条原文。效果：即使客户端把历史越滚越大，每次发给引擎的提示也有硬上限，
  冷缓存全量预填充从「十几分钟」封顶到「1~2 分钟」。

不做的事：
  不改写最近 N 条消息；不动 system 消息的当前副本；上游/摘要失败时原样
  透传（宁可慢，不丢内容）。

环境变量：
  KVMEM_PROXY_PORT=18210              监听端口
  KVMEM_PROXY_UPSTREAM_PORT=18202     上游引擎端口
  KVMEM_PROXY_THRESHOLD=45000         触发压缩的估算 token 阈值
  KVMEM_PROXY_KEEP_RECENT=10          压缩时保留最近多少条原文
  KVMEM_PROXY_KEEP_EST_MAX=15000      保留部分的估算 token 上限
  KVMEM_PROXY_SUMMARY_MAX_TOKENS=1600 摘要生成上限
  KVMEM_PROXY_REBUILD_MIN=12000       滑出 keep 窗口的新消息攒到多少「估算 token」才重建摘要
                                      （攒够前沿用旧摘要、把滑出消息原样追加 → 前缀稳定、KV 缓存命中）
  KVMEM_PROXY_LOGDIR=~/kvmem-logs     日志与 pid 文件目录（start-kvmem.sh 会传自己的 $LOGDIR）
日志：$KVMEM_PROXY_LOGDIR/kvmem-proxy.log
"""
import hashlib
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("KVMEM_PROXY_HOST", "127.0.0.1")   # 容器里需要 0.0.0.0（否则宿主端口映射够不着）
PORT = int(os.environ.get("KVMEM_PROXY_PORT", "18210"))
UP_HOST = "127.0.0.1"
UP_PORT = int(os.environ.get("KVMEM_PROXY_UPSTREAM_PORT", "18202"))
THRESHOLD = int(os.environ.get("KVMEM_PROXY_THRESHOLD", "45000"))
KEEP_RECENT = int(os.environ.get("KVMEM_PROXY_KEEP_RECENT", "10"))
KEEP_EST_MAX = int(os.environ.get("KVMEM_PROXY_KEEP_EST_MAX", "15000"))
SUMMARY_MAX_TOKENS = int(os.environ.get("KVMEM_PROXY_SUMMARY_MAX_TOKENS", "1600"))
REBUILD_MIN = int(os.environ.get("KVMEM_PROXY_REBUILD_MIN", "12000"))
UP_TIMEOUT = 1800          # 上游整请求上限（预填充最长的情况）
# 日志与 pid 的落盘目录：默认 ~/kvmem-logs，可用 KVMEM_PROXY_LOGDIR 覆盖
# （start-kvmem.sh 会把它的 $LOGDIR 传进来，避免用户自定义 LOGDIR 后两边日志分家）
LOG_DIR = os.environ.get("KVMEM_PROXY_LOGDIR") or os.path.expanduser("~/kvmem-logs")
STATE_PATH = os.path.expanduser("~/.cache/kvmem-proxy-state.json")
LOG_PATH = os.path.join(LOG_DIR, "kvmem-proxy.log")
PID_PATH = os.path.join(LOG_DIR, "kvmem-proxy.pid")

_LOG_LOCK = threading.Lock()


def log(msg):
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    with _LOG_LOCK:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------- token 估算
def _est_str(s):
    cjk = sum(1 for ch in s if "\u2e80" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")
    return int(cjk * 1.05 + (len(s) - cjk) / 3.8)


def _est_value(v):
    if isinstance(v, str):
        return 0 if v.startswith("data:") else _est_str(v)   # 媒体 data URI 不按字符算
    if isinstance(v, list):
        return sum(_est_value(x) for x in v)
    if isinstance(v, dict):
        t = v.get("type")
        if t in ("image", "image_url", "input_image"):
            return 1500        # 实测一张截图 ≈ 1.5~2k token
        if t in ("video", "input_video"):
            return 4000
        return sum(_est_value(x) for x in v.values()) + 4
    return 0


def est_tokens(msgs):
    total = 0
    for m in msgs:
        total += 8 + _est_value(m.get("content")) + _est_value(m.get("tool_calls"))
    return total


# ---------------------------------------------------------------- 压缩状态
class Compressor:
    """按会话（首个非 system 消息哈希为键）维护滚动摘要，前缀匹配则增量。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}          # key -> {"summary","covered_len","span_sig"}
        self._load()

    def _load(self):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                self.sessions = json.load(f).get("sessions", {})
        except (OSError, ValueError):
            self.sessions = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"sessions": self.sessions}, f, ensure_ascii=False)
        except OSError:
            pass

    @staticmethod
    def _sig(msgs):
        """只对非 system 消息做签名（system 提醒每轮可能变，不该打断增量链）。"""
        payload = json.dumps(
            [(m.get("role"), m.get("content")) for m in msgs if m.get("role") != "system"],
            ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # -------------------------------------------------------------- 主入口
    def maybe_compress(self, body):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or len(msgs) <= KEEP_RECENT + 1:
            return body, None
        est = est_tokens(msgs)
        if est <= THRESHOLD:
            return body, None

        # 切分：开头 system | 其余（旧段 span + 保留段 keep）
        lead_sys = []
        i = 0
        while i < len(msgs) and msgs[i].get("role") == "system":
            lead_sys.append(msgs[i]); i += 1
        rest = msgs[i:]
        if len(rest) <= KEEP_RECENT:
            return body, None
        keep = rest[-KEEP_RECENT:]
        span = rest[:-KEEP_RECENT]
        # 保留段估算超限时，从 keep 头部继续往 span 挪（至少保留 2 条）
        while est_tokens(keep) > KEEP_EST_MAX and len(keep) > 2:
            span.append(keep.pop(0))

        key = hashlib.sha256(json.dumps(
            [m.get("content") for m in lead_sys], ensure_ascii=False)[:200].encode()
        ).hexdigest()[:12] + "|" + hashlib.sha256(json.dumps(
            rest[0].get("content"), ensure_ascii=False, default=str)[:200].encode()
        ).hexdigest()[:12]

        st = self.sessions.get(key)
        sig = self._sig(span)
        mode = "compress"
        if st and st.get("covered_len", 0) <= len(span):
            old = self._sig(span[:st["covered_len"]])
            if old == st.get("span_sig"):
                mode = "incremental" if st["covered_len"] < len(span) else "reuse"

        # 2026-10-18：滑出 keep 窗口的新消息攒够（≥ REBUILD_MIN）才重建摘要。
        # 原逻辑每轮都把新滑出的 2~3 条消息重摘要一遍 → 摘要文本变化 → 主请求的
        # 前缀失效 → 每轮全量重算（实测 ~30K 提示 ≈ 2 分钟/轮）。攒够之前沿用旧
        # 摘要、把滑出消息原样追加在后面：前缀稳定 → KV 缓存命中，只算增量。
        # 注意：keep 窗口本身也在「未吸收」里，所以只按 span[covered:]（滑出部分）
        # 计数，否则阈值会被 keep 的大小顶穿、退化成每轮重建。
        if mode == "incremental":
            slid = span[st["covered_len"]:]
            slid_est = est_tokens(slid)
            if slid_est < REBUILD_MIN:
                body["messages"] = lead_sys + [{
                    "role": "user",
                    "content": "[自动压缩] 本次对话较早的历史已被压缩为以下摘要（由本地代理自动生成，内容仍为有效上下文）：\n\n"
                               + st["summary"],
                }] + slid + keep
                log(f"摘要暂缓 滑出 {slid_est} est（< {REBUILD_MIN}）→ 原样追加 est={est}")
                return body, {"mode": "defer", "est": est}

        if mode == "incremental":
            prev = st["summary"]
            new_msgs = span[st["covered_len"]:]
        else:
            prev = None
            new_msgs = span

        if mode != "reuse":
            t0 = time.time()
            summary = self._summarize(prev, new_msgs)
            took = time.time() - t0
            if summary is None:
                log(f"WARN 摘要生成失败，本次原样透传（est={est}）")
                return body, None
            self.sessions[key] = {"summary": summary,
                                  "covered_len": len(span), "span_sig": sig}
            self._save()
            log(f"摘要完成 mode={mode} span={len(span)}条 new={len(new_msgs)}条 "
                f"est={est}→~{est_tokens(lead_sys)+est_tokens(keep)+_est_str(summary)} "
                f"摘要 {len(summary)} 字 耗时 {took:.0f}s")
        else:
            summary = st["summary"]
            log(f"摘要复用 span 未变 est={est}→~{est_tokens(lead_sys)+est_tokens(keep)+_est_str(summary)}")

        body["messages"] = lead_sys + [{
            "role": "user",
            "content": "[自动压缩] 本次对话较早的历史已被压缩为以下摘要（由本地代理自动生成，内容仍为有效上下文）：\n\n"
                       + summary,
        }] + keep
        return body, {"mode": mode, "est": est}

    # ---------------------------------------------------------- 摘要生成
    @staticmethod
    def _serialize(msgs):
        out = []
        for m in msgs:
            role = m.get("role", "?")
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif isinstance(p, dict):
                        parts.append(f"[{p.get('type')} 媒体，省略]")
                text = "\n".join(parts)
            elif isinstance(c, str):
                text = c
            else:
                text = json.dumps(c, ensure_ascii=False, default=str)
            if m.get("tool_calls"):
                text += "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)[:1500]
            text = text.strip()
            if len(text) > 3000:
                text = text[:3000] + "…(截断)"
            out.append(f"### {role}\n{text or '(空)'}")
        return "\n\n".join(out)

    def _summarize(self, prev, msgs):
        sys_prompt = (
            "你在维护一个 AI 助手会话的长期记忆。给你（1）已有摘要（可能为空）和（2）新归档的较早消息。"
            "输出合并后的更新摘要：保留关键事实、决定、文件路径、命令、报错与修复办法、未完成事项、"
            "数字与名称；沿用内容原语言（混合就混合）；用简洁要点；不超过 700 字；不要编造。"
            "只输出摘要本身。")
        user = (f"[已有摘要]\n{prev or '（空，这是首次压缩）'}\n\n"
                f"[新归档消息（{len(msgs)} 条）]\n{self._serialize(msgs)}\n\n"
                f"请输出合并后的更新摘要。")
        payload = json.dumps({
            "model": "kvmem-proxy-summary",
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": user}],
            "max_tokens": SUMMARY_MAX_TOKENS, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }, ensure_ascii=False).encode()
        try:
            conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=UP_TIMEOUT)
            conn.request("POST", "/v1/chat/completions", body=payload,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read())
            conn.close()
            return (data["choices"][0]["message"].get("content") or "").strip() or None
        except Exception as e:
            log(f"WARN 摘要请求异常: {type(e).__name__}: {e}")
            return None


COMP = Compressor()


# ---------------------------------------------------------------- HTTP 层
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # 静默默认访问日志（有自己的 log）
        pass

    def _relay(self, method):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body_bytes = self.rfile.read(length) if length else b""

        note = None
        stream = False
        if body_bytes:
            try:
                parsed = json.loads(body_bytes)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("messages"), list):
                        with COMP.lock:
                            parsed, note = COMP.maybe_compress(parsed)
                        body_bytes = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                    stream = bool(parsed.get("stream"))
            except (ValueError, UnicodeDecodeError):
                pass

        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        headers["Accept-Encoding"] = "identity"
        headers["Connection"] = "close"
        try:
            conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=UP_TIMEOUT)
            conn.request(method, self.path, body=body_bytes if body_bytes else None,
                         headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            log(f"WARN 上游不可达: {e}")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            msg = json.dumps({"error": {"message": f"upstream {UP_PORT} unreachable: {e}"}}).encode()
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            self.close_connection = True
            return

        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length", "server", "date"):
                continue
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.close_connection = True

        if stream and resp.status == 200:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    # 必须用 read1：read(n) 会攒满 n 字节才返回，SSE 会变成整块倒出
                    chunk = resp.read1(8192)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, OSError):
                pass
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, OSError):
                pass
        if note:
            log(f"{method} {self.path.split('?')[0]} → {resp.status} | {note['mode']} "
                f"est {note['est']} tok")
        conn.close()

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def do_DELETE(self):
        self._relay("DELETE")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))
    log(f"kvmem-proxy 启动 :{PORT} → upstream :{UP_PORT} | 阈值 {THRESHOLD} tok | "
        f"保留最近 {KEEP_RECENT} 条(≤{KEEP_EST_MAX} tok) | 攒 {REBUILD_MIN} est 才重建摘要")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
