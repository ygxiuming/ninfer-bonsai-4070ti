"""dump_gguf_meta.py — 校验三元 GGUF 元数据并导出 prism.* JSON。

功能：
  1. 按指南复现清单校验 GGUF 元数据：
       block_size=1024、Σsign_widths=28672、weight_names=401、
       inverse=[token_embd.weight]
  2. 把 prism.hadamard.* KV 导出为 JSON，供指南 verify 的
     check_signs.py <artifact> <meta.json> 使用。

用法:
    python3 dump_gguf_meta.py <Ternary-Bonsai-2-27B-PQ2_0.gguf> <out-meta.json>

自包含：同目录 gguf_meta.py 提供解析器；仅标准库，无 numpy。
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_meta import Gguf  # noqa: E402


def _jsonable(v):
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def main() -> int:
    gguf_path, out_path = sys.argv[1], sys.argv[2]
    g = Gguf(gguf_path)
    kv = g.kv

    print("== prism.* 元数据 ==")
    for k in sorted(kv):
        if k.startswith("prism.") or k in ("general.architecture", "general.alignment"):
            v = kv[k]
            if isinstance(v, (list, tuple)) and len(v) > 8:
                head = [_jsonable(x) for x in list(v)[:4]]
                print(f"  {k}: <list len={len(v)}> head={head}")
            else:
                print(f"  {k}: {_jsonable(v)}")

    widths = kv.get("prism.hadamard.sign_widths")
    sign_vals = kv.get("prism.hadamard.sign_values")
    block_size = kv.get("prism.hadamard.block_size")
    weight_names = kv.get("prism.hadamard.weight_names")
    inverse = kv.get("prism.hadamard.inverse_weight_names")

    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        mark = "PASS" if good else "FAIL"
        print(f"  [{mark}] {name}: got={got} want={want}")

    print("== 复现清单（指南 §6 步骤 2）==")
    check("block_size", int(block_size) if block_size is not None else None, 1024)
    check("sum(sign_widths)", int(sum(widths)) if widths else None, 28672)
    check("n_weight_names", len(weight_names) if weight_names else None, 401)
    check("inverse_weight_names",
          list(inverse) if inverse else None, ["token_embd.weight"])

    # sign_values 归一化为 float 数值列表（check_signs.py 的输入格式）。
    # GGUF 里该值可能是 fp32 字节块，也可能是已解析的数值数组。
    if isinstance(sign_vals, bytes):
        n = len(sign_vals) // 4
        vals = list(struct.unpack(f"<{n}f", sign_vals[: n * 4]))
    elif isinstance(sign_vals, (list, tuple)):
        vals = [float(x) for x in sign_vals]
    else:
        raise SystemExit(f"unexpected sign_values type: {type(sign_vals)}")
    uniq = sorted(set(vals))[:5]
    print(f"sign_values: n={len(vals)} unique={uniq}")

    meta = {k: v for k, v in kv.items() if k.startswith("prism.")}
    meta["prism.hadamard.sign_values"] = vals
    meta["prism.hadamard.weight_names"] = list(weight_names or [])
    meta["prism.hadamard.inverse_weight_names"] = list(inverse or [])
    Path(out_path).write_text(json.dumps(meta), encoding="utf-8")
    size = Path(out_path).stat().st_size
    print(f"wrote {out_path} ({size:,} B)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
