"""Dump the prism.* GGUF metadata to JSON (for verify/check_signs.py) and validate
the guide's repro checklist: block_size=1024, sum(sign_widths)=28672,
n_weight_names=401, inverse=[token_embd].

Usage: python dump_gguf_meta.py <Ternary-Bonsai-2-27B-PQ2_0.gguf> <out-meta.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack import Gguf  # noqa: E402


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

    print("== prism.* metadata keys ==")
    for k in sorted(kv):
        if k.startswith("prism.") or k in ("general.architecture", "general.alignment"):
            v = kv[k]
            if isinstance(v, (list, tuple)) and len(v) > 8:
                print(f"  {k}: <list len={len(v)}> head={[_jsonable(x) for x in v[:4]]}")
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
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got={got} want={want}")

    print("== checklist (guide §6 step 2) ==")
    check("block_size", int(block_size), 1024)
    check("sum(sign_widths)", int(sum(widths)) if widths else None, 28672)
    check("n_weight_names", len(weight_names) if weight_names else None, 401)
    check("inverse_weight_names", list(inverse) if inverse else None, ["token_embd.weight"])

    # sign_values: normalize to a plain float32 list for check_signs.py
    if isinstance(sign_vals, bytes):
        vals = np.frombuffer(sign_vals, dtype=np.float32).tolist()
    elif isinstance(sign_vals, (list, tuple)):
        vals = [float(x) for x in sign_vals]
    else:
        raise SystemExit(f"unexpected sign_values type: {type(sign_vals)}")
    print(f"sign_values: n={len(vals)} unique={sorted(set(vals))[:5]}")

    meta = {}
    for k, v in kv.items():
        if k.startswith("prism."):
            meta[k] = v
    meta["prism.hadamard.sign_values"] = vals
    meta["prism.hadamard.weight_names"] = list(weight_names or [])
    meta["prism.hadamard.inverse_weight_names"] = list(inverse or [])
    Path(out_path).write_text(json.dumps(meta), encoding="utf-8")
    print(f"wrote {out_path} ({Path(out_path).stat().st_size:,} B)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
