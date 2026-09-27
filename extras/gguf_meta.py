"""自包含的最小 GGUF 头部解析器（只读）。

不依赖 pack.py / 引擎源码树，供 verify 脚本（_ternary_ref.py）与
dump_gguf_meta.py 复用。只解析头部（KV + 张量表）与数据段偏移，
不加载任何张量数据到内存（raw() 按需读取）。

GGUF v2/v3 头部布局:
  magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
  kv:   string key | u32 type | value
  张量: string name | u32 n_dims | n_dims × u64 ne | u32 type | u64 offset
数据段起始 = align_up(头部末尾, general.alignment 或 32)
"""
from __future__ import annotations

import struct
from pathlib import Path

_T_U8, _T_I8 = 0, 1
_T_U16, _T_I16 = 2, 3
_T_U32, _T_I32, _T_F32 = 4, 5, 6
_T_STR, _T_ARR = 8, 9
_T_U64, _T_I64, _T_F64 = 10, 11, 12


class Gguf:
    def __init__(self, path: str | Path):
        self.f = open(path, "rb")
        self.path = str(path)
        self.f.seek(0, 2)
        self.size = self.f.tell()
        self.f.seek(0)
        if self.f.read(4) != b"GGUF":
            raise SystemExit(f"not a GGUF file: {path}")
        self.version = self._u32()
        self.n_tensors = self._u64()
        self.n_kv = self._u64()
        self.kv: dict[str, object] = {}
        for _ in range(self.n_kv):
            key = self._str()
            self.kv[key] = self._value(self._u32())
        self.tensor: dict[str, tuple[list[int], int, int]] = {}
        order = []
        for _ in range(self.n_tensors):
            name = self._str()
            ne = [self._u64() for _ in range(self._u32())]
            tt = self._u32()
            off = self._u64()
            self.tensor[name] = (ne, tt, off)
            order.append((off, name))
        self.header_end = self.f.tell()
        self.alignment = int(self.kv.get("general.alignment", 32))
        self.data_start = -(-self.header_end // self.alignment) * self.alignment
        order.sort()
        self.tbytes: dict[str, int] = {}
        for i, (off, name) in enumerate(order):
            nxt = order[i + 1][0] if i + 1 < len(order) else (self.size - self.data_start)
            self.tbytes[name] = nxt - off

    # ---- primitives ----
    def _raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read of {n}")
        return b

    def _u32(self) -> int:
        return struct.unpack("<I", self._raw(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._raw(8))[0]

    def _str(self) -> str:
        return self._raw(self._u64()).decode("utf-8", "replace")

    def _value(self, t: int):
        if t in (_T_U8, _T_I8):
            return self._raw(1)[0]
        if t in (_T_U16, _T_I16):
            return struct.unpack("<h", self._raw(2))[0]
        if t == _T_U32:
            return self._u32()
        if t == _T_I32:
            return struct.unpack("<i", self._raw(4))[0]
        if t == _T_F32:
            return struct.unpack("<f", self._raw(4))[0]
        if t == _T_STR:
            return self._str()
        if t == _T_ARR:
            et = self._u32()
            n = self._u64()
            return [self._value(et) for _ in range(n)]
        if t == _T_U64:
            return self._u64()
        if t == _T_I64:
            return struct.unpack("<q", self._raw(8))[0]
        if t == _T_F64:
            return struct.unpack("<d", self._raw(8))[0]
        raise ValueError(f"unsupported GGUF kv type {t}")

    # ---- 公开接口（与指南 verify 脚本约定一致） ----
    @property
    def tensors(self):
        """name -> (ne 列表, GGUF 类型, 数据段内偏移)"""
        return self.tensor

    def raw(self, name: str, rows: int = 0) -> bytes:
        """按需读取一个张量的原始字节（rows 参数仅为接口兼容）。"""
        ne, tt, off = self.tensor[name]
        n = self.tbytes[name]
        self.f.seek(self.data_start + off)
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read of tensor {name}")
        return b
