"""_ternary_ref.py — 指南 verify 脚本的共享 GGUF 读取层（自包含版）。

作者发布的 verify 快照引用了未随包分发的 `_ternary_ref` 模块；本文件把
自包含的 GGUF 解析器（gguf_meta.py，不依赖 pack.py / 引擎源码树）以 verify
脚本约定的接口再导出：

    from _ternary_ref import Gguf
    g = Gguf(path)
    g.tensors[name]      -> (ne 列表, gguf 类型, 数据段内偏移)
    g.raw(name, rows)    -> 张量原始字节

只读数据源；全部判定逻辑在 verify 脚本本身（check_row_order / check_assembly）。

用法：把本文件与 gguf_meta.py 放进指南仓库的 tools/verify/ 目录
（与 check_row_order.py / check_assembly.py 同目录）。
"""
from __future__ import annotations

from gguf_meta import Gguf  # noqa: F401  再导出，接口即 pack.py 解析器的兼容子集
