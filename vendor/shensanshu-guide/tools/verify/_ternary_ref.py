"""Minimal reference GGUF reader shared by the verify scripts.

The author's published snapshot does not include `_ternary_ref.py`; this reconstruction
wraps the packer's own GGUF parser (`pack.py`, byte-for-byte identical header handling)
behind the `tensors[name] -> (ne, type, offset)` / `raw(name, rows)` read interface the
scripts expect.  It only READS source rows; all judgement logic lives in the scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLSDIR = str(Path(__file__).resolve().parent.parent)  # .../tools (pack.py lives here)
if _TOOLSDIR not in sys.path:
    sys.path.insert(0, _TOOLSDIR)

from pack import Gguf as _PackGguf  # noqa: E402


class Gguf(_PackGguf):
    """pack.py's parser exposing the verify scripts' read API."""

    @property
    def tensors(self):
        return self.tensor

    def raw(self, name: str, rows: int) -> bytes:
        ne, tt, off = self.tensor[name]
        n = self.tbytes[name]
        self.f.seek(self.data_start + off)
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read of tensor {name}")
        return b
