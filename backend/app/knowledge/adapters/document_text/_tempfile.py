from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def bytes_as_temp_file(raw: bytes, suffix: str) -> Iterator[Path]:
    """将字节写入临时文件，供仅支持路径的 LlamaIndex Reader 使用。"""
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    path = Path(path_str)
    try:
        os.write(fd, raw)
        os.close(fd)
        fd = -1
        yield path
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(path_str)
        except OSError:
            pass
