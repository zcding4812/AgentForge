"""知识库上传体积等不变量（不含 MinIO 是否配置）。"""

from __future__ import annotations


def validate_upload_byte_size(size: int, *, max_bytes: int) -> None:
    if size > max_bytes:
        raise ValueError(f"文件超过大小上限（{max_bytes} 字节）")
