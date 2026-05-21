"""MinIO（S3 兼容）对象存储：基于 aioboto3 异步 S3 客户端。"""

from __future__ import annotations

import io
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, BinaryIO, cast

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["MinioConfig", "MinioObjectStore"]

DEFAULT_CONTENT_TYPE = "application/octet-stream"
MAX_RETRY_ATTEMPTS = 3

# MinIO / 本地 S3 兼容：路径风格 + v4 签名
_BOTO_S3_CONFIG = Config(
    signature_version="s3v4",
    s3={"addressing_style": "path"},
)


@dataclass(frozen=True)
class MinioConfig:
    """MinIO 连接与桶（强类型）；由 :meth:`from_settings` 从环境装配。"""

    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool
    configured: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> MinioConfig:
        ep = settings.minio_endpoint.strip()
        configured = bool(ep and settings.minio_access_key and settings.minio_secret_key)
        return cls(
            endpoint=ep,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket_knowledge.strip() or "knowledge",
            secure=False,
            configured=configured,
        )


class MinioObjectStore:
    """S3 兼容对象存储（异步）；桶创建重试、异常与日志收口。"""

    def __init__(self, config: MinioConfig) -> None:
        self._config = config
        self._session = aioboto3.Session()

    @property
    def is_configured(self) -> bool:
        return self._config.configured

    @property
    def bucket_name(self) -> str:
        return self._config.bucket

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise RuntimeError(
                "MinIO 未启用：请配置 MINIO_ENDPOINT、MINIO_ACCESS_KEY、MINIO_SECRET_KEY",
            )

    def _endpoint_url(self) -> str:
        ep = self._config.endpoint.strip()
        if ep.startswith(("http://", "https://")):
            return ep
        scheme = "https" if self._config.secure else "http"
        return f"{scheme}://{ep}"

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "endpoint_url": self._endpoint_url(),
            "aws_access_key_id": self._config.access_key,
            "aws_secret_access_key": self._config.secret_key,
            "region_name": "us-east-1",
            "config": _BOTO_S3_CONFIG,
        }

    @asynccontextmanager
    async def _s3(self) -> AsyncIterator[Any]:
        self._require_configured()
        cm = self._session.client("s3", **self._client_kwargs())
        async with cast(AbstractAsyncContextManager[Any], cm) as s3:
            yield s3

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(
            (EndpointConnectionError, OSError, ConnectionError, TimeoutError),
        ),
        before_sleep=lambda retry_state: logger.warning(
            "MinIO 连接异常，正在重试 %s/%s",
            retry_state.attempt_number,
            MAX_RETRY_ATTEMPTS,
        ),
    )
    async def ensure_bucket_exists(self) -> None:
        """确保桶存在（幂等）；网络抖动时自动重试。"""
        async with self._s3() as s3:
            try:
                await s3.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code in ("404", "NoSuchBucket") or http_status == 404:
                    await s3.create_bucket(Bucket=self.bucket_name)
                    logger.info("MinIO 桶已创建: %s", self.bucket_name)
                else:
                    raise

    async def put_object(
        self,
        object_key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
    ) -> None:
        """上传对象（bytes 或可读 BinaryIO）；自动建桶。"""
        try:
            await self.ensure_bucket_exists()
            async with self._s3() as s3:
                if isinstance(data, bytes):
                    body: bytes | BinaryIO = data
                else:
                    data.seek(0, io.SEEK_END)
                    data.seek(0)
                    body = data

                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=body,
                    ContentType=content_type or DEFAULT_CONTENT_TYPE,
                )
            logger.debug("对象上传成功: %s", object_key)
        except ClientError as e:
            logger.error("上传对象失败 | key=%s | error=%s", object_key, e)
            raise RuntimeError(f"MinIO 上传失败: {e}") from e

    async def get_object(self, object_key: str) -> bytes:
        """读取对象内容为 bytes。"""
        try:
            async with self._s3() as s3:
                resp = await s3.get_object(Bucket=self.bucket_name, Key=object_key)
                body = resp["Body"]
                data = await body.read()
            return data
        except ClientError as e:
            logger.error("获取对象失败 | key=%s | error=%s", object_key, e)
            raise RuntimeError(f"MinIO 获取对象失败: {e}") from e

    async def object_exists(self, object_key: str) -> bool:
        try:
            async with self._s3() as s3:
                await s3.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NotFound", "NoSuchKey") or http_status == 404:
                return False
            raise

    async def remove_object(self, object_key: str) -> None:
        """删除对象；未配置或 S3 错误仅记日志（适合回滚）。"""
        if not self.is_configured:
            return
        try:
            async with self._s3() as s3:
                await s3.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.debug("对象已删除: %s", object_key)
        except ClientError as e:
            logger.warning("删除对象失败（可忽略）| key=%s | error=%s", object_key, e)

    async def presigned_get_object(self, object_key: str, *, expires_seconds: int = 3600) -> str:
        """生成 GET 预签名 URL（``expires_seconds`` 默认 1 小时）。"""
        async with self._s3() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=expires_seconds,
            )
        return url

    async def head_bucket_ok(self) -> bool:
        """探活：对已配置桶执行 ``head_bucket``；未配置或失败返回 ``False``。"""
        if not self.is_configured:
            return False
        try:
            async with self._s3() as s3:
                await s3.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception:
            logger.debug("MinIO head_bucket 探活失败", exc_info=True)
            return False
