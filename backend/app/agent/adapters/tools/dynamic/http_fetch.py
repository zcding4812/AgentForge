"""HTTP 请求工具：可配置、支持路径/查询/正文占位符；返回体对齐 ``kernel.tool_runtime`` 外部工具信封。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, ClassVar
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from app.agent.kernel.tool_runtime import (
    JSONRPC_INTERNAL_ERROR,
    build_external_tool_error_json,
    build_external_tool_json,
)

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"})
_PATH_PARAM_PATTERN = re.compile(r"\{(\w+)\}")
_DEFAULT_MAX_BODY_CHARS = 8000


class HttpToolInvokeArgs(BaseModel):
    """Agent 调用时传入的动态参数（均可省略）。"""

    path_params: dict[str, str] | None = Field(
        default=None,
        description="替换 URL 模板中的 {name}，如 {'id': '123'}",
    )
    query_params: dict[str, Any] | None = Field(
        default=None,
        description="追加到 URL 的查询参数",
    )
    body_fields: dict[str, Any] | None = Field(
        default=None,
        description="填充请求体模板中的 {field} 占位符",
    )


class HttpToolConfig(BaseModel):
    """与持久化/工厂对齐的 HTTP 工具配置（强类型）。"""

    logical_name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=512)
    url: str = Field(..., description="请求 URL，可含 {path_param} 占位符")
    method: str = Field(default="GET", pattern=r"^(GET|HEAD|POST|PUT|PATCH|DELETE)$")
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: str | None = Field(default=None, description="非 GET/HEAD 时的请求体模板")
    timeout_ms: int = Field(default=15000, ge=1000, le=300_000)
    verify_ssl: bool = Field(default=True)
    max_body_chars: int = Field(default=_DEFAULT_MAX_BODY_CHARS, ge=1000, le=100_000)
    follow_redirects: bool = Field(default=True)

    @field_validator("url", mode="after")
    @classmethod
    def _url_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("url 不能为空")
        return s

    @field_validator("method", mode="before")
    @classmethod
    def _method_upper(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().upper()
        return v


class HttpRequestTool(BaseTool):
    """可配置 HTTP 工具：异步执行、占位符、SSL/重定向/截断可配。"""

    args_schema: ClassVar[type[BaseModel]] = HttpToolInvokeArgs
    _cfg: HttpToolConfig = PrivateAttr()

    def __init__(self, cfg: HttpToolConfig) -> None:
        super().__init__(name=cfg.logical_name, description=cfg.description)
        self._cfg = cfg

    def _run(
        self,
        path_params: dict[str, str] | None = None,
        query_params: dict[str, Any] | None = None,
        body_fields: dict[str, Any] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """同步入口：桥接到 ``_arun``（Agent 异步路径走 ``ainvoke`` / ``_arun``）。"""
        return asyncio.run(
            self._arun(
                path_params=path_params,
                query_params=query_params,
                body_fields=body_fields,
                run_manager=None,
            )
        )

    async def _arun(
        self,
        path_params: dict[str, str] | None = None,
        query_params: dict[str, Any] | None = None,
        body_fields: dict[str, Any] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        cfg = self._cfg
        t0 = time.perf_counter()
        pp = path_params or {}
        qp = query_params or {}
        bf = body_fields or {}
        try:
            resolved_url = _build_resolved_url(cfg.url, pp, qp)
            content = _build_request_body(cfg.body_template, bf, cfg.method)
            timeout = cfg.timeout_ms / 1000.0
            async with httpx.AsyncClient(
                timeout=timeout,
                verify=cfg.verify_ssl,
                follow_redirects=cfg.follow_redirects,
            ) as client:
                resp = await client.request(
                    cfg.method,
                    resolved_url,
                    headers=cfg.headers or None,
                    content=content,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            body_text = _truncate_body(resp.text, cfg.max_body_chars)
            final_u = str(resp.url)
            result: dict[str, Any] = {
                "tool_name": cfg.logical_name,
                "status_code": resp.status_code,
                "url_template": cfg.url,
                "request_url": resolved_url,
                "url": final_u,
                "final_url": final_u,
                "redirected": final_u != resolved_url,
                "body": body_text,
                "response_time_ms": round(elapsed_ms, 2),
            }
            return build_external_tool_json(kind="http", result=result)
        except httpx.RequestError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning(
                "http_tool.request_error",
                extra={"tool": cfg.logical_name, "error": str(e)},
            )
            return build_external_tool_error_json(
                kind="http",
                code=JSONRPC_INTERNAL_ERROR,
                message=f"HTTP 传输失败: {e!s}",
                data={
                    "tool_name": cfg.logical_name,
                    "url_template": cfg.url,
                    "request_url": cfg.url,
                    "method": cfg.method,
                    "response_time_ms": round(elapsed_ms, 2),
                },
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception("http_tool.execution_failed", extra={"tool": cfg.logical_name})
            return build_external_tool_error_json(
                kind="http",
                code=JSONRPC_INTERNAL_ERROR,
                message=f"HTTP 工具执行失败: {e!s}",
                data={
                    "tool_name": cfg.logical_name,
                    "url_template": cfg.url,
                    "method": cfg.method,
                    "response_time_ms": round(elapsed_ms, 2),
                },
            )


def _build_resolved_url(
    url_template: str,
    path_params: dict[str, str],
    query_params: dict[str, Any],
) -> str:
    url = url_template
    for k, v in path_params.items():
        url = url.replace(f"{{{k}}}", str(v))
    missing = _PATH_PARAM_PATTERN.findall(url)
    if missing:
        raise ValueError(f"URL 缺少路径参数: {missing}")
    if query_params:
        return _merge_query_into_url(url, query_params)
    return url


def _merge_query_into_url(url: str, query_params: dict[str, Any]) -> str:
    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for k, v in query_params.items():
        key = str(k)
        if isinstance(v, (dict, list)):
            q[key] = json.dumps(v, ensure_ascii=False)
        else:
            q[key] = str(v)
    new_query = urlencode(q, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _build_request_body(
    template: str | None,
    body_fields: dict[str, Any],
    method: str,
) -> bytes | None:
    if method in ("GET", "HEAD") or not template:
        return None
    body = template
    for field_name, field_value in body_fields.items():
        placeholder = f"{{{field_name}}}"
        if placeholder in body:
            if isinstance(field_value, (dict, list)):
                rep = json.dumps(field_value, ensure_ascii=False)
            else:
                rep = str(field_value)
            body = body.replace(placeholder, rep)
    return body.encode("utf-8")


def _truncate_body(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n…(truncated)"


def _normalize_headers(headers: dict[str, Any] | None) -> dict[str, str]:
    if not headers:
        return {}
    out: dict[str, str] = {}
    for k, v in headers.items():
        out[str(k)] = v if isinstance(v, str) else str(v)
    return out


def make_http_request_tool(
    *,
    logical_name: str,
    description: str,
    url: str,
    method: str = "GET",
    headers: dict[str, Any] | None = None,
    body: str | None = None,
    timeout_s: float = 15.0,
    verify_ssl: bool = True,
    follow_redirects: bool = True,
    max_body_chars: int | None = None,
) -> BaseTool:
    """由持久化/运行时构造 HTTP 工具（无 URL 占位符时，模型可不传 path/query/body 参数）。"""
    m = method.strip().upper()
    if m not in _ALLOWED_METHODS:
        raise ValueError(f"不支持的 HTTP method: {method!r}，允许: {sorted(_ALLOWED_METHODS)}")
    hdrs = _normalize_headers(headers)
    cfg = HttpToolConfig(
        logical_name=logical_name,
        description=description,
        url=url,
        method=m,
        headers=hdrs,
        body_template=body,
        timeout_ms=max(1000, min(300_000, int(round(float(timeout_s) * 1000)))),
        verify_ssl=verify_ssl,
        follow_redirects=follow_redirects,
        max_body_chars=max_body_chars if max_body_chars is not None else _DEFAULT_MAX_BODY_CHARS,
    )
    return HttpRequestTool(cfg=cfg)


def make_http_get_tool(
    *,
    logical_name: str,
    description: str,
    url: str,
    timeout_s: float = 15.0,
) -> BaseTool:
    """兼容：仅 GET、无 headers/body。"""
    return make_http_request_tool(
        logical_name=logical_name,
        description=description,
        url=url,
        method="GET",
        headers=None,
        body=None,
        timeout_s=timeout_s,
    )
