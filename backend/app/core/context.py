import contextvars
import re
import secrets
from contextvars import Token

from fastapi import Request

# 常量见 app/core/constants/
from app.core.constants import (
    DEFAULT_REQUEST_ID,
    TRACEPARENT_FLAGS,
    TRACEPARENT_HEADER,
    TRACEPARENT_VERSION,  # 新增：补充缺失的标准化常量
    X_REQUEST_ID_HEADER,
)


class RequestTraceContext:
    """请求链路上下文，统一管理 request_id/trace_id/span_id（遵循W3C Trace Context规范）。"""

    # 正则与长度常量：提升可维护性
    _HEX_RE = re.compile(r"^[0-9a-f]+$")
    _TRACE_ID_LENGTH = 32  # 16字节 → 32位十六进制（W3C规范）
    _SPAN_ID_LENGTH = 16  # 8字节 → 16位十六进制（W3C规范）

    @classmethod
    def get_request_id(cls, request: Request) -> str:
        """安全获取request_id，兜底默认值（避免AttributeError）"""
        try:
            return getattr(request.state, "request_id", DEFAULT_REQUEST_ID) or DEFAULT_REQUEST_ID
        except AttributeError:
            return DEFAULT_REQUEST_ID

    @classmethod
    def get_trace_id(cls, request: Request) -> str:
        """安全获取trace_id，兜底默认值"""
        try:
            return getattr(request.state, "trace_id", DEFAULT_REQUEST_ID) or DEFAULT_REQUEST_ID
        except AttributeError:
            return DEFAULT_REQUEST_ID

    @classmethod
    def get_span_id(cls, request: Request) -> str:
        """安全获取span_id，兜底默认值"""
        try:
            return getattr(request.state, "span_id", DEFAULT_REQUEST_ID) or DEFAULT_REQUEST_ID
        except AttributeError:
            return DEFAULT_REQUEST_ID

    @classmethod
    def setup_from_headers(cls, request: Request) -> str:
        """从请求头初始化链路上下文，增加state容错"""
        # 容错：确保request.state存在
        if not hasattr(request, "state"):
            request.state = type("RequestState", (), {})()

        incoming_request_id = request.headers.get(X_REQUEST_ID_HEADER)
        incoming_traceparent = request.headers.get(TRACEPARENT_HEADER)

        # 标准化trace_id：优先X-Request-ID，其次traceparent，最后生成新的
        trace_id = cls._normalize_trace_id(
            incoming_request_id
        ) or cls._extract_trace_id_from_traceparent(incoming_traceparent)
        trace_id = trace_id or cls.generate_trace_id()

        span_id = cls.generate_span_id()
        request.state.trace_id = trace_id
        request.state.span_id = span_id
        # 当前阶段 request_id 复用 trace_id，便于日志与链路检索统一
        request.state.request_id = trace_id
        request.state.traceparent = cls.build_traceparent(trace_id=trace_id, span_id=span_id)

        return request.state.request_id

    @classmethod
    def build_traceparent(cls, trace_id: str, span_id: str) -> str:
        """构建符合W3C规范的traceparent，增加参数校验"""
        if not cls._normalize_trace_id(trace_id):
            raise ValueError(f"Invalid trace_id (must be 32 hex chars): {trace_id}")
        if len(span_id) != cls._SPAN_ID_LENGTH or not cls._HEX_RE.fullmatch(span_id):
            raise ValueError(f"Invalid span_id (must be 16 hex chars): {span_id}")
        return f"{TRACEPARENT_VERSION}-{trace_id}-{span_id}-{TRACEPARENT_FLAGS}"

    @classmethod
    def generate_trace_id(cls) -> str:
        """生成符合W3C规范的trace_id（32位十六进制）"""
        return secrets.token_hex(16)

    @classmethod
    def generate_span_id(cls) -> str:
        """生成符合W3C规范的span_id（16位十六进制）"""
        return secrets.token_hex(8)

    @classmethod
    def get_log_context(cls, request: Request) -> dict[str, str]:
        """新增：获取日志上下文（便于日志集成）"""
        return {
            "request_id": cls.get_request_id(request),
            "trace_id": cls.get_trace_id(request),
            "span_id": cls.get_span_id(request),
        }

    @classmethod
    def _normalize_trace_id(cls, value: str | None) -> str | None:
        """标准化trace_id：去横线、小写、校验长度和格式"""
        if value is None:
            return None
        candidate = value.strip().lower().replace("-", "")
        if len(candidate) != cls._TRACE_ID_LENGTH or not cls._HEX_RE.fullmatch(candidate):
            return None
        return candidate

    @classmethod
    def is_w3c_trace_id(cls, value: str | None) -> bool:
        """是否为 W3C trace_id（32 位小写十六进制，与 ``trace_runs.trace_id`` 列一致）。"""
        if value is None:
            return False
        v = value.strip().lower()
        return len(v) == cls._TRACE_ID_LENGTH and bool(cls._HEX_RE.fullmatch(v))

    @classmethod
    def is_w3c_span_id(cls, value: str | None) -> bool:
        """是否为 W3C span_id（16 位小写十六进制，与 ``trace_spans.span_id`` 列一致）。"""
        if value is None:
            return False
        v = value.strip().lower()
        return len(v) == cls._SPAN_ID_LENGTH and bool(cls._HEX_RE.fullmatch(v))

    @classmethod
    def _extract_trace_id_from_traceparent(cls, traceparent: str | None) -> str | None:
        """严格解析traceparent中的trace_id（符合W3C规范）"""
        if not traceparent:
            return None
        parts = traceparent.strip().split("-")
        # 严格校验格式：版本(00)-trace_id-span_id-flags
        if len(parts) != 4 or parts[0] != TRACEPARENT_VERSION:
            return None
        return cls._normalize_trace_id(parts[1])


# ========== 上下文变量定义（独立作用域，避免命名冲突） ==========
_current_request: contextvars.ContextVar[Request | None] = contextvars.ContextVar(
    "current_request", default=None
)
_db_span_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "db_span_enabled", default=True
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span_id", default=None
)


class RequestContext:
    """请求级上下文，便于非路由层访问当前request（增加上下文管理器）。"""

    @staticmethod
    def set_request(request: Request) -> Token:
        return _current_request.set(request)

    @staticmethod
    def reset_request(token: Token) -> None:
        _current_request.reset(token)

    @staticmethod
    def get_request() -> Request | None:
        return _current_request.get()

    @classmethod
    def context(cls, request: Request) -> "RequestContextManager":
        """新增：上下文管理器（with语法，自动set/reset）"""
        return RequestContextManager(request)


class RequestContextManager:
    """请求上下文管理器（简化with语法使用）"""

    def __init__(self, request: Request):
        self.request = request
        self.token: Token | None = None

    def __enter__(self) -> Request:
        self.token = RequestContext.set_request(self.request)
        return self.request

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            RequestContext.reset_request(self.token)


class TraceRuntimeContext:
    """链路运行时上下文（增加上下文管理器，简化使用）"""

    @staticmethod
    def disable_db_span() -> Token:
        return _db_span_enabled.set(False)

    @staticmethod
    def reset_db_span(token: Token) -> None:
        _db_span_enabled.reset(token)

    @staticmethod
    def is_db_span_enabled() -> bool:
        return _db_span_enabled.get()

    @staticmethod
    def set_current_span_id(span_id: str) -> Token:
        return _current_span_id.set(span_id)

    @staticmethod
    def reset_current_span_id(token: Token) -> None:
        _current_span_id.reset(token)

    @staticmethod
    def get_current_span_id() -> str | None:
        """获取当前span_id（链路嵌套时使用）"""
        return _current_span_id.get()

    @classmethod
    def disable_db_span_context(cls) -> "DbSpanDisabledContext":
        """新增：上下文管理器（自动禁用/恢复DB span）"""
        return DbSpanDisabledContext()


class DbSpanDisabledContext:
    """DB Span禁用上下文管理器（简化with语法）"""

    def __init__(self):
        self.token: Token | None = None

    def __enter__(self):
        self.token = TraceRuntimeContext.disable_db_span()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            TraceRuntimeContext.reset_db_span(self.token)
