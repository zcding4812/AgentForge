"""模型提供商：与 HTTP/ORM 无关的展示与校验规则。"""

from __future__ import annotations


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    s = str(value)
    if len(s) <= 4:
        return "****"
    return "****" + s[-4:]


def provider_has_credentials(
    *,
    auth_type: str | None,
    api_key: str | None,
    api_secret: str | None,
) -> bool:
    """启用前需有可用的鉴权信息：密钥/签名，或明确为无需密钥的接入。"""
    auth = (auth_type or "").lower()
    if auth == "none":
        return True
    if api_key and str(api_key).strip():
        return True
    if api_secret and str(api_secret).strip():
        return True
    return False


def protocol_from_api_format(api_format: str) -> str:
    if api_format in ("openai", "anthropic", "baidu", "tencent", "doubao"):
        return "openai_api"
    return "openai_api"


def api_format_from_create(*, protocol: str | None, provider_kind: str | None) -> str:
    """创建提供商时写入库表的 ``api_format`` 字段。"""
    if protocol and protocol != "openai_api":
        return provider_kind or "openai"
    return provider_kind or "openai"


def ui_types_from_db(types: set[str]) -> list[str]:
    """映射为前端多选值：chat / embedding / ocr。"""
    out: set[str] = set()
    for t in types:
        if t == "llm":
            out.add("chat")
        elif t == "embedding":
            out.add("embedding")
        else:
            out.add("ocr")
    return sorted(out)


def map_filter_model_type(ui: str | None) -> tuple[str, ...] | None:
    if not ui or ui == "all":
        return None
    m = {
        "chat": ("llm", "tts", "stt"),
        "embedding": ("embedding",),
        "ocr": ("ocr", "image"),
    }
    if ui in m:
        return m[ui]
    return (ui,)
