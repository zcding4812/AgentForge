"""模型提供商与模型列表：编排仓储、映射 DTO；持久化见 `ProviderRepository`。"""

from __future__ import annotations

import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import PROVIDER_API_MAX_PAGE_SIZE
from app.domain.providers import (
    api_format_from_create,
    map_filter_model_type,
    mask_secret,
    protocol_from_api_format,
    provider_has_credentials,
    ui_types_from_db,
)
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.sys_model_mod import SysModel, SysModelProvider
from app.repositories.provider_repo import ProviderRepository
from app.schemas.providers import (
    LlmModelCreate,
    LlmModelOut,
    LlmModelUpdate,
    ModelProviderCreate,
    ModelProviderOut,
    ModelProviderUpdate,
    ProbeResult,
)

logger = logging.getLogger(__name__)


def _to_provider_out(
    p: SysModelProvider,
    *,
    model_count: int,
    supported_types: set[str],
) -> ModelProviderOut:
    return ModelProviderOut(
        id=p.provider_code,
        name=p.provider_name,
        supported_model_types=ui_types_from_db(supported_types),
        description=None,
        enabled=p.status == 1,
        model_count=model_count,
        provider_kind=p.api_format,
        protocol=protocol_from_api_format(p.api_format),
        base_url=p.base_url or "",
        api_key_masked=mask_secret(p.api_key),
        org_id=None,
        weight=50,
        timeout_sec=30,
        max_retries=2,
        is_default=False,
    )


def _to_model_out(m: SysModel, provider: SysModelProvider) -> LlmModelOut:
    ep = (m.endpoint or "").strip() or None
    pb = (provider.base_url or "").strip() or None
    to = m.timeout if m.timeout is not None else 30
    return LlmModelOut(
        id=str(m.id),
        model_code=m.model_code,
        model_name=m.model_name,
        provider_id=provider.provider_code,
        provider_name=provider.provider_name,
        model_type=m.model_type,
        endpoint=ep,
        provider_base_url=pb,
        timeout=to,
        api_key_masked=mask_secret(provider.api_key),
        enabled=m.is_enabled == 1,
        health_status="unknown",
        health_message=None,
    )


class ProviderService:
    """模型提供商与模型列表；构造注入 ``db_manager``。"""

    def __init__(self, db_manager: SQLAlchemyDatabaseManager) -> None:
        self._db = db_manager

    async def list_providers(
        self,
        *,
        q: str | None,
        status: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[ModelProviderOut], int]:
        page = max(1, page)
        page_size = max(1, min(page_size, PROVIDER_API_MAX_PAGE_SIZE))

        rows, total = await ProviderRepository.list_providers_paginated(
            db_manager=self._db,
            q=q,
            status=status,
            page=page,
            page_size=page_size,
        )
        ids = [p.id for p in rows]
        cm, tm = await ProviderRepository.provider_counts_and_types(ids, db_manager=self._db)

        out: list[ModelProviderOut] = []
        for p in rows:
            out.append(
                _to_provider_out(
                    p,
                    model_count=cm.get(p.id, 0),
                    supported_types=tm.get(p.id, set()),
                )
            )
        return out, total

    async def create_provider(self, body: ModelProviderCreate) -> ModelProviderOut:
        api_format = api_format_from_create(
            protocol=body.protocol,
            provider_kind=body.provider_kind,
        )
        key_stripped = body.api_key.strip() if body.api_key else ""

        row = SysModelProvider(
            provider_code=body.id.strip(),
            provider_name=body.name.strip(),
            base_url=body.base_url.strip(),
            api_format=api_format,
            auth_type="api_key",
            api_key=key_stripped or None,
            api_secret=None,
            status=1 if body.enabled else 0,
        )
        if row.status == 1 and not provider_has_credentials(
            auth_type=row.auth_type,
            api_key=row.api_key,
            api_secret=row.api_secret,
        ):
            raise ValueError("启用前请先配置 API Key（或无需密钥请将鉴权类型设为 none）")

        p = await ProviderRepository.insert_provider(row, db_manager=self._db)
        cm, tm = await ProviderRepository.provider_counts_and_types([p.id], db_manager=self._db)
        return _to_provider_out(p, model_count=cm.get(p.id, 0), supported_types=tm.get(p.id, set()))

    async def update_provider(
        self,
        provider_key: str,
        body: ModelProviderUpdate,
    ) -> ModelProviderOut:
        def apply_updates(p: SysModelProvider) -> None:
            if body.name is not None:
                p.provider_name = body.name.strip()
            if body.base_url is not None:
                p.base_url = body.base_url.strip() or None
            if body.provider_kind is not None:
                p.api_format = body.provider_kind.strip()
            if body.api_secret is not None:
                p.api_secret = body.api_secret.strip() or None
            if body.api_key and body.api_key.strip():
                p.api_key = body.api_key.strip()
            if body.enabled is not None:
                p.status = 1 if body.enabled else 0

        def after_flush(p: SysModelProvider) -> None:
            if p.status == 1 and not provider_has_credentials(
                auth_type=p.auth_type,
                api_key=p.api_key,
                api_secret=p.api_secret,
            ):
                raise ValueError("启用前请先配置 API Key（或无需密钥请将鉴权类型设为 none）")

        p = await ProviderRepository.update_provider(
            provider_key,
            db_manager=self._db,
            apply_updates=apply_updates,
            after_flush=after_flush,
        )
        cm, tm = await ProviderRepository.provider_counts_and_types([p.id], db_manager=self._db)
        return _to_provider_out(p, model_count=cm.get(p.id, 0), supported_types=tm.get(p.id, set()))

    async def delete_provider(self, provider_key: str) -> None:
        await ProviderRepository.delete_provider_by_key(provider_key, db_manager=self._db)

    async def probe_provider(self, provider_key: str) -> ProbeResult:
        p = await ProviderRepository.get_provider_by_key(provider_key, db_manager=self._db)
        if not p:
            raise KeyError("not found")
        base = (p.base_url or "").rstrip("/")
        if not base:
            return ProbeResult(ok=False, message="未配置 base_url")
        url = f"{base}/models"
        headers: dict[str, str] = {}
        if p.api_key and p.auth_type in (None, "api_key", "api_key_secret"):
            headers["Authorization"] = f"Bearer {p.api_key}"
        try:
            r = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
            ok = r.status_code < 500
            return ProbeResult(ok=ok, message=f"HTTP {r.status_code}")
        except Exception as e:
            logger.warning("probe provider failed", exc_info=True)
            return ProbeResult(ok=False, message=str(e))

    async def list_models(
        self,
        *,
        q: str | None,
        provider_id: str | None,
        model_type: str | None,
        status: str | None,
        provider_enabled_only: bool,
        page: int,
        page_size: int,
    ) -> tuple[list[LlmModelOut], int]:
        page = max(1, page)
        page_size = max(1, min(page_size, PROVIDER_API_MAX_PAGE_SIZE))
        type_tuple = map_filter_model_type(model_type)

        raw, total = await ProviderRepository.list_models_paginated(
            db_manager=self._db,
            q=q,
            provider_id=provider_id,
            model_types=type_tuple,
            status=status,
            provider_enabled_only=provider_enabled_only,
            page=page,
            page_size=page_size,
        )
        items = [_to_model_out(m, p) for m, p in raw]
        return items, total

    async def create_model(self, body: LlmModelCreate) -> LlmModelOut:
        endpoint = body.endpoint.strip() if body.endpoint and body.endpoint.strip() else None
        timeout = body.timeout if body.timeout is not None else 30
        is_enabled = bool(body.enabled)

        def validate_enabled_on_provider(prov: SysModelProvider) -> None:
            if not provider_has_credentials(
                auth_type=prov.auth_type,
                api_key=prov.api_key,
                api_secret=prov.api_secret,
            ):
                raise ValueError("启用模型前请先为所属提供商配置密钥（无需密钥的接入方式除外）")

        m, prov = await ProviderRepository.create_model_row(
            db_manager=self._db,
            provider_id_key=body.provider_id,
            model_code=body.model_code,
            model_name=body.model_name,
            model_type=body.model_type,
            endpoint=endpoint,
            timeout=timeout,
            is_enabled=is_enabled,
            validate_enabled_on_provider=validate_enabled_on_provider if is_enabled else None,
        )
        return _to_model_out(m, prov)

    async def update_model(self, model_key: str, body: LlmModelUpdate) -> LlmModelOut:
        if not model_key.isdigit():
            raise KeyError("not found")
        model_pk = int(model_key)

        async def apply_updates(m: SysModel, prov: SysModelProvider, session: AsyncSession) -> None:
            if body.model_name is not None:
                m.model_name = body.model_name.strip()
            if body.model_type is not None:
                m.model_type = body.model_type.strip()
            if body.enabled is not None:
                m.is_enabled = 1 if body.enabled else 0
            if body.endpoint is not None:
                m.endpoint = body.endpoint.strip() or None
            if body.timeout is not None:
                m.timeout = body.timeout
            if body.provider_id is not None:
                np = await ProviderRepository.get_provider_in_session(
                    session, body.provider_id.strip()
                )
                if not np:
                    raise ValueError("供应商不存在")
                m.provider_id = np.id

        def after_flush(m: SysModel, prov_final: SysModelProvider) -> None:
            if m.is_enabled == 1 and not provider_has_credentials(
                auth_type=prov_final.auth_type,
                api_key=prov_final.api_key,
                api_secret=prov_final.api_secret,
            ):
                raise ValueError("启用模型前请先为所属提供商配置密钥（无需密钥的接入方式除外）")

        m, prov_final = await ProviderRepository.update_model(
            model_pk,
            db_manager=self._db,
            apply_updates=apply_updates,
            after_flush=after_flush,
        )
        return _to_model_out(m, prov_final)

    async def delete_model(self, model_key: str) -> None:
        if not model_key.isdigit():
            raise KeyError("not found")
        await ProviderRepository.delete_model_by_pk(int(model_key), db_manager=self._db)

    async def probe_model(self, model_key: str) -> ProbeResult:
        if not model_key.isdigit():
            raise KeyError("not found")
        row = await ProviderRepository.get_model_with_provider(int(model_key), db_manager=self._db)
        if not row:
            raise KeyError("not found")
        m, p = row
        base = (m.endpoint or p.base_url or "").rstrip("/")
        if not base:
            return ProbeResult(ok=False, message="未配置访问地址")
        url = f"{base}/models"
        headers: dict[str, str] = {}
        if p.api_key:
            headers["Authorization"] = f"Bearer {p.api_key}"
        try:
            r = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
            return ProbeResult(ok=r.status_code < 500, message=f"HTTP {r.status_code}")
        except Exception as e:
            return ProbeResult(ok=False, message=str(e))
