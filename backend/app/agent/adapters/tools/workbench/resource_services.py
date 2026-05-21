"""工作台资源服务：Agent / 知识库 / 工具与模型目录（同目录扁平模块）。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from app.agent.adapters.tools.workbench.tool_support import wb_err, wb_ok
from app.core.constants import PROVIDER_API_MAX_PAGE_SIZE
from app.repositories.agent_repo import AgentRepository
from app.schemas.agent import AgentCreateBody, AgentUpdateBody
from app.schemas.knowledge import KnowledgeCreateBody

if TYPE_CHECKING:
    from app.agent.adapters.tools.workbench.context import WorkbenchRuntime


class WorkbenchAgentResourceService:
    __slots__ = ("_ctx",)

    def __init__(self, ctx: WorkbenchRuntime) -> None:
        self._ctx = ctx

    async def list_agents(
        self, page: int = 1, page_size: int = 20, q: str | None = None
    ) -> dict[str, Any]:
        from app.services.agent_svc import AgentEntityService

        svc = AgentEntityService(self._ctx.db_manager, redis=self._ctx.redis)  # type: ignore[arg-type]
        data = await svc.list_agents(
            page=page,
            page_size=min(page_size, 50),
            q=q,
            workspace_namespace=self._ctx.namespace,
        )
        return wb_ok(
            {
                "total": data.total,
                "items": [x.model_dump(mode="json") for x in data.items],
            },
            message="success",
        )

    async def get_agent(self, agent_id: int) -> dict[str, Any]:
        from app.services.agent_svc import AgentEntityService

        svc = AgentEntityService(self._ctx.db_manager, redis=self._ctx.redis)  # type: ignore[arg-type]
        row = await svc.get_agent(agent_id)
        if row is None:
            return wb_err("NOT_FOUND", "Agent 不存在", data={"agent_id": agent_id})
        if row.workspace_namespace != self._ctx.namespace:
            return wb_err("NAMESPACE_MISMATCH", "Agent 不在当前工作区", data={"agent_id": agent_id})
        return wb_ok(row.model_dump(mode="json"))

    async def update_agent(self, agent_id: int, patch_json: str) -> dict[str, Any]:
        try:
            raw = json.loads(patch_json)
        except json.JSONDecodeError as e:
            return wb_err("VALIDATION_ERROR", f"JSON 无效: {e}", data={})
        if not isinstance(raw, dict):
            return wb_err("VALIDATION_ERROR", "patch_json 须为 JSON 对象", data={})
        try:
            body = AgentUpdateBody.model_validate(raw)
        except ValidationError as e:
            return wb_err("VALIDATION_ERROR", str(e), data={})

        row = await AgentRepository.get_by_id(agent_id, db_manager=self._ctx.db_manager)
        if row is None:
            return wb_err("NOT_FOUND", "Agent 不存在", data={"agent_id": agent_id})
        if row.workspace_namespace != self._ctx.namespace:
            return wb_err("NAMESPACE_MISMATCH", "Agent 不在当前工作区", data={"agent_id": agent_id})

        from app.services.agent_svc import AgentEntityService

        svc = AgentEntityService(self._ctx.db_manager, redis=self._ctx.redis)  # type: ignore[arg-type]
        try:
            out = await svc.update_agent(agent_id, body)
        except LookupError as e:
            return wb_err("NOT_FOUND", str(e), data={"agent_id": agent_id})
        except ValueError as e:
            return wb_err("UPDATE_FAILED", str(e), data={"agent_id": agent_id})

        return wb_ok(out.model_dump(mode="json"), message="updated")

    async def create_agent(self, create_json: str) -> dict[str, Any]:
        try:
            raw = json.loads(create_json)
        except json.JSONDecodeError as e:
            return wb_err("VALIDATION_ERROR", f"JSON 无效: {e}", data={})
        if not isinstance(raw, dict):
            return wb_err("VALIDATION_ERROR", "create_json 须为 JSON 对象", data={})

        raw = {**raw, "workspace_namespace": self._ctx.namespace}

        try:
            body = AgentCreateBody.model_validate(raw)
        except ValidationError as e:
            return wb_err("VALIDATION_ERROR", str(e), data={})

        from app.services.agent_svc import AgentEntityService

        svc = AgentEntityService(self._ctx.db_manager, redis=self._ctx.redis)  # type: ignore[arg-type]
        try:
            out = await svc.create_agent(body)
        except ValueError as e:
            return wb_err("CREATE_FAILED", str(e), data={})

        return wb_ok(out.model_dump(mode="json"), message="created")


class WorkbenchKnowledgeResourceService:
    __slots__ = ("_ctx",)

    def __init__(self, ctx: WorkbenchRuntime) -> None:
        self._ctx = ctx

    async def list_knowledge_bases(
        self,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        from app.services.knowledge_svc import KnowledgeBaseService

        svc = KnowledgeBaseService(self._ctx.db_manager)
        try:
            data = await svc.list_knowledge_bases(
                page=page,
                page_size=min(max(page_size, 1), 100),
                q=q,
                status=status,
                workspace_namespace=self._ctx.namespace,
            )
        except Exception as e:  # noqa: BLE001
            return wb_err("LIST_FAILED", str(e), data={})
        return wb_ok(
            {
                "total": data.total,
                "page": data.page,
                "page_size": data.page_size,
                "items": [x.model_dump(mode="json") for x in data.items],
            },
            message="success",
        )

    async def create_knowledge_base(
        self,
        name: str,
        description: str | None = None,
        retrieval_type: str = "keyword",
        storage_type: str = "keyword",
        embedding_model_config_id: int | None = None,
    ) -> dict[str, Any]:
        from app.services.knowledge_svc import KnowledgeBaseService

        try:
            body = KnowledgeCreateBody(
                name=name.strip(),
                workspace_namespace=self._ctx.namespace,
                description=description,
                retrieval_type=retrieval_type,  # type: ignore[arg-type]
                storage_type=storage_type,  # type: ignore[arg-type]
                embedding_model_config_id=embedding_model_config_id,
            )
        except Exception as e:  # noqa: BLE001
            return wb_err("VALIDATION_ERROR", str(e), data={})
        svc = KnowledgeBaseService(self._ctx.db_manager)
        try:
            out = await svc.create_knowledge_base(body)
        except ValueError as e:
            return wb_err("CREATE_FAILED", str(e), data={})
        return wb_ok(out.model_dump(mode="json"), message="created")


class WorkbenchCatalogResourceService:
    __slots__ = ("_ctx",)

    def __init__(self, ctx: WorkbenchRuntime) -> None:
        self._ctx = ctx

    async def list_registered_tools(self, namespace: str = "default") -> dict[str, Any]:
        from app.services.agent_tool_svc import AgentToolService

        svc = AgentToolService(self._ctx.db_manager)
        try:
            data = await svc.list_registered(namespace=namespace)
        except Exception as e:  # noqa: BLE001
            return wb_err("LIST_FAILED", str(e), data={})
        return wb_ok(data.model_dump(mode="json"), message="success")

    async def list_external_tools(self) -> dict[str, Any]:
        from app.services.ext_tool_svc import list_persisted_external_tools

        try:
            data = await list_persisted_external_tools(db_manager=self._ctx.db_manager)
        except Exception as e:  # noqa: BLE001
            return wb_err("LIST_FAILED", str(e), data={})
        return wb_ok(data.model_dump(mode="json"), message="success")

    async def list_sys_models(
        self,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        model_type: str = "chat",
    ) -> dict[str, Any]:
        from app.schemas.providers import paged_models
        from app.services.provider_svc import ProviderService

        svc = ProviderService(self._ctx.db_manager)
        mt = (model_type or "").strip()
        filter_type = mt if mt else "all"
        try:
            items, total = await svc.list_models(
                q=q,
                provider_id=None,
                model_type=filter_type,
                status="enabled",
                provider_enabled_only=True,
                page=max(1, page),
                page_size=min(max(page_size, 1), PROVIDER_API_MAX_PAGE_SIZE),
            )
        except Exception as e:  # noqa: BLE001
            return wb_err("LIST_FAILED", str(e), data={})
        data = paged_models(
            items, max(1, page), min(max(page_size, 1), PROVIDER_API_MAX_PAGE_SIZE), total
        )
        return wb_ok(data.model_dump(mode="json"), message="success")
