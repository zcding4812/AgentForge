"""工作台内置工具：StructuredTool 链路与 code/message/data 信封（轻量集成）。"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock

import pydantic
import pytest

from app.agent.adapters.tools.workbench import WORKBENCH_TOOL_INSTANCES
from app.agent.adapters.tools.workbench import handlers as wb_handlers
from app.agent.adapters.tools.workbench.context import WorkbenchRuntime, workbench_runtime_scope
from app.agent.adapters.tools.workbench.sub_agent_invoke import run_sub_agent_invocation
from app.agent.kernel.spec import AgentKind
from app.repositories.agent_repo import AgentRepository
from app.schemas.agent import (
    AgentDetailOut,
    AgentInvokeData,
    AgentInvokeRequest,
    AgentListData,
    AgentOut,
)
from app.services.agent_svc import AgentEntityService, AgentService


def _workbench_tool(name: str):
    for t in WORKBENCH_TOOL_INSTANCES:
        if t.name == name:
            return t
    raise AssertionError(name)


def test_workbench_tool_instances_include_kb_and_tool_catalog() -> None:
    names = {t.name for t in WORKBENCH_TOOL_INSTANCES}
    assert "workbench_update_agent" in names
    assert "workbench_create_agent" in names
    assert "workbench_list_knowledge_bases" in names
    assert "workbench_create_knowledge_base" in names
    assert "workbench_list_registered_tools" in names
    assert "workbench_list_external_tools" in names
    assert "workbench_list_sys_models" in names
    assert "workbench_invoke_sub_agents_parallel" in names
    assert len(WORKBENCH_TOOL_INSTANCES) == 11


@pytest.mark.asyncio
async def test_workbench_list_agents_coroutine_returns_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_list_agents(
        self: AgentEntityService,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        workspace_namespace: str | None = None,
    ) -> AgentListData:
        return AgentListData(items=[], total=0, page=page, page_size=page_size)

    monkeypatch.setattr(AgentEntityService, "list_agents", fake_list_agents)

    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    async with workbench_runtime_scope(rt):
        out = await wb_handlers.list_agents(page=1, page_size=20)

    assert out["code"] == "OK"
    assert out["message"] == "success"
    assert out["data"]["total"] == 0
    assert out["data"]["items"] == []


@pytest.mark.asyncio
async def test_workbench_list_agents_structured_tool_ainvoke_returns_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_list_agents(
        self: AgentEntityService,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        workspace_namespace: str | None = None,
    ) -> AgentListData:
        return AgentListData(items=[], total=0, page=page, page_size=page_size)

    monkeypatch.setattr(AgentEntityService, "list_agents", fake_list_agents)

    tool = _workbench_tool("workbench_list_agents")
    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    async with workbench_runtime_scope(rt):
        result = await tool.ainvoke({"page": 1, "page_size": 20, "q": None})

    assert isinstance(result, dict)
    assert result["code"] == "OK"
    assert "total" in result["data"] and "items" in result["data"]


@pytest.mark.asyncio
async def test_workbench_list_sys_models_coroutine_returns_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.schemas.providers import LlmModelOut

    async def fake_list_models(
        self,
        *,
        q: str | None,
        provider_id: str | None,
        model_type: str | None,
        status: str | None,
        provider_enabled_only: bool,
        page: int,
        page_size: int,
    ):
        one = LlmModelOut(
            id="9",
            model_code="x",
            model_name="X",
            provider_id="p",
            provider_name="P",
            model_type="llm",
            endpoint=None,
            provider_base_url=None,
            timeout=30,
            api_key_masked="***",
            enabled=True,
            health_status="unknown",
            health_message=None,
        )
        return [one], 1

    from app.services.provider_svc import ProviderService

    monkeypatch.setattr(ProviderService, "list_models", fake_list_models)

    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    async with workbench_runtime_scope(rt):
        out = await wb_handlers.list_sys_models(page=1, page_size=10, model_type="chat")

    assert out["code"] == "OK"
    body = out["data"]
    assert body["meta"]["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == "9"


@pytest.mark.asyncio
async def test_workbench_list_knowledge_bases_coroutine_returns_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.schemas.knowledge import KnowledgeListData
    from app.services.knowledge_svc import KnowledgeBaseService

    async def fake_list(
        self: KnowledgeBaseService,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        status: str | None = None,
        workspace_namespace: str | None = None,
    ) -> KnowledgeListData:
        return KnowledgeListData(items=[], total=0, page=page, page_size=page_size)

    monkeypatch.setattr(KnowledgeBaseService, "list_knowledge_bases", fake_list)

    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    async with workbench_runtime_scope(rt):
        out = await wb_handlers.list_knowledge_bases(page=1, page_size=20)

    assert out["code"] == "OK"
    assert out["data"]["total"] == 0
    assert out["data"]["items"] == []


@pytest.mark.asyncio
async def test_workbench_update_agent_delegates_like_patch_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    detail = AgentDetailOut(
        id=2,
        namespace_id=1,
        workspace_namespace="ns-demo",
        name="n",
        description="new",
        agent_kind=AgentKind.REACT,
        status="active",
        sys_model_id=None,
        system_prompt=None,
        config_json=None,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
    )

    async def fake_update(self, agent_id: int, body):
        assert agent_id == 2
        assert body.description == "new"
        return detail

    monkeypatch.setattr(AgentEntityService, "update_agent", fake_update)

    async def fake_get_by_id(agent_id, db_manager=None):
        m = MagicMock()
        m.workspace_namespace = "ns-demo"
        return m

    monkeypatch.setattr(AgentRepository, "get_by_id", fake_get_by_id)

    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    async with workbench_runtime_scope(rt):
        out = await wb_handlers.update_agent(2, '{"description": "new"}')

    assert out["code"] == "OK"
    assert out["data"]["description"] == "new"


@pytest.mark.asyncio
async def test_workbench_create_agent_forces_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime

    created = AgentOut(
        id=7,
        namespace_id=3,
        workspace_namespace="ns-demo",
        name="bot-a",
        description=None,
        agent_kind=AgentKind.SIMPLE_CHAT,
        status="active",
        sys_model_id=None,
        system_prompt=None,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
    )

    captured: dict = {}

    async def fake_create(self, body):
        captured["body"] = body
        return created

    monkeypatch.setattr(AgentEntityService, "create_agent", fake_create)

    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    payload = (
        '{"name": "bot-a", "agent_kind": "simple_chat", '
        '"system_prompt": "你是助手。", "workspace_namespace": "evil-other"}'
    )
    async with workbench_runtime_scope(rt):
        out = await wb_handlers.create_agent(payload)

    assert out["code"] == "OK"
    assert out["data"]["name"] == "bot-a"
    assert captured["body"].workspace_namespace == "ns-demo"


@pytest.mark.asyncio
async def test_workbench_create_agent_rejects_workbench_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    payload = '{"name": "wb", "agent_kind": "workbench", "system_prompt": "x"}'
    async with workbench_runtime_scope(rt):
        out = await wb_handlers.create_agent(payload)

    assert out["code"] == "CREATE_FAILED"


@pytest.mark.asyncio
async def test_workbench_update_agent_namespace_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_by_id(agent_id, db_manager=None):
        m = MagicMock()
        m.workspace_namespace = "other-ns"
        return m

    monkeypatch.setattr(AgentRepository, "get_by_id", fake_get_by_id)

    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=MagicMock(),
        parent_invoke=parent,
        effective_request_id="req-int",
    )
    async with workbench_runtime_scope(rt):
        out = await wb_handlers.update_agent(2, '{"description": "x"}')

    assert out["code"] == "NAMESPACE_MISMATCH"


def test_conversation_owner_requires_session_id() -> None:
    with pytest.raises(pydantic.ValidationError):
        AgentInvokeRequest(
            agent_kind=AgentKind.REACT,
            user_message="x",
            agent_id=1,
            conversation_owner_agent_id=1,
        )


@pytest.mark.asyncio
async def test_workbench_sub_agent_shares_root_session_on_invoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """子 Agent 内部 invoke 应带上根工作台的 session + owner，避免单独建子会话。"""

    row = MagicMock()
    row.id = 99
    row.agent_kind = "react"
    row.workspace_namespace = "ns-demo"
    row.status = "active"
    row.system_prompt = None
    row.config_json = {}
    row.sys_model_id = 1

    async def fake_get_by_id(aid, db_manager=None):
        if aid == 99:
            return row
        return None

    captured: dict = {}

    async def fake_invoke(self, body, request_id=None, forward_pixel_progress_to=None, **_kw):
        captured["body"] = body
        return AgentInvokeData(
            assistant_text="ok", tokens=1, conversation_session_id=body.conversation_session_id
        )

    monkeypatch.setattr(AgentRepository, "get_by_id", fake_get_by_id)
    monkeypatch.setattr(AgentService, "invoke", fake_invoke)

    parent = AgentInvokeRequest(
        agent_kind=AgentKind.WORKBENCH,
        user_message="hi",
        workspace_namespace="ns-demo",
        agent_id=1,
    )
    rt = WorkbenchRuntime(
        db_manager=MagicMock(),
        namespace="ns-demo",
        agent_service=AgentService(MagicMock()),  # type: ignore[arg-type]
        parent_invoke=parent,
        effective_request_id="req-int",
        root_conversation_session_id="sess-root-abc",
    )
    out = await run_sub_agent_invocation(rt, 99, "sub only", None)
    assert out["code"] == "OK"
    b: AgentInvokeRequest = captured["body"]
    assert b.conversation_session_id == "sess-root-abc"
    assert b.conversation_owner_agent_id == 1
    assert b.agent_id == 99
