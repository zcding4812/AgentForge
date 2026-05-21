"""Agent HTTP 路由。

能用 ``@trace_span`` 的同步返回路由尽量用装饰器；``StreamingResponse`` 在路由 return 后仍继续写流，
无法用装饰器覆盖整段执行，故 ``/invoke/stream`` 使用 ``capture_span_parent`` + 服务层子 span。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from starlette.responses import Response, StreamingResponse

from app.core.constants import X_REQUEST_ID_HEADER
from app.core.context import RequestTraceContext
from app.core.deps import DBManagerDep, OptionalRedisDep
from app.core.tracing import capture_span_parent, trace_span
from app.domain.conversation import ConversationDomainError
from app.schemas.agent import (
    AgentCreateBody,
    AgentDetailOut,
    AgentHttpToolPersistedOut,
    AgentInvokeData,
    AgentInvokeRequest,
    AgentListData,
    AgentOut,
    AgentRegisteredToolsData,
    AgentRegisterHttpToolBody,
    AgentRegisterMcpToolBody,
    AgentRuntimeHttpToolsListData,
    AgentToolRegisterResultData,
    AgentUpdateBody,
    AgentUpdateHttpToolBody,
    AgentUpdateMcpToolBody,
    DefaultPromptDetailData,
    McpProbeListToolsBody,
    McpProbeListToolsData,
    WorkbenchWorkspaceCreateBody,
)
from app.schemas.response import ApiResponse
from app.services.agent_svc import AgentEntityService, AgentService
from app.services.agent_tool_svc import AgentToolService
from app.services.ext_tool_svc import (
    create_persisted_http_tool,
    create_persisted_mcp_tool,
    delete_persisted_http_tool,
    list_persisted_external_tools,
    probe_mcp_list_tools,
    update_persisted_http_tool,
    update_persisted_mcp_tool,
)
from app.services.knowledge_svc import ChunkRepositoryDep


def get_agent_entity_service(
    db_manager: DBManagerDep,
    redis: OptionalRedisDep,
) -> AgentEntityService:
    return AgentEntityService(db_manager, redis=redis)


AgentEntityServiceDep = Annotated[AgentEntityService, Depends(get_agent_entity_service)]


def get_agent_service(
    db_manager: DBManagerDep,
    redis: OptionalRedisDep,
    chunk_repository: ChunkRepositoryDep,
) -> AgentService:
    return AgentService(
        db_manager=db_manager,
        redis=redis,
        chunk_repository=chunk_repository,
    )


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]


router = APIRouter()


@router.get(
    "",
    response_model=ApiResponse[AgentListData],
    summary="Agent 列表（分页）",
    operation_id="get_agents_list",
)
async def list_agents(
    agents: AgentEntityServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    q: Annotated[str | None, Query(description="名称或描述模糊搜索")] = None,
    agent_kind: Annotated[
        list[str] | None,
        Query(description="可重复传入，筛选 simple_chat / react / plan_execute"),
    ] = None,
    exclude_agent_kind: Annotated[
        list[str] | None,
        Query(description="可重复传入，从列表中排除的 agent_kind（如系统工作台 workbench）"),
    ] = None,
    workspace_namespace: Annotated[
        str | None,
        Query(description="按命名空间 slug 精确筛选（对应 workspace_namespace.slug）"),
    ] = None,
) -> ApiResponse[AgentListData]:
    data = await agents.list_agents(
        page=page,
        page_size=page_size,
        q=q,
        agent_kinds=agent_kind,
        exclude_agent_kinds=exclude_agent_kind,
        workspace_namespace=workspace_namespace,
    )
    return ApiResponse(message="ok", data=data)


@router.post(
    "",
    response_model=ApiResponse[AgentOut],
    summary="创建 Agent",
    operation_id="post_agents_create",
)
async def create_agent_route(
    body: AgentCreateBody,
    agents: AgentEntityServiceDep,
) -> ApiResponse[AgentOut]:
    try:
        out = await agents.create_agent(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ApiResponse(message="ok", data=out)


@router.post(
    "/workbench",
    response_model=ApiResponse[AgentOut],
    summary="创建工作区工作台（新命名空间）",
    description=(
        "在指定 ``workspace_namespace`` 下插入唯一 ``agent_kind=workbench`` 行；"
        "该命名空间已存在工作台时返回 400。普通 POST /api/agents 仍不可直接创建 workbench。"
    ),
    operation_id="post_agents_workbench_workspace",
)
async def create_workbench_workspace_route(
    body: WorkbenchWorkspaceCreateBody,
    agents: AgentEntityServiceDep,
) -> ApiResponse[AgentOut]:
    try:
        out = await agents.create_workbench_workspace(body)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return ApiResponse(message="ok", data=out)


@router.get(
    "/default-prompts/{kind}",
    response_model=ApiResponse[DefaultPromptDetailData],
    summary="按类型获取默认系统提示词全文",
    operation_id="get_default_prompt_by_kind",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def get_default_prompt_by_kind(
    request: Request, kind: str
) -> ApiResponse[DefaultPromptDetailData]:
    data = AgentService.get_default_prompt(kind)
    if data is None:
        raise HTTPException(status_code=404, detail="未知的默认提示词类型") from None
    return ApiResponse(message="ok", data=data)


@router.get(
    "/tools",
    response_model=ApiResponse[AgentRegisteredToolsData],
    summary="已注册工具列表（进程内）",
    operation_id="get_agents_registered_tools",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def get_registered_tools(
    request: Request,
    db_manager: DBManagerDep,
    namespace: Annotated[str, Query(description="工具命名空间，默认与解析一致")] = "default",
) -> ApiResponse[AgentRegisteredToolsData]:
    data = await AgentToolService(db_manager).list_registered(namespace=namespace)
    return ApiResponse(message="ok", data=data)


@router.get(
    "/tools/http",
    response_model=ApiResponse[AgentRuntimeHttpToolsListData],
    summary="持久化外部工具列表（库表，HTTP / MCP）",
    operation_id="get_agents_http_tools_list",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def list_http_tools_route(
    request: Request,
    db_manager: DBManagerDep,
) -> ApiResponse[AgentRuntimeHttpToolsListData]:
    data = await list_persisted_external_tools(db_manager=db_manager)
    return ApiResponse(message="ok", data=data)


@router.post(
    "/tools/http",
    response_model=ApiResponse[AgentToolRegisterResultData],
    summary="创建并持久化 HTTP 工具（落库 + 进程内注册）",
    description=(
        "将 HTTP 请求模板写入库表并立即注册到 LangChain 工具表；进程重启后由启动流程重放。"
        "生产环境必须鉴权并限制可访问地址，防范 SSRF。"
    ),
    operation_id="post_agents_register_http_tool",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def register_http_tool_route(
    request: Request,
    body: AgentRegisterHttpToolBody,
    db_manager: DBManagerDep,
) -> ApiResponse[AgentToolRegisterResultData]:
    try:
        data = await create_persisted_http_tool(body, db_manager=db_manager)
    except ValueError as e:
        msg = str(e)
        if "已存在" in msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg) from e
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from e
    return ApiResponse(message="ok", data=data)


@router.patch(
    "/tools/http/{tool_name}",
    response_model=ApiResponse[AgentHttpToolPersistedOut],
    summary="更新持久化 HTTP 工具",
    operation_id="patch_agents_http_tool",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def patch_http_tool_route(
    request: Request,
    tool_name: str,
    body: AgentUpdateHttpToolBody,
    db_manager: DBManagerDep,
) -> ApiResponse[AgentHttpToolPersistedOut]:
    try:
        data = await update_persisted_http_tool(tool_name, body, db_manager=db_manager)
    except LookupError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValueError as e:
        msg = str(e)
        if "版本冲突" in msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg) from e
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from e
    return ApiResponse(message="ok", data=data)


@router.delete(
    "/tools/http/{tool_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除持久化 HTTP 工具",
    operation_id="delete_agents_http_tool",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def delete_http_tool_route(
    request: Request,
    tool_name: str,
    db_manager: DBManagerDep,
) -> Response:
    try:
        await delete_persisted_http_tool(tool_name, db_manager=db_manager)
    except LookupError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/tools/mcp",
    response_model=ApiResponse[AgentToolRegisterResultData],
    summary="创建并持久化 MCP 工具（落库 + 进程内注册）",
    description="将 MCP 连接配置写入库表并立即注册占位工具；进程重启后由启动流程重放。",
    operation_id="post_agents_register_mcp_tool",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def register_mcp_tool_route(
    request: Request,
    body: AgentRegisterMcpToolBody,
    db_manager: DBManagerDep,
) -> ApiResponse[AgentToolRegisterResultData]:
    try:
        data = await create_persisted_mcp_tool(body, db_manager=db_manager)
    except ValueError as e:
        msg = str(e)
        if "已存在" in msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg) from e
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from e
    return ApiResponse(message="ok", data=data)


@router.post(
    "/tools/mcp/probe-list",
    response_model=ApiResponse[McpProbeListToolsData],
    summary="探测 MCP tools/list（不落库）",
    description="使用当前表单中的 server_url、connection_config 发起 Streamable HTTP 会话并返回 tools/list，供创建前校验。",
    operation_id="post_agents_mcp_probe_list",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def probe_mcp_list_tools_route(
    request: Request,
    body: McpProbeListToolsBody,
) -> ApiResponse[McpProbeListToolsData]:
    try:
        data = await probe_mcp_list_tools(body)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.patch(
    "/tools/mcp/{tool_name}",
    response_model=ApiResponse[AgentHttpToolPersistedOut],
    summary="更新持久化 MCP 工具",
    operation_id="patch_agents_mcp_tool",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def patch_mcp_tool_route(
    request: Request,
    tool_name: str,
    body: AgentUpdateMcpToolBody,
    db_manager: DBManagerDep,
) -> ApiResponse[AgentHttpToolPersistedOut]:
    try:
        data = await update_persisted_mcp_tool(tool_name, body, db_manager=db_manager)
    except LookupError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValueError as e:
        msg = str(e)
        if "版本冲突" in msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg) from e
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from e
    return ApiResponse(message="ok", data=data)


@router.patch(
    "/{agent_id}",
    response_model=ApiResponse[AgentDetailOut],
    summary="更新 Agent（工作区配置持久化）",
    operation_id="patch_agent",
)
async def patch_agent(
    agent_id: int,
    body: AgentUpdateBody,
    agents: AgentEntityServiceDep,
) -> ApiResponse[AgentDetailOut]:
    try:
        data = await agents.update_agent(agent_id, body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除 Agent（物理删除；工作台则级联整命名空间）",
    operation_id="delete_agent",
)
@trace_span(span_type="api", component="agent-api", trace_id_arg="request")
async def delete_agent_route(
    request: Request,
    agent_id: int,
    agents: AgentEntityServiceDep,
) -> Response:
    try:
        await agents.delete_agent(agent_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{agent_id}",
    response_model=ApiResponse[AgentDetailOut],
    summary="Agent 详情",
    operation_id="get_agent_by_id",
)
async def get_agent_by_id(
    agent_id: int,
    agents: AgentEntityServiceDep,
) -> ApiResponse[AgentDetailOut]:
    row = await agents.get_agent(agent_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    return ApiResponse(message="ok", data=row)


@router.post(
    "/invoke",
    response_model=ApiResponse[AgentInvokeData],
    summary="Agent 非流式调用（基线）",
    operation_id="post_agent_invoke",
)
@trace_span(
    span_type="api",
    component="agent-api",
    trace_id_arg="request",
)
async def invoke_agent(
    request: Request,
    body: AgentInvokeRequest,
    agent: AgentServiceDep,
) -> ApiResponse[AgentInvokeData]:
    request_id = body.request_id or RequestTraceContext.get_request_id(request)
    try:
        data = await agent.invoke(
            body,
            request_id=request_id,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, ConversationDomainError) as e:
        status = getattr(e, "status_code", 400)
        raise HTTPException(status_code=status, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.post(
    "/invoke/stream",
    summary="Agent 流式调用（SSE，text/event-stream）",
    operation_id="post_agent_invoke_stream",
    responses={
        200: {
            "description": (
                "SSE：`data: {JSON}\\n\\n`；事件含 start / progress / delta / ping / done / error。"
            ),
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type":"start","agent_kind":"simple_chat","model":"gpt-4o-mini",'
                        '"provider":"openai","config_id":null,"tool_names":[]}\n\n'
                        'data: {"type":"delta","text":"...","lane":"content"}\n\n'
                        'data: {"type":"done","assistant_text":"...","structured":null}\n\n'
                    ),
                },
            },
        },
    },
)
async def invoke_agent_stream(
    request: Request,
    body: AgentInvokeRequest,
    agent: AgentServiceDep,
) -> StreamingResponse:
    # 未使用 @trace_span：SSE 主体在 return 之后执行，装饰器无法覆盖整段流式耗时。
    request_id = body.request_id or RequestTraceContext.get_request_id(request)
    stream_span_parent = capture_span_parent(request)
    try:
        prepared = await agent.prepare_invoke(
            body,
            request_id=request_id,
            span_parent=stream_span_parent,
        )
        await agent.persist_conversation_user_for_stream(
            prepared.invoke_body,
            conversation_session_id=prepared.conversation_session_id,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, ConversationDomainError) as e:
        status = getattr(e, "status_code", 400)
        raise HTTPException(status_code=status, detail=str(e)) from e

    return StreamingResponse(
        agent.iter_invoke_stream_sse_bytes(
            body,
            prepared,
            request,
            span_parent=stream_span_parent,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            X_REQUEST_ID_HEADER: request_id or "",
        },
    )
