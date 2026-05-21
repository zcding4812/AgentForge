"""Agent 对话历史 HTTP API（会话 + 消息；与 Agent 执行内核解耦，由调用方按需写入/读取）。

多轮与 Agent 编排的衔接见 ``AgentInvokeRequest`` / ``AgentInvokeData``：须在调用 ``/api/agent/invoke`` 或流式接口时
回传 ``conversation_session_id``，否则仅传 ``agent_id`` 时每次都会新建会话。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.constants.conversation import (
    CONVERSATION_DETAIL_MESSAGES_DEFAULT_PAGE_SIZE,
    CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE,
    CONVERSATION_LIST_DEFAULT_PAGE_SIZE,
    CONVERSATION_LIST_MAX_PAGE_SIZE,
)
from app.core.deps import DBManagerDep, OptionalRedisDep
from app.domain.conversation import ConversationDomainError
from app.schemas.conversation import (
    ConversationDetailData,
    ConversationMessageAppendBody,
    ConversationMessageExecutionOut,
    ConversationMessageOut,
    ConversationSessionCreateBody,
    ConversationSessionListData,
    ConversationSessionOut,
    ConversationSessionPatchBody,
    RollingSummaryEnqueueOut,
)
from app.schemas.response import ApiResponse
from app.services.conversation_svc import ConversationService


def get_conversation_service(
    db_manager: DBManagerDep,
    redis: OptionalRedisDep,
) -> ConversationService:
    return ConversationService(db_manager, redis=redis)


ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]


router = APIRouter()


@router.post(
    "/sessions",
    response_model=ApiResponse[ConversationSessionOut],
    summary="创建对话会话",
    operation_id="post_conversation_session",
)
async def create_session(
    body: ConversationSessionCreateBody,
    svc: ConversationServiceDep,
) -> ApiResponse[ConversationSessionOut]:
    try:
        data = await svc.create_session(body)
    except ConversationDomainError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.get(
    "/sessions",
    response_model=ApiResponse[ConversationSessionListData],
    summary="分页列出对话会话（不含已软删除）",
    operation_id="get_conversation_sessions",
)
async def list_sessions(
    svc: ConversationServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=CONVERSATION_LIST_MAX_PAGE_SIZE),
    ] = CONVERSATION_LIST_DEFAULT_PAGE_SIZE,
    agent_id: Annotated[int | None, Query(description="按 agent_entity.id 筛选")] = None,
) -> ApiResponse[ConversationSessionListData]:
    data = await svc.list_sessions(
        page=page,
        page_size=page_size,
        agent_id=agent_id,
    )
    return ApiResponse(message="ok", data=data)


@router.post(
    "/sessions/{session_id}/rolling-summary",
    response_model=ApiResponse[RollingSummaryEnqueueOut],
    summary="手动触发滚动摘要（异步入队）",
    operation_id="post_conversation_session_rolling_summary",
)
async def enqueue_rolling_summary(
    session_id: str,
    svc: ConversationServiceDep,
) -> ApiResponse[RollingSummaryEnqueueOut]:
    """将摘要任务写入进程内队列；实际 LLM 在后台执行。"""
    try:
        outcome = await svc.request_manual_rolling_summary(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    if outcome == "duplicate":
        return ApiResponse(
            message="已有摘要任务在排队或执行中",
            data=RollingSummaryEnqueueOut(queued=False),
        )
    if outcome == "queue_unavailable":
        raise HTTPException(
            status_code=503,
            detail="摘要队列未就绪（服务未启动 RollingSummaryWorker）",
        ) from None
    return ApiResponse(message="ok", data=RollingSummaryEnqueueOut(queued=True))


@router.get(
    "/sessions/{session_id}",
    response_model=ApiResponse[ConversationDetailData],
    summary="按会话 ID 获取会话元数据与消息分页（第 1 页为最近一批，页内按时间正序）",
    operation_id="get_conversation_session_detail",
)
async def get_session_detail(
    session_id: str,
    svc: ConversationServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE),
    ] = CONVERSATION_DETAIL_MESSAGES_DEFAULT_PAGE_SIZE,
    messages_agent_id: Annotated[
        int | None,
        Query(
            description="仅返回并统计 agent_id 等于该值的会话消息；工作台历史可传当前 workbench 的 agent_entity.id",
        ),
    ] = None,
    include_message_metadata: Annotated[
        bool,
        Query(
            description=(
                "为 false 时不加载各消息 metadata JSON、响应中 metadata 恒为 null；"
                "执行过程等用 GET .../messages/{message_id}/execution"
            ),
        ),
    ] = True,
) -> ApiResponse[ConversationDetailData]:
    data = await svc.get_session_detail(
        session_id,
        page=page,
        page_size=page_size,
        messages_agent_id=messages_agent_id,
        include_message_metadata=include_message_metadata,
    )
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在或已删除") from None
    return ApiResponse(message="ok", data=data)


@router.get(
    "/sessions/{session_id}/messages/{message_id}/execution",
    response_model=ApiResponse[ConversationMessageExecutionOut],
    summary="按消息 ID 查询执行过程快照",
    operation_id="get_conversation_message_execution",
)
async def get_message_execution(
    session_id: str,
    message_id: int,
    svc: ConversationServiceDep,
) -> ApiResponse[ConversationMessageExecutionOut]:
    """读取该条 **助手** 消息 ``metadata`` 中的 ``process_trace`` / ``tool_history`` / ``thinking_text``。"""
    try:
        data = await svc.get_message_execution(session_id, message_id)
    except ConversationDomainError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ApiResponse[ConversationMessageOut],
    summary="追加一条消息（活跃会话；关闭后拒绝写入）",
    operation_id="post_conversation_message",
)
async def append_message(
    session_id: str,
    body: ConversationMessageAppendBody,
    svc: ConversationServiceDep,
) -> ApiResponse[ConversationMessageOut]:
    data = await svc.append_message(session_id, body)
    if data is None:
        raise HTTPException(
            status_code=400,
            detail="会话不存在、已删除、已关闭或无法写入",
        ) from None
    return ApiResponse(message="ok", data=data)


@router.patch(
    "/sessions/{session_id}",
    response_model=ApiResponse[ConversationSessionOut],
    summary="更新会话（标题、关闭）",
    operation_id="patch_conversation_session",
)
async def patch_session(
    session_id: str,
    body: ConversationSessionPatchBody,
    svc: ConversationServiceDep,
) -> ApiResponse[ConversationSessionOut]:
    data = await svc.patch_session(session_id, body)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在、已删除或状态非法") from None
    return ApiResponse(message="ok", data=data)


@router.delete(
    "/sessions/{session_id}",
    response_model=ApiResponse[dict[str, bool]],
    summary="软删除会话（列表与默认查询不可见）",
    operation_id="delete_conversation_session",
)
async def delete_session(
    session_id: str,
    svc: ConversationServiceDep,
) -> ApiResponse[dict[str, bool]]:
    ok = await svc.soft_delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或已删除") from None
    return ApiResponse(message="ok", data={"deleted": True})
