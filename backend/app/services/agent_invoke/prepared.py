"""``prepare_invoke`` 冻结结果类型。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.agent.kernel import ModelConfigSnapshot, OutputControl, PromptSlots
from app.schemas.agent import AgentInvokeRequest, KnowledgeInvokeCitation


@dataclass(frozen=True, slots=True)
class PreparedInvoke:
    """供 ``invoke`` / 流式 SSE 共用的编排快照。"""

    snapshot: ModelConfigSnapshot
    output_control: OutputControl
    tools: tuple[BaseTool | dict, ...]
    prompt_slots: PromptSlots | None
    effective_request_id: str | None
    conversation_session_id: str | None
    #: 经 ``workbench`` 等对账后的请求体；与入参一致或已校验命名空间，供运行时与落库元数据使用
    invoke_body: AgentInvokeRequest
    #: 非致命提示（如未注册工具名已忽略）；严格模式下为空（未注册则已抛错）
    tool_warnings: tuple[str, ...] = ()
    #: 知识库检索注入上下文时的分片来源，供响应与 SSE done 带回显
    knowledge_citations: tuple[KnowledgeInvokeCitation, ...] = ()
