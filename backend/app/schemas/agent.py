from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agent.kernel.spec import AgentKind


class ModelIdentityBody(BaseModel):
    provider: str = Field(default="openai", description="厂商/渠道标识")
    model_name: str = Field(default="gpt-4o-mini", description="模型名")
    deployment: str | None = Field(default=None, description="部分云厂商部署名")
    base_url_override: str | None = Field(default=None, description="可选网关覆盖")
    api_key: str | None = Field(
        default=None, description="可选 API Key；不传则使用服务端环境变量等默认凭据"
    )

    @field_validator("provider", "model_name", mode="after")
    @classmethod
    def _strip_required(cls, v: str) -> str:
        return v.strip()

    @field_validator("deployment", "base_url_override", "api_key", mode="after")
    @classmethod
    def _strip_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None


class HyperparametersBody(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = Field(
        default=None,
        description="内核快照字段；默认 ChatOpenAI 适配器不向 OpenAI Chat Completions 传 top_k（API 不支持）。",
    )
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None

    @field_validator("stop", mode="before")
    @classmethod
    def _strip_stop_sequences(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, (tuple, list)):
            return v
        out = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return tuple(out)


class ResponseConstraintsBody(BaseModel):
    response_format: Literal["text", "json_object", "json_schema"] = "text"
    json_schema_id: str | None = None
    response_json_schema: dict[str, Any] | None = Field(
        default=None,
        description="当 `response_format=json_schema` 时：下发给模型的 JSON Schema 根对象（OpenAI `response_format.json_schema.schema`）。省略或空对象时由服务端使用宽松默认可变 object，不必提前手写模板。",
    )
    parallel_tool_calls: bool | None = None

    @field_validator("json_schema_id", mode="before")
    @classmethod
    def _strip_json_schema_id(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class ToolChoiceBody(BaseModel):
    mode: Literal["auto", "none", "required", "specific"] = "auto"
    forced_tool_name: str | None = None

    @field_validator("forced_tool_name", mode="before")
    @classmethod
    def _strip_forced_tool(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class OutputControlBody(BaseModel):
    strip_thinking_blocks: bool = True
    include_tool_messages_in_raw: bool = False


class ToolCallPart(BaseModel):
    """历史轮次中的单次工具调用声明（写入对应 ``AIMessage.tool_calls``）。"""

    id: str = Field(min_length=1, description="与 ToolMessage.tool_call_id 一致")
    name: str = Field(min_length=1, description="工具名")
    arguments: str = Field(
        default="{}",
        description="JSON 对象字符串，将解析为工具参数",
    )

    @field_validator("id", "name", "arguments", mode="after")
    @classmethod
    def _strip_tool_call_fields(cls, v: str) -> str:
        return v.strip()


class ToolResultPart(BaseModel):
    """与 ``ToolCallPart.id`` 对应的工具执行结果（``ToolMessage``）。"""

    tool_call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    content: str = Field(default="", description="工具返回正文")

    @field_validator("tool_call_id", "name", "content", mode="after")
    @classmethod
    def _strip_tool_result_fields(cls, v: str) -> str:
        return v.strip()


class ChatHistoryTurn(BaseModel):
    """一轮对话：用户 Human → 助手 AI →（可选）工具返回，与时间轴顺序一致。

    多轮串联成严格 Human/AI 交替；若该轮助手调用了工具，则在对应 ``AIMessage`` 后紧跟 ``ToolMessage`` 序列。
    """

    user: str = Field(min_length=1, description="该轮用户消息")
    assistant: str = Field(
        default="",
        description="该轮助手正文；若该轮仅有工具调用链可为空",
    )
    tool_calls: list[ToolCallPart] = Field(
        default_factory=list,
        description="该轮助手消息上的工具调用列表（可为空）",
    )
    tool_results: list[ToolResultPart] = Field(
        default_factory=list,
        description="与 tool_calls 对应的工具结果，顺序建议与调用顺序一致",
    )

    @field_validator("user", "assistant", mode="after")
    @classmethod
    def _strip_turn_text(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _assistant_or_tool_calls(self) -> Self:
        if not self.assistant and not self.tool_calls:
            raise ValueError("每轮需提供 assistant 文本，或至少一条 tool_calls")
        return self


class PromptEngineeringBody(BaseModel):
    """与四类消息规范对齐：system_prompt（第 1 类）、context（第 4 类参考资料）、chat_history（第 2/3 类）。"""

    system_prompt: str | None = Field(
        default=None,
        description="第 1 类 System Prompt：角色、能力、约束、输出格式与行为准则（单条；未传用服务端默认）",
    )
    context: str | None = Field(
        default=None,
        description="第 4 类：参考上下文，写入末条 Human，位于当前用户问题之前",
    )
    chat_history: list[ChatHistoryTurn] = Field(
        default_factory=list,
        description=(
            "第 2/3 类：按时间从早到晚；每轮 Human→AI（可选 tool_calls）→ToolMessage*；服务端截断为最近 max_history_rounds 轮。"
            "续聊且传 conversation_session_id 时默认以库表为 SSOT；"
            "除非该 Agent 的 config_json.memory.allow_client_chat_history 为 true，否则忽略客户端本字段"
        ),
    )
    max_history_rounds: int = Field(
        default=10,
        ge=1,
        le=50,
        description="滑动窗口：保留最近若干轮完整历史轮次（含工具消息）",
    )
    max_history_tokens: int | None = Field(
        default=None,
        ge=256,
        le=200_000,
        description=(
            "可选：窗内历史段 token 上限（LangChain trim_messages 近似计数），与 max_history_rounds 叠加；"
            "前缀 System 与当前用户 Human 不受此上限裁剪"
        ),
    )

    @field_validator("system_prompt", "context", mode="before")
    @classmethod
    def _strip_prompt_fields(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class AgentInvokeRequest(BaseModel):
    agent_kind: AgentKind = Field(description="Agent 编排类型")
    user_message: str = Field(min_length=1, description="用户输入")
    model_identity: ModelIdentityBody = Field(default_factory=ModelIdentityBody)
    hyperparameters: HyperparametersBody = Field(default_factory=HyperparametersBody)
    response: ResponseConstraintsBody = Field(default_factory=ResponseConstraintsBody)
    tool_choice: ToolChoiceBody | None = None
    output: OutputControlBody = Field(default_factory=OutputControlBody)
    config_id: str | int | None = Field(
        default=None,
        description="可选：系统模型主键 sys_model.id；传入时服务端从库加载 endpoint/凭据，与 model_identity 手动项互斥（见接口说明）",
    )
    request_id: str | None = Field(
        default=None, description="可选：显式请求 ID；未传则使用网关注入的链路 ID"
    )
    conversation_session_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "续聊时必传：与首轮或创建会话时返回的 ID 一致。"
            "若省略且传入 agent_id，则每次请求都会新建会话（会话碎片化）；"
            "多轮连续对话须在客户端保存响应中的 conversation_session_id 并在后续请求中回传。"
            "若传入则须同时传 agent_id。通常 agent_id 须与会话行绑定的 agent_entity.id 一致；"
            "工作台子 Agent 调用时见 conversation_owner_agent_id。"
        ),
    )
    conversation_owner_agent_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "与会话行绑定的「归属」agent 主键（须等于该 session 在库中的 agent_id），"
            "而 agent_id 可为实际执行的 Agent（工作台子调用、像素办公室多角色同会话等）。"
            "与 conversation_session_id 成对出现；工作台子调用时须与根 workbench 的 agent_id 一致；"
            "HTTP 直连场景下须与归属、执行双方同一 workspace_namespace。"
        ),
    )
    conversation_user_display_name: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "可选；非空时写入本轮用户消息的 metadata.user_display_name，"
            "供工作台等多端展示发言人；与模型推理无关。"
        ),
    )
    agent_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "可选：agent_entity 主键。"
            "与 conversation_session_id 一起用于归属校验；"
            "仅传 agent_id 而不传 conversation_session_id 时，服务端为本轮自动新建会话并落库消息（见 conversation_session_id）。"
        ),
    )
    tool_names: list[str] = Field(default_factory=list, description="按名称解析的已注册工具列表")
    strict_tool_names: bool = Field(
        default=False,
        description=(
            "为 true 时：存在未在服务端注册的工具名则请求失败（400）；"
            "默认 false：忽略未注册名称，仅用已注册工具，并在响应 warnings / SSE done 中提示"
        ),
    )
    workspace_namespace: str = Field(
        default="default",
        max_length=64,
        description=(
            "工作区命名空间。"
            "当 agent_kind=workbench 时须传 agent_id，且须与入库 agent_entity.workspace_namespace 完全一致，"
            "否则拒绝请求。"
        ),
    )
    prompts: PromptEngineeringBody | None = Field(
        default=None,
        description="可选：system_prompt（第 1 类）、context（第 4 类）、chat_history（第 2/3 类）",
    )
    workbench_force_client_history: bool = Field(
        default=False,
        description=(
            "内部/程序化调用：为 true 时强制使用请求体 prompts.chat_history 组装窗内历史（PASSTHROUGH），"
            "不因存在 conversation_session_id 而改走库表合并。用于 workbench 子 Agent 继承父对话上下文。"
        ),
    )

    @field_validator("workspace_namespace", mode="after")
    @classmethod
    def _strip_workspace_namespace(cls, v: str) -> str:
        s = (v or "").strip()
        return s if s else "default"

    @field_validator("user_message", mode="after")
    @classmethod
    def _strip_user_message(cls, v: str) -> str:
        return v.strip()

    @field_validator("conversation_user_display_name", mode="after")
    @classmethod
    def _strip_conversation_user_display_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    @field_validator("config_id", mode="before")
    @classmethod
    def _normalize_config_id(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @field_validator("request_id", "conversation_session_id", mode="before")
    @classmethod
    def _strip_optional_ids(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @field_validator("tool_names", mode="before")
    @classmethod
    def _strip_tool_name_list(cls, v: Any) -> Any:
        if not v:
            return []
        if not isinstance(v, list):
            return v
        out: list[str] = []
        for x in v:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
        return out

    @model_validator(mode="after")
    def _conversation_requires_agent_id(self) -> Self:
        sid = self.conversation_session_id or ""
        if sid and self.agent_id is None:
            raise ValueError(
                "传入 conversation_session_id 时必须同时传入 agent_id（子 Agent 调用时为被执行业务 Agent 的 id；"
                "若使用 conversation_owner_agent_id 对账根会话，则该 id 为子 Agent）"
            )
        if self.conversation_owner_agent_id is not None and not sid:
            raise ValueError(
                "设置 conversation_owner_agent_id 时必须同时提供 conversation_session_id"
            )
        return self


class OrchestrationEdge(BaseModel):
    """工作台本轮对子 Agent 的调用边（由末态 ``messages`` 中 ``workbench_invoke_sub_agent`` 解析）。"""

    order: int = Field(ge=0, description="同轮内第几次子调用（从 0 起）")
    parent_agent_id: int = Field(ge=1, description="父：当前工作台 agent_entity.id")
    child_agent_id: int = Field(ge=1, description="子：被 invoke 的目标 agent_entity.id")


class WorkbenchFanInItem(BaseModel):
    """工作台 fan-in 单项：一次子调用（含并行批内拆分项）的汇总。"""

    tool: str = Field(min_length=1, description="工具名（workbench_invoke_sub_agent 或 parallel）")
    code: str = Field(min_length=1, description="工具信封 code")
    ok: bool = Field(description="是否成功（code == OK）")
    message: str = Field(default="", description="工具信封 message")
    child_agent_id: int | None = Field(default=None, ge=1, description="涉及的子 Agent id")


class WorkbenchFanInChildStat(BaseModel):
    """按子 Agent 聚合的 fan-in 统计，便于前端直接渲染状态。"""

    child_agent_id: int = Field(ge=1)
    total_calls: int = Field(ge=0)
    success_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    last_code: str | None = Field(default=None)


class WorkbenchFanInSummary(BaseModel):
    """工作台 fan-in 聚合摘要（由 LangGraph fan-in 节点生成）。"""

    total_calls: int = Field(ge=0)
    success_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    unique_child_agent_ids: list[int] = Field(default_factory=list)
    items: list[WorkbenchFanInItem] = Field(default_factory=list)
    child_stats: list[WorkbenchFanInChildStat] = Field(default_factory=list)


class AgentProcessTraceStep(BaseModel):
    """执行过程时间线：模型段与工具调用 / 返回交错，与 ``metadata.process_trace`` 项形状一致。"""

    seq: int = Field(ge=0, description="本轮内按时间序从 0 递增（含工具步骤）")
    text: str = Field(description="该段展示用正文（Markdown；工具行可含代码块）")
    phase: Literal["step", "final"] = Field(
        description="模型段：step=中间；final=与 assistant_text 一致的末段。工具段恒为 step",
    )
    kind: Literal["model", "tool_call", "tool_result"] = Field(
        default="model",
        description="model=模型输出；tool_call=请求调用工具；tool_result=工具执行返回",
    )
    tool_call_id: str | None = Field(default=None, description="工具行：关联的 tool_call_id")
    name: str | None = Field(default=None, description="工具行：工具名")
    arguments: Any | None = Field(default=None, description="tool_call 行：入参对象")
    content: str | None = Field(
        default=None,
        description="tool_result 行：原始返回字符串（可能与 text 中展示一致或供结构化消费）",
    )


class KnowledgeInvokeCitation(BaseModel):
    """本轮将知识库检索结果注入 ``prompts.context`` 时的分片来源，供前端在对话中展示。"""

    kb_id: int = Field(ge=1)
    kb_name: str = Field(min_length=1, max_length=256)
    doc_id: int = Field(ge=1)
    filename: str | None = Field(default=None, max_length=512)
    chunk_index: int | None = None
    text_snippet: str | None = Field(default=None, max_length=4000)
    match_type: str | None = Field(default=None, max_length=32)


class AgentInvokeData(BaseModel):
    assistant_text: str
    thinking_text: str | None = Field(
        default=None,
        description="从 ``<thinking>`` / ``<redacted_thinking>`` 解析的思考正文；与 assistant_text 分离；无则 null",
    )
    structured: dict | None = None
    tokens: int | None = Field(
        default=None,
        description="本轮网关 usage 总 token（多步编排时为各次调用之和）",
    )
    conversation_session_id: str | None = Field(
        default=None,
        description=(
            "本轮生效的对话会话 ID；新会话由服务端生成。"
            "多轮对话时请在后续 AgentInvokeRequest.conversation_session_id 中原样回传，避免每轮产生新会话。"
        ),
    )
    warnings: list[str] | None = Field(
        default=None,
        description="非致命提示：例如 tool_names 中未注册名称已忽略（strict_tool_names=false 时）",
    )
    knowledge_citations: list[KnowledgeInvokeCitation] | None = Field(
        default=None,
        description="本轮从绑定知识库检索并注入上下文的分片列表，用于侧栏对话展示来源",
    )
    orchestration_edges: list[OrchestrationEdge] | None = Field(
        default=None,
        description="仅 agent_kind=workbench：本轮子 Agent 调用顺序（从末态消息解析）",
    )
    workbench_fan_in: WorkbenchFanInSummary | None = Field(
        default=None,
        description="仅 agent_kind=workbench：本轮 fan-in 聚合摘要（子调用总数、成功失败与明细）",
    )
    process_trace: list[AgentProcessTraceStep] | None = Field(
        default=None,
        description=(
            "执行过程时间线：按消息顺序交错模型输出与工具调用 / 返回（seq 连续）；"
            "写入助手消息 metadata，供「执行过程」统一展示"
        ),
    )
    tool_history: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "本轮（末条 user 之后）工具调用与 ToolMessage 结果时间序列；"
            "项含 phase=call|result、name、tool_call_id、arguments|content；"
            "子 Agent 嵌套步骤可含 sub_agent_id、nesting=sub_agent；"
            "同时写入助手消息 metadata.tool_history"
        ),
    )


# --- 入库 Agent 资源（`agent_entity` 表：列表 / 创建 / 详情 / 更新）---


class WorkbenchWorkspaceCreateBody(BaseModel):
    """在全新命名空间下创建一条 ``agent_kind=workbench`` 的入库行（普通 ``POST /api/agents`` 仍禁止创建 workbench）。"""

    workspace_namespace: str = Field(
        min_length=1,
        max_length=64,
        description="工作区命名空间；创建后与子 Agent / 知识库等资源隔离",
    )
    name: str = Field(
        default="工作台",
        min_length=1,
        max_length=128,
        description="该命名空间下工作台展示名；（workspace_namespace, name）唯一",
    )
    description: str | None = Field(default=None, max_length=512)

    @field_validator("workspace_namespace", mode="after")
    @classmethod
    def _normalize_ws_create_namespace(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("workspace_namespace 不能为空")
        return s

    @field_validator("name", mode="after")
    @classmethod
    def _strip_ws_create_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("description", mode="before")
    @classmethod
    def _strip_ws_create_desc(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class AgentCreateBody(BaseModel):
    name: str = Field(
        min_length=1, max_length=128, description="展示名称；在 workspace_namespace 内唯一"
    )
    workspace_namespace: str = Field(
        default="default",
        max_length=64,
        description="工作区命名空间",
    )
    description: str | None = Field(default=None, max_length=512)
    agent_kind: AgentKind = Field(description="编排类型，与 AgentInvokeRequest.agent_kind 一致")
    system_prompt: str | None = Field(
        default=None,
        description="第 1 类系统提示；可留空，随后在详情或 config_json.prompt_system 中编辑",
    )
    sys_model_id: int | None = Field(
        default=None,
        description="可选：挂载对话模型 sys_model.id（须为已启用的 llm 类；与 AgentUpdateBody 一致）",
    )
    config_json: dict[str, Any] | None = Field(
        default=None,
        description=(
            "可选：工作区序列化配置（如 v、temperature、tool_names、memory、knowledge 等）；"
            "可与 system_prompt 二选一提供提示（本字段内 prompt_system 非空字符串）；"
            "创建时均可省略，随后在详情中补全；语义同 AgentUpdateBody.config_json"
        ),
    )

    @field_validator("workspace_namespace", mode="after")
    @classmethod
    def _strip_create_workspace_namespace(cls, v: str) -> str:
        s = (v or "").strip()
        return s if s else "default"

    @field_validator("name", mode="after")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("description", "system_prompt", mode="before")
    @classmethod
    def _strip_create_optional(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class AgentUpdateBody(BaseModel):
    """工作区保存：元数据、挂载模型、`system_prompt` 与配置快照（`config_json` 全量替换）。"""

    name: str | None = Field(default=None, max_length=128, description="展示名称，全局唯一")
    description: str | None = Field(default=None, max_length=512)
    agent_kind: AgentKind | None = Field(default=None, description="编排类型")
    sys_model_id: int | None = Field(
        default=None,
        description="sys_model.id；传 null 表示取消挂载",
    )
    system_prompt: str | None = Field(
        default=None,
        description="第 1 类系统提示；传 null 表示清空列",
    )
    config_json: dict[str, Any] | None = Field(
        default=None,
        description=(
            "前端工作区序列化配置（温度、工具等；提示词可与 system_prompt 列双写）。"
            "记忆见 ``config_json.memory``：max_history_rounds_cap、allow_client_chat_history、"
            "summarization_sys_model_id、summary_context_ratio_threshold、summary_context_ratio_urgent、"
            "summary_min_rounds_since_last、summary_min_tokens_since_last、summary_context_window_tokens"
            "（见 app.domain.agent.memory_settings）。"
            "输入规则见 ``config_json.input_filter``（ReAct 为图首节点、Simple Chat 为 ``create_agent`` 中间件）："
            "enabled、banned_keywords、banned_regex、"
            "max_user_chars、truncate_on_max、reject_message"
        ),
    )

    @field_validator("name", mode="after")
    @classmethod
    def _strip_update_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("名称不能为空")
        return s

    @field_validator("description", "system_prompt", mode="before")
    @classmethod
    def _strip_update_optional_text(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class AgentOut(BaseModel):
    id: int
    namespace_id: int = Field(description="命名空间主键，对应表 namespace.id")
    workspace_namespace: str = "default"
    name: str
    description: str | None
    agent_kind: AgentKind
    status: str
    sys_model_id: int | None
    system_prompt: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AgentDetailOut(AgentOut):
    config_json: dict[str, Any] | None = None


class AgentListData(BaseModel):
    items: list[AgentOut]
    total: int
    page: int
    page_size: int


class DefaultPromptDetailData(BaseModel):
    """某一类型的完整默认提示词（可填入工作区 system_prompt）。"""

    id: str
    label: str
    description: str
    text: str = Field(description="默认正文全文")


class AgentHttpToolConfigOut(BaseModel):
    """HTTP 工具请求配置摘要（用于列表展示）。"""

    method: str
    url: str
    headers: dict[str, str] | None = None
    body: str | None = Field(default=None, description="请求体原文")
    timeout_seconds: float


class AgentRegisteredToolOut(BaseModel):
    """进程内 LangChain 工具注册表项（与 ``tool_names`` 解析一致）。"""

    name: str = Field(description="注册名，请求体 tool_names 项与此一致")
    namespace: str = Field(default="default", description="注册命名空间")
    description: str = Field(default="", description="工具说明（BaseTool.description）")
    parameters_summary: str = Field(
        default="（无参数）",
        description="由参数 JSON Schema 生成的简要说明，便于控制台展示",
    )
    parameters_json_schema: dict[str, Any] | None = Field(
        default=None,
        description="工具参数 JSON Schema（若有）",
    )
    origin: Literal["builtin", "runtime", "http", "mcp"] = Field(
        description="builtin：进程内置；runtime：非持久化运行期；http/mcp：库表持久化的外部工具",
    )
    http_request: AgentHttpToolConfigOut | None = Field(
        default=None,
        description="当 origin=http 时返回持久化 HTTP 请求配置",
    )
    mcp_config: dict[str, Any] | None = Field(
        default=None,
        description="当 origin=mcp 时返回持久化 MCP 配置摘要",
    )


class AgentRegisteredToolsData(BaseModel):
    items: list[AgentRegisteredToolOut]
    namespace: str


_HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"})


class AgentRegisterHttpToolBody(BaseModel):
    """创建并持久化 HTTP 工具（落库 + 进程内注册）。"""

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",
        description="与 tool_names 项一致；同名则冲突",
    )
    description: str = Field(min_length=1, max_length=512)
    url: str = Field(description="请求 URL，须为 http:// 或 https://")
    method: str = Field(default="GET", max_length=16, description="HTTP 方法")
    headers: dict[str, str] | None = Field(default=None, description="额外请求头")
    body: str | None = Field(
        default=None, description="请求体模板（如 JSON 字符串）；GET/HEAD 忽略"
    )
    timeout_seconds: float = Field(default=15, ge=1, le=120)
    enabled: bool = Field(default=True, description="是否启用；仅启用的工具在启动时重放注册")
    input_schema: dict[str, Any] | None = Field(
        default=None, description="工具入参 JSON Schema（可选，用于模型侧约束）"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="工具出参 JSON Schema（可选）"
    )

    @field_validator("name", "description", mode="after")
    @classmethod
    def _strip_required(cls, v: str) -> str:
        return v.strip()

    @field_validator("method", mode="after")
    @classmethod
    def _method_upper(cls, v: str) -> str:
        m = v.strip().upper()
        if m not in _HTTP_METHODS:
            raise ValueError(f"method 须为 {sorted(_HTTP_METHODS)} 之一")
        return m

    @field_validator("url", mode="after")
    @classmethod
    def _http_url_only(cls, v: str) -> str:
        s = v.strip()
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError("url 须为 http:// 或 https://")
        return s


class AgentUpdateHttpToolBody(BaseModel):
    """更新持久化 HTTP 工具（仅提交需修改的字段）。"""

    description: str | None = Field(default=None, max_length=512)
    url: str | None = None
    method: str | None = Field(default=None, max_length=16)
    headers: dict[str, str] | None = None
    body: str | None = Field(default=None, description="请求体；置空请用 clear_body")
    timeout_seconds: float | None = Field(default=None, ge=1, le=120)
    enabled: bool | None = Field(default=None, description="是否启用")
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    version: int | None = Field(
        default=None,
        ge=1,
        description="乐观锁：传入须与当前库中 version 一致，否则 409 冲突",
    )
    clear_headers: bool = Field(default=False, description="为 true 时清空已存请求头")
    clear_body: bool = Field(default=False, description="为 true 时清空已存请求体")

    @field_validator("description", mode="after")
    @classmethod
    def _strip_desc(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip()

    @field_validator("url", mode="after")
    @classmethod
    def _url_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError("url 须为 http:// 或 https://")
        return s

    @field_validator("method", mode="after")
    @classmethod
    def _method_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        m = v.strip().upper()
        if m not in _HTTP_METHODS:
            raise ValueError(f"method 须为 {sorted(_HTTP_METHODS)} 之一")
        return m


class AgentRegisterMcpToolBody(BaseModel):
    """创建并持久化 MCP 工具（落库 + 进程内注册占位）。"""

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",
        description="与 tool_names 项一致；同名则冲突",
    )
    description: str = Field(min_length=1, max_length=512)
    transport_type: Literal["http", "sse"] = Field(
        default="http",
        description="MCP 传输类型：http=Streamable HTTP（POST）；sse=HTTP+SSE（典型 /sse）；与 connection_config 配合",
    )
    server_url: str = Field(description="MCP 服务端 URL，须为 http:// 或 https://")
    connection_config: dict[str, Any] | None = Field(
        default=None,
        description="除 server_url / transport_type 外的连接扩展（JSON 对象）",
    )
    tools_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "工具级扩展配置（JSON 对象）。MCP（transport_type=http）可设置 "
            "`mcp_tool_name` 为远端 `tools/call` 的 `name`（与 `tools/list` 中一致）。"
            "未设置时后端使用本平台注册的 `name` 作为远端名；若与远端不一致须显式填写。"
            " **推荐** 设置 `dispatch_mode: true`（亦可放在请求体顶层合并进 `mcp_config`）："
            " 注册为调度工具，须先 `step=list_tools` 拉取 `tools/list`，再 `step=call_tool`；"
            " 服务端不校验参数 JSON Schema，由大模型按 `tools/list` 结果填参。"
        ),
    )
    enabled: bool = Field(default=True, description="是否启用；仅启用的工具在启动时重放注册")
    input_schema: dict[str, Any] | None = Field(
        default=None, description="工具入参 JSON Schema（可选）"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="工具出参 JSON Schema（可选）"
    )

    @field_validator("name", "description", mode="after")
    @classmethod
    def _strip_required_mcp(cls, v: str) -> str:
        return v.strip()

    @field_validator("server_url", mode="after")
    @classmethod
    def _mcp_server_url(cls, v: str) -> str:
        s = v.strip()
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError("server_url 须为 http:// 或 https://")
        return s


class AgentUpdateMcpToolBody(BaseModel):
    """更新持久化 MCP 工具（仅提交需修改的字段）。"""

    description: str | None = Field(default=None, max_length=512)
    transport_type: Literal["http", "sse"] | None = None
    server_url: str | None = None
    connection_config: dict[str, Any] | None = None
    tools_config: dict[str, Any] | None = Field(
        default=None,
        description="同创建接口；合并更新 MCP 工具时设置 `mcp_tool_name` 等",
    )
    enabled: bool | None = Field(default=None, description="是否启用")
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    version: int | None = Field(
        default=None,
        ge=1,
        description="乐观锁：传入须与当前库中 version 一致，否则 409 冲突",
    )
    clear_connection_config: bool = Field(
        default=False, description="为 true 时清空已存 connection_config"
    )
    clear_tools_config: bool = Field(default=False, description="为 true 时清空已存 tools_config")

    @field_validator("description", mode="after")
    @classmethod
    def _strip_desc_mcp(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip()

    @field_validator("server_url", mode="after")
    @classmethod
    def _mcp_server_url_opt(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError("server_url 须为 http:// 或 https://")
        return s


class AgentExternalToolPersistedOut(BaseModel):
    """库表中的一条可配置外部工具（HTTP 或 MCP；管理接口返回）。"""

    id: int
    name: str
    description: str
    kind: Literal["http", "mcp"]
    enabled: bool
    version: int
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    url: str | None = None
    method: str | None = None
    headers: dict[str, str] | None = None
    body: str | None = None
    timeout_ms: int | None = None
    timeout_seconds: float | None = None
    transport_type: str | None = None
    server_url: str | None = None
    connection_config: dict[str, Any] | None = None
    tools_config: dict[str, Any] | None = None
    mcp_config: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class AgentRuntimeExternalToolsListData(BaseModel):
    items: list[AgentExternalToolPersistedOut]


# 兼容旧 OpenAPI/调用方命名
AgentHttpToolPersistedOut = AgentExternalToolPersistedOut
AgentRuntimeHttpToolsListData = AgentRuntimeExternalToolsListData


class AgentToolRegisterResultData(BaseModel):
    name: str


class McpProbeListToolsBody(BaseModel):
    """不落库，仅探测远端 MCP ``tools/list``（Streamable HTTP 或 HTTP+SSE）。"""

    server_url: str = Field(description="MCP 服务端 URL")
    transport_type: Literal["http", "sse"] = Field(
        default="http",
        description="http=Streamable POST；sse=HTTP+SSE（如 /sse）",
    )
    connection_config: dict[str, Any] | None = Field(
        default=None,
        description="与持久化 MCP 一致：可含 protocol_version、headers、verify_ssl、timeout_seconds 等",
    )

    @field_validator("server_url", mode="after")
    @classmethod
    def _mcp_probe_url(cls, v: str) -> str:
        s = v.strip()
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError("server_url 须为 http:// 或 https://")
        return s


class McpProbeListToolsData(BaseModel):
    tools: list[dict[str, Any]] = Field(description="``tools/list`` 返回的 tools 数组")
