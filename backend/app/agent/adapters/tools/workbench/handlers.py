"""LangChain 工具协程入口：薄适配层 + ``StructuredTool`` 工厂 / 外观。"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from app.agent.adapters.tools.workbench.context import require_workbench_runtime
from app.agent.adapters.tools.workbench.resource_services import (
    WorkbenchAgentResourceService,
    WorkbenchCatalogResourceService,
    WorkbenchKnowledgeResourceService,
)
from app.agent.adapters.tools.workbench.sub_agent_invoke import (
    run_parallel_sub_agent_invocations,
    run_sub_agent_invocation,
)
from app.agent.kernel.workbench_orchestration import (
    WORKBENCH_INVOKE_SUB_AGENT_TOOL,
    WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL,
)


async def list_agents(page: int = 1, page_size: int = 20, q: str | None = None):
    ctx = require_workbench_runtime()
    return await WorkbenchAgentResourceService(ctx).list_agents(page=page, page_size=page_size, q=q)


async def get_agent(agent_id: int):
    ctx = require_workbench_runtime()
    return await WorkbenchAgentResourceService(ctx).get_agent(agent_id)


async def update_agent(agent_id: int, patch_json: str):
    ctx = require_workbench_runtime()
    return await WorkbenchAgentResourceService(ctx).update_agent(agent_id, patch_json)


async def create_agent(create_json: str):
    ctx = require_workbench_runtime()
    return await WorkbenchAgentResourceService(ctx).create_agent(create_json)


async def invoke_sub_agent(
    agent_id: int,
    user_message: str,
    parent_messages: str | None = None,
):
    ctx = require_workbench_runtime()
    return await run_sub_agent_invocation(ctx, agent_id, user_message, parent_messages)


async def invoke_sub_agents_parallel(tasks_json: str):
    ctx = require_workbench_runtime()
    return await run_parallel_sub_agent_invocations(ctx, tasks_json)


async def list_knowledge_bases(
    page: int = 1,
    page_size: int = 20,
    q: str | None = None,
    status: str | None = None,
):
    ctx = require_workbench_runtime()
    return await WorkbenchKnowledgeResourceService(ctx).list_knowledge_bases(
        page=page, page_size=page_size, q=q, status=status
    )


async def create_knowledge_base(
    name: str,
    description: str | None = None,
    retrieval_type: str = "keyword",
    storage_type: str = "keyword",
    embedding_model_config_id: int | None = None,
):
    ctx = require_workbench_runtime()
    return await WorkbenchKnowledgeResourceService(ctx).create_knowledge_base(
        name,
        description=description,
        retrieval_type=retrieval_type,
        storage_type=storage_type,
        embedding_model_config_id=embedding_model_config_id,
    )


async def list_registered_tools(namespace: str = "default"):
    ctx = require_workbench_runtime()
    return await WorkbenchCatalogResourceService(ctx).list_registered_tools(namespace=namespace)


async def list_external_tools():
    ctx = require_workbench_runtime()
    return await WorkbenchCatalogResourceService(ctx).list_external_tools()


async def list_sys_models(
    page: int = 1,
    page_size: int = 20,
    q: str | None = None,
    model_type: str = "chat",
):
    ctx = require_workbench_runtime()
    return await WorkbenchCatalogResourceService(ctx).list_sys_models(
        page=page, page_size=page_size, q=q, model_type=model_type
    )


class WorkbenchToolFactory:
    """工厂：生成全部 ``workbench_*`` LangChain 工具实例。

    ``StructuredTool.from_function`` 会根据协程的**类型注解**生成 JSON Schema（模型填参时可见字段名/类型）；
    下述 ``description`` 为面向模型的自然语言说明，并补充「入参」摘要以便与 Schema 对照。
    """

    @staticmethod
    def create_all() -> tuple[StructuredTool, ...]:
        return (
            StructuredTool.from_function(
                coroutine=list_agents,
                name="workbench_list_agents",
                description=(
                    "列出当前工作区（workspace_namespace）下的 Agent 分页列表；可选 q 模糊搜索。"
                    "返回结构化 JSON：code/message/data，data 含 total、items。"
                    "入参：page（页码，默认 1）、page_size（每页条数，默认 20）、q（可选，模糊匹配名称等）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=get_agent,
                name="workbench_get_agent",
                description=(
                    "按数字 id 获取 Agent 详情；仅同命名空间可见。"
                    "返回 code/message/data；失败时 code 非 OK。"
                    "入参：agent_id（整数，目标 Agent 主键）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=update_agent,
                name="workbench_update_agent",
                description=(
                    "更新同命名空间内 Agent 元数据与配置，语义与 HTTP PATCH /api/agents/{agent_id} 的 AgentUpdateBody 一致。"
                    "patch_json 为 JSON 对象字符串，可含任意已定义字段的子集：name、description、agent_kind、"
                    "sys_model_id（null 表示取消挂载）、system_prompt（null 清空）、config_json（全量替换）。"
                    "至少须包含一项可更新字段；成功时 data 为更新后的 Agent 详情。"
                    "入参：agent_id（整数）、patch_json（字符串，合法 JSON 对象）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=create_agent,
                name="workbench_create_agent",
                description=(
                    "在当前工作区命名空间新建普通 Agent（语义与 POST /api/agents 的 AgentCreateBody 一致）。"
                    "create_json 为 JSON 对象字符串，须含 name、agent_kind；"
                    "**须含非空 system_prompt，或非空 config_json.prompt_system（二选一或同时）**。"
                    "**应在 create_json 中包含 sys_model_id（整数）**：创建前**先**调用 `workbench_list_sys_models`（默认 model_type=chat），从返回的**启用**项中选取 id 写入；使子 Agent 创建后即可被委派。**省略 sys_model_id 时子 Agent 可能无法完成推理调用**；禁止编造模型 id。"
                    "建议 config_json 与工作区一致：含 v、temperature、max_tokens、tool_names、memory、knowledge、"
                    "prompt_system、stream_output 等键。"
                    "可选 description。workspace_namespace 由平台固定为当前工作台，传入也会被覆盖；禁止 agent_kind=workbench。"
                    "成功时 data 为新 Agent 摘要；校验失败见 VALIDATION_ERROR。"
                    "入参：create_json（单字符串，内容为 JSON 对象）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=invoke_sub_agent,
                name=WORKBENCH_INVOKE_SUB_AGENT_TOOL,
                description=(
                    "调用同命名空间内另一 Agent 完成子任务；user_message 为发给子 Agent 的指令。"
                    "可选 parent_messages：JSON 数组字符串；每轮元素为 ChatHistoryTurn（键 user、assistant，可选 tool_*），"
                    "或 OpenAI 风格（键 role、content，role 取 user/assistant/tool）。"
                    "成功时 data 含 assistant_text、structured、tokens、sub_agent_id 等；失败见 code（如 UNAVAILABLE、INVOKE_FAILED）。"
                    "超长正文可能带 result_truncated。勿跨命名空间。"
                    "入参：agent_id（整数，子 Agent 主键）、user_message（字符串）、parent_messages（可选字符串）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=invoke_sub_agents_parallel,
                name=WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL,
                description=(
                    "在同一轮内**并发**调用多个子 Agent（平台用 asyncio 并行执行）。"
                    "参数 tasks_json：JSON **数组**字符串，每项为对象，须含 agent_id（整数）、user_message（字符串）；"
                    "可选 parent_messages（字符串，语义与同名单次调用一致）。"
                    "成功时 data.parallel 为 true，data.items 为与 tasks 顺序一致的结果列表，每项含 code/message/data（与单次调用语义一致）；"
                    "任一项失败时该项 code 非 OK，其它项仍可能成功。任务数上限见平台常量（通常 ≤8）。"
                    "若子任务相互独立、可并行缩短延迟，优先用本工具而非多次 workbench_invoke_sub_agent。"
                    "入参：tasks_json（单字符串，内容为 JSON 数组，元素字段见上文）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=list_knowledge_bases,
                name="workbench_list_knowledge_bases",
                description=(
                    "分页列出当前工作区命名空间下的知识库（只读）；可选 q、status。"
                    "data 含 total、page、page_size、items。"
                    "入参：page、page_size、q（可选）、status（可选）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=create_knowledge_base,
                name="workbench_create_knowledge_base",
                description=(
                    "在当前工作区创建知识库元数据（name 必填；向量/混合检索须填 embedding_model_config_id）。"
                    "成功时 data 为知识库详情；失败见 code（如 CREATE_FAILED、VALIDATION_ERROR）。"
                    "入参：name（字符串）、description（可选）、retrieval_type（默认 keyword）、"
                    "storage_type（默认 keyword）、embedding_model_config_id（可选整数）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=list_registered_tools,
                name="workbench_list_registered_tools",
                description=(
                    "列出平台进程内可绑定的工具名与描述（内置 + 持久化外部工具并集），供为子 Agent 配置 tool_names 时参考。"
                    "可选 namespace（默认 default）。data 与 GET /api/agents/tools 一致。"
                    "入参：namespace（可选字符串，工具注册命名空间，默认 default）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=list_external_tools,
                name="workbench_list_external_tools",
                description=(
                    "列出库表中持久化的 HTTP/MCP 外部工具配置摘要，便于了解可注册到 Agent 的远端工具。"
                    "data.items 与 GET /api/agents/tools/http 一致。"
                    "入参：无（不传参）。"
                ),
            ),
            StructuredTool.from_function(
                coroutine=list_sys_models,
                name="workbench_list_sys_models",
                description=(
                    "分页列出平台已配置的 sys_model（与 GET /api/providers/models 一致）。"
                    "**编排新建子 Agent 时请先调用本工具**，用返回中**启用**项的 id 作为 `create_json.sys_model_id`（整数）；向量/混合知识库需要 embedding 时可传 model_type=embedding。"
                    "默认 model_type=chat（对话类 llm/tts/stt）；传 all 不过滤类型。data 含 items 与 meta（分页）。"
                    "入参：page、page_size、q（可选）、model_type（默认 chat，可 embedding / all 等）。"
                ),
            ),
        )


class WorkbenchToolsFacade:
    """外观：对外提供工具元组与名称集合（隐藏工厂细节）。"""

    _instances: tuple[StructuredTool, ...] | None = None

    @classmethod
    def tool_instances(cls) -> tuple[StructuredTool, ...]:
        if cls._instances is None:
            cls._instances = WorkbenchToolFactory.create_all()
        return cls._instances

    @classmethod
    def tool_names(cls) -> frozenset[str]:
        return frozenset(t.name for t in cls.tool_instances())
