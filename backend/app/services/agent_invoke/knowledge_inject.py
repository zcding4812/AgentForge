"""invoke 前：按 Agent 绑定的知识库检索用户问题，拼入 ``PromptSlots.context``。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.agent.kernel import PromptSlots
from app.domain.agent.knowledge_binding import AgentKnowledgeBindingSettings
from app.knowledge.facades import KnowledgeRetrievalFacade
from app.repositories.knowledge_repo import KnowledgeBaseRepository
from app.schemas.agent import AgentInvokeRequest, KnowledgeInvokeCitation
from app.schemas.knowledge import KnowledgeSearchData, RetrievalType

if TYPE_CHECKING:
    from app.infrastructure.db import SQLAlchemyDatabaseManager
    from app.repositories.chunk_repo import ChunkRepository

logger = logging.getLogger(__name__)

_KB_CTX_HEADER = "【以下片段由知识库根据当前用户问题自动检索，供参考；与问题无关可忽略】"


def _override_to_api(
    s: AgentKnowledgeBindingSettings,
) -> RetrievalType | None:
    if s.retrieval_override == "inherit":
        return None
    return s.retrieval_override  # keyword | vector | hybrid


def _format_kb_block(
    *,
    kb_id: int,
    kb_name: str,
    data: KnowledgeSearchData,
    show_sources: bool,
) -> str:
    lines: list[str] = [f"### 知识库「{kb_name}」", ""]
    if data.note:
        lines.append(f"*说明：{data.note}*")
        lines.append("")
    if not data.items:
        lines.append("*（无匹配分片）*")
        return "\n".join(lines)
    for i, hit in enumerate(data.items, start=1):
        fn = hit.filename or "—"
        snip = (hit.text_snippet or "").strip()
        if not snip:
            continue
        chunk_part = hit.chunk_index if hit.chunk_index is not None else "—"
        if show_sources:
            lines.append(
                f"{i}. **{fn}**  \n{snip}\n"
                f"*来源：知识库「{kb_name}」· id={kb_id} · 文档 id={hit.doc_id} · 分片序号={chunk_part}*",
            )
        else:
            lines.append(f"{i}. **{fn}**  \n{snip}")
    return "\n".join(lines)


def _hits_to_citations(
    *,
    kb_id: int,
    kb_name: str,
    data: KnowledgeSearchData,
) -> list[KnowledgeInvokeCitation]:
    out: list[KnowledgeInvokeCitation] = []
    for hit in data.items:
        snip = (hit.text_snippet or "").strip()
        if not snip:
            continue
        out.append(
            KnowledgeInvokeCitation(
                kb_id=kb_id,
                kb_name=kb_name,
                doc_id=hit.doc_id,
                filename=hit.filename,
                chunk_index=hit.chunk_index,
                text_snippet=snip[:4000] if len(snip) > 4000 else snip,
                match_type=hit.match_type,
            ),
        )
    return out


async def enrich_prompt_slots_with_knowledge(
    db_manager: SQLAlchemyDatabaseManager | None,
    chunk_repository: ChunkRepository | None,
    body: AgentInvokeRequest,
    slots: PromptSlots | None,
    knowledge: AgentKnowledgeBindingSettings,
) -> tuple[PromptSlots | None, tuple[KnowledgeInvokeCitation, ...]]:
    if db_manager is None or chunk_repository is None:
        return slots, ()
    if not knowledge.enabled or not knowledge.knowledge_base_ids:
        return slots, ()
    q = body.user_message.strip()
    if not q:
        return slots, ()

    facade = KnowledgeRetrievalFacade(db_manager, chunk_repository)
    ro = _override_to_api(knowledge)
    blocks: list[str] = []
    citations: list[KnowledgeInvokeCitation] = []
    for kb_id in knowledge.knowledge_base_ids:
        kb_row = await KnowledgeBaseRepository.get_active_by_id(kb_id, db_manager=db_manager)
        if kb_row is None:
            logger.info("知识库绑定：跳过不存在或已删除的知识库 | kb_id=%s", kb_id)
            continue
        kb_name = (kb_row.name or "").strip() or f"#{kb_id}"
        try:
            data = await facade.retrieve(
                kb_id,
                q=q,
                limit=knowledge.top_k,
                retrieval_override=ro,
            )
        except Exception:
            logger.exception("知识库绑定：检索失败 | kb_id=%s", kb_id)
            continue
        citations.extend(_hits_to_citations(kb_id=kb_id, kb_name=kb_name, data=data))
        blocks.append(
            _format_kb_block(
                kb_id=kb_id,
                kb_name=kb_name,
                data=data,
                show_sources=knowledge.show_sources,
            ),
        )

    if not blocks:
        return slots, tuple(citations)

    injected = "\n\n".join([_KB_CTX_HEADER, *blocks])
    manual = (slots.context if slots else None) or ""
    manual = manual.strip()
    merged = f"{manual}\n\n{injected}" if manual else injected

    if slots is None:
        return (
            PromptSlots(
                system_prompt=None,
                context=merged,
                rolling_summary=None,
                omit_platform_default_system=False,
                history_turns=(),
                max_history_rounds=10,
                max_history_tokens=None,
            ),
            tuple(citations),
        )

    return (
        PromptSlots(
            system_prompt=slots.system_prompt,
            context=merged,
            rolling_summary=slots.rolling_summary,
            omit_platform_default_system=slots.omit_platform_default_system,
            workbench_workspace_agent_catalog=slots.workbench_workspace_agent_catalog,
            history_turns=slots.history_turns,
            max_history_rounds=slots.max_history_rounds,
            max_history_tokens=slots.max_history_tokens,
        ),
        tuple(citations),
    )
