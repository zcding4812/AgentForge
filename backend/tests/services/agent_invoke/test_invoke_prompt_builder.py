"""``InvokePromptBuilder``：与 DB 历史合并及 cap 行为。"""

from app.agent.kernel import HistoryTurnSlot, PromptSlots
from app.agent.kernel.spec import AgentKind
from app.schemas.agent import AgentInvokeRequest, PromptEngineeringBody
from app.services.agent_invoke.invoke_memory import InvokePromptBuilder


def _minimal_request(**kwargs: object) -> AgentInvokeRequest:
    base = dict(
        agent_kind=AgentKind.SIMPLE_CHAT,
        user_message="hello",
    )
    base.update(kwargs)
    return AgentInvokeRequest.model_validate(base)


def test_compose_with_db_history_no_prompts_only_db_turns() -> None:
    body = _minimal_request(prompts=None)
    db_turns = (HistoryTurnSlot(user="u1", assistant="a1"),)
    out = InvokePromptBuilder.compose_with_db_history(
        body,
        None,
        db_turns,
        max_rounds_cap=20,
    )
    assert out is not None
    assert out.history_turns == db_turns
    assert out.system_prompt is None
    assert out.context is None
    assert out.max_history_rounds == 10


def test_compose_with_db_history_clamps_max_to_cap() -> None:
    body = _minimal_request(
        prompts=PromptEngineeringBody(
            system_prompt="s",
            max_history_rounds=30,
        ),
    )
    db_turns = (HistoryTurnSlot(user="u", assistant="a"),)
    out = InvokePromptBuilder.compose_with_db_history(
        body,
        None,
        db_turns,
        max_rounds_cap=5,
    )
    assert out is not None
    assert out.max_history_rounds == 5  # min(30, cap=5)


def test_compose_with_db_history_merges_prev_slots_system() -> None:
    body = _minimal_request(prompts=PromptEngineeringBody(context="c"))
    prev = PromptSlots(system_prompt="sys", context=None, history_turns=(), max_history_rounds=3)
    db_turns = (HistoryTurnSlot(user="u", assistant="a"),)
    out = InvokePromptBuilder.compose_with_db_history(
        body,
        prev,
        db_turns,
        max_rounds_cap=20,
    )
    assert out is not None
    assert out.system_prompt == "sys"
    assert out.context == "c"
    assert out.history_turns == db_turns


def test_compose_with_db_history_empty_db_no_prompts_returns_none() -> None:
    body = _minimal_request(prompts=None)
    out = InvokePromptBuilder.compose_with_db_history(
        body,
        None,
        (),
        max_rounds_cap=20,
    )
    assert out is None


def test_compose_with_db_history_rolling_summary_only_no_db_turns() -> None:
    body = _minimal_request(prompts=None)
    out = InvokePromptBuilder.compose_with_db_history(
        body,
        None,
        (),
        max_rounds_cap=20,
        rolling_summary="早前讨论了登录流程",
    )
    assert out is not None
    assert out.rolling_summary == "早前讨论了登录流程"
    assert out.history_turns == ()
    assert out.system_prompt is None
