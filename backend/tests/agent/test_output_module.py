import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.adapters.output.module import AgentOutputModule
from app.agent.kernel.spec import OutputControl, ResponseConstraints


def test_finalize_strips_thinking_tags() -> None:
    mod = AgentOutputModule()
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="<thinking>secret</thinking>visible answer",
        ),
    ]
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(strip_thinking_blocks=True),
    )
    assert "secret" not in out.text
    assert "visible answer" in out.text
    assert out.thinking_text == "secret"


def test_finalize_json_object_parses_structured() -> None:
    mod = AgentOutputModule()
    messages = [AIMessage(content='{"a": 1}')]
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="json_object"),
        OutputControl(),
    )
    assert out.structured == {"a": 1}


def test_finalize_process_trace_excludes_history_assistants() -> None:
    """多轮会话：process_trace 不得把上一轮 assistant 正文再记进本轮。"""
    mod = AgentOutputModule()
    messages = [
        HumanMessage(content="第一问"),
        AIMessage(content="第一答"),
        HumanMessage(content="第二问"),
        AIMessage(content="本轮规划"),
        AIMessage(content="本轮最终"),
    ]
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
    )
    assert out.text == "本轮最终"
    assert out.process_trace is not None
    assert len(out.process_trace) == 2
    assert out.process_trace[0].text == "本轮规划"
    assert "第一答" not in (out.process_trace[0].text + out.process_trace[1].text)


def test_finalize_process_trace_multi_assistant() -> None:
    mod = AgentOutputModule()
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="先规划一步",
            tool_calls=[
                {"id": "c1", "name": "noop", "args": {}, "type": "tool_call"},
            ],
        ),
        ToolMessage(content="{}", tool_call_id="c1", name="noop"),
        AIMessage(content="最终答复"),
    ]
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
    )
    assert out.text == "最终答复"
    assert out.process_trace is not None
    assert len(out.process_trace) == 4
    assert out.process_trace[0].seq == 0
    assert out.process_trace[0].kind == "model"
    assert out.process_trace[0].phase == "step"
    assert out.process_trace[0].text == "先规划一步"
    assert out.process_trace[1].kind == "tool_call"
    assert out.process_trace[1].name == "noop"
    assert out.process_trace[2].kind == "tool_result"
    assert out.process_trace[2].name == "noop"
    assert out.process_trace[3].kind == "model"
    assert out.process_trace[3].phase == "final"
    assert out.process_trace[3].text == "最终答复"


def test_finalize_observability_empty_list_falls_back_to_suffix() -> None:
    """门面传入空切片时，应回退为末条 Human 之后，避免正文与 trace 丢失。"""
    mod = AgentOutputModule()
    full = [
        HumanMessage(content="u0"),
        AIMessage(content="历史答"),
        HumanMessage(content="子任务"),
        AIMessage(content="子答复"),
    ]
    out = mod.finalize(
        full,
        ResponseConstraints(response_format="text"),
        OutputControl(),
        observability_messages=[],
    )
    assert out.text == "子答复"
    assert out.process_trace is not None
    assert len(out.process_trace) == 1
    assert "历史答" not in out.process_trace[0].text


def test_finalize_process_trace_single_assistant_one_step() -> None:
    mod = AgentOutputModule()
    out = mod.finalize(
        [HumanMessage(content="x"), AIMessage(content="only")],
        ResponseConstraints(response_format="text"),
        OutputControl(),
    )
    assert out.text == "only"
    assert out.process_trace is not None
    assert len(out.process_trace) == 1
    assert out.process_trace[0].phase == "final"
    assert out.process_trace[0].text == "only"


def test_finalize_passes_workbench_fan_in() -> None:
    mod = AgentOutputModule()
    messages = [AIMessage(content="ok")]
    fan_in = {
        "total_calls": 2,
        "success_calls": 2,
        "failed_calls": 0,
        "unique_child_agent_ids": [1, 2],
        "items": [],
    }
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
        workbench_fan_in=fan_in,
    )
    assert out.workbench_fan_in == fan_in


def test_finalize_workbench_falls_back_when_aggregate_is_meta_tail() -> None:
    """aggregate 仅输出『任务完成』式元话语时，对外正文应回退到前文最详实步骤产出。"""
    mod = AgentOutputModule()
    long_detail = "步骤一详情\n" + ("x" * 720)
    short_meta = "本答复严格对齐用户字面诉求，已完成精准交付。如您后续需要代码片段，请随时提出。"
    messages = [
        HumanMessage(content="查询流程"),
        AIMessage(content=long_detail),
        AIMessage(content=short_meta),
    ]
    fan_in = {
        "total_calls": 1,
        "success_calls": 1,
        "failed_calls": 0,
        "unique_child_agent_ids": [1],
        "items": [],
    }
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
        workbench_fan_in=fan_in,
    )
    assert long_detail in out.text
    assert "已完成精准交付" not in out.text


def test_finalize_workbench_fallback_appends_substantive_aggregate_tail() -> None:
    """触发回退时，若 aggregate 末条仍有实质补充（非短元话术），应拼接到最终正文。"""
    mod = AgentOutputModule()
    long_detail = "主体步骤\n" + ("a" * 620)
    # 长度触发 ratio_short（相对前文足够短），且不命中元话术正则
    useful_tail = (
        "补充：仅在 new_opportunitytype 为渠道报备（5）时触发审批；"
        "其余类型不走 new_flowsubmitauto。" + ("x" * 40)
    )
    assert len(useful_tail) >= 80
    messages = [
        HumanMessage(content="q"),
        AIMessage(content=long_detail),
        AIMessage(content=useful_tail),
    ]
    fan_in = {
        "total_calls": 1,
        "success_calls": 1,
        "failed_calls": 0,
        "unique_child_agent_ids": [1],
        "items": [],
    }
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
        workbench_fan_in=fan_in,
    )
    assert long_detail in out.text
    assert useful_tail in out.text
    assert "\n---\n" in out.text


def test_finalize_tool_history_call_result_pair() -> None:
    mod = AgentOutputModule()
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_ab", "name": "demo_tool", "args": {"x": 1}, "type": "tool_call"},
            ],
        ),
        ToolMessage(content='{"ok":true}', tool_call_id="call_ab", name="demo_tool"),
        AIMessage(content="最终"),
    ]
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
    )
    assert out.tool_history is not None
    assert len(out.tool_history) == 2
    assert out.tool_history[0]["phase"] == "call"
    assert out.tool_history[0]["name"] == "demo_tool"
    assert out.tool_history[0]["arguments"] == {"x": 1}
    assert out.tool_history[1]["phase"] == "result"
    assert out.tool_history[1]["content"] == '{"ok":true}'
    assert out.process_trace is not None
    assert len(out.process_trace) == 4
    assert out.process_trace[0].kind == "model"
    assert "见下列工具步骤" in out.process_trace[0].text
    assert out.process_trace[1].kind == "tool_call"
    assert out.process_trace[1].name == "demo_tool"
    assert out.process_trace[2].kind == "tool_result"
    assert out.process_trace[3].text == "最终"


def test_finalize_expands_workbench_sub_agent_nested_history() -> None:
    """父级 finalize 应在 workbench_invoke_sub_agent 结果前插入子 Agent 的 sub_tool_history。"""
    mod = AgentOutputModule()
    inner = {
        "assistant_text": "子答",
        "sub_agent_id": 7,
        "sub_agent_kind": "react",
        "sub_tool_history": [
            {"phase": "call", "name": "inner", "arguments": {}, "tool_call_id": "k"},
            {"phase": "result", "name": "inner", "content": "{}", "tool_call_id": "k"},
        ],
    }
    envelope = {"code": "OK", "message": "m", "data": inner}
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "w",
                    "name": "workbench_invoke_sub_agent",
                    "args": {"agent_id": 7, "user_message": "hi"},
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            content=json.dumps(envelope, ensure_ascii=False),
            tool_call_id="w",
            name="workbench_invoke_sub_agent",
        ),
        AIMessage(content="汇总"),
    ]
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
    )
    assert out.tool_history is not None
    assert len(out.tool_history) == 4
    assert "[子Agent #7]" in str(out.tool_history[1].get("name", ""))
    assert out.tool_history[-1].get("name") == "workbench_invoke_sub_agent"


def test_finalize_tool_history_only_after_last_human() -> None:
    mod = AgentOutputModule()
    messages = [
        HumanMessage(content="1"),
        AIMessage(
            content="",
            tool_calls=[{"id": "a", "name": "t", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content="old", tool_call_id="a", name="t"),
        AIMessage(content="prev"),
        HumanMessage(content="2"),
        AIMessage(
            content="",
            tool_calls=[{"id": "b", "name": "t2", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content="new", tool_call_id="b", name="t2"),
        AIMessage(content="final"),
    ]
    out = mod.finalize(
        messages,
        ResponseConstraints(response_format="text"),
        OutputControl(),
    )
    assert out.tool_history is not None
    assert len(out.tool_history) == 2
    assert out.tool_history[0]["name"] == "t2"
    assert out.tool_history[1]["content"] == "new"


def test_finalize_observability_messages_drop_prefix_ai() -> None:
    """仅对「本轮图追加」做观测时，不得把前缀中误混入的其它 AIMessage 记入 process_trace。"""
    mod = AgentOutputModule()
    full = [
        HumanMessage(content="子任务"),
        AIMessage(content="不应出现在子观测里"),
        AIMessage(content="子真实输出一"),
        AIMessage(content="子真实输出二"),
    ]
    obs = [
        AIMessage(content="子真实输出一"),
        AIMessage(content="子真实输出二"),
    ]
    out_default = mod.finalize(
        full,
        ResponseConstraints(response_format="text"),
        OutputControl(),
    )
    assert out_default.process_trace is not None
    assert any("不应出现" in step.text for step in out_default.process_trace)

    out_slice = mod.finalize(
        full,
        ResponseConstraints(response_format="text"),
        OutputControl(),
        observability_messages=obs,
    )
    assert out_slice.process_trace is not None
    blob = "".join(s.text for s in out_slice.process_trace)
    assert "不应出现" not in blob
    assert "子真实输出一" in blob
    assert "子真实输出二" in blob
