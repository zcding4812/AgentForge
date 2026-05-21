from app.domain.agent import (
    DEFAULT_ALLOW_CLIENT_CHAT_HISTORY,
    DEFAULT_COMPRESS_BATCH_ROUNDS,
    DEFAULT_MAX_HISTORY_ROUNDS_CAP,
    DEFAULT_MIN_TAIL_RAW_ROUNDS,
    DEFAULT_SUMMARY_CONTEXT_RATIO_THRESHOLD,
    DEFAULT_SUMMARY_CONTEXT_RATIO_URGENT,
    DEFAULT_SUMMARY_CONTEXT_WINDOW_TOKENS,
    DEFAULT_SUMMARY_MIN_ROUNDS_SINCE_LAST,
    DEFAULT_SUMMARY_MIN_TOKENS_SINCE_LAST,
    AgentMemorySettingsParser,
    parse_agent_memory_settings,
)


def test_defaults_when_empty() -> None:
    s = AgentMemorySettingsParser().parse(None)
    assert s.max_history_rounds_cap == DEFAULT_MAX_HISTORY_ROUNDS_CAP
    assert s.allow_client_chat_history is DEFAULT_ALLOW_CLIENT_CHAT_HISTORY
    assert s.summarization_sys_model_id is None
    assert s.summary_context_ratio_threshold == DEFAULT_SUMMARY_CONTEXT_RATIO_THRESHOLD
    assert s.summary_context_ratio_urgent == DEFAULT_SUMMARY_CONTEXT_RATIO_URGENT
    assert s.summary_min_rounds_since_last == DEFAULT_SUMMARY_MIN_ROUNDS_SINCE_LAST
    assert s.summary_min_tokens_since_last == DEFAULT_SUMMARY_MIN_TOKENS_SINCE_LAST
    assert s.summary_context_window_tokens == DEFAULT_SUMMARY_CONTEXT_WINDOW_TOKENS
    assert s.min_tail_raw_rounds == DEFAULT_MIN_TAIL_RAW_ROUNDS
    assert s.compress_batch_rounds == DEFAULT_COMPRESS_BATCH_ROUNDS


def test_memory_block() -> None:
    s = AgentMemorySettingsParser().parse(
        {"memory": {"max_history_rounds_cap": 12, "allow_client_chat_history": True}},
    )
    assert s.max_history_rounds_cap == 12
    assert s.allow_client_chat_history is True


def test_memory_full_block() -> None:
    s = AgentMemorySettingsParser().parse(
        {
            "memory": {
                "max_history_rounds_cap": 30,
                "allow_client_chat_history": True,
                "summarization_sys_model_id": 5,
                "summary_context_ratio_threshold": 0.35,
                "summary_context_ratio_urgent": 0.72,
                "summary_min_rounds_since_last": 15,
                "summary_min_tokens_since_last": 800,
            },
        },
    )
    assert s.max_history_rounds_cap == 30
    assert s.summarization_sys_model_id == 5
    assert abs(s.summary_context_ratio_threshold - 0.35) < 1e-9
    assert abs(s.summary_context_ratio_urgent - 0.72) < 1e-9
    assert s.summary_min_rounds_since_last == 15
    assert s.summary_min_tokens_since_last == 800


def test_cap_clamped() -> None:
    s = AgentMemorySettingsParser().parse({"memory": {"max_history_rounds_cap": 999}})
    assert s.max_history_rounds_cap == 200


def test_module_level_parse_alias() -> None:
    assert parse_agent_memory_settings(None) == AgentMemorySettingsParser().parse(None)
