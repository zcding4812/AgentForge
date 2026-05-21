"""全量基线：从空库一步到位创建所有表与种子数据。

Revision ID: schema_baseline
Revises:
Create Date: 2026-05-20

新建库执行 ``alembic upgrade head`` 即可。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision: str = "schema_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _trace_dt():
    bind = op.get_bind()
    if bind.dialect.name == "mysql":
        return mysql.DATETIME(fsp=6)
    return sa.DateTime(timezone=False)


def _sys_dt():
    return sa.DateTime(timezone=False)


def _long_text():
    bind = op.get_bind()
    if bind.dialect.name == "mysql":
        return mysql.LONGTEXT()
    return sa.Text()


def _mysql_kw(**extra: str) -> dict:
    return {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", **extra}


def _conv_kw() -> dict:
    return _mysql_kw(**{"mysql_collate": "utf8mb4_unicode_ci"})


# ── upgrade ──────────────────────────────────────────────────────────

def upgrade() -> None:
    dt6 = _trace_dt()
    dt = _sys_dt()
    lt = _long_text()
    ts_default = sa.text("CURRENT_TIMESTAMP")

    bind = op.get_bind()
    is_mysql = bind.dialect.name == "mysql"
    ts_update = sa.text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP") if is_mysql else ts_default

    # ── trace_runs ───────────────────────────────────────────────────

    op.create_table(
        "trace_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("trace_id", sa.String(32), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("trace_name", sa.String(128), nullable=False),
        sa.Column("service_name", sa.String(64), nullable=False),
        sa.Column("http_method", sa.String(16), server_default=sa.text("'GET'"), nullable=False),
        sa.Column("http_path", sa.String(255), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("started_at", dt6, nullable=False),
        sa.Column("ended_at", dt6, nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("agent_task_id", sa.String(64), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.String(1024), nullable=True),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_runs_trace_id", "trace_runs", ["trace_id"], unique=True)
    op.create_index("ix_trace_runs_status", "trace_runs", ["status"])
    op.create_index("ix_trace_runs_session_id", "trace_runs", ["session_id"])
    op.create_index("ix_trace_runs_agent_task_id", "trace_runs", ["agent_task_id"])

    # ── trace_spans ──────────────────────────────────────────────────

    op.create_table(
        "trace_spans",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("trace_id", sa.String(32), nullable=False),
        sa.Column("span_id", sa.String(16), nullable=False),
        sa.Column("parent_span_id", sa.String(16), nullable=True),
        sa.Column("span_name", sa.String(128), nullable=False),
        sa.Column("span_type", sa.String(32), nullable=False),
        sa.Column("component", sa.String(64), nullable=True),
        sa.Column("service_name", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("started_at", dt6, nullable=False),
        sa.Column("ended_at", dt6, nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False),
        sa.Column("is_slow", sa.Boolean(), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.String(1024), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_spans_trace_id", "trace_spans", ["trace_id"])
    op.create_index("ix_trace_spans_span_id", "trace_spans", ["span_id"], unique=True)
    op.create_index("ix_trace_spans_parent_span_id", "trace_spans", ["parent_span_id"])
    op.create_index("ix_trace_spans_status", "trace_spans", ["status"])

    # ── trace_span_logs ──────────────────────────────────────────────

    op.create_table(
        "trace_span_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("trace_id", sa.String(32), nullable=False),
        sa.Column("span_id", sa.String(16), nullable=False),
        sa.Column("log_level", sa.String(16), nullable=False),
        sa.Column("event_name", sa.String(128), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("occurred_at", dt6, nullable=False),
        sa.Column("seq_no", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_span_logs_trace_id", "trace_span_logs", ["trace_id"])
    op.create_index("ix_trace_span_logs_span_id", "trace_span_logs", ["span_id"])

    # ── sys_model_provider ───────────────────────────────────────────

    op.create_table(
        "sys_model_provider",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("provider_code", sa.String(64), nullable=False),
        sa.Column("provider_name", sa.String(64), nullable=False),
        sa.Column("base_url", sa.String(255), nullable=True),
        sa.Column("api_format", sa.String(32), server_default=sa.text("'openai'"), nullable=False),
        sa.Column("auth_type", sa.String(32), server_default=sa.text("'api_key'"), nullable=True),
        sa.Column("api_key", sa.Text(), nullable=True),
        sa.Column("api_secret", sa.Text(), nullable=True),
        sa.Column("status", sa.SmallInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("create_time", dt, server_default=ts_default, nullable=False),
        sa.Column("update_time", dt, server_default=ts_update, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_code", name="sys_model_provider_provider_code_uc"),
        **_mysql_kw(),
    )

    # ── sys_model ────────────────────────────────────────────────────

    op.create_table(
        "sys_model",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("provider_id", sa.BigInteger(), nullable=False),
        sa.Column("model_code", sa.String(128), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("model_type", sa.String(32), nullable=False),
        sa.Column("endpoint", sa.String(255), nullable=True),
        sa.Column("timeout", sa.Integer(), server_default=sa.text("30"), nullable=True),
        sa.Column("is_enabled", sa.SmallInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("create_time", dt, server_default=ts_default, nullable=False),
        sa.Column("update_time", dt, server_default=ts_update, nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["sys_model_provider.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id", "model_code", name="uk_provider_model"),
        **_mysql_kw(),
    )
    op.create_index("ix_sys_model_provider_id", "sys_model", ["provider_id"])

    # ── namespace ────────────────────────────────────────────────────
    # workbench_agent_id FK 在 agent_entity 建表后再加（互相引用）

    op.create_table(
        "namespace",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("workbench_agent_id", sa.BigInteger(), nullable=True),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uk_namespace_slug"),
        sa.UniqueConstraint("workbench_agent_id", name="uk_namespace_workbench_agent_id"),
        **_mysql_kw(),
    )
    op.create_index("ix_namespace_slug", "namespace", ["slug"])

    # ── agent_entity ─────────────────────────────────────────────────

    op.create_table(
        "agent_entity",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("namespace_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.String(512), nullable=True),
        sa.Column("agent_kind", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), server_default=sa.text("'active'"), nullable=False),
        sa.Column("sys_model_id", sa.BigInteger(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.ForeignKeyConstraint(
            ["namespace_id"], ["namespace.id"],
            name="fk_agent_entity_namespace_id", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["sys_model_id"], ["sys_model.id"], ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("namespace_id", "name", name="uk_agent_entity_namespace_name"),
        **_mysql_kw(),
    )
    op.create_index("ix_agent_entity_namespace_id", "agent_entity", ["namespace_id"])
    op.create_index("ix_agents_name", "agent_entity", ["name"])
    op.create_index("ix_agents_agent_kind", "agent_entity", ["agent_kind"])
    op.create_index("ix_agents_status", "agent_entity", ["status"])
    op.create_index("ix_agents_sys_model_id", "agent_entity", ["sys_model_id"])

    # namespace.workbench_agent_id → agent_entity.id（解决循环引用）
    op.create_foreign_key(
        "fk_namespace_workbench_agent_id",
        "namespace", "agent_entity",
        ["workbench_agent_id"], ["id"],
        ondelete="SET NULL",
    )

    # ── conversation_session ─────────────────────────────────────────

    op.create_table(
        "conversation_session",
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("agent_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("status", sa.SmallInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.Column("deleted_at", dt, nullable=True),
        sa.Column("summary", lt, nullable=True),
        sa.Column("summary_version", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("summary_job_status", sa.SmallInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("summary_anchor_turn_index", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agent_entity.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
        **_conv_kw(),
    )
    op.create_index("ix_agent_conv_session_agent_id", "conversation_session", ["agent_id"])
    op.create_index("ix_agent_conv_session_updated_at", "conversation_session", ["updated_at"])
    op.create_index("ix_agent_conv_session_deleted_at", "conversation_session", ["deleted_at"])

    # ── conversation_message ─────────────────────────────────────────

    op.create_table(
        "conversation_message",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("agent_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", lt, nullable=False),
        sa.Column("content_type", sa.String(32), server_default=sa.text("'text'"), nullable=False),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("turn_index", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("reply_message_id", sa.BigInteger(), nullable=True),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["conversation_session.session_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["agent_id"], ["agent_entity.id"],
            name="fk_conv_msg_agent_id", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["reply_message_id"], ["conversation_message.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        **_conv_kw(),
    )
    op.create_index("ix_agent_conv_msg_session_created", "conversation_message", ["session_id", "created_at"])
    op.create_index("ix_agent_conv_msg_session_turn", "conversation_message", ["session_id", "turn_index"])
    op.create_index("ix_agent_conv_msg_agent_id", "conversation_message", ["agent_id"])
    op.create_index("ix_agent_conv_msg_session_agent", "conversation_message", ["session_id", "agent_id"])

    # ── runtime_external_tool（基表） ────────────────────────────────

    op.create_table(
        "runtime_external_tool",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("description", sa.String(512), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("1"), nullable=False),
        sa.Column("input_schema", sa.JSON(), nullable=True),
        sa.Column("output_schema", sa.JSON(), nullable=True),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uk_runtime_external_tool_name"),
        **_mysql_kw(),
    )
    op.create_index("idx_runtime_external_tool_kind_enabled", "runtime_external_tool", ["kind", "enabled"])

    # ── runtime_http_tool（子表） ────────────────────────────────────

    op.create_table(
        "runtime_http_tool",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("method", sa.String(16), server_default=sa.text("'GET'"), nullable=False),
        sa.Column("headers_json", sa.JSON(), nullable=True),
        sa.Column("request_body_template", lt, nullable=True),
        sa.Column("timeout_ms", sa.Integer(), server_default=sa.text("15000"), nullable=False),
        sa.ForeignKeyConstraint(["id"], ["runtime_external_tool.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── runtime_mcp_tool（子表） ─────────────────────────────────────

    op.create_table(
        "runtime_mcp_tool",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("transport_type", sa.String(32), server_default=sa.text("'http'"), nullable=False),
        sa.Column("server_url", sa.String(2048), nullable=False),
        sa.Column("connection_config_json", sa.JSON(), nullable=True),
        sa.Column("tools_config_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["id"], ["runtime_external_tool.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_runtime_mcp_tool_transport", "runtime_mcp_tool", ["transport_type"])

    # ── knowledge_base ───────────────────────────────────────────────

    op.create_table(
        "knowledge_base",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("namespace_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("slug", sa.String(128), nullable=False),
        sa.Column("description", sa.String(512), nullable=True),
        sa.Column("storage_type", sa.String(32), server_default=sa.text("'vector'"), nullable=False),
        sa.Column("retrieval_type", sa.String(32), server_default=sa.text("'vector'"), nullable=False),
        sa.Column("status", sa.String(32), server_default=sa.text("'empty'"), nullable=False),
        sa.Column("embedding_model_config_id", sa.BigInteger(), nullable=True),
        sa.Column("milvus_collection", sa.String(256), nullable=True),
        sa.Column("minio_prefix", sa.String(512), nullable=True),
        sa.Column("chunk_method", sa.String(32), server_default=sa.text("'length'"), nullable=False),
        sa.Column("chunk_size", sa.Integer(), server_default=sa.text("512"), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), server_default=sa.text("50"), nullable=False),
        sa.Column("chunk_separator", sa.String(64), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("deleted_at", dt, nullable=True),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.ForeignKeyConstraint(
            ["namespace_id"], ["namespace.id"],
            name="fk_knowledge_base_namespace_id", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_model_config_id"], ["sys_model.id"], ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("namespace_id", "slug", name="uk_knowledge_base_namespace_slug"),
        **_mysql_kw(),
    )
    op.create_index("ix_knowledge_base_namespace_id", "knowledge_base", ["namespace_id"])
    op.create_index("ix_knowledge_base_name", "knowledge_base", ["name"])
    op.create_index("ix_knowledge_base_status", "knowledge_base", ["status"])
    op.create_index("ix_knowledge_base_deleted_at", "knowledge_base", ["deleted_at"])

    # ── knowledge_document ───────────────────────────────────────────

    op.create_table(
        "knowledge_document",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("kb_id", sa.BigInteger(), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("object_key", sa.String(1024), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("mime", sa.String(128), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("chunk_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("content_version", sa.String(128), server_default=sa.text("''"), nullable=False),
        sa.Column("deleted_at", dt, nullable=True),
        sa.Column("chunk_method", sa.String(32), nullable=True),
        sa.Column("chunk_size", sa.Integer(), nullable=True),
        sa.Column("chunk_overlap", sa.Integer(), nullable=True),
        sa.Column("chunk_separator", sa.String(64), nullable=True),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_base.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        **_mysql_kw(),
    )
    op.create_index("ix_knowledge_document_kb_id", "knowledge_document", ["kb_id"])
    op.create_index("ix_knowledge_document_deleted_at", "knowledge_document", ["deleted_at"])

    # ── knowledge_task ───────────────────────────────────────────────

    op.create_table(
        "knowledge_task",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("kb_id", sa.BigInteger(), nullable=False),
        sa.Column("doc_id", sa.BigInteger(), nullable=True),
        sa.Column("content_version", sa.String(128), server_default=sa.text("''"), nullable=False),
        sa.Column("task_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("started_at", dt, nullable=True),
        sa.Column("finished_at", dt, nullable=True),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(256), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("checkpoint_json", sa.JSON(), nullable=True),
        sa.Column("last_heartbeat_at", dt, nullable=True),
        sa.Column("created_at", dt, server_default=ts_default, nullable=False),
        sa.Column("updated_at", dt, server_default=ts_update, nullable=False),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_base.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["doc_id"], ["knowledge_document.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", name="uk_knowledge_task_task_id"),
        sa.UniqueConstraint("idempotency_key", name="uk_knowledge_task_idempotency_key"),
        **_mysql_kw(),
    )
    op.create_index("ix_knowledge_task_kb_id", "knowledge_task", ["kb_id"])
    op.create_index("ix_knowledge_task_doc_id", "knowledge_task", ["doc_id"])
    op.create_index("ix_knowledge_task_status", "knowledge_task", ["status"])
    op.create_index("ix_knowledge_task_task_type", "knowledge_task", ["task_type"])

    # ── 种子数据（仅 MySQL） ─────────────────────────────────────────

    if not is_mysql:
        return

    op.execute(sa.text("""
INSERT INTO sys_model_provider
(provider_code, provider_name, base_url, api_format, auth_type, status)
VALUES
('openai',    'OpenAI',         'https://api.openai.com/v1',                          'openai',   'api_key',        0),
('anthropic', 'Anthropic',      'https://api.anthropic.com',                          'anthropic','api_key',        0),
('tongyi',    '阿里通义千问',    'https://dashscope.aliyuncs.com/compatible-mode/v1',  'openai',   'api_key',        0),
('wenxin',    '百度文心一言',    'https://aip.baidubce.com',                           'baidu',    'api_key_secret', 0),
('doubao',    '字节豆包',       'https://ark.cn-beijing.volces.com/api/v1',            'openai',   'api_key',        0),
('hunyuan',   '腾讯混元',       'https://hunyuan.cloud.tencent.com',                  'tencent',  'api_key_secret', 0),
('ollama',    '本地Ollama',     'http://localhost:11434/v1',                           'openai',   'none',           0),
('zhipu',     '智谱AI',         'https://open.bigmodel.cn/api/paas/v4/',              'openai',   'api_key',        0),
('moonshot',  '月之暗面Kimi',   'https://api.moonshot.cn/v1',                          'openai',   'api_key',        0)
"""))

    op.execute(sa.text("""
INSERT INTO sys_model (provider_id, model_code, model_name, model_type, is_enabled) VALUES
(1, 'gpt-3.5-turbo',               'GPT-3.5 Turbo',            'llm',       0),
(1, 'gpt-4o',                      'GPT-4o',                   'llm',       0),
(1, 'gpt-4-turbo',                 'GPT-4 Turbo',              'llm',       0),
(1, 'text-embedding-3-small',      'Embedding 3 Small',        'embedding', 0),
(1, 'tts-1',                       'TTS语音合成',               'tts',       0),
(1, 'whisper-1',                   '语音识别',                  'stt',       0),
(2, 'claude-3-5-sonnet',           'Claude 3.5 Sonnet',        'llm',       0),
(2, 'claude-3-opus',               'Claude 3 Opus',            'llm',       0),
(3, 'qwen-turbo',                  '通义千问Turbo',             'llm',       0),
(3, 'qwen-plus',                   '通义千问Plus',              'llm',       0),
(3, 'text-embedding-v1',           '通义向量模型',              'embedding', 0),
(3, 'qwen-max',                    '通义千问 Max',              'llm',       0),
(3, 'qwen-long',                   '通义千问 Long',             'llm',       0),
(3, 'qwen2.5-72b-instruct',       'Qwen2.5 72B Instruct',     'llm',       0),
(3, 'qwen2.5-32b-instruct',       'Qwen2.5 32B Instruct',     'llm',       0),
(3, 'qwen2.5-14b-instruct',       'Qwen2.5 14B Instruct',     'llm',       0),
(3, 'qwen2.5-7b-instruct',        'Qwen2.5 7B Instruct',      'llm',       0),
(3, 'qwen2.5-coder-32b-instruct', 'Qwen2.5 Coder 32B',       'llm',       0),
(3, 'qwen2.5-coder-7b-instruct',  'Qwen2.5 Coder 7B',        'llm',       0),
(3, 'qwen-vl-plus',               '通义千问 VL Plus',          'llm',       0),
(3, 'qwen-vl-max',                '通义千问 VL Max',           'llm',       0),
(3, 'qwen2.5-vl-72b-instruct',    'Qwen2.5 VL 72B Instruct', 'llm',       0),
(3, 'text-embedding-v2',           '通义向量 v2',              'embedding', 0),
(4, 'ernie-4.0',                   '文心4.0',                  'llm',       0),
(4, 'ernie-3.5',                   '文心3.5',                  'llm',       0),
(5, 'doubao-pro-128k',             '豆包Pro 128K',             'llm',       0),
(5, 'doubao-lite',                 '豆包Lite',                 'llm',       0),
(6, 'hunyuan-lite',                '混元Lite',                 'llm',       0),
(6, 'hunyuan-standard',            '混元标准版',               'llm',       0),
(7, 'llama3',                      'Llama 3',                  'llm',       0),
(7, 'qwen',                        'Qwen本地版',               'llm',       0),
(7, 'nomic-embed-text',            '本地向量模型',              'embedding', 0),
(8, 'glm-4',                       'GLM-4',                    'llm',       0),
(8, 'glm-3-turbo',                 'GLM-3 Turbo',              'llm',       0),
(9, 'moonshot-v1-128k',            'Kimi 128K',                'llm',       0),
(9, 'moonshot-v1-8k',              'Kimi 8K',                  'llm',       0)
"""))

    op.execute(sa.text("""
INSERT INTO namespace (slug) VALUES ('default')
"""))


# ── downgrade ────────────────────────────────────────────────────────

def downgrade() -> None:
    op.drop_table("knowledge_task")
    op.drop_table("knowledge_document")
    op.drop_table("knowledge_base")
    op.drop_table("runtime_mcp_tool")
    op.drop_table("runtime_http_tool")
    op.drop_table("runtime_external_tool")
    op.drop_table("conversation_message")
    op.drop_table("conversation_session")
    # 先断开循环 FK 再删表
    op.drop_constraint("fk_namespace_workbench_agent_id", "namespace", type_="foreignkey")
    op.drop_table("agent_entity")
    op.drop_table("namespace")
    op.drop_table("sys_model")
    op.drop_table("sys_model_provider")
    op.drop_table("trace_span_logs")
    op.drop_table("trace_spans")
    op.drop_table("trace_runs")
