from app.services.monitor_svc import redact_url_for_display


def test_redact_mysql_style_url() -> None:
    s = "mysql+pymysql://user:secret@127.0.0.1:3306/ai_agents"
    out = redact_url_for_display(s)
    assert "secret" not in out
    assert "user:***@" in out
    assert "127.0.0.1:3306" in out


def test_redact_redis_password_only() -> None:
    s = "redis://:abc@localhost:6379/0"
    out = redact_url_for_display(s)
    assert "abc" not in out
    assert ":***@localhost" in out


def test_sqlite_memory_short() -> None:
    assert redact_url_for_display("sqlite+aiosqlite:///:memory:") == "sqlite :memory:"
