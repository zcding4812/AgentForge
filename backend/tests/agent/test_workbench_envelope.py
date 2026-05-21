from app.agent.adapters.tools.workbench.tool_support import wb_err, wb_ok


def test_wb_ok_shape() -> None:
    o = wb_ok({"total": 1, "items": [1]})
    assert o["code"] == "OK"
    assert o["message"] == "success"
    assert o["data"]["total"] == 1


def test_wb_err_empty_data() -> None:
    e = wb_err("NOT_FOUND", "missing")
    assert e["code"] == "NOT_FOUND"
    assert e["message"] == "missing"
    assert e["data"] == {}
