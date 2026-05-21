"""运行期可注册的工具工厂（数据驱动，非热加载任意 Python）。"""

from app.agent.adapters.tools.dynamic.http_fetch import make_http_get_tool, make_http_request_tool
from app.agent.adapters.tools.dynamic.mcp_stub import make_mcp_placeholder_tool

__all__ = ["make_http_get_tool", "make_http_request_tool", "make_mcp_placeholder_tool"]
