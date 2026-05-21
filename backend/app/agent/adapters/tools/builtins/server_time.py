"""内置工具：返回当前服务器时间（无副作用）。

时刻取自单次 ``time.time()``，换算为东八区与 UTC；输出为 **标准墙钟字符串**（``YYYY-MM-DD HH:MM:SS``）。
若与真实世界时间不符，属 **宿主机/容器系统时钟** 问题，需在环境侧配置 NTP。

**工具契约（供模型 / 调用方对齐）**

- **输入**：无参数。LangChain 侧为 **空对象**（不接收任何字段）；调用时勿传参。
- **输出**：单一字符串（UTF-8 纯文本），**固定三行**：
  1. 固定前缀行：``【进程实时时间】以下为单次采样结果，请勿替换为示例日期。``
  2. ``北京时间（UTC+8）: YYYY-MM-DD HH:MM:SS``
  3. ``UTC: YYYY-MM-DD HH:MM:SS``
  模型应答若需引用本工具结果，应 **原样引用** 上述时间行，勿改写为示例日期。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone

from langchain_core.tools import tool

# 中国标准时间（与 IANA Asia/Shanghai 一致；不依赖系统 tz 数据库）
_CN_STD = timezone(timedelta(hours=8))

_FMT = "%Y-%m-%d %H:%M:%S"


@tool("server_time")
async def server_time_tool() -> str:
    """查询进程当前时间（非示例）。

    **输入**：无参数（空 schema）。

    **输出**：UTF-8 纯文本，三行结构见模块文档；时间行为 ``YYYY-MM-DD HH:MM:SS`` 墙钟格式。
    """
    ts = time.time()
    utc = datetime.fromtimestamp(ts, tz=UTC)
    cn = utc.astimezone(_CN_STD)
    return (
        "【进程实时时间】以下为单次采样结果，请勿替换为示例日期。\n"
        f"北京时间（UTC+8）: {cn.strftime(_FMT)}\n"
        f"UTC: {utc.strftime(_FMT)}"
    )
