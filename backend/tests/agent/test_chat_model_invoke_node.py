import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from app.agent.adapters.graph.nodes.simple_chat.invoke import ChatModelInvokeNode


def test_chat_model_invoke_node_calls_model() -> None:
    async def _run() -> None:
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="hi"))
        node = ChatModelInvokeNode(model)
        out = await node({"messages": []})
        model.ainvoke.assert_awaited_once()
        assert out["messages"][0].content == "hi"

    asyncio.run(_run())
