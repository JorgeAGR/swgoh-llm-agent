"""The tool-calling agent loop and a streaming event interface over it."""

import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from swgoh_agent.agent.prompts import build_system_prompt
from swgoh_agent.agent.tools import build_tools
from swgoh_agent.catalog import Catalog
from swgoh_agent.config import Settings
from swgoh_agent.knowledge.index import KnowledgeIndex

# Logs the agent's reasoning trace (tool calls and results); the CLI can route
# it to a file with --log-file.
trace = logging.getLogger("swgoh_agent.trace")

RECURSION_LIMIT = 40


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_llm(settings: Settings) -> ChatOpenAI:
    kwargs: dict[str, Any] = {}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        **kwargs,
    )


class SwgohAgent:
    def __init__(self, settings: Settings, catalog: Catalog, index: KnowledgeIndex):
        self.settings = settings
        self.catalog = catalog
        tools = build_tools(catalog, index)
        self.llm = build_llm(settings).bind_tools(tools)
        self.system_prompt = SystemMessage(build_system_prompt(catalog))

        builder = StateGraph(State)
        builder.add_node("agent", self._call_model)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        # Conversation history is kept per thread_id in memory.
        self.graph = builder.compile(checkpointer=InMemorySaver())

    async def _call_model(self, state: State) -> dict:
        response = await self.llm.ainvoke([self.system_prompt, *state["messages"]])
        return {"messages": [response]}

    async def astream(self, message: str, thread_id: str) -> AsyncIterator[dict]:
        """Run one user turn, yielding UI-friendly events.

        Events: {"type": "token", "text"}, {"type": "tool_call", "id", "name",
        "args"}, {"type": "tool_result", "id", "name", "content"}, {"type": "done"}.
        """
        trace.info("[%s] USER: %s", thread_id, message)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        inputs = {"messages": [HumanMessage(message)]}
        async for mode, payload in self.graph.astream(
            inputs, config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, meta = payload
                if (
                    isinstance(chunk, AIMessageChunk)
                    and meta.get("langgraph_node") == "agent"
                    and (text := chunk.text)
                ):
                    yield {"type": "token", "text": text}
                continue
            for update in payload.values():
                for msg in (update or {}).get("messages", []):
                    for event in self._message_events(msg, thread_id):
                        yield event
        yield {"type": "done"}

    @staticmethod
    def _message_events(msg: AnyMessage, thread_id: str) -> list[dict]:
        if isinstance(msg, AIMessage):
            if msg.text:
                trace.info("[%s] ASSISTANT: %s", thread_id, msg.text)
            for call in msg.tool_calls:
                trace.info(
                    "[%s] TOOL CALL %s(%s)", thread_id, call["name"], call["args"]
                )
            return [
                {
                    "type": "tool_call",
                    "id": c["id"],
                    "name": c["name"],
                    "args": c["args"],
                }
                for c in msg.tool_calls
            ]
        if isinstance(msg, ToolMessage):
            trace.info("[%s] TOOL RESULT %s: %.300s", thread_id, msg.name, msg.text)
            return [
                {
                    "type": "tool_result",
                    "id": msg.tool_call_id,
                    "name": msg.name,
                    "content": msg.text,
                }
            ]
        return []


def create_agent(settings: Settings) -> SwgohAgent:
    catalog = Catalog.load(settings.units_path)
    index = KnowledgeIndex(settings)
    if index.count() == 0:
        raise RuntimeError(
            "The knowledge index is empty. Run `swgoh-agent index build`."
        )
    return SwgohAgent(settings, catalog, index)
