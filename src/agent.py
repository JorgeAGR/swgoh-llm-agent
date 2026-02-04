import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from chatbot import State, chatbot
from model import tools
import logging
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
import asyncio

# 3. Define the Nodes
builder = StateGraph(State)

# Add our nodes: the chatbot and the prebuilt ToolNode
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

# Set the flow: Start -> Chatbot
builder.add_edge(START, "chatbot")

# Conditional Edge: After the chatbot runs, should we call tools or end?
builder.add_conditional_edges(
    "chatbot",
    tools_condition, # This checks if the LLM generated a tool call
)

# After tools run, always go back to the chatbot to process the results
builder.add_edge("tools", "chatbot")

# Compile the graph
agent = builder.compile()

# --- SETUP LOGGING ---
logging.basicConfig(
    filename='agent_reasoning.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='a' # 'a' for append, 'w' to overwrite each run
)


def log_and_print(message):
    """Helper to both print to console and write to the log file."""
    print(message)
    logging.info(message)


async def run_agent_with_logging(user_input: str):
    log_and_print(f"\n{'='*20} NEW SESSION: {datetime.now()} {'='*20}")
    log_and_print(f"USER INPUT: {user_input}")
    
    inputs = {"messages": [("user", user_input)]}
    
    # 1. We use 'async for' to handle the async generator from astream
    async for chunk in agent.astream(inputs, stream_mode="updates"):
        for node_name, data in chunk.items():
            log_and_print(f"\n>>> NODE TRANSITION: Entering [{node_name}]")
            
            messages = data.get("messages", [])
            for msg in messages:
                # 1. Capture Agent Reasoning
                if isinstance(msg, AIMessage) and msg.content:
                    log_and_print(f"[REASONING]: {msg.content}")
                
                # 2. Capture Tool Calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        log_and_print(f"[ACTION]: Calling Tool '{tc['name']}' with args: {tc['args']}")
                
                # 3. Capture Tool Outputs
                if isinstance(msg, ToolMessage):
                    raw_content = msg.content
                    display_content = (raw_content[:200] + '...') if len(raw_content) > 200 else raw_content
                    log_and_print(f"[TOOL RESULT]: {display_content}")

    log_and_print(f"\n{'='*20} SESSION ENDED {'='*20}\n")


if __name__ == "__main__":
    asyncio.run(run_agent_with_logging("What is the ideal team for Jedi Master Mace Windu?"))