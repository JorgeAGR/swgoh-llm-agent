from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from model import llm_with_tools


SYSTEM_PROMPT = """You are a Star Wars Galaxy of Heroes assistant. 
Your goal is to provide accurate character data from the provided tools. Don't be afraid to use tools multiple times, especially when having to consider multiple characters.
Only answer with the data provided by the tools. Do not make up any data.

TOOL USAGE RULES:
1. Use 'find_relevant_units' to get a list of characters and their tags. Use this as a guide to choose appropriate characters to answer the user's question.
2. Use 'get_character_data' with that EXACT URL to fetch stats and abilities for a character.

INTERPRETATION RULES:
- Remember that team sizes are limited to 5 characters.
- Battles are limited to 5 minutes.
- Zetas and Omicrons are elite upgrades. If 'is_zeta' is True, identify it as a Zeta ability. Likewise for 'is_omicron'.
- Always provide the 'mods_data_url' if the user asks for modding advice.
- Query as many characters as needed to answer the user's question. Pay particular attention to tags to maximize synergy.
- Be concise but stay in character as a helpful tactical droid."""


# This tracks the list of messages in the conversation
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    """The node that calls the LLM with a system prompt"""
    # Prepend the system prompt to the existing conversation history
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # Invoke the model with the augmented message list
    response = llm_with_tools.invoke(messages)
    
    # Return the response to be added to the state
    return {"messages": [response]}