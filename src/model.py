from agent_tools import find_character, get_character_data, list_all_characters, find_relevant_units
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# tools = [find_character, get_character_data, list_all_characters, find_relevant_units]
tools = [find_relevant_units, get_character_data]
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
llm_with_tools = llm.bind_tools(tools)