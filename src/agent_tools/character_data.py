import pandas as pd
import json
from langchain.tools import tool
import os
import asyncio

# Load your dataframes once at the start
data_path = os.path.join(os.path.dirname(__file__), "../../data")
df_all = pd.read_parquet(os.path.join(data_path, "swgoh_units.parquet"))
df_details = pd.read_parquet(os.path.join(data_path, "character_details.parquet"))


@tool
def list_all_characters() -> list:
    """
    Returns a list of all characters in the dataset including their names and tags.
    """
    return df_all[['name', 'tags']].to_dict(orient='records')


@tool
def find_character(character_name: str) -> str:
    """
    Search for a character by name and return their unique URL.
    Use this when the user mentions a character but you don't have their URL yet.
    """
    # Simple case-insensitive match
    match = df_all[df_all['name'].str.contains(character_name, case=False, na=False)]
    
    if not match.empty:
        # Return the first match found
        return match.iloc[0]['character_url']
    else:
        return f"Character '{character_name}' not found."


@tool
async def get_character_data(character_url: str) -> str:
    """
    Retrieves the full row of data (stats, abilities, tags, and links) for a 
    character using their specific URL. URL must be exact.
    """
    # Search the details table using the URL as the key
    def fetch():
        match = df_details[df_details['character_url'] == character_url]
        
        if not match.empty:
            # Convert the row to a dictionary and then to JSON for the LLM
            data_dict = match.iloc[0].to_dict()
            
            # Parquet stores lists/dicts natively; ensure they are serializable
            return json.dumps(data_dict, default=str)
        else:
            return "No detailed data found for this URL."
    
    return await asyncio.to_thread(fetch)