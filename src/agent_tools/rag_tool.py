from langchain.tools import tool
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
import tqdm
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
data_path = os.path.join(os.path.dirname(__file__), "../../data")

# 1. Load your source data
df_all = pd.read_parquet(os.path.join(data_path, "swgoh_units.parquet"))
df_details = pd.read_parquet(os.path.join(data_path, "character_details.parquet"))

# Set up embeddings and cache
store = LocalFileStore(os.path.join(data_path, "embedding_cache"))
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
# The 'namespace' ensures different models don't mix up their vectors
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, 
    store,
    namespace=embeddings.model,
    key_encoder="sha256"
)

# Initialize summarizer model
summarizer_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

def summarize_ability(description: str) -> str:
    """Uses Gemini 3 Flash to provide a concise summary of an ability description."""
    if not description:
        return ""
    prompt = f"Summarize the following SWGOH character ability description in one to three concise sentences, focusing on the core mechanics and effects. Limit your response to only the summary of the ability, given only the information shown. Make it high level and as snappy as possible. Omit exact quantities to keep things brief. Keep the format simple, no titles or bullet points, just provide the summary:\n\n{description}"
    try:
        response = summarizer_llm.invoke(prompt)
        return response.content[0]['text']
    except Exception as e:
        print(f"Error summarizing ability: {e}")
        return description  # Fallback to original description if error


def format_list(list_to_format: list) -> str:
    list_str = ""
    for item in list_to_format:
        list_str += f"{item}, "
    list_str = list_str[:-2]
    return list_str


def process_character(character_row: pd.Series, character_details: pd.Series) -> Document:
    abilities_text_list = []
    for ability in character_details['abilities']:
        if ability['description'] == "Placeholder":
            continue
        if ability['is_zeta'] and ability['is_omicron']:
            ability_material = ' [Zeta,Omicron]'
        elif ability['is_zeta']:
            ability_material = ' [Zeta]'
        elif ability['is_omicron']:
            ability_material = ' [Omicron]'
        elif ability['is_ultimate']:
            ability_material = ' [Ultimate]'
        else:
            ability_material = ''
        
        # Summarize the ability description
        summary = summarize_ability(ability['description'])
        ability_text = f"{ability['ability_name']} ({ability['ability_type']}){ability_material}: {summary}"
        abilities_text_list.append(ability_text)
        
    abilities_combined = "\n".join(abilities_text_list)
    ability_classes_str = format_list(character_details['ability_classes'])
    tags = format_list(character_row['tags'])
    
    content = f"Character: {character_row['name']}\nTags: {tags}\nAbility Classes: {ability_classes_str}\nAbilities:\n{abilities_combined}"
    
    # We keep the URL in metadata for the second 'Tool' step later
    doc = Document(page_content=content, metadata={"url": character_row['character_url']})
    return doc


# Check if vector store already exists
vectordb_path = os.path.join(data_path, "swgoh_vectordb")

if os.path.exists(vectordb_path) and os.listdir(vectordb_path):
    # Load existing vector store
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=vectordb_path,
        embedding_function=cached_embedder
    )
else:
    # Create new vector store (first time only)
    print("Creating new vector store...")

    # Transform rows into "Documents"
    documents = []
    print(f"Processing {len(df_all)} characters and summarizing abilities...")
    for _, row in tqdm.tqdm(df_all.iterrows(), total=len(df_all)):
        # We combine key fields into a text block for the model to "read"
        character_details = df_details[df_details['character_url'] == row['character_url']].iloc[0]
        doc = process_character(row, character_details)
        documents.append(doc)

    # Load your existing vector store
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=cached_embedder, 
        persist_directory=os.path.join(data_path, "swgoh_vectordb")
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


@tool
def find_relevant_units(query: str) -> str:
    """
    Searches the SWGOH database for character descriptions, mechanics, or tags.
    Use this to identify WHICH characters might be relevant to the user's request.
    Input should be a natural language search query.
    """
    # 1. Fetch relevant documents from Chroma
    docs = retriever.invoke(query)
    
    # 2. Format the output for the LLM
    # We combine the content and include the URL from metadata so the 
    # agent knows which URL to pass to the 'get_exact_stats' tool.
    results = []
    for doc in docs:
        char_info = f"Content: {doc.page_content}\nCharacter URL: {doc.metadata.get('url', 'Unknown')}"
        results.append(char_info)
    
    return "\n\n---\n\n".join(results)