from langchain.tools import tool
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
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
    for _, row in df_all.iterrows():
        # We combine key fields into a text block for the model to "read"
        character_details = df_details[df_details['character_url'] == row['character_url']].iloc[0]
        abilities_dict = {}
        for ability in character_details['abilities']:
            abilities_dict[ability['name']] = ability['description']
        content = f"Character: {row['name']}. Tags: {row['tags']}. Ability Classes: {character_details['ability_classes'].to_string()}"
        
        # We keep the URL in metadata for the second 'Tool' step later
        doc = Document(page_content=content, metadata={"url": row['character_url']})
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