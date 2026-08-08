"""
Central config for the QA Test Case Generation Agent.
Change these if your Ollama model names or folder locations differ.
"""

# Folder where you drop your requirement PDFs (2 to start)
DATA_DIR = "data"

# Where Chroma stores its local vector DB
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "requirements"

# Ollama model names
LLM_MODEL = "phi3:mini"          # your installed generation model
EMBED_MODEL = "nomic-embed-text"  # you'll need to pull this: `ollama pull nomic-embed-text`

# Ollama server (default local address)
OLLAMA_URL = "http://localhost:11434"

# Chunking
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks

# Retrieval
TOP_K = 4  # how many chunks to retrieve per query
