from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil


# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        raise ValueError("No Markdown files found in ./data folder")

    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".md"):
            path = os.path.join(DATA_PATH, filename)
            print(f"Loading: {path}")
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents")
    return documents

def split_text(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    if chunks:
        print("\n Sample chunk preview:")
        print(chunks[0].page_content[:200], "...\n")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("Creating Chroma Vector DB (Local Embeddings)...")

    # FREE offline embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks → {CHROMA_PATH}")

if __name__ == "__main__":
    main()
