from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load .env file (optional for future API logic)
load_dotenv()

CHROMA_PATH = "chroma"

def main():
    query = input("Ask a question: ")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("\n🔍 Searching relevant chunks...\n")
    results = db.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(doc.page_content)
        print("-" * 80)

if __name__ == "__main__":
    main()
