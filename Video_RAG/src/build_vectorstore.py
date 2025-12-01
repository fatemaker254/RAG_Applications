import os, json, shutil
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import TRANSCRIPT_DIR, CHROMA_DIR

model = SentenceTransformer("BAAI/bge-base-en")  # GPU accelerated

def load_docs():
    docs = []
    for file in os.listdir(TRANSCRIPT_DIR):
        if file.endswith(".json"):
            data = json.load(open(os.path.join(TRANSCRIPT_DIR, file)))
            for seg in data["segments"]:
                docs.append(Document(
                    page_content=seg["text"],
                    metadata={"source": file, "start": seg["start"]}
                ))
    return docs

def main():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embedding_fn = lambda texts: model.encode(texts, convert_to_numpy=True)

    db = Chroma.from_documents(
        chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR
    )
    db.persist()
    print(f"📦 Stored {len(chunks)} chunks in vector DB")

if __name__ == "__main__":
    main()
