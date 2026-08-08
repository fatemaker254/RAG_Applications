"""
Step 2: Embed chunks using a local Ollama embedding model and store
them in a local Chroma vector database. Also provides retrieval.
"""

import requests
import chromadb
from config import OLLAMA_URL, EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME, TOP_K


def get_embedding(text: str):
    """Call Ollama's embedding endpoint for a single piece of text."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    if resp.status_code != 200:
        # Surface Ollama's actual error message (e.g. model not found,
        # context length exceeded, out of memory) instead of a bare HTTPError.
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(
            f"Ollama embedding call failed (status {resp.status_code}): {detail}\n"
            f"Text preview that caused it: {text[:200]!r}"
        )
    return resp.json()["embedding"]


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_vectorstore(chunks):
    """
    Embeds every chunk and stores it in a persistent local Chroma collection.
    Wipes and rebuilds the collection each time (fine for a small capstone dataset).
    """
    client = get_client()

    # Fresh collection each run so re-ingesting doesn't duplicate data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []
    skipped = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}...")
        try:
            emb = get_embedding(chunk["text"])
        except RuntimeError as e:
            print(f"  SKIPPED chunk {i} due to error:\n  {e}\n")
            skipped.append(i)
            continue
        ids.append(f"chunk_{i}")
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"], "section": chunk["section"]})

    if not ids:
        raise RuntimeError("No chunks were successfully embedded. Check the errors above.")

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Stored {len(ids)} chunks in Chroma collection '{COLLECTION_NAME}'.")
    if skipped:
        print(f"Note: {len(skipped)} chunk(s) were skipped (indices: {skipped}).")
    return collection


def load_vectorstore():
    """Load an already-built collection (skip re-embedding)."""
    client = get_client()
    return client.get_collection(COLLECTION_NAME)


def retrieve(collection, query: str, top_k: int = TOP_K):
    """Embed the query and pull back the most relevant chunks."""
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": doc, "source": meta["source"], "section": meta["section"]})
    return retrieved