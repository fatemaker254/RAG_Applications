from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    # Load local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # Generate embeddings
    word1 = "apple"
    word2 = "apple"
    vector1 = embeddings.embed_query(word1)
    vector2 = embeddings.embed_query(word2)

    print(f"Vector length: {len(vector1)}")

    # Convert to numpy arrays for cosine similarity
    v1 = np.array(vector1).reshape(1, -1)
    v2 = np.array(vector2).reshape(1, -1)

    similarity = cosine_similarity(v1, v2)[0][0]

    print(f"\nCosine similarity between '{word1}' and '{word2}': {similarity:.4f}")

    if similarity > 0.7:
        print("Highly related concepts")
    elif similarity > 0.4:
        print("Somewhat related")
    else:
        print("Not related")

if __name__ == "__main__":
    main()
