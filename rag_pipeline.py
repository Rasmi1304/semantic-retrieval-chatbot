from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st


@st.cache_resource
def build_index():
    # Load embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load documents
    with open("data/docs.txt", "r", encoding="utf-8") as f:
        documents = f.read().split("\n\n")

    # Generate embeddings
    doc_embeddings = embed_model.encode(documents)
    faiss.normalize_L2(doc_embeddings)

    # Create FAISS index using cosine similarity
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(doc_embeddings))

    return embed_model, index, documents


def generate_answer(query, top_k=3):
    embed_model, index, documents = build_index()

    query_embedding = embed_model.encode([query])
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(np.array(query_embedding), top_k)

    top_score = scores[0][0]

    if top_score < 0.4:
        return None, None

    results = []
    for i in range(top_k):
        doc = documents[indices[0][i]]
        score = float(scores[0][i])
        results.append((doc, score))

    return results[0], results[1:]