import streamlit as st
from rag_pipeline import generate_answer

st.set_page_config(page_title="Semantic Retrieval System", layout="centered")

st.title("🔎 Semantic Retrieval Chatbot")
st.markdown("Lightweight Retrieval-Augmented System using MiniLM + FAISS")

st.divider()

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Searching knowledge base..."):
        top_result, other_results = generate_answer(query)

    if top_result is None:
        st.error("No relevant information found.")
    else:
        doc, score = top_result

        st.success("Best Match")
        st.write(doc)

        st.progress(min(score, 1.0))
        st.caption(f"Confidence Score: {score:.4f}")

        if other_results:
            with st.expander("See other possible matches"):
                for doc, score in other_results:
                    st.markdown("---")
                    st.write(doc)
                    st.caption(f"Score: {score:.4f}")

st.divider()

with st.expander("⚙️ System Architecture"):
    st.markdown("""
- Transformer Embeddings: all-MiniLM-L6-v2  
- Vector Database: FAISS  
- Similarity Metric: Cosine Similarity  
- Retrieval Strategy: Top-K Ranked Semantic Search  
- Offline and Lightweight Implementation  
""")