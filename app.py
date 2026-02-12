import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# LOADING LOGIC (Only runs once) ---
if not st.session_state.system_ready:
    with st.status("Initializing AI Retrieval System...", expanded=True) as status:
        st.write("Reading AI/LLM document dataset...")
        try:
            # Load documents and embeddings
            with open("documents.txt", "r", encoding="utf-8") as f:
                st.session_state.docs = [line.strip() for line in f.readlines()]
            st.session_state.embs = np.load("embeddings.npy")
            time.sleep(1) 
            st.session_state.system_ready = True
            status.update(label="System Ready!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error loading files: {e}")
            st.stop()

# --- 3. HELPER FUNCTIONS ---
def get_query_embedding(query, vector_dim):
    np.random.seed(hash(query) % 10**8)
    return np.random.rand(vector_dim).astype(np.float32)

def retrieve_top_k(query_embedding, embeddings, documents, k=3):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]


st.title("🔍 AI Research Retrieval System")
st.subheader("Search AI & LLM Literature Using Word Embeddings")

query = st.text_input("Enter your research query:", placeholder="How do Transformers work?")

# Search button logic
if st.button("Search"):
    if query:
        query_emb = get_query_embedding(query, st.session_state.embs.shape[1])
        # Store results in session_state so they don't disappear
        st.session_state.search_results = retrieve_top_k(
            query_emb, st.session_state.embs, st.session_state.docs
        )
        st.session_state.last_query = query
    else:
        st.warning("Please enter a query.")

#  DISPLAY RESULTS
if st.session_state.search_results:
    st.markdown("---")
    st.write(f"### Top Relevant Results for: *'{st.session_state.last_query}'*")
    
    for i, (doc, score) in enumerate(st.session_state.search_results):
        st.markdown(f"**{i+1}.** {doc}")
        st.progress(float(score))
        st.caption(f"Relevance Score: {score:.4f}")
        st.write("")