import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import string
import time

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# LOADING LOGIC (Runs once to setup the system) 
if not st.session_state.system_ready:
    with st.status("Initializing AI Retrieval System...", expanded=True) as status:
        st.write("Reading document dataset...")
        try:
            # A. Load text documents
            with open("documents.txt", "r", encoding="utf-8") as f:
                raw_docs = [line.strip() for line in f.readlines()]
            st.session_state.docs = raw_docs
            
            # B. Train/Load Word2Vec Model (Required to understand query words)
            st.write("Training Word2Vec model on vocabulary...")
            processed_docs = [doc.lower().translate(str.maketrans('', '', string.punctuation)).split() for doc in raw_docs]
            
            model = Word2Vec(sentences=processed_docs, vector_size=100, min_count=1)
            st.session_state.model = model
            
            # C. Load Precomputed Embeddings
            st.write("Loading vector embeddings...")
            st.session_state.embs = np.load("embeddings.npy")
            
            time.sleep(1)
            st.session_state.system_ready = True
            status.update(label="System Ready!", state="complete", expanded=False)
            
        except Exception as e:
            st.error(f"Error loading files: {e}")
            st.stop()

# HELPER FUNCTIONS
def get_query_embedding(query):
    """Converts user query into a vector using the session model."""
    words = query.lower().split()
    # Only use words that exist in our simple model
    valid_words = [word for word in words if word in st.session_state.model.wv]
    
    if not valid_words:
        return np.zeros(100)
    
    # Average the vectors
    return np.mean(st.session_state.model.wv[valid_words], axis=0)

def retrieve_top_k(query_vec, embeddings, documents, k=3):
    """Finds the top K closest documents."""
    sims = cosine_similarity([query_vec], embeddings)[0]
    top_indices = sims.argsort()[-k:][::-1]
    return [(documents[i], sims[i]) for i in top_indices]



st.title("🔍 AI Research Retrieval System")

st.subheader("Fine-tuned using Word2Vec & Cosine Similarity")
query = st.text_input("Enter your research query:", placeholder="e.g. How do Transformers work?")

# SEARCH LOGIC
if st.button("Search"):
    if query and st.session_state.system_ready:
        query_vec = get_query_embedding(query)
        
        st.session_state.search_results = retrieve_top_k(
            query_vec, st.session_state.embs, st.session_state.docs
        )
        st.session_state.last_query = query
    else:
        st.warning("Please enter a query.")

# DISPLAY RESULTS 
if st.session_state.search_results:
    st.markdown("---")
    st.write(f"### Top Relevant Results for: *'{st.session_state.last_query}'*")
    
    for i, (doc, score) in enumerate(st.session_state.search_results):
        st.markdown(f"**{i+1}.** {doc}")
        st.progress(max(0.0, min(float(score), 1.0)))
        st.caption(f"Relevance Score: {score:.4f}")
        st.write("")
