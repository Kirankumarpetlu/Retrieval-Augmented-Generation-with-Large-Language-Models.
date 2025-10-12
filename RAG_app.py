import os
import uuid
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# CONFIGURATION

st.set_page_config(page_title="RAG ", layout="wide")

GROQ_API_KEY = " paste your groq api " 
os.makedirs("sessions", exist_ok=True)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# UTIL FUNCTIONS

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def chunk_text(text, chunk_size=500):
    """Splits text into word chunks."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_session():
    """Creates a unique session directory."""
    session_id = str(uuid.uuid4())
    session_path = os.path.join("sessions", session_id)
    os.makedirs(session_path, exist_ok=True)
    return session_id, session_path

def build_faiss_index(chunks, session_path):
    """Builds a FAISS index safely, checking for empty embeddings."""
    if not chunks or len(chunks) == 0:
        raise ValueError(" No valid text chunks found. Your PDF might not contain readable text.")

    # Compute embeddings
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings)

    # Validate embeddings shape
    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError(f" Invalid embeddings shape: {embeddings.shape}")

    # Normalize embeddings
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, os.path.join(session_path, "faiss.index"))
    np.save(os.path.join(session_path, "chunks.npy"), np.array(chunks, dtype=object))

    st.success(f" FAISS index built successfully with {len(chunks)} chunks.")
    return index

def load_faiss_index(session_path):
    """Loads an existing FAISS index and corresponding chunks."""
    index = faiss.read_index(os.path.join(session_path, "faiss.index"))
    chunks = np.load(os.path.join(session_path, "chunks.npy"), allow_pickle=True)
    return index, chunks

def query_rag(query, index, chunks, top_k=3):
    """Queries FAISS index and gets relevant context from Gemma (via Groq API)."""
    if not query.strip():
        return " Please enter a valid question."

    # Encode query
    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    # Search top_k relevant chunks
    D, I = index.search(q_emb, top_k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n\n".join(retrieved)

    if not context.strip():
        return " No relevant content found in the document."

    # Build prompt
    prompt = f"""
You are an assistant that answers questions using only the provided context.
Context:
{context}

Question: {query}
Answer:
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Groq API error: {e}"


# STREAMLIT UI

st.title("Retrieval-Augmented Generation(RAG) ")

uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

if uploaded_file:
    try:
        session_id, session_path = create_session()
        st.info(f"Session created: `{session_id}`")

        # Extract text
        text = extract_text_from_pdf(uploaded_file)
        st.write(f" Extracted {len(text)} characters from PDF.")

        if len(text.strip()) == 0:
            st.error(" No readable text found. Your PDF might be image-based. Please use an OCR-processed version.")
        else:
            # Build FAISS index
            chunks = chunk_text(text)
            index = build_faiss_index(chunks, session_path)

            # Question input
            user_query = st.text_input("Ask your question about the document:")
            if user_query:
                with st.spinner("🤔 Thinking..."):
                    answer = query_rag(user_query, index, chunks)
                st.markdown("###  Answer:")
                st.write(answer)

    except Exception as e:
        st.error(f" Error: {str(e)}")

else:
    st.info("Please upload a PDF to start.")

st.markdown("""
---
<div style='text-align: center; font-size: 14px; color: grey;'>
Developed by <b>Kiran Kumar Petlu</b>
</div>
""", unsafe_allow_html=True)
