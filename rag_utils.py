import os
import uuid
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF, return empty string if unreadable."""
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    except Exception as e:
        print(f"PDF read error: {e}")
    return text.strip()


def chunk_text(text, chunk_size=500):
    """Split text into chunks of words."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def create_session():
    """Create unique session folder."""
    session_id = str(uuid.uuid4())
    session_path = os.path.join("sessions", session_id)
    os.makedirs(session_path, exist_ok=True)
    return session_id, session_path


def build_faiss_index(texts, session_path):
    """Build FAISS index safely from multiple texts."""
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    if not all_chunks:
        all_chunks = ["No readable text found."]  # fallback chunk

    embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(session_path, "faiss.index"))
    np.save(os.path.join(session_path, "chunks.npy"), np.array(all_chunks, dtype=object))

    return index, all_chunks, embeddings


def load_faiss_index(session_path):
    """Load FAISS index and chunks from session folder."""
    index = faiss.read_index(os.path.join(session_path, "faiss.index"))
    chunks = np.load(os.path.join(session_path, "chunks.npy"), allow_pickle=True)
    return index, chunks


def retrieve_context(query, index, chunks, top_k=3):
    """Retrieve top-k chunks for a query and calculate retrieval accuracy."""
    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    accuracy = float(np.mean(D[0])) * 100 if D.size > 0 else 0.0
    return retrieved, accuracy
