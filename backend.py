from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import os
from rag_utils import extract_text_from_pdf, create_session, build_faiss_index, retrieve_context

app = FastAPI(title="RAG Backend API")
GROQ_API_KEY = "Paste LLM's API key" #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
client = Groq(api_key=GROQ_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

@app.post("/ingest/")
async def ingest_files(files: list[UploadFile]):
    """Ingest multiple PDFs."""
    session_id, session_path = create_session()
    texts = []

    for file in files:
        content = await file.read()
        pdf_path = os.path.join(session_path, file.filename)
        with open(pdf_path, "wb") as f:
            f.write(content)
        text = extract_text_from_pdf(pdf_path)
        if text:
            texts.append(text)

    # build index even if no text
    index, chunks, embeddings = build_faiss_index(texts, session_path)
    sessions[session_id] = {"index": index, "chunks": chunks, "embeddings": embeddings}

    return {"session_id": session_id, "chunks": len(chunks)}


@app.post("/query/")
async def query_doc(session_id: str, query: str):
    """Answer a query using RAG."""
    data = sessions.get(session_id)
    if not data:
        return {"error": "Invalid session_id"}

    index, chunks, embeddings = data["index"], data["chunks"], data["embeddings"]
    retrieved, accuracy = retrieve_context(query, index, chunks)

    context = "\n".join(retrieved)
    prompt = f"Use the context below to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    answer = completion.choices[0].message.content.strip()

    return {"answer": answer, "retrieval_accuracy": round(accuracy, 2), "context": retrieved}
