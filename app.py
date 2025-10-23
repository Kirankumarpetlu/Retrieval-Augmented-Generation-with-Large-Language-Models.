import requests
import streamlit as st

st.set_page_config(page_title="RAG System", layout="wide")
BACKEND_URL = "http://127.0.0.1:8000"

st.title("knowledge based search engine ")

uploaded_files = st.file_uploader("📂 Upload multiple PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    if st.button("Ingest Documents"):
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
        try:
            resp = requests.post(f"{BACKEND_URL}/ingest/", files=files, timeout=60)
            data = resp.json()
            if "session_id" in data:
                st.session_state.session_id = data["session_id"]
                st.success(f"✅ Documents ingested! Total chunks: {data['chunks']}")
            else:
                st.error("❌ Ingestion failed. Backend response missing keys.")
        except Exception as e:
            st.error(f"❌ Error connecting to backend: {e}")

if "session_id" in st.session_state:
    query = st.text_input("💬 Ask a question:")
    if query:
        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query/",
                        params={
                            "session_id": st.session_state.session_id,
                            "query": query
                        },
                        timeout=60
                    )
                    data = resp.json()
                    if "answer" in data:
                        st.write("### 🧠 Answer:")
                        st.write(data["answer"])
                        with st.expander("📄 Retrieved Context"):
                            for i, ctx in enumerate(data["context"]):
                                st.markdown(f"**Chunk {i+1}:** {ctx[:400]}...")
                    else:
                        st.error("❌ Query failed. Backend returned no answer.")
                except Exception as e:
                    st.error(f"❌ Error connecting to backend: {e}")

st.markdown("""
---
<div style='text-align: center; font-size: 14px; color: grey;'>
Developed by <b>Kiran Kumar P</b>
</div>
""", unsafe_allow_html=True)
