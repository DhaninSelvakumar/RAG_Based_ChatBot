# chatBot_UI_rag.py
import streamlit as st
import requests
import time

RAG_URL = "http://localhost:8010"

st.set_page_config(page_title="TinyLlama + RAG", page_icon="📚")
st.title("📚 TinyLlama + RAG (Local)")
st.write("Ground answers in your own documents. Upload files, then ask questions.")

# ===== Ingestion UI =====
st.subheader("Ingest documents")
with st.expander("Upload files (txt, md, pdf)"):
    ns = st.text_input("Namespace (optional)", value="default")
    files = st.file_uploader("Choose files", type=["txt", "md", "pdf"], accept_multiple_files=True)
    if st.button("Ingest") and files:
        with st.spinner("Embedding & indexing..."):
            files_to_send = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            resp = requests.post(f"{RAG_URL}/ingest", files=files_to_send, data={"namespace": ns})
        if resp.status_code == 200:
            st.success(f"Ingested {resp.json().get('chunks_added')} chunks into {resp.json().get('collection')}")
        else:
            st.error(f"Ingest error: {resp.text}")

# ===== Chat state =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for role, text in st.session_state.messages:
    st.markdown(f"**{role.title()}:** {text}")

# ===== Chat box =====
use_rag = st.checkbox("Use RAG", value=True)
top_k = st.number_input("Top-k context chunks", min_value=1, max_value=10, value=4, step=1)
user_input = st.text_input("Ask a question:")

if st.button("Send") and user_input:
    st.session_state.messages.append(("user", user_input))
    with st.spinner("Thinking..."):
        try:
            payload = {"prompt": user_input, "use_rag": use_rag, "top_k": int(top_k)}
            resp = requests.post(f"{RAG_URL}/generate", json=payload, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("response", "")
                sources = data.get("sources", [])
            else:
                answer = f"⚠️ Error from RAG service: {resp.text}"
                sources = []
        except Exception as e:
            answer = f"⚠️ Connection error: {e}"
            sources = []

    # Typing effect
    placeholder = st.empty()
    streamed_text = ""
    for ch in answer:
        streamed_text += ch
        placeholder.markdown(f"**Assistant:** {streamed_text}")
        time.sleep(0.01)

    st.session_state.messages.append(("assistant", answer))

    # Show sources
    if sources:
        st.markdown("### Sources")
        for s in sources:
            st.markdown(
                f"- `{s.get('filename')}` (ns: `{s.get('namespace')}` • chunk: {s.get('chunk_index')} • score: {s.get('score'):.3f})"
            )

    st.rerun()