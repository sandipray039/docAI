"""
DocAI - Streamlit frontend for the RAG API.
"""

import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/api")
TIMEOUT = int(os.getenv("API_TIMEOUT", "300"))


st.set_page_config(page_title="DocAI - Chat with your PDFs", page_icon="📄", layout="wide")
st.title("📄 DocAI - Chat with your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []


def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=TIMEOUT)


def api_post(path: str, **kw):
    return requests.post(f"{API_URL}{path}", timeout=TIMEOUT, **kw)


def api_delete(path: str):
    return requests.delete(f"{API_URL}{path}", timeout=TIMEOUT)


with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest", use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                r = api_post("/upload", files=files)
                if r.ok:
                    data = r.json()
                    st.success(f"Indexed {data['chunks_added']} chunks from {data['pages']} pages.")
                else:
                    st.error(f"{r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("Indexed")
    try:
        r = api_get("/documents")
        docs = r.json() if r.ok else []
    except Exception as e:
        docs = []
        st.error(f"Cannot reach API: {e}")

    if not docs:
        st.caption("No documents yet.")
    else:
        for d in docs:
            cols = st.columns([4, 1])
            cols[0].write(f"**{d['source']}**  \n_{d['chunks']} chunks_")
            if cols[1].button("🗑", key=f"del_{d['source']}"):
                api_delete(f"/documents/{d['source']}")
                st.rerun()

    st.divider()
    sources = ["(all)"] + [d["source"] for d in docs]
    chosen = st.selectbox("Filter by source", sources)
    top_k = st.slider("Top K", min_value=1, max_value=15, value=5)

    if st.button("Reset index", type="secondary", use_container_width=True):
        api_post("/reset")
        st.session_state.messages = []
        st.rerun()


# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — p.{s.get('page','?')}")
                    st.caption(s.get("snippet", ""))


prompt = st.chat_input("Ask something about your PDFs...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            payload = {
                "question": prompt,
                "top_k": top_k,
                "source": None if chosen == "(all)" else chosen,
            }
            try:
                r = api_post("/query", json=payload)
                if r.ok:
                    data = r.json()
                    placeholder.markdown(data["answer"])
                    if data.get("sources"):
                        with st.expander("Sources"):
                            for s in data["sources"]:
                                st.markdown(f"**{s['source']}** — p.{s.get('page','?')}")
                                st.caption(s.get("snippet", ""))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": data["answer"], "sources": data.get("sources", [])}
                    )
                else:
                    placeholder.error(f"{r.status_code}: {r.text}")
            except Exception as e:
                placeholder.error(f"Request failed: {e}")
