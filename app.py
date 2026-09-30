import streamlit as st
from src.graph import rag_app

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title(" RAG Chatbot")
st.caption("Powered by LangGraph, Pinecone, and strictly grounded on the Konverge AI eBook.")

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            with st.expander(" Retrieved Context & Confidence"):
                st.write(f"**Confidence Score:** `{msg['metadata']['confidence']}`")
                for c in msg["metadata"]["chunks"]:
                    st.markdown(f"- **Page {c['page']} (Score: {c['score']})**: {c['content'][:250]}...")

prompt = st.chat_input("Ask a question about Agentic AI eBook")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Traversing LangGraph & retrieving context..."):
            res = rag_app.invoke({"question": prompt, "context": [], "final_answer": "", "confidence_or_score": 0.0})
            st.markdown(res["final_answer"])
            
            with st.expander("Retrieved Context & Confidence"):
                st.write(f"**Confidence Score:** `{res['confidence_or_score']}`")
                for c in res["context"]:
                    st.markdown(f"- **Page {c['page']} (Score: {c['score']})**: {c['content'][:250]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": res["final_answer"],
        "metadata": {
            "confidence": res["confidence_or_score"],
            "chunks": res["context"]
        }
    })