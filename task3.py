import streamlit as st
import numpy as np
import faiss
import io
from sentence_transformers import SentenceTransformer

# ========================
# Session State Initialize
# ========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ========================
# Custom CSS for Dark Mode
# ========================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1c1c1c;
        border-right: 2px solid #444;
        color: #ffffff;
    }

    /* Headings */
    h1, h2, h3 {
        color: #ffffff;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border: none;
        margin-bottom: 6px;
        background-color: #333333;
        color: #ffffff;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        transition: all 0.2s ease-in-out;
        background-color: #555555;
    }

    /* Chat history box */
    .chat-box {
        background-color: #1e1e1e;
        border: 1px solid #444;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        color: #ffffff;
    }

    /* Input box */
    input[type="text"], textarea {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# Sidebar - Controls
# ========================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    uploaded_file = st.file_uploader("📂 Upload Knowledge Base", type=["txt"])

    if uploaded_file:
        # Load knowledge base
        kb_text = uploaded_file.read().decode("utf-8")
        st.session_state.knowledge_base = [
            line.strip() for line in kb_text.split("\n") if line.strip()
        ]

        # Encode embeddings
        embeddings = st.session_state.model.encode(
            st.session_state.knowledge_base, convert_to_numpy=True
        )
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        st.session_state.faiss_index = index
        st.success("✅ Knowledge base uploaded and indexed!")

    # 🆕 New Chat
    if st.button("🆕 New Chat"):
        st.session_state.chat_history = []
        st.success("✨ Started a new chat!")

    # 🗑️ Clear Chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.success("✅ Chat history cleared!")

    # 💾 Download Chat
    if st.button("💾 Download Chat"):
        chat_text = "\n\n".join(
            [f"Q: {q}\n{a}" for q, a in st.session_state.chat_history]
        )
        if chat_text.strip():
            buffer = io.BytesIO(chat_text.encode("utf-8"))
            st.download_button(
                "⬇️ Save Chat History",
                buffer,
                file_name="chat_history.txt",
                mime="text/plain",
            )
        else:
            st.warning("⚠️ Chat history is empty!")

    # 📝 Chat History
    st.markdown("## 📝 Chat History")
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(
                f"<div class='chat-box'><b>Q{i}:</b> {q}<br><i>{a}</i></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No chat history yet. Start asking questions!")

# ========================
# Retrieval Function
# ========================
def retrieve_top_k_facts(question, k=3):
    if not st.session_state.faiss_index:
        return ["⚠️ Please upload a knowledge base first!"]

    model = st.session_state.model
    question_vec = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(question_vec)

    distances, indices = st.session_state.faiss_index.search(question_vec, k)
    top_facts = [st.session_state.knowledge_base[i] for i in indices[0]]
    return top_facts

# ========================
# Main App
# ========================
st.title("🤖 RAG Chatbot - Dark Mode")
st.write("Ask a question about your uploaded knowledge base and get the most relevant facts!")

# ✅ Form with Submit button
with st.form(key="qa_form"):
    user_question = st.text_input("❓ Enter your question:")
    submit_button = st.form_submit_button("🚀 Submit")

if submit_button and user_question:
    with st.spinner("🔎 Searching for the most relevant facts..."):
        results = retrieve_top_k_facts(user_question)

    st.success("🎯 Top relevant facts:")
    for i, fact in enumerate(results, start=1):
        st.markdown(f"**{i}.** {fact}")

    # Save in chat history
    st.session_state.chat_history.append((user_question, "\n".join(results)))
