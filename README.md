🤖 RAG Chatbot – Task 3

This is a Retrieval-Augmented Generation (RAG) Chatbot built using Streamlit, SentenceTransformers, and FAISS.
It retrieves the most relevant facts from your uploaded knowledge base file based on the questions you ask.

🚀 Features

📂 Upload Knowledge Base (.txt file)

🔎 Semantic Search using Sentence Transformers + FAISS

💬 Ask Questions and get context-aware answers

📝 Chat History (view, clear, start new chat)

💾 Download Chat History as a text file

🎨 Custom UI Styling with modern look and feel

🛠️ Tech Stack

Streamlit
 – Web UI

SentenceTransformers
 – Embedding Model

FAISS
 – Vector Search

NumPy
 – Numerical Computations


 ⚙️ Installation
1. Clone the repository:git clone https://github.com/sundasmustaf69/task3.git
cd task3

2. Create and activate a virtual environment:python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Mac/Linux

3.Install dependencies:pip install -r requirements.txt

Run the app:streamlit run app.py
