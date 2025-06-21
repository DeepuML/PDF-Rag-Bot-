<p align="center">
  <h1>🧠 AI-Powered Multi-PDF Search Engine with RAG</h1>
</p>

<p align="center">
  An intelligent <b>multi-PDF search assistant</b> that allows users to upload PDFs, ask questions, and get context-aware answers using <b>semantic search + LLMs</b> with <b>Retrieval-Augmented Generation (RAG)</b>.
</p>

<p align="center">
  Built with <b>Streamlit</b>, <b>FAISS</b>, <b>EURI API</b>, and <b>pdfplumber</b>.
</p>

---

## 🚀 Features

- 📄 Upload and parse multiple PDF files  
- 🧠 Chunk and embed document content with `text-embedding-3-small` (via EURI API)  
- 🔍 Perform semantic similarity search using FAISS  
- 💬 Ask questions and receive contextual answers using RAG  
- 📊 Visualize vector relationships with PCA  
- 💾 Keeps chat memory for follow-up questions  

---

## 📁 Project Structure

AI_PDF_Search_RAG/

├── app.py # Main Streamlit application

├── .env # Environment variables (EURI API Keys)

├── requirements.txt # Python dependencies

└── README.md # This file


## ⚙️ Setup Instructions

### 1. Clone the Repository : 

git clone https://github.com/yourusername/AI_PDF_Search_RAG.git

cd AI_PDF_Search_RAG

2. Create a Virtual Environment : 

python -m venv venv

source venv/bin/activate  # or venv\\Scripts\\activate on Windows

3. Install Dependencies : 

pip install -r requirements.txt

4. Add API Keys
 
Create a .env file:

EURI_API_KEY=your_euri_api_key

EURI_CHAT_URL=https://api.euron.one/api/v1/euri/alpha/chat/completions

EURI_EMBED_URL=https://api.euron.one/api/v1/euri/alpha/embeddings

🧠 How It Works: 

PDF Extraction with pdfplumber

Text Chunking (sliding window)

Embedding using EURI embedding API

Semantic Search via FAISS

Context-Aware QA using EURI Chat model (RAG)

2D Vector Visualization with PCA

📦 Dependencies:

streamlit

pdfplumber

faiss-cpu

requests

numpy

matplotlib

scikit-learn

python-dotenv

✅ Future Enhancements:
 
DOCX and TXT support

Persistent vector storage

Chat memory persistence

OCR support for scanned PDFs

File-specific filtering and source highlighting

<p align="center"> Built with ❤️ by <b>[Deepu]</b> </p> ```









