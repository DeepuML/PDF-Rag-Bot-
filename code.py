import pdfplumber
import numpy as np
import faiss
import requests
import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()


EURI_API_KEY=os.getenv('EURI_API_KEY')
EURI_CHAT_URL=os.getenv('EURI_CHAT_URL')
EURI_EMBED_URL = os.getenv('EURI_EMBED_URL')

conversation_memory=[]

def extract_text_from_pdf(pdf_path):
    full_text= ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n" 
    return full_text

def split_text(text,chunk_size=5000,overlap=1000):
    chunks=[]
    start=0
    while start<len(text):
        end=start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_euri_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    res = requests.post(EURI_EMBED_URL, headers=headers, json=payload)
    return np.array([d["embedding"] for d in res.json()["data"]])

def build_vector_store(chunks):
    embeddings = get_euri_embeddings(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings

def retrieve_context(question, chunks, index, embeddings, top_k=3):
    q_embed = get_euri_embeddings([question])[0]
    D, I = index.search(np.array([q_embed]), top_k)
    return "\n\n".join([chunks[i] for i in I[0]])



def ask_euri_with_context(question, context, memory=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions from a document."}
    ]
    if memory:
        messages.extend(memory)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.3
    }

    res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)
    reply = res.json()['choices'][0]['message']['content']
    memory.append({"role": "user", "content": question})
    memory.append({"role": "assistant", "content": reply})
    return reply

def rewrite_question(user_input):
    messages = [
        {"role": "system", "content": "You are an assistant that rephrases unclear or poorly worded questions into clear, concise ones."},
        {"role": "user", "content": user_input}
    ]

    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4.1-nano",  
        "messages": messages,
        "temperature": 0.2
    }

    try:
        res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)
        res.raise_for_status()
        rewritten = res.json()['choices'][0]['message']['content']
        return rewritten.strip()
    except Exception as e:
        st.error(f"Question rewriting failed: {e}")
        return user_input 



# Streamlit UI
st.title("📄 PDF Knowledge Extraction RAG Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_question = st.text_input("Ask a question about the document")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    full_text = extract_text_from_pdf("temp.pdf")
    chunks = split_text(full_text)
    index, embeddings = build_vector_store(chunks)

    st.success("✅ PDF loaded and indexed.")

    if user_question:
        rewritten_question = rewrite_question(user_question)
        context = retrieve_context(rewritten_question, chunks, index, embeddings)
        response = ask_euri_with_context(rewritten_question, context, conversation_memory)

        # 💬 Display
        print("🧠 Conversation Memory:", conversation_memory)
        st.markdown("### ✅ Answer:")
        st.write(response)


