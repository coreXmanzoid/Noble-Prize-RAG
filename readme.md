# 🧠 CSV RAG Knowledge Base Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Gradio**, **FAISS**, **Sentence Transformers**, and **Groq LLMs**.  
The application allows users to ask questions from a **CSV-based knowledge base** hosted on Google Drive.

---

## 🚀 Live Demo
> Deployed on **Hugging Face Spaces**

---

## 🖼️ Screenshot
<!-- Replace the link below with your uploaded screenshot URL -->
![App Screenshot](https://i.ibb.co/jktbR26r/Project-Screen-Shot.png)

---

## ✨ Features
- 📄 CSV-based knowledge ingestion (Google Drive supported)
- 🔍 Semantic search using **FAISS**
- 🧠 Embeddings via **Sentence Transformers**
- ⚡ Fast inference with **Groq (LLaMA 3.3 70B)**
- 🎨 Modern, responsive Gradio UI
- 🚫 Strict grounding (no hallucinations outside context)

---

## 🏗️ Tech Stack
- **Python**
- **Gradio** (UI)
- **FAISS (CPU)** (Vector Search)
- **Sentence-Transformers** (Embeddings)
- **Groq API** (LLM)
- **Pandas** (CSV parsing)

---

## 📁 Project Structure
```text
.
├── app.py
├── requirements.txt
├── README.md
```

---

## 🔐 Environment Variables
This project requires a Groq API key.

### Hugging Face Spaces
Go to:
```
Settings → Secrets
```
Add:
```
GROQ_API_KEY=your_groq_api_key
```

---

## 📦 Installation (Local / Colab)

```bash
pip install -r requirements.txt
```

Run the app:
```bash
python app.py
```

---

## 📊 CSV Knowledge Base
- CSV file is hosted on **Google Drive**
- Public access is required
- Each row is converted into searchable semantic chunks

Example format:
```csv
column1,column2,column3
value1,value2,value3
```

---

## 🧪 How It Works
1. CSV is downloaded from Google Drive
2. Rows are chunked and embedded
3. FAISS indexes embeddings
4. User query → semantic search
5. Relevant context sent to Groq LLM
6. Answer returned strictly from knowledge base

---

## ⚠️ Limitations
- FAISS index is rebuilt on startup
- Large CSV files may increase cold-start time
- Single CSV source (can be extended)

---

## 🔮 Future Improvements
- Multi-CSV support
- Upload CSV directly from UI
- Chat history
- Source citations per answer
- Persistent FAISS index

---

## 👨‍💻 Author
**Hammad Ashraf (F24BDOCS1E02086)**  

BS Computer Science | Python & Flask Web Developer

---

## 📄 License
MIT License
