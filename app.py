import os
import pandas as pd
import faiss
import gradio as gr
import gdown

from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------- CONFIG ----------------
DATA_URL = "https://drive.google.com/uc?id=1PPxR62m3xMar6XWfE4mLXpI_tfvlt-lb"
CSV_PATH = "nobel_data.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5


# ---------------- INIT ----------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedder = SentenceTransformer(EMBEDDING_MODEL)


# ---------------- LOAD DATA ----------------
gdown.download(DATA_URL, CSV_PATH, quiet=True)
df = pd.read_csv(CSV_PATH)

for col in ["year", "category", "full_name", "motivation"]:
    df[col] = df[col].astype(str).fillna("")

df["document"] = (
    "In " + df["year"] +
    ", the Nobel Prize in " + df["category"] +
    " was awarded to " + df["full_name"] +
    ". Motivation: " + df["motivation"]
)

df = df[df["document"].str.len() > 50]
documents = df["document"].tolist()


# ---------------- FAISS ----------------
embeddings = embedder.encode(documents, convert_to_numpy=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# ---------------- RAG ----------------
def retrieve(query):
    q_emb = embedder.encode([str(query)], convert_to_numpy=True)
    distances, indices = index.search(q_emb, TOP_K)

    if distances[0].mean() > 1.5:
        return []

    return [documents[i] for i in indices[0]]


def ask_llm(context, question):
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say:
"I don't know based on the Nobel Prize data."
Context:
{context}
Question:
{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def chat(query):
    retrieved = retrieve(query)

    if not retrieved:
        return (
            "### ❌ Answer Not Found\n\n"
            "I don't know based on the Nobel Prize data."
        )

    context = "\n\n".join(retrieved)
    answer = ask_llm(context, query)

    return f"### ✅ Answer\n\n{answer}"


# ---------------- UI (ACCESSIBLE & CLEAN) ----------------
css = """
body {
    background: #0f172a;
    color: #f8fafc;
    font-family: Inter, system-ui;
}
#container {
    max-width: 900px;
    margin: auto;
    padding-top: 20px;
}
h1, h3 {
    color: #e5e7eb;
}
.gr-textbox textarea {
    background: #020617;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 12px;
}
.gr-button {
    background: #4f46e5;
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 10px 16px;
}
.gr-button:hover {
    background: #4338ca;
}
#answer-box {
    background: #ecfeff;
    color: #042f2e;
    border-left: 6px solid #14b8a6;
    border-radius: 12px;
    padding: 18px;
    margin-top: 18px;
    font-size: 16px;
    line-height: 1.6;
}
"""

with gr.Blocks(css=css) as app:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """
            # 🏆 Nobel Prize Knowledge Assistant
            Ask questions strictly from the Nobel Prize knowledge base.
            """
        )

        query = gr.Textbox(
            label="🔍 Your Question",
            placeholder="Who won the Nobel Prize in Physics in 1921?"
        )

        ask_btn = gr.Button("Get Answer")

        answer = gr.Markdown(elem_id="answer-box")

        ask_btn.click(chat, inputs=query, outputs=answer)

app.launch()
