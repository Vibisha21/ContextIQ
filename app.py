import os
import re
import json
import pickle
import streamlit as st
from PyPDF2 import PdfReader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import pipeline

# ----------------------
# CONFIG
# ----------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
VECTORSTORE_PATH = "vectorstore.pkl"

MAX_CONTEXT_CHARS = 800
MIN_FLASHCARDS = 2

# ----------------------
# HELPERS
# ----------------------
def load_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def split_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def save_vectorstore(vs):
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vs, f)


def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            return pickle.load(f)
    return None


def deduplicate_text(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(dict.fromkeys(lines))


def safe_context(text):
    return deduplicate_text(text)[:MAX_CONTEXT_CHARS]


def extract_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def unique_flashcards(cards):
    seen = set()
    unique = []
    for c in cards:
        q = c["question"].lower().strip()
        if q not in seen:
            unique.append(c)
            seen.add(q)
    return unique


def smart_fallback_flashcards(context, min_q=2):
    """
    Always generates meaningful questions from the material itself.
    """
    cards = []
    sentences = re.split(r"[.\n]", context)

    # 1️⃣ Definition-based
    for s in sentences:
        s = s.strip()
        if len(s) < 40:
            continue

        if " is " in s.lower():
            term = s.split(" is ")[0].strip()
            cards.append({
                "question": f"What is {term}?",
                "answer": s
            })

        if len(cards) >= min_q:
            return cards

    # 2️⃣ Explanation-based (last resort, still grounded)
    keywords = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", context)
    for kw in keywords:
        cards.append({
            "question": f"Explain {kw}.",
            "answer": context.split(".")[0]
        })
        if len(cards) >= min_q:
            return cards

    return cards


def is_analysis_question(q):
    return any(k in q.lower() for k in ["list", "topics", "overview", "summarize"])

# ----------------------
# MODEL LOADERS
# ----------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=200
    )
    return HuggingFacePipeline(pipeline=pipe)

# ----------------------
# PROMPTS
# ----------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

TOPIC_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
From the material below, list the major topics covered.
Use bullet points.
Do not repeat topics.

Material:
{context}

Topics:
"""
)

FLASHCARD_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
From the context, generate exam-oriented question–answer pairs.

Rules:
- Questions must be UNIQUE
- Short, factual answers
- Output ONLY valid JSON

Format:
[
  {{
    "question": "string",
    "answer": "string"
  }}
]

Context:
{context}
"""
)

# ----------------------
# STREAMLIT UI
# ----------------------
st.set_page_config(page_title="ContextIQ: Exam Prep Assistant", layout="wide")
st.title("📚 ContextIQ: Exam Preparation Assistant")

# ----------------------
# SESSION STATE
# ----------------------
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# ----------------------
# FILE UPLOAD
# ----------------------
uploaded_files = st.file_uploader(
    "Upload PDF notes",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    raw_text = load_pdf_text(uploaded_files)
    texts = split_text(raw_text)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    save_vectorstore(vectorstore)

    st.success("✅ Documents indexed successfully")

# ----------------------
# LOAD VECTORSTORE
# ----------------------
vectorstore = load_vectorstore()

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # ----------------------
    # ASK QUESTION
    # ----------------------
    st.subheader("💬 Ask a Question")
    user_q = st.text_input("Ask from your notes")

    if user_q:
        docs = retriever.invoke(user_q)
        context = safe_context("\n".join(d.page_content for d in docs))

        if is_analysis_question(user_q):
            answer = (TOPIC_PROMPT | llm).invoke({"context": context})
        else:
            answer = qa_chain.invoke(user_q)["result"]

        st.success(answer)

    # ----------------------
    # TOOLS
    # ----------------------
    st.subheader("🛠 Tools")
    col1, col2 = st.columns(2)

    # LIST TOPICS (UNCHANGED)
    with col1:
        if st.button("📌 List Topics"):
            docs = retriever.invoke("overview syllabus table of contents")
            context = safe_context("\n".join(d.page_content for d in docs))

            topics = (TOPIC_PROMPT | llm).invoke({"context": context})

            st.markdown("### 📚 Topics in the Material")
            st.write(topics)

    # FLASHCARDS (NEVER FAILS)
    with col2:
        if st.button("🧠 Generate Flashcards"):
            docs = retriever.invoke("definitions key concepts")
            context = safe_context("\n".join(d.page_content for d in docs))

            raw = llm.invoke(FLASHCARD_PROMPT.format(context=context))
            cards = extract_json(raw) or []

            cards = unique_flashcards(cards)

            if len(cards) < MIN_FLASHCARDS:
                cards.extend(
                    smart_fallback_flashcards(context, MIN_FLASHCARDS)
                )
                cards = unique_flashcards(cards)

            st.session_state.flashcards = cards[:MIN_FLASHCARDS]
            st.success("✅ Flashcards generated")

    # ----------------------
    # FLASHCARDS DISPLAY
    # ----------------------
    if st.session_state.flashcards:
        st.subheader("📇 Flashcards")

        for i, c in enumerate(st.session_state.flashcards, 1):
            st.write(f"**Q{i}. {c['question']}**")
            st.write(c["answer"])

else:
    st.info("Upload PDFs to start")
