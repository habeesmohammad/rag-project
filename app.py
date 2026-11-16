import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import time
import hashlib

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ──────────────────────────────────────────────────────── */
:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #161625;
    --bg-card: #1c1c30;
    --bg-card-hover: #22223a;
    --accent-primary: #6c63ff;
    --accent-secondary: #ff6584;
    --accent-gradient: linear-gradient(135deg, #6c63ff 0%, #a855f7 50%, #ff6584 100%);
    --text-primary: #e8e8f0;
    --text-secondary: #9090b0;
    --text-muted: #5e5e80;
    --border-color: rgba(108, 99, 255, 0.15);
    --shadow-glow: 0 0 30px rgba(108, 99, 255, 0.1);
    --radius: 16px;
    --radius-sm: 10px;
}

/* ── Global Styles ───────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132a 0%, #0d0d1e 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
    color: var(--text-primary) !important;
}

/* ── Headings ────────────────────────────────────────────────────────────── */
h1, h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* ── File Uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(108, 99, 255, 0.3) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    background: rgba(108, 99, 255, 0.04) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-primary) !important;
    background: rgba(108, 99, 255, 0.08) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--accent-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.45) !important;
}

/* ── Text Input ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
[data-testid="stChatInput"] textarea {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.75rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.15) !important;
}

/* ── Chat Messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.5rem !important;
    margin-bottom: 1rem !important;
    box-shadow: var(--shadow-glow) !important;
    transition: transform 0.2s ease !important;
}

[data-testid="stChatMessage"]:hover {
    transform: translateY(-1px) !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-color) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Alerts  ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-color) !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow-glow) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
.stSpinner > div {
    color: var(--accent-primary) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--accent-primary);
    border-radius: 3px;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {
    border-color: var(--border-color) !important;
}

/* ── Success / Info ──────────────────────────────────────────────────────── */
.stSuccess {
    background: rgba(16, 185, 129, 0.08) !important;
    border-color: rgba(16, 185, 129, 0.25) !important;
}

/* ── Hero Header ─────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    margin-bottom: 2rem;
    font-weight: 400;
}

/* ── Stat Cards ──────────────────────────────────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-glow);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent-primary);
    box-shadow: 0 8px 30px rgba(108, 99, 255, 0.2);
}

.stat-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}

.stat-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* ── Source Card ──────────────────────────────────────────────────────────── */
.source-card {
    background: rgba(108, 99, 255, 0.06);
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: var(--radius-sm);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

.source-badge {
    display: inline-block;
    background: var(--accent-gradient);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    letter-spacing: 0.05em;
}

/* ── Empty State ─────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}

.empty-state-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.empty-state-text {
    font-size: 1rem;
    color: var(--text-secondary);
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Pulse animation for processing ──────────────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(108, 99, 255, 0.2); }
    50% { box-shadow: 0 0 24px rgba(108, 99, 255, 0.5); }
}

.processing-indicator {
    animation: pulse-glow 2s ease-in-out infinite;
    border-radius: var(--radius);
}

/* ── Sidebar Document Chip ───────────────────────────────────────────────── */
.doc-chip {
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--text-primary);
}

.doc-chip-icon {
    font-size: 1.1rem;
}

.doc-chip-meta {
    font-size: 0.72rem;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "vectorstore": None,
        "chat_history": [],
        "documents_loaded": [],
        "total_chunks": 0,
        "total_pages": 0,
        "embeddings_model": None,
        "processing": False,
        "file_hashes": set(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─── Helpers ───────────────────────────────────────────────────────────────────
def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def get_embeddings():
    """Singleton embedding model to avoid re-loading."""
    if st.session_state.embeddings_model is None:
        st.session_state.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return st.session_state.embeddings_model


def process_pdf(uploaded_file):
    """Process a single uploaded PDF and merge into the vectorstore."""
    raw = uploaded_file.getvalue()
    h = file_hash(raw)

    if h in st.session_state.file_hashes:
        return False, "Document already loaded."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        num_pages = len(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        embeddings = get_embeddings()

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            new_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore.merge_from(new_store)

        st.session_state.file_hashes.add(h)
        st.session_state.documents_loaded.append(
            {"name": uploaded_file.name, "pages": num_pages, "chunks": len(chunks)}
        )
        st.session_state.total_chunks += len(chunks)
        st.session_state.total_pages += num_pages

        return True, f"Processed **{uploaded_file.name}** — {num_pages} pages, {len(chunks)} chunks"
    except Exception as e:
        return False, f"Error processing file: {e}"
    finally:
        os.unlink(tmp_path)


def ask_question(question: str):
    """Run retrieval QA and return answer + sources."""
    try:
        llm = OllamaLLM(model="llama2", temperature=0.3, num_predict=512)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
        )
        result = qa.invoke({"query": question})
        return result["result"], result.get("source_documents", [])
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg:
            return (
                "⚠️ **Could not connect to Ollama.** Make sure it is running:\n\n"
                "```\nollama serve\n```\n\n"
                "Then ensure the `llama2` model is pulled:\n\n"
                "```\nollama pull llama2\n```",
                [],
            )
        return f"⚠️ An error occurred: {error_msg}", []


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 RAG Intelligence")
    st.markdown("---")

    # Upload section
    st.markdown("#### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if file_hash(uf.getvalue()) not in st.session_state.file_hashes:
                with st.spinner(f"Processing {uf.name}..."):
                    success, msg = process_pdf(uf)
                    if success:
                        st.success(msg, icon="✅")
                    else:
                        st.warning(msg, icon="⚠️")

    # Loaded documents list
    if st.session_state.documents_loaded:
        st.markdown("---")
        st.markdown("#### 📚 Loaded Documents")
        for doc in st.session_state.documents_loaded:
            st.markdown(
                f"""<div class="doc-chip">
                    <span class="doc-chip-icon">📄</span>
                    <div>
                        <div>{doc['name']}</div>
                        <div class="doc-chip-meta">{doc['pages']} pages · {doc['chunks']} chunks</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Clear all button
        if st.button("🗑️  Clear All Documents", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.documents_loaded = []
            st.session_state.total_chunks = 0
            st.session_state.total_pages = 0
            st.session_state.chat_history = []
            st.session_state.file_hashes = set()
            st.session_state.embeddings_model = None
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:var(--text-muted);font-size:0.75rem;">'
        "Powered by LangChain · Ollama · FAISS</p>",
        unsafe_allow_html=True,
    )


# ─── Main Content ─────────────────────────────────────────────────────────────
# Hero header
st.markdown(
    '<div class="hero-title">🧠 Document Intelligence</div>'
    '<div class="hero-subtitle">'
    "Upload PDFs and ask questions — powered by RAG with local LLMs"
    "</div>",
    unsafe_allow_html=True,
)

# If no documents loaded, show empty state
if st.session_state.vectorstore is None:
    st.markdown(
        """<div class="empty-state">
            <div class="empty-state-icon">📄</div>
            <div class="empty-state-title">No documents loaded yet</div>
            <div class="empty-state-text">
                Upload one or more PDF files via the sidebar to build a knowledge
                base, then ask any question about your documents.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    # Stats row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-value">{len(st.session_state.documents_loaded)}</div>
                <div class="stat-label">Documents</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-icon">📑</div>
                <div class="stat-value">{st.session_state.total_pages}</div>
                <div class="stat-label">Pages</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-icon">🧩</div>
                <div class="stat-value">{st.session_state.total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history display
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("sources"):
                with st.expander("📖 View Sources", expanded=False):
                    for i, src in enumerate(entry["sources"]):
                        page_num = src.metadata.get("page", "?")
                        st.markdown(
                            f"""<div class="source-card">
                                <span class="source-badge">SOURCE {i + 1} · PAGE {page_num}</span>
                                <div>{src.page_content[:300]}{'…' if len(src.page_content) > 300 else ''}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # Chat input
    if question := st.chat_input("Ask anything about your documents…"):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents…"):
                answer, sources = ask_question(question)
            st.markdown(answer)

            if sources:
                with st.expander("📖 View Sources", expanded=False):
                    for i, src in enumerate(sources):
                        page_num = src.metadata.get("page", "?")
                        st.markdown(
                            f"""<div class="source-card">
                                <span class="source-badge">SOURCE {i + 1} · PAGE {page_num}</span>
                                <div>{src.page_content[:300]}{'…' if len(src.page_content) > 300 else ''}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

        # Save to history
        st.session_state.chat_history.append(
            {"question": question, "answer": answer, "sources": sources}
        )