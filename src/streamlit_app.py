import os
import streamlit as st
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
import pytesseract
from pdf2image import convert_from_bytes
import platform
from dotenv import load_dotenv
 
# =====================================
# Load Environment Variables
# =====================================
load_dotenv()
 
api_key = os.getenv("Api_key")
hf_token = os.getenv("Huggingface_api_key")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
 
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    POPPLER_PATH = r"poppler-26.02.0\Library\bin"
else:
    POPPLER_PATH = None
 
# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# =====================================
# Custom CSS
# =====================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# =====================================
# Guard: API Key Check
# =====================================
if not api_key:
    st.error("`Api_key` not found. Please add your Groq API key as a HuggingFace Space secret named `Api_key`.")
    st.stop()
 
# =====================================
# Session State
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
 
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
 
# =====================================
# LLM — streaming enabled
# =====================================
@st.cache_resource
def load_model(key):
    return ChatGroq(
        api_key=key,
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024,
        streaming=True,
    )
 
model = load_model(api_key)
 
# =====================================
# Embeddings
# =====================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
    )
 
# =====================================
# Prompt  (history + context + question)
# =====================================
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I don't know based on the uploaded document."
 
Previous Conversation:
{history}
 
Context:
{context}
 
Question:
{question}
 
Answer:"""
)
 
# =====================================
# Retrieval
# =====================================
@traceable
def retrieve_relevant_chunks(vector_store, query):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )
    return retriever.invoke(query)
 
# =====================================
# Document Text Extraction Helpers
# =====================================
def extract_pdf(uploaded_file):
    """Returns list of {source, page, text} dicts. Uses OCR per-page as fallback."""
    docs = []
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io_from_bytes(pdf_bytes))
    total_pages = len(reader.pages)
    page_images = None  # lazy — only rendered if needed
 
    for page_no, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
 
        if len(page_text.strip()) > 50:
            docs.append({"source": uploaded_file.name, "page": page_no + 1, "text": page_text})
        else:
            # Render images lazily once
            if page_images is None:
                page_images = convert_from_bytes(pdf_bytes, dpi=150, poppler_path=POPPLER_PATH)
            try:
                ocr_text = pytesseract.image_to_string(page_images[page_no], config="--psm 6")
                if ocr_text.strip():
                    docs.append({"source": uploaded_file.name, "page": page_no + 1, "text": ocr_text})
            except Exception as e:
                st.warning(f"OCR failed on page {page_no + 1} of {uploaded_file.name}: {e}")
 
    return docs, total_pages
 
 
def extract_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if text.strip():
        return [{"source": uploaded_file.name, "page": 1, "text": text}]
    return []
 
 
def extract_pptx(uploaded_file):
    prs = Presentation(uploaded_file)
    docs = []
    for slide_no, slide in enumerate(prs.slides, start=1):
        text = "\n".join(
            shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()
        )
        if text.strip():
            docs.append({"source": uploaded_file.name, "page": slide_no, "text": text})
    return docs
 
 
def extract_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    if text.strip():
        return [{"source": uploaded_file.name, "page": 1, "text": text}]
    return []
 
 
# BytesIO helper so PdfReader gets a seekable stream from bytes
import io
def io_from_bytes(b: bytes):
    return io.BytesIO(b)
 
# =====================================
# Sidebar
# =====================================
with st.sidebar:
    st.title(" Documents")
 
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
    )
 
    if uploaded_files:
        file_id = "-".join(f"{f.name}-{f.size}" for f in uploaded_files)
 
        if st.session_state.processed_file != file_id:
 
            with st.spinner("Processing Documents..."):
                all_documents = []
                total_pages = 0
 
                for uploaded_file in uploaded_files:
                    ext = uploaded_file.name.split(".")[-1].lower()
                    try:
                        if ext == "pdf":
                            docs, pages = extract_pdf(uploaded_file)
                            all_documents.extend(docs)
                            total_pages += pages
                        elif ext == "docx":
                            all_documents.extend(extract_docx(uploaded_file))
                        elif ext == "pptx":
                            all_documents.extend(extract_pptx(uploaded_file))
                        elif ext == "txt":
                            all_documents.extend(extract_txt(uploaded_file))
                    except Exception as e:
                        st.warning(f"Could not process {uploaded_file.name}: {e}")
 
                if not all_documents:
                    st.error("No text could be extracted from the uploaded files.")
                    st.stop()
 
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                )
 
                all_chunks = []
                metadatas = []
 
                for doc in all_documents:
                    chunks = text_splitter.split_text(doc["text"])
                    all_chunks.extend(chunks)
                    metadatas.extend(
                        [{"source": doc["source"], "page": doc["page"]} for _ in chunks]
                    )
 
                if not all_chunks:
                    st.error("Chunking produced no results.")
                    st.stop()
 
                try:
                    embedding_model = load_embeddings()
                except Exception as e:
                    st.error(f"Embedding model error: {e}")
                    st.stop()
 
                try:
                    st.session_state.vector_store = FAISS.from_texts(
                        texts=all_chunks,
                        embedding=embedding_model,
                        metadatas=metadatas,
                    )
                except Exception as e:
                    st.error(f"FAISS indexing error: {e}")
                    st.stop()
 
                st.session_state.processed_file = file_id
                st.session_state.messages = []
 
            st.success(" Documents Indexed Successfully")
 
            sources = sorted({m["source"] for m in metadatas})
            for src in sources:
                st.write(f"• {src}")
 
            col1, col2, col3 = st.columns(3)
            col1.metric("Files", len(uploaded_files))
            col2.metric("Pages", total_pages)
            col3.metric("Chunks", len(all_chunks))
 
        else:
            st.info("Using cached vector store")
 
    st.divider()
 
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
 
# =====================================
# Header
# =====================================
st.markdown(
    """
    <div class="main-header">
        <h1> RAG Document Q&A</h1>
        <p>Upload a document and ask questions about its contents</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# =====================================
# Chat History Display
# =====================================
if not st.session_state.messages:
    st.info("📎 Upload a document from the sidebar and start asking questions.")
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# =====================================
# Build conversation history string
# (last 10 messages = 5 turns)
# =====================================
def build_history(messages, n=10):
    recent = messages[-n:] if len(messages) > n else messages
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) if lines else "None"
 
# =====================================
# Chat Input & Answer Generation
# =====================================
question = st.chat_input("Ask a question about your document...")
 
if question:
    if st.session_state.vector_store is None:
        st.warning(" Please upload a document first.")
    else:
        # Append & display user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
 
        # Generate streamed answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_chunks = retrieve_relevant_chunks(
                    st.session_state.vector_store, question
                )
                context = "\n\n".join(c.page_content for c in relevant_chunks)
                history_text = build_history(st.session_state.messages[:-1])  # exclude current question
 
                formatted_prompt = prompt.format(
                    history=history_text,
                    context=context,
                    question=question,
                )
 
            # Stream tokens into a placeholder
            placeholder = st.empty()
            answer = ""
            for chunk in model.stream(formatted_prompt):
                if chunk.content:
                    answer += chunk.content
                    placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
 
            # Sources expander
            with st.expander(" Sources Used"):
                for i, chunk in enumerate(relevant_chunks, start=1):
                    source = chunk.metadata.get("source", "Unknown")
                    page = chunk.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i}** |  `{source}` — Page {page}")
                    st.write(chunk.page_content[:500])
                    st.divider()
 
        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()