import os
import streamlit as st
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
import pytesseract
from pdf2image import convert_from_bytes
from concurrent.futures import ThreadPoolExecutor
import platform
from dotenv import load_dotenv

print("MAIN.PY STARTED")
# =====================================
# Load Environment Variables
# =====================================
load_dotenv()

api_key = os.getenv("Api_key")#use your any api key for LLM
if not api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()
    
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
# Page Config & Memory 
# =====================================
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
memory = ConversationBufferWindowMemory(
    k=5,
    return_messages=True
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
# Session State
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# =====================================
# LLM
# =====================================
model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=1024,
    streaming=True
)

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
# Prompt
# =====================================
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Previous Conversation:
{history}

Context:
{context}

Question:
{question}

Answer:
""")

# =====================================
# Retrieval Function
# =====================================
@traceable
def retrieve_relevant_chunks(vector_store, query):

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    return retriever.invoke(query)

# =====================================
# Sidebar
# =====================================
with st.sidebar:

    st.title(" Documents")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True
    )

    
    if uploaded_files:

        file_id = "-".join(
            [f"{file.name}-{file.size}" for file in uploaded_files]
        )

        if st.session_state.processed_file != file_id:

            with st.spinner("Processing Documents..."):

                documents = []

                total_pages = 0

                for uploaded_file in uploaded_files:

                    extension = (
                        uploaded_file.name
                        .split(".")[-1]
                        .lower()
                    )

                    text = ""

                    try:

                        # ------------------
                        # PDF
                        # ------------------
                        if extension == "pdf":
                        
                            uploaded_file.seek(0)
                            pdf_bytes = uploaded_file.read()
                        
                            reader = PdfReader(uploaded_file)
                        
                            total_pages += len(reader.pages)
                        
                            page_images = None
                        
                            for page_no, page in enumerate(reader.pages):
                        
                                page_text = page.extract_text() or ""
                        
                                # Normal text page
                                if len(page_text.strip()) > 50:
                        
                                    documents.append(
                                        {
                                            "source": uploaded_file.name,
                                            "page": page_no + 1,
                                            "text": page_text,
                                        }
                                    )
                        
                                else:
                        
                                    # Convert PDF pages to images only once
                                    if page_images is None:
                        
                                        page_images = convert_from_bytes(
                                            pdf_bytes,
                                            dpi=150,
                                            poppler_path=POPPLER_PATH,
                                        )
                        
                                    try:
                        
                                        ocr_text = pytesseract.image_to_string(
                                            page_images[page_no],
                                            config="--psm 6"
                                        )
                        
                                        if ocr_text.strip():
                        
                                            documents.append(
                                                {
                                                    "source": uploaded_file.name,
                                                    "page": page_no + 1,
                                                    "text": ocr_text,
                                                }
                                            )
                        
                                    except Exception as e:
                        
                                        st.warning(
                                            f"OCR failed on page {page_no + 1}: {e}"
                                        )

                        # ------------------
                        # DOCX
                        # ------------------
                        elif extension == "docx":

                            doc = Document(uploaded_file)

                            text = "\n".join(
                                para.text
                                for para in doc.paragraphs
                            )

                        # ------------------
                        # PPTX
                        # ------------------
                        elif extension == "pptx":

                            prs = Presentation(uploaded_file)

                            for slide in prs.slides:

                                for shape in slide.shapes:

                                    if hasattr(shape, "text"):
                                        text += shape.text + "\n"

                        # ------------------
                        # TXT
                        # ------------------
                        elif extension == "txt":

                            text = (
                                uploaded_file
                                .read()
                                .decode("utf-8")
                            )

                        if text.strip():

                            documents.append(
                                {
                                    "source": uploaded_file.name,
                                    "page": page_no + 1,
                                    "text": text,
                                }
                            )

                    except Exception as e:

                        st.warning(
                            f"Could not process "
                            f"{uploaded_file.name}: {e}"
                        )

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )

                all_chunks = []

                metadatas = []

                for doc in documents:

                    chunks = text_splitter.split_text(
                        doc["text"]
                    )

                    all_chunks.extend(chunks)

                    metadatas.extend(
                    [
                        {
                            "source": doc["source"],
                            "page": doc["page"]
                        }
                        for _ in chunks
                    ]
                )
                if not all_chunks:
                    st.error("No text could be extracted from the uploaded files.")
                    st.stop()
                try:
                    embedding_model = load_embeddings()
                except Exception as e:
                    st.error(f"Embedding error: {e}")
                    st.stop()
                
                try:
                    st.session_state.vector_store = FAISS.from_texts(
                        texts=all_chunks,
                        embedding=embedding_model,
                        metadatas=metadatas,
                    )
                except Exception as e:
                    st.error(f"FAISS error: {e}")
                    st.stop()

                st.session_state.processed_file = file_id
                st.session_state.messages = []

            st.success("✅ Documents Indexed Successfully")
            st.write("Indexed Documents:")

            sources = set()

            for meta in metadatas:
                sources.add(meta["source"])

            for source in sources:
                st.write(source)

            st.write(f"Total Files Indexed: {len(sources)}")

            st.metric(
                "Files",
                len(uploaded_files)
            )

            st.metric(
                "Pages",
                total_pages
            )

            st.metric(
                "Chunks",
                len(all_chunks)
            )

        else:
            st.info("Using cached vector store")


    if st.button(
        " Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()
    

# =====================================
# Header
# =====================================
st.markdown(
    """
    <div class="main-header">
        <h1> PDF Question Answering</h1>
        <p>Upload a PDF and ask questions about its contents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================
# Chat History
# =====================================
if not st.session_state.messages:

    st.info(
        " Upload a PDF from the sidebar and start asking questions."
    )

history = "\n".join(
    [
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.messages[-10:]
    ]
)
# =====================================
# Chat Input
# =====================================
@traceable(name="generate_answer")
def generate_answer(prompt_text):
    return model.invoke(prompt_text)
question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        # Show User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate Answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                relevant_chunks = retrieve_relevant_chunks(
                    st.session_state.vector_store,
                    question,
                )

                context = "\n\n".join(
                    chunk.page_content
                    for chunk in relevant_chunks
                )

                formatted_prompt = prompt.format(
                history=history,
                context=context,
                question=question,
            )
                
                placeholder = st.empty()

                answer = ""
                
                for chunk in model.stream(formatted_prompt):
                
                    if chunk.content:
                
                        answer += chunk.content
                
                        placeholder.markdown(answer)

            with st.expander("📚 Sources Used"):

                for i, chunk in enumerate(
                    relevant_chunks,
                    start=1
                ):
                    source = chunk.metadata.get(
                        "source",
                        "Unknown"
                    )
                    
                    page = chunk.metadata.get(
                        "page",
                        "?"
                    )
                    
                    st.markdown(
                        f"📄 {source} | Page {page}"
                    )
                    
                    st.write(chunk.page_content[:500])
        # Save Assistant Message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()
