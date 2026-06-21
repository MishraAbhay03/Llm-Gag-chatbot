---
title: RAG Document Q&A
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Multi-document RAG chatbot with OCR, Vision AI, Hybrid Retri
license: mit
---

# 📄 RAG Document Q&A with Vision, OCR & Hybrid Retrieval

An advanced Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, FAISS, BM25, OCR, and Vision-Language Models.

The application allows users to upload multiple documents and ask natural language questions while receiving citation-aware answers grounded in the uploaded content.

## ✨ Features

### 📚 Multi-Document Support

* Upload multiple PDF, DOCX, PPTX, and TXT files
* Query across all uploaded documents simultaneously
* File-level and page-level source tracking

### 🔍 Hybrid Retrieval

* FAISS vector search
* BM25 keyword search
* Reciprocal Rank Fusion (RRF)
* Maximum Marginal Relevance (MMR)

This combination improves retrieval accuracy for both semantic and keyword-based queries.

### 🧠 Query Expansion

The system automatically rewrites complex questions into more retrieval-friendly queries while preserving the original intent.

### 📄 OCR Support

Scanned PDFs and image-based documents are automatically processed using Tesseract OCR.

### 👁️ Vision Understanding

The application can analyze:

* Diagrams
* Flowcharts
* Screenshots
* Charts
* Tables
* Visual document content

Visual content is described using a Vision-Language Model and added to the retrieval index.

### 📑 Document Summarization

Generate structured summaries containing:

* Document Overview
* Key Topics
* Important Entities

### 📌 Citation-Aware Answers

Responses include document and page references whenever information is retrieved from uploaded content.

Example:

According to Annual_Report.pdf (Page 12), the company achieved a 23% increase in revenue.

---

## 🏗️ Architecture

Upload Documents
↓
Text Extraction
↓
OCR Fallback
↓
Vision Analysis (Images / Diagrams)
↓
Chunking
↓
BGE Embeddings
↓
FAISS Vector Store
+
BM25 Index
↓
Hybrid Retrieval (RRF)
↓
Groq LLM
↓
Citation-Aware Response

---

## 🛠️ Tech Stack

### LLM

* Groq API
* Llama 3.1 8B Instant
* Llama 4 Scout Vision

### Retrieval

* LangChain
* FAISS
* BM25
* Reciprocal Rank Fusion

### Embeddings

* BAAI/bge-base-en-v1.5

### OCR & Vision

* Tesseract OCR
* PDF2Image
* PDFPlumber

### Frontend

* Streamlit

---

## 📂 Supported File Types

* PDF
* DOCX
* PPTX
* TXT

---

## 🔐 Environment Variables

Create the following Hugging Face Space Secrets:

Api_key=YOUR_GROQ_API_KEY

Huggingface_api_key=YOUR_HF_TOKEN

---

## 🚀 Deployment

Designed for deployment on:

* Hugging Face Spaces
* Docker Spaces
* Streamlit Cloud

---

## 👨‍💻 Author

AbhayKumar Mishra

M.Sc. Data Science & Artificial Intelligence

Focused on:
- AI Engineering
- Retrieval-Augmented Generation (RAG)
- Machine Learning
- MLOps
- Generative AI
- LLM Applications