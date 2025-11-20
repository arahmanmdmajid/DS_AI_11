
# AskKSA — Smart Bilingual Saudi Services Assistant

A modern **Retrieval-Augmented Generation (RAG)**-powered chatbot for Iqama, visa, and Absher guidance. It utilizes a curated dataset of Absher-related articles, and **Google’s Gemini 2.5 Flash** model to provide accurate, context-based, bilingual answers (English + Urdu).

✅ **Live Demo:** [Ask KSA](https://askksa.streamlit.app/)

---

## 📚 Table of Contents

- [Overview](#overview)
- [What Problem Does It Solve?](#what-problem-does-it-solve)
- [How It Solves These Problems](#how-it-solves-these-problems)
- [Features](#features)
  - [Core Features](#core-features)
  - [Technical Features](#technical-features)
- [Tech Stack](#tech-stack)
- [How It Works (Architecture)](#how-it-works-architecture)
- [Project Structure](#project-structure)
- [Installation & Local Setup](#installation--local-setup)
- [Live Demo](#live-demo)

---

## Overview

**AskKSA** is an AI-powered assistant that answers questions related to **Saudi Arabia’s Iqama system, visas, visit visas, fines, and Absher services**. The chatbot is built with **Streamlit** and deployed on **Streamlit Cloud** with a modern UI.

---

## What Problem Does It Solve?

Saudi expatriates frequently face challenges such as:

* Not knowing the exact steps for Absher services
* Confusion around Iqama renewal, status checking, and MOI processes
* A lack of clear guidance in multiple languages
* Difficulty navigating government portals
* Scattered information across the web

AskKSA solves these problems by providing:

* A **single, clean interface** for common Saudi-government queries
* Accurate answers **only from validated sources**
* **Bilingual responses** (English/Urdu)
* Step-by-step instructions extracted from real content
* A helpful UI that’s friendly for non-technical users

---

## How It Solves These Problems

AskKSA uses the **RAG (Retrieval Augmented Generation)** approach:

1. User asks a question
2. System retrieves the **most relevant article-chunks** from a FAISS vector index
3. The context is fed into **Gemini 2.5 Flash**
4. Model produces a **truthful, grounded answer**
5. User sees a modern chat interface with helpful features

This ensures:

* No hallucinations
* High accuracy
* Answers grounded in your curated dataset

---

## Features

### Core Features

* Bilingual responses (English & Urdu)
* Urdu answers use **Noto Nastaliq Urdu** + right alignment
* Auto language detection or forced language mode
* Helpful “👍 Helpful / 👎 Not Helpful” feedback
* Shows which articles were used as sources
* Friendly bot avatar

### Technical Features

* Streamlit-based web UI
* FAISS vector index for semantic search
* Sentence-Transformers embeddings (`BAAI/bge-m3`)
* Modern CSS-injected design
* Efficient resource caching on Streamlit Cloud
* Strong per-question language control to avoid incorrect-language replies

---

## Tech Stack

### **Frontend / UI**

* Streamlit
* RTL Urdu support
* Google Fonts (Noto Nastaliq Urdu)

### **Backend / AI**

* Google Gemini 2.5 Flash (via `google-genai`)
* Sentence Transformers (`BAAI/bge-m3`)
* FAISS (vector similarity search)

### **Data / RAG**

* Scrapped and Curated `rag_ready_abshir_services.json`
* Pre-computed:

  * `faiss_index_ip.bin`
  * `chunks.json`
  * `chunks_metadata.json`

---

## How It Works (Architecture)

```text
           ┌──────────────────────────────┐
           │     User Question (Eng/Urdu) │
           └──────────────────────────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │  Embedding (BGE-M3)    │
             └────────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   FAISS Retrieval   │
               └─────────────────────┘
                          │
                          ▼
       ┌────────────────────────────────────┐
       │ Top-K Relevant Absher Article Text │
       └────────────────────────────────────┘
                          │
                          ▼
              ┌────────────────────┐
              │ Gemini 2.5 Flash   │
              │  (context-injected)│
              └────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Final Answer + Sources      │
           └──────────────────────────────┘
```

---

## Project Structure

```text
askksa/
│
├── app.py                    # Main Streamlit app
├── bot.png                   # Chatbot avatar
├── chunks.json               # RAG chunks
├── chunks_metadata.json      # Metadata for chunks
├── faiss_index_ip.bin        # Vector index
├── rag_ready_abshir_services.json  # Original dataset
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation & Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/arahmanmdmajid/DS_AI_11
cd DS_AI_11
cd askksa
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key

Create a `.env` file or set an environment variable:

```bash
export GOOGLE_API_KEY="your_key_here"
```

### 5. Run the app locally

```bash
streamlit run app.py
```

---

## Live Demo

🚀 **Live Streamlit Cloud App:** `https://askksa.streamlit.app/`

