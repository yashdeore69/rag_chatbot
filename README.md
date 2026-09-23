<div align="center">

# Secure RAG Chatbot

**A locally-running, document-grounded QA chatbot with five layers of AI security — prompt injection detection, content moderation, relevance filtering, strict context-bound generation, and hallucination validation.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000?style=flat-square)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-F46A25?style=flat-square)](https://www.trychroma.com/)
[![LangChain](https://img.shields.io/badge/LangChain-framework-1C3C3C?style=flat-square)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## Overview

This project is a secure Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own PDF documents — and refuses to answer anything else. Every query passes through five sequential security layers before a response is returned, making it resistant to prompt injection, jailbreaks, unsafe content, and hallucination. All models run locally via [Ollama](https://ollama.com/); no API keys or internet connection required.

It was built around a Machine Learning course's Unit 3 notes (Probabilistic Models, Decision Trees) and can be adapted to any PDF corpus.

---

## How It Works

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 1 — Prompt Guard 2 (meta-llama/Prompt-Guard-2-86M)│
│ Detects prompt injection & jailbreak attempts            │
│ Pattern matching + binary ML classification              │
└────────────┬────────────────────────────────────────────┘
             │ passes
             ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2 — Llama Guard 3 (llama-guard3:1b via Ollama)    │
│ Content safety moderation across 13 harm categories      │
└────────────┬────────────────────────────────────────────┘
             │ passes
             ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 3 — ChromaDB Retrieval                            │
│ Finds top-k chunks; filters by relevance threshold 0.3  │
└────────────┬────────────────────────────────────────────┘
             │ context found
             ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 4 — LLaMA 3.2 3B (Ollama) with strict prompt      │
│ Context-bound generation; no external knowledge allowed  │
└────────────┬────────────────────────────────────────────┘
             │ response generated
             ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 5 — Hallucination Validation                       │
│ Pattern detection + Llama Guard grounding check         │
│ Blocks responses with confidence < 30% of being grounded│
└────────────┬────────────────────────────────────────────┘
             │ validated
             ▼
        Final response + cited sources
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| [Ollama](https://ollama.com/) | Runs LLMs locally |
| `llama3.2:3b` | Generation model |
| `llama-guard3:1b` | Content moderation |
| `nomic-embed-text:137m-v1.5-fp16` | Embeddings |
| Hugging Face account | To download Prompt Guard 2 (gated model) |

---

## Quick Start

### 1 — Install Ollama and pull models

```bash
# Install Ollama from https://ollama.com/
ollama pull llama3.2:3b
ollama pull llama-guard3:1b
ollama pull nomic-embed-text:137m-v1.5-fp16
```

### 2 — Clone and install Python dependencies

```bash
git clone https://github.com/yashdeore69/rag_chatbot.git
cd rag_chatbot
pip install langchain langchain-chroma langchain-ollama langchain-community
pip install transformers torch
```

### 3 — Authenticate with Hugging Face (for Prompt Guard 2)

Prompt Guard 2 is a gated model — request access at [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) and then login:

```bash
pip install huggingface_hub
huggingface-cli login
```

### 4 — Add your PDF documents

```bash
mkdir data
# Copy your PDF files into the data/ folder
cp your_notes.pdf data/
```

### 5 — Build the vector database

```bash
python database.py
```

To clear and rebuild from scratch:

```bash
python database.py --reset
```

### 6 — Ask a question

```bash
python query.py "What is a Naive Bayes classifier?"
```

**Disable individual security layers for testing:**

```bash
# Disable Llama Guard (content moderation)
python query.py "Explain decision trees" --disable-guard

# Disable Prompt Guard (injection detection)
python query.py "Explain decision trees" --disable-prompt-guard

# Disable both (raw RAG, no security)
python query.py "Explain decision trees" --disable-guard --disable-prompt-guard
```

---

## Security Layers in Detail

### Layer 1 — Prompt Guard 2

`prompt_guard.py` uses Meta's [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) (86M parameter classifier) to detect prompt injection and jailbreak attempts. Detection uses two signals combined:

- **Pattern matching** — catches explicit manipulation phrases (`ignore previous instructions`, `pretend you are`, `system prompt`, etc.)
- **ML binary classification** — BENIGN / MALICIOUS with probability scores

The blocking logic requires both to agree, or the ML model to be above 85% confident alone. This reduces false positives on legitimate queries.

### Layer 2 — Llama Guard 3

`llama_guard.py` wraps `llama-guard3:1b` running via Ollama. It checks both the incoming query and the outgoing response against 13 harm categories including violent crimes, hate speech, self-harm, and sexual content.

### Layer 3 — Relevance Filtering

After vector similarity search (top-5 chunks from ChromaDB), chunks below a **0.3 relevance threshold** are dropped. If no chunks pass, the bot returns a canned refusal rather than hallucinating an answer.

### Layer 4 — Context-Bound Prompt Engineering

The system prompt explicitly instructs the LLM to answer only from the provided context, refuse out-of-scope questions, and never reveal its instructions or engage in roleplay.

### Layer 5 — Hallucination Validation

After generation, `llama_guard.py`'s `validate_rag_response()` runs:

- **Pattern checks** for LLM meta-commentary, external citations, and fabricated numbers
- **Llama Guard grounding check** — the model judges whether the response is grounded in the retrieved context
- **Confidence scoring** — responses blocked only if grounding confidence < 30%, reducing over-blocking of legitimate paraphrasing

---

## Project Structure

```
rag_chatbot/
β"œβ"€β"€ database.py               # PDF ingestion, chunking, ChromaDB population
β"œβ"€β"€ query.py                  # Main entry point — 5-layer query pipeline
β"œβ"€β"€ get_embedding_function.py # Returns Ollama nomic-embed-text embeddings
β"œβ"€β"€ llama_guard.py            # Llama Guard 3 wrapper: safety + hallucination checks
β"œβ"€β"€ prompt_guard.py           # Prompt Guard 2 wrapper: injection detection
β"œβ"€β"€ data/                     # Your PDF files go here (not committed)
└── chroma/                   # Vector database (auto-generated, not committed)
```

---

## Configuration

Key settings to change for your own use case:

| Setting | File | Default | Description |
|---|---|---|---|
| Embedding model | `get_embedding_function.py` | `nomic-embed-text:137m-v1.5-fp16` | Swap for any Ollama embedding model |
| Generation model | `query.py` | `llama3.2:3b` | Any Ollama LLM |
| Guard model | `llama_guard.py` | `llama-guard3:1b` | Llama Guard model variant |
| Prompt Guard model | `prompt_guard.py` | `meta-llama/Llama-Prompt-Guard-2-86M` | HF model ID |
| Chunk size | `database.py` | `800` chars | Tune for your document type |
| Chunk overlap | `database.py` | `80` chars | |
| Retrieval top-k | `query.py` | `5` | Chunks retrieved per query |
| Relevance threshold | `query.py` | `0.3` | Min relevance score to keep a chunk |
| Hallucination block threshold | `llama_guard.py` | `0.3` | Grounding confidence below this = block |
| Injection ML threshold | `prompt_guard.py` | `0.5` | Malicious probability to flag |
| Data path | `database.py` | `data/` | Folder containing your PDFs |
| Chroma path | `database.py` / `query.py` | `chroma/` | Persisted vector DB directory |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-ollama` | LLM and embedding integration via Ollama |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-community` | PDF directory loader (`PyPDFDirectoryLoader`) |
| `langchain-core` | Prompt templates, document types |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` for chunking |
| `transformers` | Loads Prompt Guard 2 from Hugging Face |
| `torch` | Runs Prompt Guard 2 inference locally |

---

## License

MIT © 2026