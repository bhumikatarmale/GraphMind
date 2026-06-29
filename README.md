# 🧠 GraphMind: Knowledge Notebook

GraphMind is an advanced local Hybrid RAG (Retrieval-Augmented Generation) and Knowledge Graph Q&A system. Inspired by **NotebookLM**, it integrates a dark bento-grid theme, offline-first execution, multi-format document ingestion, and source-scoped context filtering with dynamic task-based model routing.

---

## 🌟 Key Features

1. **NotebookLM-Style Source Panel**:
   * Centralized left sidebar to upload, preview, and toggle active documents.
   * Scopes vector search context and knowledge graph traversals strictly to selected sources.
   * Clean source deletion: dropping a file clears its chunks from ChromaDB and sweeps its unique edges/orphaned nodes from the Knowledge Graph.
   * Inline previewers to view chunk summaries of ingested files.

2. **Multi-Format Ingestion**:
   * Out-of-the-box support for **PDF (`fitz` / PyMuPDF)**, **Word (`.docx`)**, **PowerPoint (`.pptx`)**, and **Images (`.jpg`, `.png`, `.jpeg`)**.
   * Vision-based OCR/descriptions (via local `moondream` or cloud `gemini-2.5-flash`) for scanned diagrams and images.

3. **Dynamic Task-Based Model Routing**:
   * Routes processing to the most efficient local model to minimize VRAM usage on laptop GPUs:
     * **Fast (Entity Extraction & Router):** `gemma3:1b` (0.8 GB, runs 100% in VRAM at 70+ tokens/sec, speeding up ingestion by ~10x).
     * **Reasoning/Validation (Q&A Synthesis):** `gemma3:4b` (3.3 GB, runs on GPU at 26+ tokens/sec for rich context synthesis).
     * **Vision (Multimodal OCR):** `moondream:latest` (1.7 GB).
     * **Embedding (Semantic Vectors):** `nomic-embed-text` (274 MB).

4. **Interactive Persistent Chat UX**:
   * Chat message conversation thread with message history.
   * Instant markdown answer exports (`.md` file downloads) and a "Clear History" button.

5. **Clarity-Optimized Knowledge Graphs**:
   * Automatically extracts entity triples `(Subject, Relation, Object)`.
   * Prunes duplicate/weak edges and caps interactive pyvis renderings to the top 80 most central nodes to prevent browser freezes.

---

## 🏗️ Project Structure

```
GraphMind/
├── .streamlit/
│   └── config.toml         # Locked dark mode UI theme configuration
├── src/
│   ├── __init__.py         # Package initializer
│   ├── agents.py           # Query routing & retrieval coordination
│   ├── config.py           # Settings, paths, and model prompt templates
│   ├── evaluation.py       # Metrics and benchmarking pipelines
│   ├── graph_store.py      # NetworkX store and pyvis visualization engine
│   ├── ingestion.py        # Document text extraction and splitting
│   ├── llm.py              # LLM connectors (Ollama & Gemini API)
│   ├── source_registry.py  # JSON registry tracking active files
│   └── vector_store.py     # ChromaDB client & vector embeddings manager
├── app.py                  # Main Streamlit dashboard interface
├── requirements.txt        # Python library dependencies
├── .env.example            # Environment variables configuration template
└── .gitignore              # Files excluded from version control
```

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.10+ installed on your system.

To run models locally with GPU acceleration, install **[Ollama](https://ollama.com/)** and pull the following models:
```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull moondream
ollama pull nomic-embed-text
```

### 2. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Copy the configuration template to create your `.env` file:
```bash
cp .env.example .env
```
Open `.env` and configure your preferred provider (`ollama` or `gemini`):
```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 4. Running the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

---

## ⚙️ Development Configs
You can fine-tune text chunking sizes, model targets, paths, and extraction prompts in the [src/config.py](src/config.py) module.