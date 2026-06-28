# Implementation Plan: GraphMind (Graph-based RAG Q&A System)

GraphMind is a local-first document Q&A system that combines vector search (ChromaDB) and a Knowledge Graph (NetworkX) with an agentic router (LangGraph/State-Machine) and an LLM generator (Gemini API / Ollama) to answer simple, relational, and global queries with source citations.

---

## User Review Required

> [!IMPORTANT]
> **Ollama vs. Gemini API Fallback**
> Ollama is not installed in the path of this system. To ensure the application is immediately usable, the codebase will support **both** local Ollama and the Gemini API (via `google-genai`). The Streamlit UI will allow the user to input their Gemini API Key or customize their local Ollama endpoint.
>
> **Lightweight Dependencies**
> To avoid Windows C++ compilation errors and massive downloads, we will implement a clean, custom Python recursive text splitter and use standard, pure Python PDF parsers (`pypdf` / `pdfplumber`). Tesseract OCR support will be integrated as an optional feature.

---

## Open Questions

1. **Gemini API Key:** Do you have a Gemini API key from Google AI Studio that we can use for development? If so, we can save it in a `.env` file for instant testing.
2. **Evaluation Framework:** For testing and evaluation, would you like a custom LLM-as-a-judge benchmark script, or do you prefer installing the full RAGAS framework (which requires several LangChain integrations)? *(Recommended: Custom LLM-as-a-judge since it is lightweight and yields comparable rating metrics without installation friction).*

---

## Proposed Changes

We will create a structured Python project with the following folder structure:
- `requirements.txt`: Python package requirements.
- `.env`: Environment configurations.
- `app.py`: Streamlit main dashboard.
- `src/config.py`: Application configurations and env loading.
- `src/llm.py`: Abstract LLM wrapper supporting Gemini & Ollama.
- `src/ingestion.py`: PDF parser & text splitter.
- `src/vector_store.py`: ChromaDB manager for vector search.
- `src/graph_store.py`: NetworkX KG builder, entity extractor, & traverser.
- `src/agents.py`: Query router and generation agent.
- `src/evaluation.py`: Benchmark evaluator.

---

### Ingestion & Storage

#### [NEW] [requirements.txt](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/requirements.txt)
Dependencies required for building the Streamlit UI, local vector DB, Knowledge Graph, model interaction, and visualization.

#### [NEW] [src/config.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/src/config.py)
Configuration settings, system prompts, directory setup, and environment variables loader.

#### [NEW] [src/llm.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/src/llm.py)
A unified client to route LLM calls. If `GEMINI_API_KEY` is provided, it uses `gemini-2.5-flash` or `gemini-1.5-flash` (via `google-genai`). Otherwise, it connects to local Ollama (`llama3.1`).

#### [NEW] [src/ingestion.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/src/ingestion.py)
Extracts text from PDFs using `pypdf`. Performs chunking using a custom recursive text splitter. Recognizes scanned PDFs and extracts their text using OCR/multimodal APIs.

#### [NEW] [src/vector_store.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/src/vector_store.py)
Initializes ChromaDB using `PersistentClient` and saves embeddings. Uses Gemini embedding API when running on Gemini, and standard `sentence-transformers/all-MiniLM-L6-v2` or Ollama embeddings when local.

---

### Knowledge Graph & Retrieval

#### [NEW] [src/graph_store.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/src/graph_store.py)
- Uses LLM prompts to extract `(subject, predicate, object)` triples from text chunks.
- Builds a `networkx.DiGraph` representing the knowledge base.
- Provides interactive visual rendering with `pyvis` HTML generation.
- Implements graph traversal: for a query, identifies relevant entities, traverses up to 2 hops, and gathers neighboring nodes and edges.

#### [NEW] [src/agents.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/src/agents.py)
- Classifies queries into: **SIMPLE** (facts found in single chunks), **COMPLEX** (relationships requiring multi-hop traversal), **GLOBAL** (summarizing themes across the document), or **HYBRID** (both).
- Routes the query to vector search, graph store, or both.
- Synthesizes the final answer using the model, generating clean markdown citations pointing to source files, pages, and graph relationships.

---

### Frontend Dashboard & Evaluation

#### [NEW] [app.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/app.py)
- Streamlit application displaying:
  1. **Upload Section:** Multi-file uploader with progress tracking.
  2. **Chat Q&A:** Beautiful chat UI showcasing agent routing, confidence scores, and source citations.
  3. **Knowledge Graph Visualizer:** Embedded interactive graph where users can click on nodes and relations.
  4. **Evaluation Panel:** Compare RAG vs. Graph-RAG performance.

#### [NEW] [src/evaluation.py](file:///c:/Users/tarma/OneDrive/New%20folder/Graphmind/src/evaluation.py)
Benchmark evaluator that takes a set of test questions, runs them through Vector-only RAG, Graph-only RAG, and GraphMind's Hybrid RAG, and uses the LLM to score their correctness and comprehensiveness.

---

## Verification Plan

### Automated Tests
1. **Model Connection Check:** A script to verify connection to Gemini/Ollama.
   ```bash
   python -c "import src.llm; print(src.llm.test_connection())"
   ```
2. **Ingestion Check:** Test pdf text extraction, chunking, and ChromaDB insertion on a sample PDF.
3. **Graph Extraction Check:** Verify that triples are correctly extracted as JSON from sample text.

### Manual Verification
1. **Start Streamlit UI:**
   ```bash
   streamlit run app.py
   ```
2. **Verification Steps:**
   - Open Streamlit in browser.
   - Set up API Key or local host in sidebar.
   - Upload a test PDF.
   - Verify that chunks are created and the Knowledge Graph is visualized under the "Knowledge Graph" tab.
   - Ask a simple query (vector search) and verify the source citation.
   - Ask a multi-hop query (e.g. "How does Entity A relate to Entity B?") and verify the traversal and answer.
   - Compare performance in the Evaluation tab.
