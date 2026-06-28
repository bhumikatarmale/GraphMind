# 📘 GraphMind — Complete Structured Plan

> **Scope:** 1 semester (4 months) | **Team:** 3 members | **Starting knowledge:** Learning from scratch | **Primary tool:** Antigravity + Ollama

---

## 1. 📚 LEARNING ROADMAP — What to Learn & When

### Month 0 (Pre-start / Week 1-2): Foundations

| Topic | What to Learn | Resource | Time | Who |
|-------|--------------|----------|:----:|:---:|
| Python for ML | pandas, numpy, file I/O, API calls, JSON handling | Kaggle "Intro to Python" (free) | 3 days | All |
| What is an LLM? | Prompting, tokens, temperature, system prompts | Andrej Karpathy "Intro to LLMs" YouTube (1hr) | 1 day | All |
| What is RAG? | Embeddings, vector search, chunking, retrieval pipeline | DeepLearning.AI "LangChain: Chat with Your Data" (free short course) | 2 days | All |
| What is a Knowledge Graph? | Nodes, edges, triples (subject→predicate→object), NetworkX basics | "Graph Algorithms" chapter in NetworkX docs + YouTube tutorial | 2 days | All |
| LangChain Basics | Chains, agents, tools, memory | LangChain official tutorial (Python quickstart) | 2 days | Member 1 |
| ChromaDB Basics | Embedding, storing, querying vectors | ChromaDB getting started guide | 1 day | Member 2 |
| Streamlit Basics | Building web apps in Python | Streamlit 30-day challenge (Day 1-5 only) | 1 day | Member 3 |

### Month 1-4: Learn As You Build (Just-in-Time Learning)

| When You Need It | Topic | Resource |
|-----------------|-------|----------|
| Month 1 | Chunking strategies (recursive, semantic) | LangChain Text Splitters documentation |
| Month 1 | Sentence embeddings | HuggingFace Sentence-Transformers docs |
| Month 2 | LLM entity extraction | "NER with LLMs" tutorial on HuggingFace blog |
| Month 2 | Graph databases (Neo4j vs NetworkX) | NetworkX tutorial (start simple, upgrade later) |
| Month 3 | LangGraph (agentic workflows) | LangGraph official documentation + examples |
| Month 3 | OCR / Document parsing | Unstructured.io quickstart guide |
| Month 4 | RAGAS evaluation | RAGAS docs + "Evaluating RAG" DeepLearning.AI course |

---

## 2. 🏗️ ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│              (Streamlit Web Dashboard)                   │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │ Upload   │  │ Ask Question │  │ View Knowledge   │  │
│   │ Documents│  │ (Text Input) │  │ Graph (Visual)   │  │
│   └────┬─────┘  └──────┬───────┘  └──────────────────┘  │
└────────┼───────────────┼────────────────────────────────┘
         │               │
         ▼               ▼
┌─────────────────┐  ┌──────────────────────────────────┐
│  INGESTION      │  │  QUERY ENGINE                     │
│  PIPELINE       │  │                                   │
│                 │  │  ┌─────────────────────────────┐  │
│  PDF/Image      │  │  │    AGENTIC ROUTER           │  │
│     ↓           │  │  │  (LangGraph)                │  │
│  Text Extraction│  │  │                             │  │
│  (Unstructured) │  │  │  Decides:                   │  │
│     ↓           │  │  │  • Simple query → Vector DB │  │
│  Chunking       │  │  │  • Complex query → Graph    │  │
│     ↓           │  │  │  • Both → Hybrid            │  │
│  ┌──────────┐   │  │  └──────┬──────────────────────┘  │
│  │Embeddings│   │  │         │                         │
│  │(Sentence │   │  │    ┌────┴────┐                    │
│  │ BERT)    │   │  │    ▼         ▼                    │
│  └────┬─────┘   │  │ ┌───────┐ ┌────────────┐         │
│       │         │  │ │Vector │ │ Graph      │         │
│       ▼         │  │ │Search │ │ Traversal  │         │
│  ┌─────────┐    │  │ │(Chroma│ │ (NetworkX) │         │
│  │ChromaDB │    │  │ │  DB)  │ │            │         │
│  │(Vectors)│    │  │ └───┬───┘ └─────┬──────┘         │
│  └─────────┘    │  │     └─────┬─────┘                │
│       │         │  │           ▼                       │
│  Entity/Relation│  │  ┌─────────────────┐              │
│  Extraction     │  │  │  LLM Generator  │              │
│  (LLM-based)    │  │  │  (Ollama/Llama  │              │
│       │         │  │  │   3.1 8B)       │              │
│       ▼         │  │  │  + Citations    │              │
│  ┌──────────┐   │  │  └────────┬────────┘              │
│  │Knowledge │   │  │           │                       │
│  │Graph     │   │  │           ▼                       │
│  │(NetworkX)│   │  │  ┌─────────────────┐              │
│  └──────────┘   │  │  │  Answer + Source│              │
│                 │  │  │  Citations      │              │
└─────────────────┘  └──┴─────────────────┴──────────────┘
```

---

## 3. 🛠️ TECH STACK

| Layer | Tool | Why This (Not Alternatives) | Cost |
|-------|------|---------------------------|:----:|
| **LLM** | Ollama + Llama 3.1 8B | Runs locally, free, no API key needed. 8B is enough for extraction + QA. | Free |
| **Vector DB** | ChromaDB | Simplest local vector DB. No server setup. Python-native. | Free |
| **Graph Store** | NetworkX (start) → Neo4j (later) | NetworkX = pure Python, zero setup, perfect for learning. Upgrade to Neo4j only if needed. | Free |
| **Embeddings** | all-MiniLM-L6-v2 (Sentence-BERT) | 80MB model, fast, good quality. Runs on CPU. | Free |
| **Agent Framework** | LangGraph | Official LangChain agent framework. Better than CrewAI for our use case (routing logic). | Free |
| **Document Parsing** | Unstructured.io | Handles PDFs, images, tables in one library. Free open-source version. | Free |
| **OCR (scans)** | Tesseract OCR | Free, open-source, works with Unstructured. | Free |
| **Frontend** | Streamlit | Build web dashboards in pure Python. No React/JS needed. | Free |
| **Graph Visualization** | pyvis (NetworkX → HTML) | Interactive graph visualization in browser. One-liner integration. | Free |
| **Evaluation** | RAGAS | Standard RAG evaluation framework. Measures faithfulness, relevance, precision. | Free |
| **Dev Environment** | Antigravity | Writes code, runs commands, debugs, builds UI — primary development tool. | — |

**Total cost: ₹0.** Everything is open-source and runs on a regular laptop.

---

## 4. 👤 USER FLOW DIAGRAM

```
┌──────────┐
│  START   │
└────┬─────┘
     │
     ▼
┌────────────────┐     ┌───────────────────┐
│ Upload Documents│────▶│ System ingests:    │
│ (PDF, images,  │     │ 1. Extracts text   │
│  scanned docs) │     │ 2. Creates chunks  │
│                │     │ 3. Generates embeds│
└────────────────┘     │ 4. Extracts entities│
                       │ 5. Builds KG edges │
                       └─────────┬─────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │ Knowledge Graph  │
                       │ + Vector Store   │
                       │ READY            │
                       └────────┬────────┘
                                │
     ┌──────────────────────────┘
     │
     ▼
┌────────────────┐
│ User asks a    │
│ question       │
│ (natural lang) │
└────────┬───────┘
         │
         ▼
┌─────────────────────────┐
│ Router classifies query:│
│                         │
│ "What does doc X say    │
│  about topic Y?"        │
│  → SIMPLE → Vector DB   │
│                         │
│ "How does finding in    │
│  doc A relate to doc B?"│
│  → COMPLEX → Graph      │
│                         │
│ "Summarize all docs on  │
│  topic Z"               │
│  → GLOBAL → Graph       │
│    communities           │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────┐
│ Retrieve context     │
│ (chunks + graph      │
│  neighbors)          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ LLM generates answer │
│ with SOURCE CITATIONS│
│                      │
│ "Based on Document 3,│
│  page 7: ..."        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Display to user:     │
│ • Answer text        │
│ • Highlighted sources│
│ • Confidence score   │
│ • Knowledge graph    │
│   visualization      │
└──────────────────────┘
```

---

## 5. 📅 DETAILED 4-MONTH PLAN (Weekly Breakdown)

### MONTH 1: Basic RAG System (Foundation)

| Week | Task | Tool Used | Deliverable |
|:----:|------|:---------:|-------------|
| 1 | Learn Python, LLM basics, set up Ollama + ChromaDB | Antigravity (setup) | Ollama running locally, can chat with Llama 3.1 |
| 2 | Build PDF ingestion: upload → extract text → chunk → embed → store in ChromaDB | Antigravity (code) | Upload a PDF, it gets stored in vector DB |
| 3 | Build query engine: user question → embed → search ChromaDB → pass to LLM → answer | Antigravity (code) | Type question → get answer from uploaded PDF |
| 4 | Add source citations + Streamlit UI | Antigravity (UI) | **MVP DEMO: Upload PDF → Ask question → Get cited answer** |

**Month 1 Deliverable:** A working Streamlit app that does basic RAG. This alone is demo-able.

### MONTH 2: Add Knowledge Graph Memory

| Week | Task | Tool Used | Deliverable |
|:----:|------|:---------:|-------------|
| 5 | Learn entity extraction. Write LLM prompts to extract (entity, relation, entity) triples from text chunks | Antigravity (code) | Given a paragraph, extract entities and relations as JSON |
| 6 | Build knowledge graph: store triples in NetworkX. Visualize with pyvis. | Antigravity (code) | Interactive graph visualization of extracted entities |
| 7 | Build graph traversal retrieval: for a query, find relevant nodes → traverse neighbors → collect context | Antigravity (code) | Query "How does X relate to Y?" answered via graph traversal |
| 8 | Integrate graph + vector retrieval. Build simple router: short queries → vector, complex queries → graph | Antigravity (code) | **HYBRID SYSTEM: Both retrieval methods working** |

**Month 2 Deliverable:** Knowledge graph auto-built from documents. Can answer multi-hop questions.

### MONTH 3: Agentic Router + Multimodal

| Week | Task | Tool Used | Deliverable |
|:----:|------|:---------:|-------------|
| 9 | Build LangGraph agent that dynamically decides: vector search vs graph traversal vs both | Antigravity (code) | Agent routes queries intelligently |
| 10 | Add self-correction: if retrieval confidence < threshold, agent retries with different strategy | Antigravity (code) | Agent retries when uncertain |
| 11 | Add multimodal: handle scanned PDFs using Tesseract OCR + Unstructured.io | Antigravity (code) | System handles both text and scanned PDFs |
| 12 | Polish UI: add graph visualization panel, confidence scores, source highlighting | Antigravity (UI) | **FULL FEATURE DEMO** |

**Month 3 Deliverable:** Complete system with agentic routing, graph memory, multimodal support.

### MONTH 4: Evaluation + Documentation + Final Demo

| Week | Task | Tool Used | Deliverable |
|:----:|------|:---------:|-------------|
| 13 | Create evaluation benchmark: 50 documents, 100 questions (50 simple, 50 multi-hop) | Manual + Antigravity | Gold-standard test set |
| 14 | Run RAGAS evaluation: compare GraphMind vs ChromaDB-only RAG vs NotebookLM | Antigravity (code) | Comparison table with metrics |
| 15 | Write project report (IEEE format) + prepare presentation | Antigravity (docs) | IEEE-format report draft |
| 16 | Final demo preparation + practice + submission | All | **FINAL SUBMISSION** |

**Month 4 Deliverable:** Evaluation results, IEEE report, final presentation, working demo.

---

## 6. 🔧 TOOL USAGE PLAN — What Tool Does What

| Task | Primary Tool | How |
|------|:------------:|-----|
| Writing all Python code | **Antigravity** | I write the entire codebase — ingestion, graph, retrieval, agents, UI |
| Running/testing code | **Antigravity** | I run terminal commands, see errors, fix them in real-time |
| Building Streamlit dashboard | **Antigravity** | I create the full UI and open it in browser |
| Installing packages | **Antigravity** | `pip install langchain chromadb networkx pyvis streamlit` |
| Debugging errors | **Antigravity** | I read stack traces, identify bugs, fix code |
| Web research (APIs, docs) | **Antigravity** | I search web for latest docs, tutorials, solutions |
| Running Ollama (LLM) | **Your terminal** | `ollama run llama3.1` — you start it once, I use it via API |
| Literature review research | **Antigravity** | I find papers, summarize them, fill your spreadsheet |
| Writing project report | **Antigravity** | I draft IEEE-format sections |
| Testing on real documents | **You** | Upload your own PDFs/documents to test |
| Final presentation | **You** | Present the demo that Antigravity built |

---

## 7. 📝 PRT (Project Research Template)

### Research Questions (Feynman-Style)

| # | Research Question | Why It Matters | How to Answer | Status |
|:-:|------------------|---------------|---------------|:------:|
| 1 | **What is RAG and why does flat vector search fail for complex queries?** | Foundation — you must understand the problem before the solution | Read Paper #3 (RAG Survey). Build basic RAG. Test with multi-hop question. See it fail. | ⬜ |
| 2 | **What is a knowledge graph? How do you represent "Professor X teaches Course Y at University Z" as a graph?** | Core concept — if you can't think in graphs, you can't build one | Read NetworkX tutorial. Draw 5 triples on paper. Code them in Python. | ⬜ |
| 3 | **How do LLMs extract entities and relations from raw text?** | This is how your graph gets built automatically | Read Paper #9. Write a prompt that extracts triples from a paragraph. Test on 10 paragraphs. | ⬜ |
| 4 | **What is the difference between vector similarity search and graph traversal? When is each better?** | This is your project's core hypothesis | Build both. Test same 20 questions on both. Compare answers. | ⬜ |
| 5 | **How does Microsoft's GraphRAG use community detection for global summarization?** | Key technique you're adapting | Read Paper #4. Explain it in your own words. Identify what you'll simplify. | ⬜ |
| 6 | **What is an "agent" in LangGraph? How does it decide which tool to use?** | Month 3 core feature | Read LangGraph docs. Build a toy agent that chooses between 2 tools. | ⬜ |
| 7 | **How do you evaluate a RAG system? What are faithfulness, relevance, and context precision?** | Without evaluation, you can't prove your system works | Read RAGAS docs. Run evaluation on your basic RAG. Understand each metric. | ⬜ |
| 8 | **What are the limitations of LLM-extracted knowledge graphs?** | Intellectual honesty — know your weaknesses | Read Paper #9 accuracy numbers. Test on your own docs. Document where extraction fails. | ⬜ |

### Research Validation Checklist

- [ ] Can I explain RAG to a non-technical person in 2 minutes?
- [ ] Can I draw the system architecture from memory?
- [ ] Can I explain WHY graph memory is better than vector-only for multi-hop queries?
- [ ] Can I name 3 limitations of my approach honestly?
- [ ] Have I tested the system on documents I've never seen before?
- [ ] Can I explain every metric in my evaluation table?

---

## 8. ⚠️ RISK MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| LLM entity extraction is inaccurate (55-70% F1) | High | Medium | Add validation step: cross-check extracted relations against source text. Flag low-confidence triples. |
| Knowledge graph gets too noisy with wrong relations | Medium | High | Implement confidence scores on edges. Low-confidence edges contribute less to answers. |
| Ollama/Llama 3.1 is too slow on laptop | Medium | Medium | Use quantized model (Q4). If still slow, fall back to Gemini API free tier. |
| Can't show improvement over basic RAG | Low | High | Carefully design multi-hop test questions where vector RAG is known to fail. This is where graph shines. |
| Scope creep — trying to do too much | High | High | **Rule: MVP first, always.** Basic RAG (Month 1) is your safety net. Everything after is bonus. |
| Team member can't contribute equally | Medium | Medium | Clear role assignment: Member 1 (backend/graph), Member 2 (ingestion/eval), Member 3 (UI/docs). |

---

## 9. 🎯 SUCCESS CRITERIA

| Criteria | Minimum (Pass) | Target (Good Grade) | Stretch (Publication) |
|----------|:-:|:-:|:-:|
| Documents supported | Text PDFs only | Text + scanned PDFs | + images, tables, handwritten |
| Query types | Simple factual | + Multi-hop reasoning | + Global summarization |
| Retrieval method | Vector-only RAG | Vector + Graph hybrid | + Agentic routing |
| Evaluation | Manual testing (20 Qs) | RAGAS on 100 Qs | + Comparison with NotebookLM |
| Graph visualization | None | Static graph view | Interactive explorable graph |
| Report | Basic project report | IEEE format with literature review | Submittable to IEEE conference |
