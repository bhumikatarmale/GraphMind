# 📋 PROJECT PROPOSAL FORM (Revised)

---

**Project Title:** GraphMind: A Knowledge Notebook with Graph-Based Memory for Intelligent Document Question Answering

---

**Abstract:**

- **Problem:** Professionals (lawyers, doctors, researchers) waste hours searching through PDFs, scanned reports, and documents. Tools like Google's NotebookLM use vector-based search, which finds similar text but fails at connecting information across documents (e.g., "How does finding in Report A contradict Report B?").

- **Gap:** Vector-based RAG (Retrieval-Augmented Generation) treats documents as isolated text chunks. It cannot perform multi-hop reasoning, understand relationships between entities, or summarize themes across large collections.

- **Proposed Solution:** We build GraphMind — a document Q&A system that uses a **knowledge graph** as its memory instead of a flat vector database. When documents are uploaded, the system extracts entities (people, concepts, dates) and their relationships, building a graph that grows with each document. For answering queries, an intelligent router decides whether to use semantic search (vector), relational search (graph traversal), or both.

- **Key Features:** (1) Automatic knowledge graph construction from uploaded documents using LLMs. (2) Hybrid retrieval — vector search for simple queries, graph traversal for multi-hop queries. (3) Multimodal support — handles text PDFs and scanned images. (4) Source citations with every answer. (5) Runs locally — no data leaves the user's machine.

- **Expected Outcome:** A working web application where users upload documents, ask natural language questions, and receive accurate, cited answers. We benchmark against standard vector-RAG and target 25-40% improvement on multi-hop queries.

- **Tools:** Python, LangChain/LangGraph, NetworkX/Neo4j, ChromaDB, Ollama (Llama 3.1), Streamlit, RAGAS evaluation framework.

---

**References:**

1. Pan, S., Luo, L., et al. "Unifying Large Language Models and Knowledge Graphs: A Roadmap." *IEEE Trans. on Knowledge and Data Engineering (TKDE)*, Vol. 36, No. 7, 2024.

2. Jia, N., et al. "Knowledge Graph Enhanced RAG for Failure Mode and Effects Analysis." *IEEE Trans. on Knowledge and Data Engineering (TKDE)*, 2024.

3. Wu, J., et al. "Graph Neural Networks for Information Retrieval: A Survey." *IEEE Trans. on Neural Networks and Learning Systems (TNNLS)*, 2023.

4. Zhao, W., et al. "A Comprehensive Survey on Document Understanding with Deep Learning." *IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI)*, 2023.

5. Gao, Y., et al. "A Survey on Retrieval-Augmented Generation for Large Language Models." *IEEE Access*, 2024.

6. Edge, D., Trinh, H., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *Microsoft Research*, arXiv:2404.16130, 2024.

7. Zhu, Y., et al. "A Survey on Multimodal Knowledge Graphs: Construction, Completion, and Applications." *IEEE/CAA Journal of Automatica Sinica*, Vol. 11, No. 2, 2024.

8. Li, J., et al. "Retrieval-Augmented Generation for AI-Generated Content: A Survey." *IEEE Intl. Conf. on Big Data*, 2024.

9. Chen, W., et al. "Entity Extraction and Knowledge Graph Construction from Unstructured Text Using LLMs." *IEEE Access*, 2024.

10. Zhang, R., et al. "Question Answering over Knowledge Graphs with LLMs: A Survey." *IEEE DSAA Conference*, 2024.

11. Fan, T., et al. "MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation." *IEEE/ACM Proceedings*, arXiv:2501.06713, 2025.

12. Kumar, S., et al. "Analyzing Embedding Models for Vector Databases." *IEEE ICTBIG Conference*, 2023.

---

**Whether Interdisciplinary (Yes/No):** No

**If Yes, Name and sign of the Expert Guide from another Department:** ____________________________________

---

**Name & Signature:**

Name of member 1: ____________________________________

Name of member 2: ____________________________________

Name of member 3: ____________________________________

---

**Name and Signature of Co-Guide (if any):** ____________________________________

**Name and Signature of Guide:** ____________________________________
