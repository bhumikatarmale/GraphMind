import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="GraphMind - Knowledge Notebook",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Base application styling */
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b, #0f172a);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Text color for sidebar */
[data-testid="stSidebar"] .stMarkdown {
    color: #cbd5e1;
}

/* Styled headers */
h1, h2, h3 {
    background: linear-gradient(90deg, #00adb5, #00fff2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* Premium card wrappers */
.premium-card {
    background-color: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

/* Button stylings */
.stButton>button {
    background: linear-gradient(90deg, #00adb5 0%, #00818a 100%);
    color: #ffffff;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 173, 181, 0.4);
    color: #ffffff;
    border: none;
}

.stButton>button:active {
    transform: translateY(0);
}

/* Status logs */
.log-box {
    background-color: #020617;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 0.75rem;
    font-family: monospace;
    color: #10b981;
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# Imports from src
from src.vector_store import VectorStore
from src.graph_store import GraphStore
from src.agents import QueryAgent
from src import ingestion
from src import llm

# Initialize session state for DB & Graph connections
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "graph_store" not in st.session_state:
    st.session_state.graph_store = GraphStore()
if "query_agent" not in st.session_state:
    st.session_state.query_agent = QueryAgent(
        st.session_state.vector_store,
        st.session_state.graph_store
    )

# Sidebar Header & Model Config
st.sidebar.markdown("# 🧠 GraphMind Config")
st.sidebar.markdown("Configure your AI model and engine settings below:")

# Select LLM Provider
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Ollama", "Gemini API"],
    index=0 if os.getenv("LLM_PROVIDER", "ollama") == "ollama" else 1
)

st.session_state.llm_provider = "ollama" if llm_provider == "Ollama" else "gemini"

if st.session_state.llm_provider == "gemini":
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Generate a free key at Google AI Studio"
    )
    st.session_state.gemini_key = gemini_key
    st.session_state.gemini_model = st.sidebar.text_input("Gemini Model", value="gemini-2.5-flash")
    st.session_state.gemini_embed = st.sidebar.text_input("Gemini Embed Model", value="text-embedding-004")
else:
    ollama_url = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")
    ollama_model = st.sidebar.text_input("Ollama LLM Model", value="llama3.2")
    ollama_embed = st.sidebar.text_input("Ollama Embedding Model", value="nomic-embed-text")
    st.session_state.ollama_url = ollama_url
    st.session_state.ollama_model = ollama_model
    st.session_state.ollama_embed = ollama_embed

# Connection Test button
if st.sidebar.button("Test LLM Connection"):
    with st.sidebar:
        with st.spinner("Connecting..."):
            success, msg = llm.test_connection()
            if success:
                st.success(msg)
            else:
                st.error(msg)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Statistics")
try:
    num_nodes = st.session_state.graph_store.graph.number_of_nodes()
    num_edges = st.session_state.graph_store.graph.number_of_edges()
    num_chunks = len(st.session_state.vector_store.collection.get().get("ids", []))
except Exception:
    num_nodes = 0
    num_edges = 0
    num_chunks = 0

st.sidebar.metric("Document Chunks", num_chunks)
st.sidebar.metric("KG Entities (Nodes)", num_nodes)
st.sidebar.metric("KG Relations (Edges)", num_edges)

if st.sidebar.button("Reset Database & Graph", type="secondary"):
    st.session_state.vector_store.reset()
    st.session_state.graph_store.reset()
    st.toast("Database and Graph have been reset successfully.")
    st.rerun()

# Main Layout
st.markdown("# 🧠 GraphMind: Knowledge Notebook")
st.markdown("##### *An Intelligent Document Q&A Assistant with Graph-Based Memory*")

# Navigation tabs
tab_qa, tab_ingest, tab_graph, tab_eval = st.tabs([
    "💬 Intelligent Q&A", 
    "🚀 Ingest Documents", 
    "🕸️ Knowledge Graph", 
    "📊 Evaluation Benchmark"
])

# Tab 1: Q&A Engine
with tab_qa:
    st.markdown("### Ask a Question")
    st.markdown("Ask anything about your ingested documents. The agent will automatically route the query and traverse the knowledge memory to find the answer.")
    
    query = st.text_input("Your Question", placeholder="How does finding in document X relate to document Y?")
    
    if st.button("Query Engine"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            if num_chunks == 0:
                st.error("No documents ingested yet! Please go to the 'Ingest Documents' tab to upload some files first.")
            else:
                with st.spinner("Agentic router retrieving context and synthesizing answer..."):
                    try:
                        # Execute agent
                        result = st.session_state.query_agent.answer_query(query)
                        
                        # Layout answer
                        st.markdown("### 📝 Answer")
                        
                        # Premium card for answer
                        st.markdown(f"""
                        <div class="premium-card">
                            {result['answer']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_met1, col_met2 = st.columns(2)
                        with col_met1:
                            st.metric("Routed Category", result["category"])
                        with col_met2:
                            st.metric("Confidence Score", f"{result['confidence']}%")
                        
                        # Expanders for tracing
                        with st.expander("🔍 Show Agent Routing & Logs"):
                            st.markdown(f"**Routed Category:** `{result['category']}`")
                            st.markdown(f"**Classification Reasoning:** *{result['reasoning']}*")
                            st.markdown(f"**Chunks Retrived:** {len(result['vector_chunks'])}")
                            st.markdown(f"**Relations Discovered:** {len(result['graph_relations'])}")
                            
                        with st.expander("📄 Show Retrieved Document Chunks (Vector DB)"):
                            for idx, chunk in enumerate(result["vector_chunks"]):
                                st.markdown(f"**Chunk {idx+1} (Source: {chunk['source']}, Page: {chunk['page']})**")
                                st.info(chunk["text"])
                                
                        with st.expander("🕸️ Show Traversed Subgraph Relations (KG)"):
                            if not result["graph_relations"]:
                                st.write("No relation triples traversed for this query.")
                            else:
                                for idx, rel in enumerate(result["graph_relations"]):
                                    st.write(f"- **({rel['subject']})** --`{rel['relation']}`--> **({rel['object']})** (Sources: {', '.join(rel['sources'])})")
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")

# Tab 2: Ingestion Pipeline
with tab_ingest:
    st.markdown("### Ingest Documents")
    st.markdown("Upload your PDF files (text books, reports, research papers). The system will split the text, embed it in ChromaDB, extract knowledge triples, and merge them into the Knowledge Graph.")
    
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Start Ingest & Index"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            # Temporary folder to save PDFs
            temp_dir = os.path.join(os.getcwd(), "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            
            logs = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                logs.append(f"Saved upload: {uploaded_file.name}")
                log_container.code("\n".join(logs))
                
                # Parse PDF
                status_text.text(f"Processing PDF text from: {uploaded_file.name}...")
                try:
                    chunks = ingestion.parse_pdf(file_path)
                    logs.append(f"Extracted {len(chunks)} text chunks.")
                    log_container.code("\n".join(logs))
                    
                    # Store in ChromaDB
                    status_text.text("Generating embeddings and indexing in Vector DB...")
                    st.session_state.vector_store.add_chunks(chunks)
                    logs.append("Vector database indexing complete.")
                    log_container.code("\n".join(logs))
                    
                    # Extract KG Triples
                    status_text.text("Extracting entities and building Knowledge Graph memory...")
                    total_chunks = len(chunks)
                    for c_idx, chunk in enumerate(chunks):
                        status_text.text(f"Extracting relations from chunk {c_idx+1}/{total_chunks}...")
                        st.session_state.graph_store.add_relations_from_chunk(chunk)
                    
                    logs.append(f"Successfully constructed knowledge graph memory for {uploaded_file.name}.")
                    log_container.code("\n".join(logs))
                    
                except Exception as e:
                    logs.append(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    log_container.code("\n".join(logs))
                    st.error(f"Failed to ingest document: {uploaded_file.name}. Error: {str(e)}")
                
                # Remove temporary file
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                
                # Update progress
                progress = int((idx + 1) / len(uploaded_files) * 100)
                progress_bar.progress(progress)
            
            # Save graph
            st.session_state.graph_store.save()
            status_text.text("All documents processed and knowledge store successfully updated!")
            st.success("Ingestion complete!")
            st.rerun()

# Tab 3: Knowledge Graph Visualizer
with tab_graph:
    st.markdown("### Interactive Knowledge Graph")
    st.markdown("Explore the graph memory constructed from your uploaded files. Click and drag nodes, zoom, and highlight connection paths.")
    
    if num_nodes == 0:
        st.info("No nodes in the graph to visualize. Please ingest documents to populate the knowledge graph.")
    else:
        with st.spinner("Generating interactive graph network..."):
            html_path = st.session_state.graph_store.generate_visualization_html()
            
            # Load HTML and embed
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                # Render using iframe
                components.html(html_content, height=600)
            else:
                st.error("Failed to generate graph HTML file.")
        
        # Display degree table
        st.markdown("### 📊 Entity Importance (Node Degree)")
        nodes_sorted = sorted(
            [(node, st.session_state.graph_store.graph.degree(node)) for node in st.session_state.graph_store.graph.nodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        cols = st.columns(3)
        for i, (node, deg) in enumerate(nodes_sorted[:9]):
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"**{node}** (Degree: `{deg}`)")

# Tab 4: Evaluation Benchmark
with tab_eval:
    st.markdown("### Evaluation Benchmark (GraphRAG vs Vector RAG)")
    st.markdown("Compare the performance of our GraphMind Hybrid engine against standard Vector-only RAG.")
    
    # We load evaluation page
    st.markdown("""
    <div class="premium-card">
        <h4>Benchmark metrics target: 25-40% improvement on multi-hop questions</h4>
        <p>A multi-hop question (e.g. "How does the founder of Company X relate to Project Y?") requires connecting information from different pages. Vector search retrieves isolated chunks and fails to establish links, whereas the Knowledge Graph traces relationships directly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Run Simulation Benchmark"):
        with st.spinner("Running evaluation benchmark on sample multi-hop questions..."):
            # We can import and run evaluation
            from src import evaluation
            eval_results = evaluation.run_comparison(
                st.session_state.vector_store,
                st.session_state.graph_store
            )
            
            # Display results
            st.markdown("### 📈 Evaluation Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Vector-only RAG Accuracy", f"{eval_results['vector_accuracy']}%")
            col2.metric("GraphMind Hybrid Accuracy", f"{eval_results['hybrid_accuracy']}%")
            col3.metric("Improvement Margin", f"+{eval_results['improvement']}%")
            
            st.markdown("### Detailed Comparison Results")
            st.table(eval_results["details"])
