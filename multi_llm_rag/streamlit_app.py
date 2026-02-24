"""
Multi-LLM RAG with Groq API - Modern Streamlit Frontend
Beautiful, modern UI for document upload and chatbot interaction
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Load environment
load_dotenv(os.path.join(project_root, ".env"))

# Page configuration
st.set_page_config(
    page_title="Multi-Model RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom header */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin: -1rem -4rem 2rem -4rem;
    }

    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }

    /* Chat messages */
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease;
    }

    .chat-message-user {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        color: white;
        margin-left: 2rem;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
    }

    .chat-message-assistant {
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        margin-right: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-right: 0.75rem;
    }

    .chat-avatar-user {
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
    }

    .chat-avatar-assistant {
        background: linear-gradient(135deg, #10b981, #059669);
    }

    /* Input container */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem 2rem;
        background: rgba(15, 15, 26, 0.95);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #7c3aed;
        background: rgba(124, 58, 237, 0.1);
    }

    /* Sidebar */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
    }

    /* File chips */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(124, 58, 237, 0.2);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 2rem;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #e2e8f0;
    }

    /* Model badges */
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .model-badge-reasoning {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .model-badge-code {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .model-badge-fast {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(124, 58, 237, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(124, 58, 237, 0.7);
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 1.5rem 0;
    }

    /* Loading animation */
    .typing-indicator {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
    }

    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        animation: typing 1.4s infinite;
    }

    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

# Import RAG components
try:
    from ingestion.document_loader import (
        load_documents,
        chunk_documents,
        create_vector_store,
        load_existing_vector_store
    )
    from ingestion.database_loader import create_sql_retriever, get_table_info
    from llm.model_router import ModelRouter
    from retrieval.retriever import HybridRetriever
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    st.error(f"Error loading RAG components: {e}")

# ============== Configuration ==============
CHROMA_DIR = os.path.join(project_root, "chroma_db")
DATA_DIR = os.path.join(project_root, "data")
DB_PATH = os.path.join(DATA_DIR, "mydatabase.db")

# ============== Session State ==============
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'router' not in st.session_state:
    st.session_state.router = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# ============== Helper Functions ==============
def initialize_rag():
    """Initialize the RAG system"""
    if not os.getenv("GROQ_API_KEY"):
        return False, "GROQ_API_KEY not found. Please add it to the .env file."

    try:
        # Initialize model router
        router = ModelRouter()

        # Check for existing vector store or create new one
        if os.path.exists(CHROMA_DIR):
            vectorstore = load_existing_vector_store(CHROMA_DIR)
        else:
            # Create empty vector store
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_texts(
                texts=["Welcome to Multi-Model RAG"],
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )

        # Initialize SQL agent if database exists
        sql_agent = None
        if os.path.exists(DB_PATH):
            try:
                sql_agent = create_sql_retriever(DB_PATH)
            except Exception as e:
                st.warning(f"Database connection failed: {e}")

        # Create hybrid retriever
        retriever = HybridRetriever(vectorstore, sql_agent)

        # Store in session state
        st.session_state.router = router
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever
        st.session_state.rag_initialized = True

        return True, "RAG system initialized successfully!"

    except Exception as e:
        return False, f"Initialization failed: {str(e)}"


def process_uploaded_files(files):
    """Process uploaded files and add to vector store"""
    if not files:
        return 0

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    all_documents = []

    for file in files:
        # Save uploaded file temporarily
        file_path = os.path.join(DATA_DIR, file.name)
        os.makedirs(DATA_DIR, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Load and process
        try:
            docs = load_documents([file_path])
            chunks = chunk_documents(docs)
            all_documents.extend(chunks)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if all_documents:
        # Create embeddings and add to vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load existing or create new
        if os.path.exists(CHROMA_DIR):
            vectorstore = load_existing_vector_store(CHROMA_DIR)
        else:
            vectorstore = Chroma.from_texts(
                texts=["Welcome"],
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )

        # Add new documents
        vectorstore.add_documents(all_documents)

        # Update session state
        st.session_state.vectorstore = vectorstore

        if st.session_state.retriever:
            st.session_state.retriever.vectorstore = vectorstore

        st.session_state.uploaded_files.extend([f.name for f in files])

    return len(all_documents)


def query_rag(user_query: str):
    """Query the RAG system"""
    if not st.session_state.rag_initialized:
        return "RAG system not initialized. Please check your API key."

    try:
        retriever = st.session_state.retriever
        router = st.session_state.router

        # Retrieve context
        results = retriever.hybrid_search(user_query)

        # Determine task type
        task_type = router.decide_task_type(user_query)

        # Format context
        context = retriever.format_context(results)

        # Generate response
        response = router.route_with_context(
            query=user_query,
            context=context,
            task_type=task_type
        )

        return response, task_type

    except Exception as e:
        return f"Error processing query: {str(e)}", "fast"


# ============== Sidebar ==============
with st.sidebar:
    st.markdown("### 🤖 Configuration")

    # API Key Status
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        st.success("✅ Groq API Key configured")
    else:
        st.error("❌ Groq API Key missing")
        st.markdown("""
        <small>Get your free API key from:</small>
        <a href="https://console.groq.com/keys" target="_blank">https://console.groq.com/keys</a>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Model Information
    st.markdown("### 📊 Active Models")

    models_info = [
        ("⚡ Fast", "llama-3.1-8b-instant", "Simple queries"),
        ("🧠 Reasoning", "mixtral-8x7b-32768", "Analysis & explanation"),
        ("💻 Code", "llama-3.1-70b-versatile", "Programming tasks"),
    ]

    for icon, model, desc in models_info:
        st.markdown(f"""
        <div class="sidebar-section">
            <div style="font-weight: 600; margin-bottom: 0.25rem;">{icon} {model}</div>
            <small style="color: #94a3b8;">{desc}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Statistics
    st.markdown("### 📈 Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.messages)}</div>
            <div class="stat-label">Messages</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.uploaded_files)}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Actions
    st.markdown("### 🔧 Actions")

    if st.button("🔄 Reinitialize RAG", use_container_width=True):
        with st.spinner("Initializing..."):
            success, msg = initialize_rag()
            if success:
                st.success(msg)
            else:
                st.error(msg)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("📁 Clear Documents", use_container_width=True):
        st.session_state.uploaded_files = []
        if os.path.exists(CHROMA_DIR):
            import shutil
            shutil.rmtree(CHROMA_DIR)
        st.session_state.rag_initialized = False
        st.rerun()


# ============== Main Content ==============
# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">
        <span>🤖</span>
        <span>Multi-Model RAG Assistant</span>
    </div>
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="color: #94a3b8; font-size: 0.85rem;">Powered by</span>
        <span style="background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">Groq API</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize RAG if not done
if not st.session_state.rag_initialized:
    if api_key and api_key != "your_groq_api_key_here":
        with st.spinner("Initializing RAG system..."):
            success, msg = initialize_rag()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    else:
        st.warning("⚠️ Please configure your Groq API key in the .env file")

# Upload Section
st.markdown("### 📁 Upload Documents")

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type=['pdf', 'txt', 'docx'],
    accept_multiple_files=True,
    help="Supported formats: PDF, TXT, DOCX"
)

if uploaded_files:
    col1, col2 = st.columns([3, 1])
    with col1:
        # Display uploaded file chips
        file_chips = " ".join([
            f'<span class="file-chip">📄 {f.name}</span>'
            for f in uploaded_files
        ])
        st.markdown(file_chips, unsafe_allow_html=True)

    with col2:
        if st.button("⚡ Process Files", type="primary"):
            with st.spinner("Processing documents..."):
                num_chunks = process_uploaded_files(uploaded_files)
                st.success(f"✅ Added {num_chunks} document chunks to knowledge base!")

# Show already uploaded files
if st.session_state.uploaded_files:
    st.markdown("**Loaded Documents:**")
    files_str = " ".join([
        f'<span class="file-chip">📄 {f}</span>'
        for f in set(st.session_state.uploaded_files)
    ])
    st.markdown(files_str, unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Chat Section
st.markdown("### 💬 Chat")

# Display chat messages
chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message chat-message-user">
                <div style="display: flex; align-items: flex-start;">
                    <div class="chat-avatar chat-avatar-user">👤</div>
                    <div>{message["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Check for task type badge
            task_type = message.get("task_type", "fast")
            badge_class = f"model-badge-{task_type}"
            badge_icon = {"reasoning": "🧠", "code": "💻", "fast": "⚡"}.get(task_type, "⚡")

            st.markdown(f"""
            <div class="chat-message chat-message-assistant">
                <div style="display: flex; align-items: flex-start;">
                    <div class="chat-avatar chat-avatar-assistant">🤖</div>
                    <div style="flex: 1;">
                        <div style="margin-bottom: 0.5rem;">{message["content"]}</div>
                        <span class="model-badge {badge_class}">{badge_icon} {task_type}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
st.markdown('<div class="input-container">', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Ask a question...",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        key="chat_input"
    )

with col2:
    send_button = st.button("Send", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Process user input
if user_input and send_button:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Query RAG
    with st.spinner("Thinking..."):
        result = query_rag(user_input)

        if isinstance(result, tuple):
            response, task_type = result
        else:
            response = result
            task_type = "fast"

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "task_type": task_type
    })

    # Rerun to display new messages
    st.rerun()

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #94a3b8;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h3>Welcome to Multi-Model RAG Assistant</h3>
        <p>Upload documents and start chatting to get answers from your knowledge base.</p>
        <p style="font-size: 0.85rem; margin-top: 1rem;">
            💡 <strong>Tips:</strong><br>
            • Upload PDF, TXT, or DOCX files above<br>
            • The system will automatically route your questions to the best model<br>
            • Use reasoning tasks for analysis, code tasks for programming
        </p>
    </div>
    """, unsafe_allow_html=True)

# Add some spacing at the bottom for the input
st.markdown("<br><br><br>", unsafe_allow_html=True)
