"""
Multi-LLM RAG Application
Main entry point for the RAG system using Groq API
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import modules
from ingestion.document_loader import (
    load_documents,
    chunk_documents,
    create_vector_store,
    load_existing_vector_store
)
from ingestion.database_loader import create_sql_retriever, get_table_info
from llm.model_router import ModelRouter
from retrieval.retriever import HybridRetriever


# ============== Configuration ==============
# Update these paths as needed
PDF_FILES = [
    "data/sample.pdf",
    "data/document.txt",
]
DB_PATH = "data/mydatabase.db"
TABLE_NAME = "users"  # Optional: for direct table queries
CHROMA_DIR = "./chroma_db"


class MultiLLMRAG:
    """
    Multi-LLM RAG System using Groq API

    Features:
    - Document retrieval (PDF, TXT, DOCX)
    - SQL database querying
    - Model routing (reasoning/fast/code)
    - Hybrid search
    """

    def __init__(self, ingest_data: bool = True):
        """
        Initialize the RAG system

        Args:
            ingest_data: Whether to ingest documents (False to use existing vector store)
        """
        print("=" * 50)
        print("Initializing Multi-LLM RAG with Groq API")
        print("=" * 50)

        # Verify API key
        if not os.getenv("GROQ_API_KEY"):
            print("ERROR: GROQ_API_KEY not found in .env file")
            print("Get your API key from: https://console.groq.com/keys")
            sys.exit(1)

        # Initialize model router
        print("\n[1/4] Initializing model router...")
        self.router = ModelRouter()
        print("      Models loaded: llama-3.1-8b-instant, mixtral-8x7b, llama-3.1-70b")

        # Initialize vector store
        print("\n[2/4] Setting up vector store...")
        if ingest_data:
            self._ingest_documents()
        else:
            self._load_existing_store()

        # Initialize SQL agent (optional)
        print("\n[3/4] Setting up SQL agent...")
        if os.path.exists(DB_PATH):
            try:
                self.sql_agent = create_sql_retriever(DB_PATH, TABLE_NAME)
                db_info = get_table_info(DB_PATH)
                print(f"      Connected to database: {len(db_info['tables'])} tables")
                for table in db_info['tables']:
                    print(f"        - {table}")
            except Exception as e:
                print(f"      Warning: Could not connect to database: {e}")
                self.sql_agent = None
        else:
            print(f"      Database not found at {DB_PATH}")
            print(f"      Create a database to enable SQL queries")
            self.sql_agent = None

        # Initialize hybrid retriever
        print("\n[4/4] Setting up hybrid retriever...")
        self.retriever = HybridRetriever(self.vectorstore, self.sql_agent)

        print("\n" + "=" * 50)
        print("System initialized successfully!")
        print("=" * 50)

    def _ingest_documents(self):
        """Load and index documents"""
        existing_files = [f for f in PDF_FILES if os.path.exists(f)]

        if not existing_files:
            print("      No documents found to ingest.")
            print("      Place PDF, TXT, or DOCX files in the data/ folder")
            print("      Creating empty vector store...")
            # Create empty vector store
            from ingestion.document_loader import create_vector_store
            self.vectorstore = create_vector_store([], CHROMA_DIR)
            return

        print(f"      Loading {len(existing_files)} documents...")
        documents = load_documents(existing_files)

        print(f"      Chunking {len(documents)} documents...")
        chunks = chunk_documents(documents)

        print(f"      Creating vector embeddings...")
        self.vectorstore = create_vector_store(chunks, CHROMA_DIR)

    def _load_existing_store(self):
        """Load existing vector store"""
        if os.path.exists(CHROMA_DIR):
            self.vectorstore = load_existing_vector_store(CHROMA_DIR)
            print(f"      Loaded existing vector store")
        else:
            print(f"      No existing vector store found")
            print("      Run with ingest_data=True to create one")
            sys.exit(1)

    def query(self, user_query: str, use_sql: bool = None, task_type: str = None):
        """
        Query the RAG system

        Args:
            user_query: The user's question
            use_sql: Override SQL usage (None for auto-detect)
            task_type: Override task type (None for auto-detect)

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant context
        print(f"\nQuery: {user_query}")
        results = self.retriever.hybrid_search(user_query, use_sql)

        # Step 2: Determine task type if not specified
        if task_type is None:
            task_type = self.router.decide_task_type(user_query)

        # Step 3: Format context
        context = self.retriever.format_context(results)

        # Step 4: Generate response using appropriate model
        response = self.router.route_with_context(
            query=user_query,
            context=context,
            task_type=task_type
        )

        # Step 5: Extract sources
        sources = []
        if results["documents"]:
            for doc in results["documents"]:
                if hasattr(doc, 'metadata') and doc.metadata:
                    sources.append(doc.metadata)
                elif hasattr(doc, 'source'):
                    sources.append({"source": doc.source})

        return {
            "answer": response,
            "task_type": task_type,
            "used_sql": results["used_sql"],
            "sources": sources,
            "context_docs": len(results["documents"])
        }

    def add_documents(self, file_paths: list):
        """Add new documents to the vector store"""
        print(f"\nAdding {len(file_paths)} new documents...")
        documents = load_documents(file_paths)
        chunks = chunk_documents(documents)

        # Add to existing vector store
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        from langchain_community.vectorstores import Chroma
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

        print(f"Added {len(chunks)} new chunks to vector store")

    def get_database_info(self):
        """Get information about the connected database"""
        if os.path.exists(DB_PATH):
            return get_table_info(DB_PATH)
        return None


# ============== Main Execution ==============
def main():
    """Run example queries"""
    # Initialize RAG system
    rag = MultiLLMRAG(ingest_data=False)

    # Example queries
    print("\n" + "=" * 50)
    print("Example Queries")
    print("=" * 50)

    example_queries = [
        # "What is the total revenue in the database?",
        # "Explain the key concepts from the documents",
        # "Write a Python function to sort a list",
        "What information do you have?",
    ]

    for query in example_queries:
        result = rag.query(query)

        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        print(f"Task Type: {result['task_type']}")
        print(f"Used SQL: {result['used_sql']}")
        print(f"Documents Retrieved: {result['context_docs']}")
        print(f"\nAnswer:\n{result['answer']}")

        if result['sources']:
            print(f"\nSources:")
            for src in result['sources'][:3]:
                print(f"  - {src}")


def interactive_mode():
    """Run in interactive mode"""
    rag = MultiLLMRAG(ingest_data=False)

    print("\n" + "=" * 50)
    print("Multi-LLM RAG - Interactive Mode")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            query = input("\nYou: ").strip()

            if query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if not query:
                continue

            result = rag.query(query)

            print(f"\n[Task: {result['task_type']}]")
            print(f"Answer: {result['answer']}")

            if result['sources']:
                print(f"\nSources:")
                for src in result['sources'][:2]:
                    print(f"  - {src}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
