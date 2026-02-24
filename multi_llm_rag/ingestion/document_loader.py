"""
Document Loader Module
Handles loading and chunking of documents (PDF, TXT, DOCX)
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_document(file_path: str):
    """Load a single document based on file extension"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return loader.load()


def load_documents(file_paths: list):
    """Load multiple documents"""
    documents = []
    for path in file_paths:
        try:
            docs = load_document(path)
            documents.extend(docs)
            print(f"Loaded: {path} ({len(docs)} pages)")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return documents


def chunk_documents(documents: list, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks, persist_directory="./chroma_db"):
    """Create Chroma vector store with HuggingFace embeddings"""
    # Using a lightweight but effective embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Vector store created with {len(chunks)} chunks")
    return vectorstore


def load_existing_vector_store(persist_directory="./chroma_db"):
    """Load an existing vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
