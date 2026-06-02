import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Setup paths relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "documents")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")

def get_embeddings():
    """Initialize and return the HuggingFace embeddings model."""
    # Using all-MiniLM-L6-v2 which is lightweight and performant
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_documents():
    """Loads PDFs, chunks them, and stores the embeddings in Chroma."""
    print(f"Loading documents from {DOCS_DIR}...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()
    
    if not documents:
        print("No documents found to ingest.")
        return None

    print(f"Loaded {len(documents)} document pages.")

    # Chunk the documents
    # 1000 chunk size with 200 overlap is standard for PDFs to maintain context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    print("Chunking documents...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embedding model...")
    embeddings = get_embeddings()

    print(f"Storing embeddings to persistent vector store at {VECTOR_STORE_DIR}...")
    # Create or update the Chroma vector store with persistent directory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    
    print("Ingestion complete!")
    return vector_store

def get_vector_store():
    """Loads the existing persistent vector store without re-ingesting."""
    embeddings = get_embeddings()
    # Check if the persistence directory exists
    if not os.path.exists(VECTOR_STORE_DIR) or not os.listdir(VECTOR_STORE_DIR):
        print("Warning: Vector store directory is empty or does not exist. You may need to run ingest_documents().")
        
    return Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings
    )

def retrieve_context(query: str, k: int = 3) -> str:
    """
    Retrieves the top-k most relevant chunks for a given query and 
    concatenates them into a single string to be passed to the LLM.
    """
    db = get_vector_store()
    results = db.similarity_search(query, k=k)
    
    # Concatenate the content of the retrieved chunks into a single string
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context

if __name__ == "__main__":
    # If this script is run directly, perform the ingestion
    ingest_documents()
