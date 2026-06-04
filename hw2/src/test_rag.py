import sys
import os
# Add the project root to sys.path so 'hw2' can be resolved
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hw2.src.rag import retrieve_context

def test_rag():
    print("Testing RAG System (Retrieval Only)...\n")
    
    query = "How is logSFR_HEC calculated?"
    print(f"Query: '{query}'\n")
    
    # Retrieve the context (top k chunks)
    context = retrieve_context(query)
    
    print("=== Retrieved Context ===")
    print(context)
    print("\n=========================")

if __name__ == "__main__":
    test_rag()
