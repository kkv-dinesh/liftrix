"""
Test script to verify RAG implementation
"""

from rag_util import RAGUtil
import pandas as pd

def test_rag():
    # Initialize RAG
    rag = RAGUtil()
    
    # Test queries
    test_queries = [
        "Tell me about shipments from Hamburg to Berlin",
        "What are the most expensive ethanol shipments?",
        "Show me shipments with high volume",
        "What's the typical cost for shipping between Rotterdam and Paris?"
    ]
    
    print("Testing RAG Implementation...")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Get relevant chunks
        relevant_chunks = rag.retrieve_relevant_context(query, top_k=3)
        
        print("Retrieved Context:")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"\n{i}. Relevance Score: {chunk['similarity']:.3f}")
            print(f"   Context: {chunk['text']}")
            print(f"   Metadata: {chunk['metadata']}")
        
        print("\n" + "-" * 30)

if __name__ == "__main__":
    test_rag()