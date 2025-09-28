# rag_util.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

class RAGUtil:
    def __init__(self, data_path: str = None):
        """Initialize RAG utility with data path and load the model."""
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), '../data/freight_data.csv')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = self._load_and_process_data()
        self.chunks = []
        self.embeddings = None
        self._prepare_chunks()
        self._compute_embeddings()

    def _load_and_process_data(self) -> pd.DataFrame:
        """Load and process the freight data."""
        try:
            df = pd.read_csv(self.data_path)
            # Add Shipment_ID if not present
            if 'Shipment_ID' not in df.columns:
                df['Shipment_ID'] = df.index + 1
            return df
        except Exception as e:
            print(f"[RAGUtil] Error loading data: {e}")
            return pd.DataFrame()

    def _prepare_chunks(self):
        """Prepare text chunks from the freight data."""
        if self.df.empty:
            return

        # Create meaningful chunks from each shipment record
        for _, row in self.df.iterrows():
            chunk = (f"Shipment {row['Shipment_ID']}: A {row['Product_Type']} shipment from {row['Origin']} "
                    f"to {row['Destination']} on {row['Shipment_Date']} with {row['Weight_kg']:.2f}kg weight, "
                    f"{row['Volume_m3']:.2f}m³ volume, costing ${row['Cost_USD']:.2f}.")
            self.chunks.append({
                'text': chunk,
                'metadata': {
                    'shipment_id': row['Shipment_ID'],
                    'origin': row['Origin'],
                    'destination': row['Destination'],
                    'product_type': row['Product_Type'],
                    'cost': row['Cost_USD']
                }
            })

    def _compute_embeddings(self):
        """Compute embeddings for all chunks."""
        if not self.chunks:
            return
        texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant context chunks for a given query.
        
        Args:
            query: The user's query
            top_k: Number of most relevant chunks to retrieve
            
        Returns:
            List of dictionaries containing relevant chunks and their metadata
        """
        if not self.chunks or self.embeddings is None:
            return []

        # Compute query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute similarities
        similarities = np.inner(self.embeddings, query_embedding)
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            relevant_chunks.append({
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'similarity': float(similarities[idx])
            })
        
        return relevant_chunks

    def format_context_for_llm(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not relevant_chunks:
            return ""
        
        context = "Here is relevant information from our freight database:\n\n"
        for chunk in relevant_chunks:
            context += f"- {chunk['text']}\n"
        context += "\nBased on this context, "
        return context