# src/embeddings/embedder.py
import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
import pickle

class JobEmbedder:
    """Class for generating embeddings from job descriptions."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with embedding model name."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}
    
    def generate_embeddings(self, job_ids: List[int], texts: List[str]) -> Dict[int, np.ndarray]:
        """Generate embeddings for a list of job descriptions."""
        # Generate embeddings in batches
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Store embeddings with their job IDs
        for i, job_id in enumerate(job_ids):
            self.embeddings[job_id] = embeddings[i]
        
        return self.embeddings
    
    def save_embeddings(self, file_path: str) -> None:
        """Save embeddings to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def load_embeddings(self, file_path: str) -> Dict[int, np.ndarray]:
        """Load embeddings from a file."""
        with open(file_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        return self.embeddings