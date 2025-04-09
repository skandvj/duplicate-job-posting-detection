# src/vector_search/vector_search.py
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Tuple, Optional

class VectorSearch:
    """Class for vector-based similarity search."""
    
    def __init__(self, dimension: int, index_type: str = 'flat'):
        """Initialize vector search with embedding dimension and index type."""
        self.dimension = dimension
        self.index_type = index_type
        self.job_ids = []
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the FAISS index."""
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
        else:
            # Default to flat
            self.index = faiss.IndexFlatL2(self.dimension)
            
    def add_batch(self, job_ids: List[int], embeddings: np.ndarray):
        """Add a batch of embeddings to the index."""
        logging.info(f"Adding {len(job_ids)} vectors to the index")
        self.index.add(embeddings)
        self.job_ids.extend(job_ids)
        logging.info(f"Index now contains {self.index.ntotal} vectors")
        
    def save_index(self, file_path: str):
        """Save the index to disk."""
        logging.info(f"Saving index to {file_path}.index")
        faiss.write_index(self.index, f"{file_path}.index")
        with open(f"{file_path}.ids", 'wb') as f:
            pickle.dump(self.job_ids, f)
        logging.info("Index saved successfully")
        
    def load_index(self, file_path: str):
        """Load the index from disk."""
        self.index = faiss.read_index(f"{file_path}.index")
        with open(f"{file_path}.ids", 'rb') as f:
            self.job_ids = pickle.load(f)
        logging.info(f"Loaded index with {self.index.ntotal} vectors")
        
    def search(self, query_embedding: np.ndarray, k: int = 10):
        """Search for similar vectors."""
        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        # Map indices to job IDs
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.job_ids):
                results.append((self.job_ids[idx], float(distances[0][i])))
                
        return results
        
    def find_duplicates(self, threshold: float = 0.85, batch_size: int = 1000, max_duplicates: int = 10000):
        """Find potential duplicates in batches with early stopping."""
        logging.info(f"Starting duplicate search with threshold {threshold}")
        duplicates = []
        total_vectors = self.index.ntotal
        
        # Process in batches to avoid memory issues
        for start_idx in range(0, min(10000, total_vectors), batch_size):
            end_idx = min(start_idx + batch_size, total_vectors)
            logging.info(f"Processing batch of vectors {start_idx} to {end_idx-1}")
            
            # For each vector in the batch
            for i in range(start_idx, end_idx):
                if i % 100 == 0:
                    logging.info(f"  Processing vector {i}")
                
                # Get the vector
                vector = self.index.reconstruct(i)
                vector = vector.reshape(1, -1)  # Reshape to 2D
                
                # Search for similar vectors
                distances, indices = self.index.search(vector, 20)  # Get more to filter
                
                # Filter results
                for j, idx in enumerate(indices[0][1:]):  # Skip first (self)
                    if idx <= i:  # Only keep one direction to avoid duplicates
                        continue
                        
                    distance = distances[0][j+1]
                    similarity = 1.0 - distance / 2  # Convert L2 distance to similarity
                    
                    if similarity >= threshold:
                        job_id1 = self.job_ids[i]
                        job_id2 = self.job_ids[idx]
                        duplicates.append((job_id1, job_id2, float(similarity)))
                        
                        if len(duplicates) % 100 == 0:
                            logging.info(f"Found {len(duplicates)} duplicates so far")
                            
                        # Early stopping if we found enough duplicates
                        if len(duplicates) >= max_duplicates:
                            logging.info(f"Reached maximum number of duplicates ({max_duplicates}). Stopping early.")
                            return duplicates
            
            # Log progress after each batch
            logging.info(f"Completed batch. Found {len(duplicates)} duplicates so far")
            
        return duplicates