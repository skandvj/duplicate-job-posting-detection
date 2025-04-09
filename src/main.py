# src/main.py
import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from data.loader import JobDataLoader
from embeddings.embedder import JobEmbedder
from vector_search.vector_search import VectorSearch
from evaluation.evaluator import DuplicateEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Main function to run the duplicate job posting detector."""
    # Get configuration from environment variables
    data_path = os.getenv('DATA_PATH', 'data/jobs.csv.zip')
    embeddings_path = os.getenv('EMBEDDINGS_PATH', 'data/embeddings.pkl')
    index_path = os.getenv('INDEX_PATH', 'data/faiss_index')
    similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))
    
    # Step 1: Load and preprocess data
    logger.info("Loading and preprocessing job data...")
    loader = JobDataLoader(data_path)
    jobs_df = loader.preprocess()
    logger.info(f"Loaded {len(jobs_df)} job postings")
    
    # Step 2: Generate embeddings
    logger.info("Generating embeddings...")
    embedder = JobEmbedder()
    
    # Try to load existing embeddings, generate if not found
    try:
        embeddings = embedder.load_embeddings(embeddings_path)
        logger.info(f"Loaded {len(embeddings)} existing embeddings")
    except (FileNotFoundError, EOFError):
        logger.info("No existing embeddings found, generating new ones")
        embeddings = embedder.generate_embeddings(
            job_ids=jobs_df['lid'].tolist(),
            texts=jobs_df['clean_description'].tolist()
        )
        embedder.save_embeddings(embeddings_path)
        logger.info(f"Generated and saved {len(embeddings)} embeddings")

    
    # When building the vector search
    logger.info("Building vector search index...")
    # Get embedding dimension from first embedding
    first_embedding = next(iter(embeddings.values()))
    dimension = len(first_embedding)
    
    vector_search = VectorSearch(dimension=dimension, index_type='flat')  # Use flat for simplicity first
    
    # Prepare data for index
    logger.info("Preparing vectors for indexing...")
    job_ids = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()))
    logger.info(f"Prepared {len(vectors)} vectors with dimension {dimension}")
    
    # Add to index
    logger.info("Adding vectors to index...")
    vector_search.add_batch(job_ids, vectors)
    vector_search.save_index(index_path)
    logger.info("Built and saved vector search index")
    
    # Step 4: Find duplicate job postings - with a high threshold to start
    logger.info(f"Finding duplicate job postings with threshold {similarity_threshold}...")
    duplicates = vector_search.find_duplicates(threshold=similarity_threshold, 
                                               batch_size=500,
                                               max_duplicates=5000)  # Limit for testing
    logger.info(f"Found {len(duplicates)} potential duplicate pairs")


    try:
        vector_search.load_index(index_path)
        logger.info("Loaded existing vector index")
    except (FileNotFoundError, EOFError, RuntimeError):  # Add RuntimeError to catch FAISS errors
        logger.info("No existing index found, building new one")
        # Prepare data for index
        job_ids = list(embeddings.keys())
        vectors = np.array(list(embeddings.values()))
    
    # Add to index
    vector_search.add_batch(job_ids, vectors)
    vector_search.save_index(index_path)
    logger.info("Built and saved vector search index")
    
    # Step 4: Find duplicate job postings
    logger.info(f"Finding duplicate job postings with threshold {similarity_threshold}...")
    duplicates = vector_search.find_duplicates(threshold=similarity_threshold)
    logger.info(f"Found {len(duplicates)} potential duplicate pairs")
    
    # Step 5: Save results
    if duplicates:
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(duplicates, columns=['job_id_1', 'job_id_2', 'similarity'])
        
        # Sort by similarity (descending)
        results_df = results_df.sort_values('similarity', ascending=False)
        
        # Save to CSV
        results_df.to_csv('data/duplicate_jobs.csv', index=False)
        logger.info("Saved duplicate pairs to data/duplicate_jobs.csv")
        
        # Display sample of duplicates
        print("\nSample of potential duplicate job postings:")
        print(results_df.head(10))
    else:
        logger.info("No duplicates found with the current threshold")

if __name__ == "__main__":
    main()