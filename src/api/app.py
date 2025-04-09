# src/api/app.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
import logging
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import JobDataLoader
from embeddings.embedder import JobEmbedder
from vector_search.vector_search import VectorSearch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Job Duplicate Detector API",
    description="An API for finding duplicate job postings using text embeddings and vector search",
    version="1.0.0"
)

# Data models
class JobPostingBase(BaseModel):
    job_id: str  
    job_title: str
    job_description: str

class JobPostingCreate(JobPostingBase):
    pass

class JobPosting(JobPostingBase):
    similarity: Optional[float] = None

class SimilarityResponse(BaseModel):
    job_id: str  
    similar_jobs: List[JobPosting]

class DuplicatePair(BaseModel):
    job_id_1: str  
    job_id_2: str  
    similarity: float

class DuplicatesResponse(BaseModel):
    duplicate_pairs: List[DuplicatePair]
    total: int

# Global variables to store our components
data_loader = None
embedder = None
vector_search = None

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup."""
    global data_loader, embedder, vector_search
    
    try:
        # Get configuration from environment variables
        data_path = os.getenv('DATA_PATH', 'data/jobs.csv.zip')
        embeddings_path = os.getenv('EMBEDDINGS_PATH', 'data/embeddings.pkl')
        index_path = os.getenv('INDEX_PATH', 'data/faiss_index')
        
        # Step 1: Load and preprocess data
        logger.info("Loading and preprocessing job data...")
        data_loader = JobDataLoader(data_path)
        jobs_df = data_loader.preprocess()
        logger.info(f"Loaded {len(jobs_df)} job postings")
        
        # Step 2: Initialize embedder
        logger.info("Loading embedder...")
        embedder = JobEmbedder()
        
        # Try to load existing embeddings, generate if not found
        try:
            embeddings = embedder.load_embeddings(embeddings_path)
            logger.info(f"Loaded {len(embeddings)} existing embeddings")
        except (FileNotFoundError, EOFError):
            logger.info("No existing embeddings found, generating new ones")
            embeddings = embedder.generate_embeddings(
                job_ids=jobs_df['job_id'].tolist(),
                texts=jobs_df['clean_description'].tolist()
            )
            embedder.save_embeddings(embeddings_path)
            logger.info(f"Generated and saved {len(embeddings)} embeddings")
        
        # Step 3: Build vector index
        logger.info("Loading vector search index...")
        # Get embedding dimension from first embedding
        first_embedding = next(iter(embeddings.values()))
        dimension = len(first_embedding)
        
        vector_search = VectorSearch(dimension=dimension, index_type='hnsw')
        
        # Try to load existing index, build if not found
        try:
            vector_search.load_index(index_path)
            logger.info("Loaded existing vector index")
        except (FileNotFoundError, EOFError, RuntimeError):
            logger.info("No existing index found, building new one")
            # Prepare data for index
            job_ids = list(embeddings.keys())
            vectors = np.array(list(embeddings.values()))
            
            # Add to index
            vector_search.add_batch(job_ids, vectors)
            vector_search.save_index(index_path)
            logger.info("Built and saved vector search index")
            
        logger.info("API startup complete")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

@app.get("/jobs/{job_id}", summary="Get job posting by ID", response_model=JobPosting)
async def get_job(job_id: int):
    """Get a job posting by its ID."""
    global data_loader
    
    if data_loader is None:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    jobs_df = data_loader.data
    
    if jobs_df is None:
        raise HTTPException(status_code=503, detail="Job data not loaded")
    
    job = jobs_df[jobs_df['job_id'] == job_id]
    
    if job.empty:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    
    job_row = job.iloc[0]
    
    return JobPosting(
        job_id=job_row['job_id'],
        job_title=job_row['job_title'],
        job_description=job_row['job_description']
)

@app.get("/jobs/{job_id}/similar", summary="Find similar job postings", response_model=SimilarityResponse)
async def find_similar_jobs(
    job_id: int, 
    threshold: float = Query(0.8, description="Similarity threshold (0-1)"),
    limit: int = Query(10, description="Maximum number of similar jobs to return")
):
    """Find similar job postings to the given job ID."""
    global data_loader, embedder, vector_search
    
    if data_loader is None or embedder is None or vector_search is None:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    jobs_df = data_loader.data
    
    if jobs_df is None:
        raise HTTPException(status_code=503, detail="Job data not loaded")
    
    # Check if job_id exists
    job = jobs_df[jobs_df['job_id'] == job_id]
    
    if job.empty:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    
    # Get embedding for the job
    job_row = job.iloc[0]
    job_description = job_row['clean_description']
    
    # Generate embedding if not already in embedder
    if job_id not in embedder.embeddings:
        embedding = embedder.model.encode([job_description])[0]
    else:
        embedding = embedder.embeddings[job_id]
    
    # Search for similar jobs
    search_results = vector_search.search(embedding, k=limit+1)  # +1 to account for self
    
    # Filter by threshold and remove self
    similar_jobs = []
    for similar_id, distance in search_results:
        if similar_id != job_id:  # Skip self
            # Convert distance to similarity (for L2 distance)
            similarity = 1.0 - distance / 2
            
            if similarity >= threshold:
                similar_job = jobs_df[jobs_df['job_id'] == similar_id]
                
                if not similar_job.empty:
                    similar_row = similar_job.iloc[0]
                    similar_jobs.append(JobPosting(
                        job_id=int(similar_row['job_id']),
                        job_title=similar_row['job_title'],
                        job_description=similar_row['job_description'],
                        similarity=float(similarity)
                    ))
    
    return SimilarityResponse(
        job_id=job_id,
        similar_jobs=similar_jobs
    )

@app.get("/duplicates", summary="Find all duplicate job postings", response_model=DuplicatesResponse)
async def find_duplicates(
    threshold: float = Query(0.85, description="Similarity threshold (0-1)"),
    limit: int = Query(100, description="Maximum number of duplicate pairs to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Find all duplicate job postings based on the similarity threshold."""
    global vector_search, data_loader
    
    if vector_search is None or data_loader is None:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    # Find duplicates
    all_duplicates = vector_search.find_duplicates(threshold=threshold)
    
    # Sort by similarity (descending)
    all_duplicates.sort(key=lambda x: x[2], reverse=True)
    
    # Apply pagination
    paginated_duplicates = all_duplicates[offset:offset+limit]
    
    # Format response
    duplicate_pairs = [
        DuplicatePair(job_id_1=id1, job_id_2=id2, similarity=similarity)
        for id1, id2, similarity in paginated_duplicates
    ]
    
    return DuplicatesResponse(
        duplicate_pairs=duplicate_pairs,
        total=len(all_duplicates)
    )

@app.get("/duplicates", summary="Find all duplicate job postings", response_model=DuplicatesResponse)
async def find_duplicates(
    threshold: float = Query(0.85, description="Similarity threshold (0-1)"),
    limit: int = Query(100, description="Maximum number of duplicate pairs to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Find all duplicate job postings based on the similarity threshold."""
    global vector_search, data_loader
    
    if vector_search is None or data_loader is None:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    try:
        # Find duplicates with a smaller batch size to avoid memory issues
        all_duplicates = vector_search.find_duplicates(
            threshold=threshold, 
            batch_size=500,
            max_duplicates=10000  # Limit to avoid excessive processing
        )
        
        # Sort by similarity (descending)
        all_duplicates.sort(key=lambda x: x[2], reverse=True)
        
        # Ensure offset is within bounds
        total_duplicates = len(all_duplicates)
        safe_offset = min(offset, total_duplicates-1) if total_duplicates > 0 else 0
        
        # Apply pagination
        end_index = min(safe_offset + limit, total_duplicates)
        paginated_duplicates = all_duplicates[safe_offset:end_index] if total_duplicates > 0 else []
        
        # Format response
        duplicate_pairs = [
            DuplicatePair(job_id_1=id1, job_id_2=id2, similarity=float(similarity))
            for id1, id2, similarity in paginated_duplicates
        ]
        
        return DuplicatesResponse(
            duplicate_pairs=duplicate_pairs,
            total=total_duplicates
        )
    
    except Exception as e:
        logger.error(f"Error in duplicates endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/jobs", summary="Add a new job posting", status_code=201)
# async def add_job(job: JobPostingCreate):
#     """Add a new job posting and find potential duplicates."""
#     global data_loader, embedder, vector_search
    
#     if data_loader is None or embedder is None or vector_search is None:
#         raise HTTPException(status_code=503, detail="Service not fully initialized")
    
#     # Clean the job description
#     clean_description = data_loader._clean_text(job.job_description)
    
#     # Generate embedding
#     embedding = embedder.model.encode([clean_description])[0]
    
#     # Add to embeddings
#     embedder.embeddings[job.job_id] = embedding
    
#     # Add to index
#     vector_search.add_batch([job.job_id], np.array([embedding]))
    
#     # Add to dataframe
#     new_row = {
#         'job_id': job.job_id,
#         'job_title': job.job_title,
#         'job_description': job.job_description,
#         'clean_description': clean_description
#     }
#     data_loader.data = data_loader.data.append(new_row, ignore_index=True)
    
#     return {"message": "Job posting added successfully", "job_id": job.job_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)