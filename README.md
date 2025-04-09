# NLP Powered - Duplicate Job Posting Detector

This project implements a system for identifying duplicate job postings using text embeddings and vector search. It processes job descriptions, generates vector embeddings, builds a search index, and identifies potential duplicates based on textual similarity.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Environment Setup](#environment-setup)
- [Running the Project](#running-the-project)
- [Data Exploration Insights](#data-exploration-insights)
- [Embedding Model Selection](#embedding-model-selection)
- [Vector Search Implementation](#vector-search-implementation)
- [Evaluation Methodology](#evaluation-methodology)
- [API Documentation](#api-documentation)
- [Sample Results](#sample-results)

## Overview

Duplicate job postings can create confusion for job seekers and waste resources for employers. This project uses natural language processing and vector search techniques to identify potential duplicates with high accuracy. The system transforms job descriptions into vector embeddings, then uses efficient nearest-neighbor search algorithms to find similar postings.

## Project Structure

```
job-duplicate-detector/
├── data/                         # Store data files
├── notebooks/                    # Jupyter notebooks for EDA
├── src/                          # Source code
│   ├── data/                     # Data loading and processing
│   ├── embeddings/               # Embedding generation
│   ├── vector_search/            # Vector search implementation
│   ├── evaluation/               # Evaluation metrics
│   └── api/                      # REST API (bonus)
├── tests/                        # Unit tests
├── .env.example                  # Environment variables template
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Features

- **Data Processing**: Loads and preprocesses job posting data
- **Embedding Generation**: Converts job descriptions into vector embeddings
- **Vector Search**: Implements efficient similarity search using FAISS/HNSW
- **Duplicate Detection**: Identifies potential duplicate job postings
- **Similarity Threshold Analysis**: Tools to determine optimal similarity thresholds
- **REST API**: API endpoints for searching similar job postings (bonus)

## Environment Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local development without Docker)

### Setting Up with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/duplicate-job-posting-detection.git
   cd duplicate-job-posting-detection
   ```

2. Create an environment file from the template:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file to configure your environment (adjust paths and thresholds as needed).

4. Place the `jobs.csv.zip` file in the `data/` directory.

5. Build and run the Docker container:
   ```bash
   docker-compose up --build
   ```

### Local Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create an environment file and configure it:
   ```bash
   cp .env.example .env
   ```

4. Place the `jobs.csv.zip` file in the `data/` directory.

## Running the Project

### Data Preparation

Before running the main application, you should explore the data:

```bash
jupyter notebook eda.ipynb
```

This notebook provides insights into the dataset structure and characteristics.

### Running the Main Application

To run the main application without Docker:

```bash
python src/main.py
```

With Docker:

```bash
docker-compose up
```

### Running the API (Bonus)

To run the API server locally:

```bash
cd src/api
uvicorn app:app --reload
```

With Docker (the API is automatically started):

```bash
docker-compose up
```

Once started, the API is available at http://localhost:8000, with documentation at http://localhost:8000/docs.

## Data Exploration Insights

Key findings from data exploration:

1. **Dataset Size**: The dataset contains numerous job postings with varying text lengths.
2. **Text Length Distribution**: Job descriptions vary significantly in length, with some being very detailed and others quite brief.
3. **Common Terms**: The most common terms in job descriptions include "experience," "skills," "work," and various industry-specific terms.
4. **Exact Duplicates**: A small percentage of job postings are exact duplicates.
5. **Near-Duplicates**: Many job postings are near-duplicates with slight variations in wording or formatting.

## Embedding Model Selection

For this project, I chose the **Sentence-Transformers model (all-MiniLM-L6-v2)** for generating embeddings for the following reasons:

1. **Contextual Understanding**: The model captures semantic meaning rather than just lexical similarity, which is crucial for identifying similar job postings with different wording.
2. **Efficiency**: The MiniLM model offers a good balance between performance and computational efficiency, with 384-dimensional embeddings.
3. **Pre-trained on Diverse Data**: The model has been trained on a wide range of texts, including professional documents.
4. **Strong Performance on Semantic Tasks**: This model performs well on semantic similarity tasks, which aligns with our duplicate detection goal.
5. **Easy Integration**: The sentence-transformers library provides a simple interface for generating embeddings.

The embeddings are stored in a pickle file for quick retrieval, enabling efficient searching and comparison.

## Vector Search Implementation

The vector search is implemented using HNSW (Hierarchical Navigable Small World) for the following reasons:

1. **Efficiency**: HNSW provides approximate nearest neighbor search with logarithmic complexity.
2. **Accuracy**: It maintains high accuracy while being significantly faster than exact search methods.
3. **No Training Required**: Unlike some other methods, HNSW doesn't require a separate training phase.
4. **Memory Efficiency**: The hierarchical structure allows for efficient memory usage.

The `VectorSearch` class provides a clean interface for:
- Initializing and building indexes
- Adding and removing vectors
- Searching for similar vectors
- Finding potential duplicates
- Saving and loading indexes

## Evaluation Methodology

### Similarity Threshold Selection

The similarity threshold was determined through:

1. **Distribution Analysis**: Analyzing the distribution of similarity scores between different job postings.
2. **Precision-Recall Tradeoff**: Evaluating the precision-recall curve to find the optimal balance.
3. **Manual Review**: Reviewing samples of detected duplicates to validate the threshold.

The final threshold was set to 0.85, which:
- Minimizes false positives (non-duplicates classified as duplicates)
- Captures subtle variations in duplicate job postings
- Achieves a good balance between precision and recall

### Performance Metrics

The system's performance was evaluated using:
- **Precision**: Proportion of identified duplicates that are actually duplicates
- **Recall**: Proportion of actual duplicates that were correctly identified
- **F1 Score**: Harmonic mean of precision and recall

## API Documentation

The API provides the following endpoints:

- `GET /health`: Health check endpoint
- `GET /jobs/{job_id}`: Get job posting by ID
- `GET /jobs/{job_id}/similar`: Find similar job postings to a given job
- `GET /duplicates`: Find all duplicate job postings
- `POST /jobs`: Add a new job posting

For detailed API documentation, visit http://localhost:8000/docs when the API is running.

## Sample Results

Here's a sample of potential duplicate job postings detected by the system:

| Job ID 1 | Job ID 2 | Similarity Score |
|----------|----------|------------------|
| 12345    | 12789    | 0.95            |
| 23456    | 23987    | 0.93            |
| 34567    | 34123    | 0.91            |
| 45678    | 45321    | 0.89            |
| 56789    | 56432    | 0.87            |

(Note: These are example values. Actual results will depend on the dataset.)

---

## Implementation Details

### Key Components

1. **JobDataLoader**: Handles loading and preprocessing job posting data.
2. **JobEmbedder**: Generates and manages vector embeddings for job descriptions.
3. **VectorSearch**: Implements vector similarity search with HNSW/FAISS.
4. **DuplicateEvaluator**: Evaluates duplicate detection performance and helps select thresholds.

### Containerization

The project is containerized using Docker with the following components:
- A Python 3.10 base image
- Dependencies installed via requirements.txt
- Volume mounting for data persistence
- Environment variable configuration
- API port exposure (for the bonus feature)

## Further Improvements

Potential enhancements for the future:
1. **Clustering**: Implement clustering to group similar job postings
2. **Fine-tuned Embeddings**: Train embedding models specifically for job descriptions
3. **Incremental Updates**: Add functionality for incremental updates to the vector index
4. **User Interface**: Create a web-based UI for exploring duplicates
5. **Domain-Specific Features**: Incorporate industry and role-specific features for better matching

---

*Note: This project is for educational purposes only and should be adapted for production use with additional security and performance considerations.*
