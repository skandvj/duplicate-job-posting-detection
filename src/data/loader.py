# src/data/loader.py
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

class JobDataLoader:
    """Class for loading and preprocessing job posting data."""
    
    def __init__(self, data_path: str):
        """Initialize with path to data file."""
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load job posting data."""
        # Check if file is zipped
        if self.data_path.endswith('.zip'):
            self.data = pd.read_csv(self.data_path, compression='zip')
        else:
            self.data = pd.read_csv(self.data_path)
        return self.data
    
    def preprocess(self) -> pd.DataFrame:
        """Preprocess job data."""
        if self.data is None:
            self.load_data()
        
        # Remove rows with missing job descriptions
        self.data = self.data.dropna(subset=['jobDescRaw'])
        
        # Clean text (basic preprocessing)
        self.data['clean_description'] = self.data['jobDescRaw'].apply(self._clean_text)
        
        return self.data
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = text.lower()
        # Additional cleaning steps can be added here
        
        return text