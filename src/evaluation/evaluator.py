# src/evaluation/evaluator.py
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

class DuplicateEvaluator:
    """Class for evaluating duplicate detection performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.similarity_scores = []
        self.is_duplicate = []
    
    def add_pair(self, similarity: float, duplicate: bool) -> None:
        """Add a pair with its similarity score and duplicate status."""
        self.similarity_scores.append(similarity)
        self.is_duplicate.append(duplicate)
    
    def add_pairs_batch(self, similarities: List[float], duplicates: List[bool]) -> None:
        """Add a batch of pairs."""
        self.similarity_scores.extend(similarities)
        self.is_duplicate.extend(duplicates)
    
    def analyze_threshold(self, visualize: bool = True) -> Dict:
        """Analyze similarity threshold for duplicate detection."""
        if not self.similarity_scores:
            return {}
        
        # Convert lists to numpy arrays
        similarities = np.array(self.similarity_scores)
        duplicates = np.array(self.is_duplicate, dtype=bool)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(duplicates, similarities)
        
        # Find the threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        # Calculate additional metrics
        average_precision = average_precision_score(duplicates, similarities)
        
        # Visualize if requested
        if visualize:
            plt.figure(figsize=(10, 8))
            
            # Plot threshold distribution
            plt.subplot(2, 1, 1)
            plt.hist(similarities, bins=50, alpha=0.7)
            plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
            plt.title('Distribution of Similarity Scores')
            plt.xlabel('Similarity Score')
            plt.ylabel('Count')
            plt.legend()
            
            # Plot precision-recall curve
            plt.subplot(2, 1, 2)
            plt.plot(recall, precision, label=f'AP: {average_precision:.3f}')
            plt.scatter(recall[best_idx], precision[best_idx], color='r', label=f'Best F1: {f1_scores[best_idx]:.3f}')
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('threshold_analysis.png')
        
        return {
            'best_threshold': float(best_threshold),
            'best_f1': float(f1_scores[best_idx]) if best_idx < len(f1_scores) else float(f1_scores[-1]),
            'average_precision': float(average_precision),
            'precision_at_best': float(precision[best_idx]) if best_idx < len(precision) else float(precision[-1]),
            'recall_at_best': float(recall[best_idx]) if best_idx < len(recall) else float(recall[-1])
        }
    
    def evaluate_threshold(self, threshold: float) -> Dict:
        """Evaluate performance at a specific threshold."""
        if not self.similarity_scores:
            return {}
        
        # Convert lists to numpy arrays
        similarities = np.array(self.similarity_scores)
        duplicates = np.array(self.is_duplicate, dtype=bool)
        
        # Make predictions using the threshold
        predictions = similarities >= threshold
        
        # Calculate metrics
        true_positives = np.sum(predictions & duplicates)
        false_positives = np.sum(predictions & ~duplicates)
        false_negatives = np.sum(~predictions & duplicates)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }