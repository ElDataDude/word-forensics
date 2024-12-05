"""Statistical calculator module for document similarity analysis."""

import numpy as np
from scipy.stats import norm
from typing import Dict
import logging

class StatisticalCalculator:
    """Handles statistical calculations for document similarity analysis."""
    
    @staticmethod
    def calculate_distance(features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate Euclidean distance between two feature vectors."""
        try:
            if features1.shape != features2.shape:
                raise ValueError(f"Feature shapes don't match: {features1.shape} vs {features2.shape}")
            return float(np.linalg.norm(features1 - features2))
        except Exception as e:
            logging.error(f"Error calculating distance: {e}")
            raise

    @staticmethod
    def calculate_pairwise_distances(features: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between all feature vectors."""
        try:
            n_samples = features.shape[0]
            distances = []
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    dist = StatisticalCalculator.calculate_distance(features[i], features[j])
                    if dist > 0:  # Only include non-zero distances
                        distances.append(dist)
            return np.array(distances)
        except Exception as e:
            logging.error(f"Error calculating pairwise distances: {e}")
            raise

    @staticmethod
    def calculate_statistics(target_distance: float, reference_distances: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures for similarity analysis."""
        try:
            if len(reference_distances) == 0:
                raise ValueError("No reference distances provided")
                
            # Basic statistics
            mean_dist = float(np.mean(reference_distances))
            std_dist = float(np.std(reference_distances)) if len(reference_distances) > 1 else 1.0
            
            # Z-score (positive means more similar than average)
            z_score = (mean_dist - target_distance) / std_dist if std_dist > 0 else 0
            
            # Percentile (what percentage of reference pairs are less similar)
            percentile = 100 * (np.sum(reference_distances > target_distance) / len(reference_distances))
            
            # Likelihood ratio based on z-score (higher z-score = higher likelihood of same origin)
            likelihood = norm.cdf(z_score) if z_score != 0 else 0.5
            
            # Log detailed statistics
            logging.debug("\nDetailed Statistical Results:")
            logging.debug(f"Target Distance: {target_distance:.3f}")
            logging.debug(f"Reference Statistics:")
            logging.debug(f"  Mean Distance: {mean_dist:.3f}")
            logging.debug(f"  Std Distance: {std_dist:.3f}")
            logging.debug(f"  Min Distance: {np.min(reference_distances):.3f}")
            logging.debug(f"  Max Distance: {np.max(reference_distances):.3f}")
            logging.debug(f"Calculated Metrics:")
            logging.debug(f"  Z-score: {z_score:.3f}")
            logging.debug(f"  Percentile: {percentile:.1f}%")
            logging.debug(f"  Likelihood Ratio: {likelihood*100:.1f}%")
            
            return {
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'z_score': z_score,
                'percentile': percentile,
                'likelihood_ratio': likelihood
            }
        except Exception as e:
            logging.error(f"Error calculating statistics: {e}")
            raise

    @staticmethod
    def interpret_results(stats: Dict[str, float], n_references: int) -> Dict[str, str]:
        """Generate human-readable interpretations of statistical results."""
        try:
            interpretations = {
                'percentile_interpretation': (
                    f"The documents are more similar than {stats['percentile']:.1f}% of reference pairs"
                ),
                'z_score_interpretation': (
                    f"The similarity is {abs(stats['z_score']):.1f} standard deviations "
                    f"{'more' if stats['z_score'] > 0 else 'less'} than average"
                ),
                'likelihood_interpretation': (
                    f"There is a {stats['likelihood_ratio']*100:.1f}% chance these documents "
                    "share an origin based on statistical analysis"
                ),
                'confidence_note': (
                    "Note: Statistical analysis is based on a limited reference set"
                    if n_references < 10 else ""
                )
            }
            return interpretations
        except Exception as e:
            logging.error(f"Error generating interpretations: {e}")
            raise
