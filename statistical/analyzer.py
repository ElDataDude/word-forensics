"""Main statistical analyzer module for document forensics."""

import logging
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from .calculator import StatisticalCalculator
from .feature_extractor import DocumentFeatureExtractor

class ForensicStatisticalAnalyzer:
    """Main class for statistical analysis of document forensics."""
    
    def __init__(self, reference_dir: Path, cache_dir: Optional[Path] = None):
        """Initialize the analyzer with reference directory and optional cache directory."""
        self.reference_dir = Path(reference_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.feature_extractor = DocumentFeatureExtractor()
        self.reference_docs = []
        self.reference_features = None
        self.analyzer = None

    def set_analyzer(self, analyzer: Any) -> None:
        """Set the document analyzer instance."""
        self.analyzer = analyzer

    def _analyze_reference_doc(self, docx_path: Path) -> Optional[Dict]:
        """Analyze a single reference document."""
        try:
            if not self.analyzer:
                raise ValueError("Analyzer not set")
            return self.analyzer.analyze_document(docx_path)
        except Exception as e:
            logging.warning(f"Failed to analyze reference document {docx_path}: {e}")
            return None

    def _load_or_create_reference_features(self) -> None:
        """Load cached reference features or create new ones."""
        if not self.analyzer:
            raise ValueError("Analyzer not set. Call set_analyzer() first.")
            
        if self.cache_dir:
            cache_file = self.cache_dir / "reference_features.joblib"

        # Process reference documents
        docx_files = list(self.reference_dir.glob('**/*.docx'))
        if not docx_files:
            raise ValueError(f"No .docx files found in {self.reference_dir}")
        
        # Analyze each reference document
        self.reference_docs = []
        for docx_path in docx_files:
            try:
                doc_data = self._analyze_reference_doc(docx_path)
                if doc_data:
                    self.reference_docs.append(doc_data)
            except Exception as e:
                logging.warning(f"Failed to analyze {docx_path}: {e}")
        
        if not self.reference_docs:
            raise ValueError("No valid reference documents found")
            
        # Extract features
        try:
            self.reference_features = self.feature_extractor.fit_transform(self.reference_docs)
            if self.reference_features is None:
                raise ValueError("Feature extraction returned None")
            if len(self.reference_features) == 0:
                raise ValueError("Feature extraction returned empty array")
                
            # Cache the results
            if self.cache_dir:
                cache_data = {
                    'features': self.reference_features,
                    'docs': self.reference_docs,
                    'feature_names': self.feature_extractor.feature_names,
                    'scaler': self.feature_extractor.scaler,
                    'pca': self.feature_extractor.pca
                }
                try:
                    joblib.dump(cache_data, cache_file)
                except Exception as e:
                    logging.warning(f"Failed to cache features: {e}")
                
        except Exception as e:
            logging.error(f"Failed to extract features: {e}")
            raise ValueError(f"Failed to extract features from reference documents: {e}")

    def analyze_similarity(self, target_data: Dict, same_origin_data: Dict) -> Dict[str, Any]:
        """Analyze statistical similarity between target and same-origin documents."""
        error_response = {
            'statistical_summary': {
                'similarity_percentile': 0.0,
                'z_score': 0.0,
                'likelihood_ratio': 0.0,
                'reference_sample_size': len(self.reference_docs) if self.reference_docs else 0
            },
            'interpretation': {
                'percentile_interpretation': "Statistical analysis failed",
                'z_score_interpretation': "Statistical analysis failed",
                'likelihood_interpretation': "Statistical analysis failed",
                'confidence_note': ""
            }
        }

        try:
            # Ensure we have reference features
            if self.reference_features is None:
                self._load_or_create_reference_features()
            
            if not self.reference_docs or len(self.reference_docs) < 3:
                msg = "Error: Need at least 3 reference documents for statistical analysis"
                logging.error(msg)
                error_response['interpretation']['confidence_note'] = msg
                return error_response

            # Transform documents
            target_features = self.feature_extractor.transform(target_data)
            same_origin_features = self.feature_extractor.transform(same_origin_data)
            
            if target_features is None or same_origin_features is None:
                raise ValueError("Feature transformation returned None")
            if len(target_features) == 0 or len(same_origin_features) == 0:
                raise ValueError("Feature transformation returned empty array")

            # Calculate distances using Euclidean distance
            target_same_origin_dist = StatisticalCalculator.calculate_distance(
                target_features[0], same_origin_features[0])
            
            # Calculate pairwise distances between reference documents
            reference_distances = StatisticalCalculator.calculate_pairwise_distances(self.reference_features)
            if len(reference_distances) == 0:
                raise ValueError("No valid reference distances calculated")

            # Calculate statistics
            stats = StatisticalCalculator.calculate_statistics(target_same_origin_dist, reference_distances)
            
            # Validate statistics
            required_keys = ['mean_distance', 'std_distance', 'z_score', 'percentile', 'likelihood_ratio']
            missing_keys = [k for k in required_keys if k not in stats]
            if missing_keys:
                raise ValueError(f"Missing statistics: {missing_keys}")

            # Generate interpretations
            interpretations = StatisticalCalculator.interpret_results(stats, len(self.reference_docs))
            
            # Get top contributing features
            try:
                if len(target_features[0]) == len(same_origin_features[0]):
                    feature_importance = pd.DataFrame({
                        'feature': self.feature_extractor.feature_names[:len(target_features[0])],
                        'importance': np.abs(target_features[0] - same_origin_features[0])
                    })
                    top_features = feature_importance.nlargest(5, 'importance')
                    # Convert numpy values to Python native types
                    top_features = top_features.astype({'importance': float})
                else:
                    logging.debug("Feature vectors have different lengths - skipping feature importance calculation")
                    top_features = pd.DataFrame()
            except Exception as e:
                logging.debug(f"Error calculating feature importance: {e}")
                top_features = pd.DataFrame()
            
            # Ensure all numpy values are converted to Python native types
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return [float(x) for x in obj]
                return obj
            
            # Convert PCA explained variance to native Python list
            pca_variance = []
            if hasattr(self.feature_extractor.pca, 'explained_variance_ratio_'):
                pca_variance = [float(x) for x in self.feature_extractor.pca.explained_variance_ratio_]
            
            return {
                'statistical_summary': {
                    'target_distance': float(target_same_origin_dist),
                    'mean_distance': float(stats['mean_distance']),
                    'std_distance': float(stats['std_distance']),
                    'z_score': float(stats['z_score']),
                    'similarity_percentile': float(stats['percentile']),
                    'likelihood_ratio': float(stats['likelihood_ratio']),
                    'reference_sample_size': int(len(self.reference_docs)),
                    'feature_count': int(len(self.feature_extractor.feature_names)),
                    'pca_dimensions': int(self.feature_extractor.pca.n_components_ if hasattr(self.feature_extractor.pca, 'n_components_') else 0)
                },
                'interpretation': interpretations,
                'top_contributing_features': [
                    {'feature': str(row['feature']), 'importance': float(row['importance'])}
                    for _, row in top_features.iterrows()
                ],
                'pca_explained_variance': pca_variance
            }
            
        except Exception as e:
            logging.error(f"Error in statistical analysis: {e}")
            error_response['interpretation']['confidence_note'] = f"Error: {str(e)}"
            return error_response
