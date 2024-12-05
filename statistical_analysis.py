"""Statistical analysis module for Word document forensics."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import pandas as pd
from pathlib import Path
import json
import joblib
from typing import Dict, List, Tuple, Any, Optional
import logging

class DocumentFeatureExtractor:
    """Extracts numerical features from document analysis results."""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep components explaining 95% of variance
        
    def _extract_metadata_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from metadata."""
        metadata = doc_data.get('metadata_analysis', {})
        features = {}
        
        # Count non-empty metadata fields
        features['metadata_completeness'] = sum(1 for v in metadata.values() if v)
        
        # Binary features for presence of key metadata
        key_fields = ['author', 'last_modified_by', 'company', 'template']
        for field in key_fields:
            features[f'has_{field}'] = 1.0 if metadata.get(field) else 0.0
            
        return features
    
    def _extract_binary_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from binary analysis."""
        binary = doc_data.get('binary_analysis', {})
        features = {}
        
        # Path statistics
        for path_type in ['user_paths', 'template_paths', 'system_markers']:
            paths = binary.get(path_type, [])
            features[f'{path_type}_count'] = len(paths)
            features[f'{path_type}_avg_length'] = np.mean([len(p) for p in paths]) if paths else 0
        
        # Binary signature features
        signatures = binary.get('binary_signatures', [])
        features['signature_count'] = len(signatures)
        
        return features
    
    def _extract_content_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from content analysis."""
        content = doc_data.get('content_analysis', {})
        features = {}
        
        # Style and font statistics
        styles = content.get('styles', [])
        fonts = content.get('fonts_used', [])
        features['style_count'] = len(styles)
        features['font_count'] = len(fonts)
        
        # Text statistics
        text = content.get('text_content', '')
        features['text_length'] = len(text)
        features['word_count'] = len(text.split()) if isinstance(text, str) else 0
        
        return features
    
    def _extract_ooxml_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from OOXML analysis."""
        ooxml = doc_data.get('ooxml_analysis', {})
        features = {}
        
        # Structure statistics
        features['part_count'] = len(ooxml.get('parts', []))
        features['relationship_count'] = len(ooxml.get('relationships', []))
        
        return features
    
    def extract_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract all numerical features from a document analysis."""
        features = {}
        features.update(self._extract_metadata_features(doc_data))
        features.update(self._extract_binary_features(doc_data))
        features.update(self._extract_content_features(doc_data))
        features.update(self._extract_ooxml_features(doc_data))
        return features
    
    def fit_transform(self, reference_docs: List[Dict]) -> np.ndarray:
        """Fit the feature extractor and transform reference documents."""
        # Extract features from all reference documents
        feature_dicts = [self.extract_features(doc) for doc in reference_docs]
        
        # Create feature matrix
        self.feature_names = list(feature_dicts[0].keys())
        X = np.array([[d[f] for f in self.feature_names] for d in feature_dicts])
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        return X_pca
    
    def transform(self, doc_data: Dict) -> np.ndarray:
        """Transform a single document's features."""
        features = self.extract_features(doc_data)
        X = np.array([[features[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

class ForensicStatisticalAnalyzer:
    """Performs statistical analysis of document similarities."""
    
    def __init__(self, reference_dir: Path, cache_dir: Optional[Path] = None):
        self.reference_dir = Path(reference_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.feature_extractor = DocumentFeatureExtractor()
        self.reference_features = None
        self.reference_docs = []
        self.analyzer = None  # Will be set to the parent WordForensicAnalyzer
        
    def set_analyzer(self, analyzer):
        """Set the parent analyzer for processing reference documents."""
        self.analyzer = analyzer
    
    def _analyze_reference_doc(self, docx_path: Path) -> Dict:
        """Analyze a single reference document."""
        if not self.analyzer:
            raise ValueError("Parent analyzer not set. Call set_analyzer() first.")
        
        analysis = {
            "metadata_analysis": self.analyzer.extract_metadata(docx_path),
            "content_analysis": self.analyzer.analyze_content(docx_path),
            "ooxml_analysis": self.analyzer.analyze_ooxml_structure(docx_path),
            "binary_analysis": self.analyzer.analyze_binary_content(docx_path)
        }
        return analysis
    
    def _load_or_create_reference_features(self) -> Tuple[np.ndarray, List[Dict]]:
        """Load cached reference features or create new ones."""
        if self.cache_dir:
            cache_file = self.cache_dir / 'reference_features.joblib'
            if cache_file.exists():
                logging.info("Loading cached reference features")
                return joblib.load(cache_file)
        
        logging.info("Analyzing reference documents...")
        # Load and analyze all reference documents
        reference_docs = []
        for docx_file in self.reference_dir.glob('*.docx'):
            try:
                doc_data = self._analyze_reference_doc(docx_file)
                reference_docs.append(doc_data)
                logging.info(f"Analyzed reference document: {docx_file.name}")
            except Exception as e:
                logging.error(f"Error analyzing {docx_file}: {e}")
        
        # Check minimum number of reference documents
        if len(reference_docs) < 3:
            raise ValueError(f"Need at least 3 reference documents for statistical analysis, found {len(reference_docs)}")
        
        logging.info(f"Successfully analyzed {len(reference_docs)} reference documents")
        
        # Extract features
        reference_features = self.feature_extractor.fit_transform(reference_docs)
        
        # Cache results
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump((reference_features, reference_docs),
                       self.cache_dir / 'reference_features.joblib')
            logging.info("Cached reference features for future use")
        
        return reference_features, reference_docs
    
    def analyze_similarity(self, target_data: Dict, same_origin_data: Dict) -> Dict[str, Any]:
        """Analyze statistical similarity between target and same-origin documents."""
        try:
            # Load or create reference features
            if self.reference_features is None:
                self.reference_features, self.reference_docs = self._load_or_create_reference_features()
            
            # Transform target and same-origin documents
            target_features = self.feature_extractor.transform(target_data)
            same_origin_features = self.feature_extractor.transform(same_origin_data)
            
            # Calculate distances
            all_features = np.vstack([self.reference_features, target_features, same_origin_features])
            distances = pairwise_distances(all_features)
            
            # Get target-same_origin distance and all reference distances
            target_same_origin_dist = distances[-2, -1]
            reference_distances = distances[:-2, :-2]
            
            # Calculate statistics
            mean_ref_dist = np.mean(reference_distances[reference_distances > 0])
            std_ref_dist = np.std(reference_distances[reference_distances > 0])
            
            # Calculate z-score (how many standard deviations from mean)
            z_score = (mean_ref_dist - target_same_origin_dist) / std_ref_dist if std_ref_dist > 0 else 0
            
            # Calculate percentile of similarity
            all_distances = reference_distances[np.triu_indices_from(reference_distances, k=1)]
            if len(all_distances) > 0:
                percentile = 100 * (1 - (np.sum(all_distances <= target_same_origin_dist) / len(all_distances)))
            else:
                percentile = 0
            
            # Determine likelihood ratio
            # How many reference pairs are more similar than target-same_origin pair
            more_similar_pairs = np.sum(all_distances <= target_same_origin_dist)
            total_pairs = len(all_distances) if len(all_distances) > 0 else 1
            likelihood_ratio = (total_pairs - more_similar_pairs) / total_pairs
            
            # Get top contributing features
            feature_importance = pd.DataFrame({
                'feature': self.feature_extractor.feature_names,
                'importance': np.abs(self.feature_extractor.pca.components_[0])
            })
            top_features = feature_importance.nlargest(5, 'importance')
            
            return {
                'statistical_summary': {
                    'similarity_percentile': float(percentile),
                    'z_score': float(z_score),
                    'likelihood_ratio': float(likelihood_ratio),
                    'reference_sample_size': len(self.reference_docs)
                },
                'interpretation': {
                    'percentile_interpretation': f"The documents are more similar than {percentile:.1f}% of reference pairs",
                    'z_score_interpretation': f"The similarity is {abs(z_score):.1f} standard deviations {'more' if z_score > 0 else 'less'} than average",
                    'likelihood_interpretation': f"There is a {likelihood_ratio*100:.1f}% chance these documents share an origin based on statistical analysis",
                    'confidence_note': "Note: Statistical analysis is based on a limited reference set" if len(self.reference_docs) < 10 else ""
                },
                'top_contributing_features': top_features.to_dict('records'),
                'pca_explained_variance': list(self.feature_extractor.pca.explained_variance_ratio_)
            }
        except ValueError as ve:
            return {
                'error': str(ve),
                'status': 'insufficient_data',
                'recommendation': 'Add more reference documents to enable statistical analysis'
            }
        except Exception as e:
            logging.error(f"Statistical analysis error: {str(e)}")
            return {
                'error': str(e),
                'status': 'analysis_failed',
                'recommendation': 'Check the error message and ensure all documents are properly formatted'
            }
