"""Feature extraction module for document analysis."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import logging
from datetime import datetime

class DocumentFeatureExtractor:
    """Extracts numerical features from document analysis results."""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.min_variance_threshold = 1e-10
        self.min_total_variance = 1e-6
        
    def extract_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract all numerical features from a document analysis."""
        features = {}
        features.update(self._extract_metadata_features(doc_data))
        features.update(self._extract_binary_features(doc_data))
        features.update(self._extract_content_features(doc_data))
        features.update(self._extract_ooxml_features(doc_data))
        
        # Log feature values for debugging
        logging.debug("Extracted raw features:")
        for name, value in sorted(features.items()):
            logging.debug(f"  {name}: {value}")
            
        return features

    def _extract_metadata_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from metadata."""
        metadata = doc_data.get('metadata', {})
        features = {}
        
        # More granular metadata features
        features['title_length'] = len(metadata.get('title', ''))
        features['author_length'] = len(metadata.get('author', ''))
        features['company_length'] = len(metadata.get('company', ''))
        features['last_modified_by_length'] = len(metadata.get('last_modified_by', ''))
        
        # Creation and modification times as timestamps
        try:
            created = metadata.get('created')
            modified = metadata.get('last_modified')
            features['creation_time'] = float(datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").timestamp()) if created else 0
            features['modified_time'] = float(datetime.strptime(modified, "%Y-%m-%dT%H:%M:%SZ").timestamp()) if modified else 0
            features['time_between_create_modify'] = features['modified_time'] - features['creation_time']
        except:
            features['creation_time'] = 0
            features['modified_time'] = 0
            features['time_between_create_modify'] = 0
        
        # Revision number
        features['revision_number'] = float(metadata.get('revision', 0))
        
        return features
    
    def _extract_binary_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from binary analysis."""
        binary = doc_data.get('binary', {})
        features = {}
        
        # Path statistics
        for path_type in ['user_paths', 'template_paths', 'system_markers']:
            paths = binary.get(path_type, [])
            features[f'{path_type}_count'] = float(len(paths))
            if paths:
                features[f'{path_type}_min_length'] = float(min(len(p) for p in paths))
                features[f'{path_type}_max_length'] = float(max(len(p) for p in paths))
                features[f'{path_type}_avg_length'] = float(sum(len(p) for p in paths)) / len(paths)
            else:
                features[f'{path_type}_min_length'] = 0.0
                features[f'{path_type}_max_length'] = 0.0
                features[f'{path_type}_avg_length'] = 0.0
        
        # Binary signature features
        signatures = binary.get('binary_signatures', [])
        features['signature_count'] = float(len(signatures))
        if signatures:
            features['avg_signature_length'] = float(sum(len(s.get('signature', '')) for s in signatures)) / len(signatures)
        else:
            features['avg_signature_length'] = 0.0
        
        return features
    
    def _extract_content_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from content analysis."""
        content = doc_data.get('content', {})
        features = {}
        
        # Style and font statistics
        styles = content.get('styles', [])
        fonts = content.get('fonts_used', [])
        features['style_count'] = float(len(styles))
        features['font_count'] = float(len(fonts))
        
        # Text statistics
        text = content.get('text_content', '')
        if isinstance(text, str):
            features['text_length'] = float(len(text))
            features['word_count'] = float(len(text.split()))
            features['avg_word_length'] = float(sum(len(w) for w in text.split())) / max(len(text.split()), 1)
            features['unique_word_ratio'] = float(len(set(text.lower().split()))) / max(len(text.split()), 1)
        else:
            features['text_length'] = 0.0
            features['word_count'] = 0.0
            features['avg_word_length'] = 0.0
            features['unique_word_ratio'] = 0.0
        
        return features
    
    def _extract_ooxml_features(self, doc_data: Dict) -> Dict[str, float]:
        """Extract numerical features from OOXML analysis."""
        ooxml = doc_data.get('ooxml', {})
        features = {}
        
        # Structure statistics
        parts = ooxml.get('parts', [])
        relationships = ooxml.get('relationships', [])
        
        features['part_count'] = float(len(parts))
        features['relationship_count'] = float(len(relationships))
        
        if parts:
            features['avg_part_name_length'] = float(sum(len(p) for p in parts)) / len(parts)
            features['unique_part_ratio'] = float(len(set(parts))) / len(parts)
        else:
            features['avg_part_name_length'] = 0.0
            features['unique_part_ratio'] = 0.0
            
        if relationships:
            features['avg_relationship_length'] = float(sum(len(str(r)) for r in relationships)) / len(relationships)
            features['unique_relationship_ratio'] = float(len(set(str(r) for r in relationships))) / len(relationships)
        else:
            features['avg_relationship_length'] = 0.0
            features['unique_relationship_ratio'] = 0.0
        
        return features

    def _filter_features(self, X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Filter out low-variance features."""
        variances = np.var(X, axis=0)
        non_constant = variances > self.min_variance_threshold
        
        if not any(non_constant):
            logging.warning("All features have near-zero variance!")
            return X, list(range(X.shape[1]))
            
        kept_indices = np.where(non_constant)[0]
        X_filtered = X[:, non_constant]
        
        # Log variance information
        for idx, (var, keep) in enumerate(zip(variances, non_constant)):
            if keep:
                logging.debug(f"Keeping feature {self.feature_names[idx]}: variance = {var:.6f}")
            else:
                logging.debug(f"Dropping feature {self.feature_names[idx]}: variance = {var:.6f}")
                
        return X_filtered, kept_indices

    def fit_transform(self, reference_docs: List[Dict]) -> np.ndarray:
        """Fit the feature extractor and transform reference documents."""
        if not reference_docs:
            raise ValueError("No reference documents provided")

        # Extract initial features to get feature names
        features = self.extract_features(reference_docs[0])
        self.feature_names = list(features.keys())
        
        # Build feature matrix
        feature_matrix = []
        for doc in reference_docs:
            doc_features = self.extract_features(doc)
            feature_vector = [doc_features[f] for f in self.feature_names]
            feature_matrix.append(feature_vector)
            
        X = np.array(feature_matrix)
        logging.debug(f"Initial feature matrix shape: {X.shape}")
        
        # Filter low-variance features
        X_filtered, kept_indices = self._filter_features(X)
        self.feature_names = [self.feature_names[i] for i in kept_indices]
        logging.debug(f"Filtered feature matrix shape: {X_filtered.shape}")
        
        # Scale features
        self.scaler.fit(X_filtered)
        X_scaled = self.scaler.transform(X_filtered)
        logging.debug("Scaled feature variances:")
        for name, var in zip(self.feature_names, np.var(X_scaled, axis=0)):
            logging.debug(f"  {name}: {var:.6f}")
        
        # Try PCA if we have enough variance
        total_variance = np.sum(np.var(X_scaled, axis=0))
        if total_variance > self.min_total_variance:
            try:
                X_pca = self.pca.fit_transform(X_scaled)
                logging.debug(f"PCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")
                logging.debug(f"Explained variance ratios: {self.pca.explained_variance_ratio_}")
                return X_pca
            except Exception as e:
                logging.warning(f"PCA failed: {e}. Using scaled features.")
                return X_scaled
        else:
            logging.warning(f"Insufficient total variance ({total_variance:.6f}). Using scaled features.")
            return X_scaled

    def transform(self, doc_data: Dict) -> np.ndarray:
        """Transform a single document's features."""
        if not self.feature_names:
            raise ValueError("Feature extractor not fitted")
            
        # Extract and select features
        all_features = self.extract_features(doc_data)
        feature_vector = [all_features[f] for f in self.feature_names]
        X = np.array([feature_vector])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA if available
        if hasattr(self.pca, 'components_'):
            return self.pca.transform(X_scaled)
        return X_scaled
